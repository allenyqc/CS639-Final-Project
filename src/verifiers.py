"""
Rule-based verifiers for LLM-generated ML code.

Four general-purpose checks covering the major categories of ML code defects:
  1. check_data_separation  — train/test boundary violations (leakage)
  2. check_test_isolation   — test set used beyond final evaluation
  3. check_evaluation_rigor — wrong metrics or flawed evaluation protocol
  4. check_code_safety      — security and Python hygiene issues
"""

from __future__ import annotations

import re
from typing import Callable


# ---------------------------------------------------------------------------
# Lightweight AST helpers
# ---------------------------------------------------------------------------

def _line_of(code: str, pattern: str, flags: int = 0) -> int | None:
    """Return the 1-based line number of the first regex match, or None."""
    m = re.search(pattern, code, flags)
    if m is None:
        return None
    return code[: m.start()].count("\n") + 1


def _all_lines_of(code: str, pattern: str, flags: int = 0) -> list[int]:
    """Return all 1-based line numbers where pattern matches."""
    return [code[: m.start()].count("\n") + 1 for m in re.finditer(pattern, code, flags)]


def _split_line(code: str) -> int | None:
    """Find the line where train/test splitting occurs."""
    for pat in [
        r'train_test_split\(',
        r'TimeSeriesSplit\(',
        r'GroupShuffleSplit\(',
        r'GroupKFold\(',
        r'StratifiedKFold\(',
        r'KFold\(',
    ]:
        ln = _line_of(code, pat)
        if ln is not None:
            return ln
    # Manual split heuristics: X_train = ..., iloc[:
    ln = _line_of(code, r'X_train\s*=')
    return ln


def _has_time_context(code: str) -> bool:
    """True if code is clearly a time-series / temporal prediction task."""
    strong = bool(re.search(
        r'TimeSeriesSplit|timeseries|time_series'
        r'|lag[_\s(].*shift|rolling\s*\(|\.shift\(|ewm\('
        r'|forecast|MACD|RSI|Bollinger|stock.*price'
        r'|temporal_split|chronolog',
        code, re.IGNORECASE,
    ))
    if strong:
        return True
    has_date_col = bool(re.search(r'["\']date["\']|\.date|_date\b', code))
    has_lag_features = bool(re.search(r'lag|rolling|shift\(', code, re.IGNORECASE))
    return has_date_col and has_lag_features


def _has_group_context(code: str) -> bool:
    return bool(re.search(
        r'patient_id|user_id|subject_id|group_id|customer_id'
        r'|participant_id|student_id|client_id',
        code, re.IGNORECASE,
    ))


# ---------------------------------------------------------------------------
# 1. DATA SEPARATION — train/test boundary violations
# ---------------------------------------------------------------------------

def check_data_separation(code: str) -> str | None:
    """
    Detect preprocessing or feature engineering that violates train/test
    separation.  Covers: data leakage, preprocessing order, TF-IDF/embedding
    leakage, augmentation leakage, group leakage, temporal leakage, and
    stacking leakage.
    """
    split_ln = _split_line(code)

    # --- Temporal leakage: time context + random shuffle ---
    if _has_time_context(code):
        uses_tts = bool(re.search(r'train_test_split\(', code))
        has_shuffle_false = bool(re.search(r'shuffle\s*=\s*False', code))
        has_temporal_split = bool(re.search(
            r'TimeSeriesSplit|\.iloc\s*\[\s*:|\.loc\s*\[.*:.*cutoff', code))
        if uses_tts and not has_shuffle_false and not has_temporal_split:
            return (
                "data_separation: time-series context uses train_test_split "
                "without shuffle=False; use a chronological cutoff or TimeSeriesSplit"
            )

    # --- Group leakage: grouped entities + non-group-aware split ---
    if _has_group_context(code):
        has_group_split = bool(re.search(
            r'GroupKFold|GroupShuffleSplit|LeaveOneGroupOut'
            r'|groups\s*=', code))
        has_plain_split = bool(re.search(
            r'train_test_split\(|(?<!Group)KFold\(|StratifiedKFold\(', code))
        if has_plain_split and not has_group_split:
            return (
                "data_separation: data contains entity groups (patient/user) "
                "but split ignores group boundaries; use GroupKFold or pass groups="
            )

    # --- Fit-before-split: .fit() on full data before the split point ---
    if split_ln is not None:
        lines = code.split("\n")
        fit_lines = _all_lines_of(
            code, r'\.(fit_transform|fit_resample|fit)\(')
        for fl in fit_lines:
            if fl >= split_ln:
                break
            line_text = lines[fl - 1] if fl <= len(lines) else ""
            if re.search(r'X_train|X_val|x_train|train_', line_text):
                continue
            if re.search(r'LabelEncoder|label_enc|le_', line_text, re.IGNORECASE):
                continue
            return (
                "data_separation: .fit() called before train/test split on "
                "non-training data; split first, then fit only on training set"
            )

    # --- TF-IDF / Vectorizer on full corpus before split ---
    if split_ln and re.search(r'Vectorizer|CountVec|Tfidf', code):
        vec_fit_lines = _all_lines_of(
            code, r'(?:vectorizer|vec|tfidf|count_vec)\w*\.fit', re.IGNORECASE)
        for vl in vec_fit_lines:
            if vl < split_ln:
                return (
                    "data_separation: text vectorizer fitted on full corpus "
                    "before split; fit only on training documents"
                )

    # --- Embedding trained on full corpus before split ---
    if split_ln and re.search(r'Word2Vec|FastText|Doc2Vec', code):
        emb_lines = _all_lines_of(code, r'Word2Vec\(|FastText\(|Doc2Vec\(')
        for el in emb_lines:
            if el < split_ln:
                return (
                    "data_separation: word embeddings trained on full corpus "
                    "before split; train embeddings only on training documents"
                )

    # --- Augmentation before split ---
    if split_ln and re.search(r'augment|ImageDataGenerator|albumentations',
                              code, re.IGNORECASE):
        lines = code.split("\n")
        aug_lines = _all_lines_of(code, r'augment', re.IGNORECASE)
        for al in aug_lines:
            if al < split_ln:
                line_text = lines[al - 1] if al <= len(lines) else ""
                if re.search(r'^\s*(def |import |#|class |""")', line_text):
                    continue
                return (
                    "data_separation: data augmentation applied before split; "
                    "augment only the training set after splitting"
                )

    # --- Stacking leakage: .predict(X_train) without out-of-fold ---
    if re.search(r'stack|meta.?learner|meta.?model|blending', code, re.IGNORECASE):
        has_train_pred = bool(re.search(
            r'\.predict(?:_proba)?\(\s*X_train\b', code))
        has_oof = bool(re.search(
            r'cross_val_predict|KFold|StratifiedKFold', code))
        if has_train_pred and not has_oof:
            return (
                "data_separation: stacking uses in-sample predictions as "
                "meta-features; use cross_val_predict or a fold loop for OOF"
            )

    # --- Direct fit on full dataframe variable (df, data, X, features) ---
    if re.search(
        r'\.fit(?:_transform|_resample)?\(\s*(?:df|data|dataset|features'
        r'|X_full|X_all|X\s*[\),])',
        code,
    ):
        if not re.search(r'LabelEncoder', code):
            return (
                "data_separation: preprocessor fitted on full/unnamed data; "
                "fit only on the training partition"
            )

    return None


# ---------------------------------------------------------------------------
# 2. TEST ISOLATION — test set used beyond final evaluation
# ---------------------------------------------------------------------------

def check_test_isolation(code: str) -> str | None:
    """
    Detect test data being used for training, tuning, early stopping, or
    calibration.  Covers: test misuse, early stopping leak, calibration leak.
    """
    # --- .fit() on test data ---
    if re.search(r'\.fit\s*\(\s*X_test', code):
        return (
            "test_isolation: model or transformer fitted on X_test; "
            "test data must only be used for final evaluation"
        )

    # --- Threshold / hyperparameter selection on test labels ---
    if re.search(r'y_test', code):
        threshold_on_test = (
            re.search(r'(precision_recall_curve|roc_curve)\s*\(\s*y_test', code)
            or re.search(r'best_threshold.*y_test|optimal.*threshold.*y_test', code)
            or (re.search(r'for\s+\w+\s+in\s+\w*[Tt]hreshold', code)
                and re.search(r'y_test', code))
        )
        if threshold_on_test:
            return (
                "test_isolation: threshold selected using y_test; "
                "tune thresholds on a validation split, not the test set"
            )

    # --- Early stopping on test data ---
    if re.search(r'validation_data\s*=\s*[\(\[]?\s*X_test', code):
        return (
            "test_isolation: test data used as validation_data for early "
            "stopping; create a separate validation split"
        )
    if re.search(r'early_stop|EarlyStopping|patience', code, re.IGNORECASE):
        uses_test = bool(re.search(r'X_test|y_test', code))
        uses_val = bool(re.search(r'X_val|y_val|X_valid|val_data', code))
        if uses_test and not uses_val:
            return (
                "test_isolation: early stopping monitors test set; "
                "use a separate validation split for early stopping"
            )

    # --- Calibration on test data ---
    if re.search(r'calibrat', code, re.IGNORECASE):
        if re.search(r'\.fit\s*\(\s*X_test', code):
            return (
                "test_isolation: calibration model fitted on test data; "
                "use a held-out calibration set from training data"
            )

    return None


# ---------------------------------------------------------------------------
# 3. EVALUATION RIGOR — metric/protocol problems
# ---------------------------------------------------------------------------

def check_evaluation_rigor(code: str) -> str | None:
    """
    Detect evaluation methodology flaws: wrong metrics for the domain,
    reporting only training performance, missing holdout after hyperparameter
    search, or uncorrected multiple comparisons.
    """
    # --- Survival data with classification accuracy ---
    if re.search(r'survival|censored|hazard|cox|lifelines', code, re.IGNORECASE):
        has_accuracy = bool(re.search(r'accuracy_score', code))
        has_proper = bool(re.search(
            r'concordance_index|c_index|brier_score|log_rank', code, re.IGNORECASE))
        if has_accuracy and not has_proper:
            return (
                "evaluation_rigor: accuracy used on survival/censored data; "
                "use concordance index or time-dependent AUC"
            )

    # --- Multi-label with only single-label accuracy ---
    if re.search(r'multi.?label|OneVsRest|MultiOutput|label_binarize',
                 code, re.IGNORECASE):
        has_accuracy_only = bool(re.search(r'accuracy_score', code))
        has_ml_metric = bool(re.search(
            r'hamming_loss|jaccard_score|average.*samples'
            r'|average.*macro|classification_report', code))
        if has_accuracy_only and not has_ml_metric:
            return (
                "evaluation_rigor: single-label accuracy on multi-label data; "
                "use hamming loss, sample-averaged F1, or macro F1"
            )

    # --- Ranking task with classification/regression metrics ---
    if re.search(r'ndcg|learn.?to.?rank|relevance.*score|retrieval',
                 code, re.IGNORECASE):
        has_wrong = bool(re.search(r'accuracy_score|mean_squared_error', code))
        has_rank_metric = bool(re.search(
            r'ndcg_score|dcg_score|mean_reciprocal|precision_at_k', code))
        if has_wrong and not has_rank_metric:
            return (
                "evaluation_rigor: classification/regression metric on ranking "
                "task; use NDCG, MAP, or MRR"
            )

    # --- GridSearchCV best_score_ as final result without holdout ---
    if re.search(r'GridSearchCV|RandomizedSearchCV', code):
        reports_inner = bool(re.search(r'best_score_', code))
        has_holdout = bool(re.search(
            r'\.(score|predict)\s*\(\s*X_(test|hold)'
            r'|_score\s*\(\s*y_test', code))
        if reports_inner and not has_holdout:
            return (
                "evaluation_rigor: GridSearchCV best_score_ reported as final "
                "result; evaluate on a separate held-out test set"
            )

    # --- Multiple statistical tests without correction ---
    stat_tests = re.findall(
        r'(?:ttest_ind|ttest_rel|mannwhitneyu|wilcoxon|chi2_contingency'
        r'|kruskal|ranksums|ks_2samp|fisher_exact|proportions_ztest)\s*\(',
        code,
    )
    if len(stat_tests) >= 2:
        has_correction = bool(re.search(
            r'multipletests|bonferroni|holm|benjamini|fdrcorrection'
            r'|alpha\s*/\s*\d',
            code, re.IGNORECASE))
        if not has_correction:
            return (
                "evaluation_rigor: multiple statistical tests without correction "
                "for multiple comparisons; apply Bonferroni or BH correction"
            )

    # --- Only accuracy, no other metric (with imbalanced hint) ---
    if re.search(r'accuracy_score', code):
        has_other = bool(re.search(
            r'f1_score|roc_auc|precision_score|recall_score'
            r'|classification_report|matthews_corrcoef|cohen_kappa'
            r'|mean_squared_error|r2_score', code))
        only_train = (
            bool(re.search(r'\.(score|predict)\s*\(\s*X_train', code))
            and not re.search(r'\.(score|predict)\s*\(\s*X_(test|val)', code)
        )
        if only_train:
            return (
                "evaluation_rigor: model evaluated only on training data; "
                "report held-out test performance"
            )

    return None


# ---------------------------------------------------------------------------
# 4. CODE SAFETY — security and Python hygiene
# ---------------------------------------------------------------------------

def check_code_safety(code: str) -> str | None:
    """
    Detect security vulnerabilities and Python anti-patterns: hardcoded
    credentials, unsafe deserialization, mutable defaults, broad exceptions.
    """
    # --- Hardcoded credentials ---
    cred_patterns = [
        r'["\'](?:sk-|api[_-]?key|bearer\s)[A-Za-z0-9_\-]{10,}["\']',
        r'(?:password|passwd|secret|token|api_key)\s*=\s*["\'][^"\']{4,}["\']',
        r'(?:mysql|postgres|mongodb|redis)://\w+:\w+@',
    ]
    for p in cred_patterns:
        if re.search(p, code, re.IGNORECASE):
            return (
                "code_safety: hardcoded credentials in source; "
                "use environment variables or a secrets manager"
            )

    # --- Unsafe deserialization ---
    if re.search(r'pickle\.load\s*\(|joblib\.load\s*\(|torch\.load\s*\(', code):
        has_safety = bool(re.search(
            r'hmac|hashlib|checksum|verify|signature'
            r'|weights_only\s*=\s*True|safetensors', code, re.IGNORECASE))
        if not has_safety:
            return (
                "code_safety: pickle/joblib load without integrity check; "
                "verify file hash or use safer formats (ONNX, safetensors)"
            )

    # --- Mutable default arguments ---
    if re.search(r'def\s+\w+\s*\([^)]*=\s*[\[\{]', code):
        return (
            "code_safety: mutable default argument (list or dict); "
            "use None and initialize inside the function body"
        )

    # --- Bare or overly broad except ---
    if re.search(r'except\s*:', code):
        return (
            "code_safety: bare except clause; "
            "catch only specific expected exceptions"
        )
    if re.search(r'except\s+(Exception|BaseException)\s*:', code):
        return (
            "code_safety: overly broad except clause; "
            "catch only the specific exceptions expected"
        )

    return None


# ---------------------------------------------------------------------------
# Checker registry and runner
# ---------------------------------------------------------------------------

_CHECKERS: dict[str, Callable[[str], str | None]] = {
    "data_separation": check_data_separation,
    "test_isolation": check_test_isolation,
    "evaluation_rigor": check_evaluation_rigor,
    "code_safety": check_code_safety,
}

# Map task-level violation_type → which checker(s) to run
_VIOLATION_TO_CHECKER: dict[str, list[str]] = {
    "data_leakage": ["data_separation"],
    "preprocessing_order": ["data_separation"],
    "temporal_leakage": ["data_separation"],
    "stacking_leakage": ["data_separation"],
    "group_leakage": ["data_separation"],
    "tfidf_leakage": ["data_separation"],
    "embedding_leakage": ["data_separation"],
    "augmentation_leakage": ["data_separation"],
    "test_misuse": ["test_isolation"],
    "early_stopping_leak": ["test_isolation"],
    "calibration_leakage": ["test_isolation"],
    "metric_misuse": ["evaluation_rigor"],
    "nested_cv_missing": ["evaluation_rigor"],
    "multiple_testing": ["evaluation_rigor"],
    "survival_metric_misuse": ["evaluation_rigor"],
    "multilabel_metric_misuse": ["evaluation_rigor"],
    "ranking_metric_misuse": ["evaluation_rigor"],
    "mutable_default_arg": ["code_safety"],
    "broad_exception": ["code_safety"],
    "unsafe_deserialization": ["code_safety"],
    "credential_exposure": ["code_safety"],
}


def run_verifiers(code: str, violation_type: str | None = None) -> dict:
    """
    Run verifier checks against generated code.

    If violation_type is known, only the relevant checker(s) run.
    Otherwise all checkers run.

    Returns dict: has_violation, violations (list[str]), feedback (str).
    """
    if violation_type and violation_type in _VIOLATION_TO_CHECKER:
        checker_names = _VIOLATION_TO_CHECKER[violation_type]
    else:
        checker_names = list(_CHECKERS.keys())

    messages: list[str] = []
    for name in checker_names:
        result = _CHECKERS[name](code)
        if result is not None:
            messages.append(result)

    return {
        "has_violation": bool(messages),
        "violations": [m.split(":")[0] for m in messages],
        "feedback": "\n".join(messages),
    }
