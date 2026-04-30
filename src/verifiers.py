from __future__ import annotations

import re


def check_data_leakage(code: str) -> str | None:
    # .fit() / .fit_transform() / .fit_resample() on anything that isn't X_train / X_val
    pattern = r'\.(fit|fit_transform|fit_resample)\(\s*X(?!_train\b)(?!_val\b)|\.fit(?:_transform|_resample)?\(\s*(data|df|dataset|features)\b'
    if re.search(pattern, code):
        return "data_leakage: preprocessor fitted on full or test data; fit only on X_train"
    return None


def check_preprocessing_order(code: str) -> str | None:
    # Any .fit() call that appears before train_test_split() in the source
    fit_pos = re.search(r'\.(fit|fit_transform|fit_resample)\(', code)
    split_pos = re.search(r'train_test_split\(', code)
    if fit_pos and split_pos and fit_pos.start() < split_pos.start():
        return "preprocessing_order: preprocessing fitted before train_test_split; split first, then fit on training data only"
    return None


def check_test_misuse(code: str) -> str | None:
    # .fit() called with X_test or y_test as the first argument
    if re.search(r'\.fit\s*\(\s*(X_test|y_test)', code):
        return "test_misuse: model or selector fitted on test data"
    # Threshold selection loop that references y_test (double-dipping)
    if re.search(r'for\s+\w+\s+in\s+\w*[Tt]hreshold', code) and 'y_test' in code:
        return "test_misuse: classification threshold selected on test labels; use a validation split instead"
    # SelectKBest / mutual_info fitted on full data (X, y) rather than X_train, y_train
    if re.search(r'(SelectKBest|mutual_info)\b', code) and re.search(r'\.fit\s*\(\s*X\b', code):
        return "test_misuse: feature selector fitted on full dataset including test labels"
    return None


def check_metric_misuse(code: str) -> str | None:
    # Only accuracy, no complementary metric
    if re.search(r'accuracy_score', code) and not re.search(
        r'f1_score|roc_auc_score|classification_report|precision_score|recall_score|average_precision', code
    ):
        return "metric_misuse: only accuracy reported; add F1, AUC, or classification_report for imbalanced/multi-class datasets"
    # Predict / score called only on training data with no test evaluation
    trains_only = re.search(r'\.(score|predict)\s*\(\s*X_train', code)
    has_test_eval = re.search(r'\.(score|predict)\s*\(\s*X_test', code)
    if trains_only and not has_test_eval:
        return "metric_misuse: model evaluated on training data only; report held-out test performance"
    return None


def check_mutable_default_arg(code: str) -> str | None:
    if re.search(r'def\s+\w+\s*\([^)]*=\s*[\[\{]', code):
        return "mutable_default_arg: mutable default argument in function definition; use None and initialise inside the function body"
    return None


def check_broad_exception(code: str) -> str | None:
    if re.search(r'except\s*:', code):
        return "broad_exception: bare except clause; catch only specific exceptions (e.g. FileNotFoundError, json.JSONDecodeError)"
    if re.search(r'except\s+(Exception|BaseException)\s*:', code):
        return "broad_exception: overly broad except clause; catch only the specific exceptions the function expects"
    return None


_CHECKERS: dict[str, callable] = {
    "data_leakage": check_data_leakage,
    "preprocessing_order": check_preprocessing_order,
    "test_misuse": check_test_misuse,
    "metric_misuse": check_metric_misuse,
    "mutable_default_arg": check_mutable_default_arg,
    "broad_exception": check_broad_exception,
}


def run_verifiers(code: str, violation_type: str | None = None) -> dict:
    """
    Run verifier checks against generated code.

    If violation_type matches a known checker, only that checker is run
    (avoids false positives across unrelated tasks). Otherwise all checkers run.

    Returns a dict with keys: has_violation (bool), violations (list[str]), feedback (str).
    """
    if violation_type and violation_type in _CHECKERS:
        checks = [_CHECKERS[violation_type]]
    else:
        checks = list(_CHECKERS.values())

    messages: list[str] = []
    for check in checks:
        result = check(code)
        if result is not None:
            messages.append(result)

    return {
        "has_violation": bool(messages),
        "violations": [m.split(":")[0] for m in messages],
        "feedback": "\n".join(messages),
    }
