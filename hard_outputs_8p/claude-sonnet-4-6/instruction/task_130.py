```python
"""
Medical Data Anonymization Pipeline for ML Research

This module implements a comprehensive data anonymization pipeline that:
- Removes direct identifiers
- Generalizes quasi-identifiers
- Verifies and enforces k-anonymity
- Trains a classifier on anonymized data
- Reports utility and privacy metrics
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

@dataclass
class PrivacyMetrics:
    k_anonymity_level: int
    information_loss: float
    suppressed_records: int
    suppression_rate: float
    quasi_identifier_combinations: int
    generalization_steps: dict = field(default_factory=dict)


@dataclass
class ModelResults:
    train_f1_macro: float
    test_f1_macro: float
    test_balanced_accuracy: float
    test_roc_auc: Optional[float]
    cv_f1_scores: list
    cv_mean_f1: float
    cv_std_f1: float
    classification_report_str: str
    feature_importances: Optional[dict]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "name", "date_of_birth", "zip_code", "gender",
    "diagnosis", "treatment", "outcome", "insurance_id", "physician_name",
}

DIRECT_IDENTIFIERS = ["name", "insurance_id"]
QUASI_IDENTIFIERS = ["age_bracket", "zip_prefix", "gender"]
SENSITIVE_ATTRIBUTES = ["diagnosis", "treatment", "outcome"]


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame has all required columns."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")


def _compute_age(dob_series: pd.Series) -> pd.Series:
    """Compute age in years from a date-of-birth Series."""
    today = pd.Timestamp.today()
    dob = pd.to_datetime(dob_series, errors="coerce")
    age = (today - dob).dt.days // 365
    return age


def _age_to_bracket(age: pd.Series, bracket_size: int = 10) -> pd.Series:
    """Convert numeric age to decade brackets (e.g., '30-39')."""
    lower = (age // bracket_size) * bracket_size
    upper = lower + bracket_size - 1
    bracket = lower.astype(str) + "-" + upper.astype(str)
    bracket[age.isna()] = "Unknown"
    return bracket


def _generalize_zip(zip_series: pd.Series, digits: int = 3) -> pd.Series:
    """Keep only the first `digits` digits of a zip code."""
    return zip_series.astype(str).str[:digits].str.replace(r"\D", "", regex=True)


def _compute_k_anonymity(df: pd.DataFrame, quasi_ids: list[str]) -> int:
    """Return the minimum group size across all quasi-identifier combinations."""
    available_qi = [q for q in quasi_ids if q in df.columns]
    if not available_qi:
        return len(df)
    group_sizes = df.groupby(available_qi).size()
    return int(group_sizes.min()) if not group_sizes.empty else 0


def _compute_information_loss(
    original_df: pd.DataFrame,
    anonymized_df: pd.DataFrame,
    suppressed_count: int,
) -> float:
    """
    Estimate information loss as a normalized score in [0, 1].

    Loss components:
    - Suppression loss: fraction of records removed
    - Generalization loss: fraction of quasi-identifier entropy lost
      (approximated by comparing unique-value counts)
    """
    n_original = len(original_df)
    if n_original == 0:
        return 1.0

    suppression_loss = suppressed_count / n_original

    # Generalization loss: compare unique values before/after for QIs
    gen_losses = []
    for col_orig, col_anon in [
        ("date_of_birth", "age_bracket"),
        ("zip_code", "zip_prefix"),
    ]:
        if col_orig in original_df.columns and col_anon in anonymized_df.columns:
            orig_unique = original_df[col_orig].nunique()
            anon_unique = anonymized_df[col_anon].nunique()
            if orig_unique > 0:
                gen_losses.append(1.0 - (anon_unique / orig_unique))

    gen_loss = float(np.mean(gen_losses)) if gen_losses else 0.0
    total_loss = 0.5 * suppression_loss + 0.5 * gen_loss
    return round(min(total_loss, 1.0), 4)


# ---------------------------------------------------------------------------
# Core anonymization steps
# ---------------------------------------------------------------------------

def remove_direct_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: Remove columns that are direct identifiers."""
    cols_to_drop = [c for c in DIRECT_IDENTIFIERS if c in df.columns]
    logger.info("Removing direct identifiers: %s", cols_to_drop)
    return df.drop(columns=cols_to_drop)


def generalize_quasi_identifiers(
    df: pd.DataFrame,
    age_bracket_size: int = 10,
    zip_digits: int = 3,
) -> pd.DataFrame:
    """Step 2: Generalize quasi-identifiers."""
    df = df.copy()

    # Generalize date_of_birth → age_bracket
    if "date_of_birth" in df.columns:
        age = _compute_age(df["date_of_birth"])
        df["age_bracket"] = _age_to_bracket(age, bracket_size=age_bracket_size)
        df.drop(columns=["date_of_birth"], inplace=True)
        logger.info("Generalized date_of_birth → age_bracket (size=%d)", age_bracket_size)

    # Generalize zip_code → zip_prefix
    if "zip_code" in df.columns:
        df["zip_prefix"] = _generalize_zip(df["zip_code"], digits=zip_digits)
        df.drop(columns=["zip_code"], inplace=True)
        logger.info("Generalized zip_code → zip_prefix (digits=%d)", zip_digits)

    return df


def enforce_k_anonymity(
    df: pd.DataFrame,
    k: int = 5,
    quasi_ids: Optional[list[str]] = None,
    max_generalization_rounds: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """
    Step 3: Verify and enforce k-anonymity.

    Strategy:
    1. Check current k-anonymity level.
    2. If not satisfied, progressively widen age brackets.
    3. If still not satisfied, suppress (drop) records in groups smaller than k.

    Returns the anonymized DataFrame and a dict of generalization steps taken.
    """
    if quasi_ids is None:
        quasi_ids = QUASI_IDENTIFIERS

    available_qi = [q for q in quasi_ids if q in df.columns]
    generalization_steps: dict = {}
    current_k = _compute_k_anonymity(df, available_qi)
    logger.info("Initial k-anonymity level: %d (target k=%d)", current_k, k)

    # --- Progressive generalization of age_bracket ---
    bracket_sizes = [10, 20, 30]  # start at 10, widen if needed
    round_idx = 0

    while current_k < k and round_idx < max_generalization_rounds:
        if "age_bracket" in df.columns and round_idx < len(bracket_sizes) - 1:
            new_bracket_size = bracket_sizes[round_idx + 1]
            logger.info(
                "k-anonymity not satisfied. Widening age brackets to size %d.",
                new_bracket_size,
            )
            # Re-derive age from age_bracket midpoint (approximate)
            # Since we no longer have DOB, we widen existing brackets
            df = _widen_age_brackets(df, new_bracket_size)
            generalization_steps[f"round_{round_idx + 1}_age_bracket_size"] = new_bracket_size
            current_k = _compute_k_anonymity(df, available_qi)
            logger.info("k-anonymity after generalization round %d: %d", round_idx + 1, current_k)
        round_idx += 1

    # --- Suppression of small groups ---
    suppressed_count = 0
    if current_k < k:
        logger.warning(
            "k-anonymity still not satisfied (k=%d). Suppressing small groups.", current_k
        )
        group_sizes = df.groupby(available_qi).transform("size")
        mask = group_sizes >= k
        suppressed_count = int((~mask).sum())
        df = df[mask].copy()
        current_k = _compute_k_anonymity(df, available_qi)
        generalization_steps["suppressed_records"] = suppressed_count
        logger.info(
            "Suppressed %d records. New k-anonymity level: %d", suppressed_count, current_k
        )

    generalization_steps["final_k"] = current_k
    return df, generalization_steps


def _widen_age_brackets(df: pd.DataFrame, new_size: int) -> pd.DataFrame:
    """
    Re-bucket existing age_bracket strings into wider brackets.
    Parses the lower bound of the current bracket and re-assigns.
    """
    df = df.copy()

    def _rebucket(bracket: str) -> str:
        if bracket == "Unknown":
            return "Unknown"
        try:
            lower = int(bracket.split("-")[0])
            new_lower = (lower // new_size) * new_size
            new_upper = new_lower + new_size - 1
            return f"{new_lower}-{new_upper}"
        except (ValueError, IndexError):
            return "Unknown"

    df["age_bracket"] = df["age_bracket"].apply(_rebucket)
    return df


# ---------------------------------------------------------------------------
# Feature engineering (fit only on train)
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "outcome",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features from target. Drop physician_name (indirect identifier).
    """
    feature_df = df.drop(columns=[target_col, "physician_name"], errors="ignore").copy()
    target = df[target_col].copy()
    return feature_df, target


def _get_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify categorical and numeric columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return cat_cols, num_cols


# ---------------------------------------------------------------------------
# Train / validation / test split (stratified, no data leakage)
# ---------------------------------------------------------------------------

def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train / validation / test sets using stratified sampling.

    Splitting order:
    1. Hold out test set first.
    2. Split remaining into train and validation.

    No transformers are fit here — this is purely an index split.
    """
    from sklearn.model_selection import train_test_split

    # Step 1: hold out test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Step 2: split train/val from trainval
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val_size,
        stratify=y_trainval,
        random_state=random_state,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(
    cat_cols: list[str],
    num_cols: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer that handles categorical and numeric features."""
    transformers = []

    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    random_state: int = 42,
) -> ModelResults:
    """
    Train a RandomForestClassifier on the anonymized training data.

    - Preprocessing is fit ONLY on X_train.
    - Validation set is used for early stopping / threshold selection (not used here
      for hyperparameter tuning, but kept separate from test).
    - Test set is used ONLY for final metric reporting.
    - Reports macro F1, balanced accuracy, and ROC-AUC (where applicable).
    """
    cat_cols, num_cols = _get_column_types(X_train)

    preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)

    # Encode target
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    n_classes = len(le.classes_)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

    # Fit ONLY on training data
    pipeline.fit(X_train, y_train_enc)

    # --- Validation metrics (for monitoring, not for test-set decisions) ---
    y_val_pred = pipeline.predict(X_val)
    val_f1 = f1_score(y_val_enc, y_val_pred, average="macro", zero_division=0)
    logger.info("Validation macro F1: %.4f", val_f1)

    # --- Cross-validation on training data (inner CV, no test leakage) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train_enc, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    logger.info(
        "CV macro F1: %.4f ± %.4f", cv_scores.mean(), cv_scores.std()
    )

    # --- Training metrics ---
    y_train_pred = pipeline.predict(X_train)
    train_f1 = f1_score(y_train_enc, y_train_pred, average="macro", zero_division=0)

    # --- Final test evaluation (only here, never before) ---
    y_test_pred = pipeline.predict(X_test)
    test_f1 = f1_score(y_test_enc, y_test_pred, average="macro", zero_division=0)
    test_bal_acc = balanced_accuracy_score(y_test_enc, y_test_pred)

    # ROC-AUC (only for binary or with probability support)
    test_roc_auc = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_test_proba = pipeline.predict_proba(X_test)
            if n_classes == 2:
                test_roc_auc = roc_auc_score(y_test_enc, y_test_proba[:, 1])
            else:
                test_roc_auc = roc_auc_score(
                    y_test_enc, y_test_proba, multi_class="ovr", average="macro"
                )
        except ValueError as exc:
            logger.warning("Could not compute ROC-AUC: %s", exc)

    report_str = classification_report(
        y_test_enc, y_test_pred, target_names=le.classes_.astype(str), zero_division=0
    )
    logger.info("Test classification report:\n%s", report_str)

    # Feature importances
    feature_importances = None
    try:
        ohe = pipeline.named_steps["preprocessor"].named_transformers_.get("cat")
        feature_names_cat = (
            ohe.get_feature_names_out(cat_cols).tolist() if ohe is not None else []
        )
        feature_names = feature_names_cat + num_cols
        importances = pipeline.named_steps["classifier"].feature_importances_
        if len(importances) == len(feature_names):
            feature_importances = dict(
                sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
    except (AttributeError, KeyError) as exc:
        logger.warning("Could not extract feature importances: %s", exc)

    return ModelResults(
        train_f1_macro=round(train_f1, 4),
        test_f1_macro=round(test_f1, 4),
        test_balanced_accuracy=round(test_bal_acc, 4),
        test_roc_auc=round(test_roc_auc, 4) if test_roc_auc is not None else None,
        cv_f1_scores=cv_scores.tolist(),
        cv_mean_f1=round(float(cv_scores.mean()), 4),
        cv_std_f1=round(float(cv_scores.std()), 4),
        classification_report_str=report_str,
        feature_importances=feature_importances,
    )


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def anonymize_and_train(
    patient_df: pd.DataFrame,
    k: int = 5,
    target_col: str = "outcome",
    age_bracket_size: int = 10,
    zip_digits: int = 3,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    max_generalization_rounds: int = 3,
) -> tuple[pd.DataFrame, PrivacyMetrics, ModelResults]:
    """
    Full anonym