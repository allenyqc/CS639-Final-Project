```python
"""
Medical Dataset Anonymization Pipeline for ML Research

This module implements a comprehensive data anonymization pipeline that:
- Removes direct identifiers
- Generalizes quasi-identifiers
- Verifies and enforces k-anonymity
- Trains ML models on anonymized data
- Reports privacy and utility metrics
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, Tuple, Any, Optional
from collections import Counter
import hashlib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _compute_age(dob: Any, reference_date: Optional[date] = None) -> Optional[int]:
    """Return age in years from a date-of-birth value."""
    if reference_date is None:
        reference_date = date.today()
    if pd.isna(dob):
        return None
    if isinstance(dob, str):
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                dob = datetime.strptime(dob, fmt).date()
                break
            except ValueError:
                continue
        else:
            return None
    if isinstance(dob, datetime):
        dob = dob.date()
    try:
        age = (reference_date - dob).days // 365
        return max(0, age)
    except Exception:
        return None


def _age_to_bracket(age: Optional[int], bracket_size: int = 10) -> str:
    """Map an integer age to a decade bracket string, e.g. '30-39'."""
    if age is None or pd.isna(age):
        return "Unknown"
    lower = (int(age) // bracket_size) * bracket_size
    upper = lower + bracket_size - 1
    return f"{lower}-{upper}"


def _generalize_zip(zip_code: Any, digits: int = 3) -> str:
    """Keep only the first *digits* characters of a ZIP code."""
    if pd.isna(zip_code):
        return "***"
    zc = str(zip_code).strip().replace(" ", "")
    return zc[:digits] + "*" * max(0, len(zc) - digits) if len(zc) >= digits else zc


def _information_loss(original: pd.Series, generalized: pd.Series) -> float:
    """
    Estimate information loss as the ratio of unique values lost.
    Returns a value in [0, 1] where 0 = no loss, 1 = total loss.
    """
    orig_unique = original.nunique()
    gen_unique = generalized.nunique()
    if orig_unique == 0:
        return 0.0
    loss = 1.0 - (gen_unique / orig_unique)
    return round(max(0.0, min(1.0, loss)), 4)


def _compute_k_anonymity(df: pd.DataFrame, quasi_identifiers: list) -> int:
    """
    Return the minimum equivalence-class size (k) for the given
    quasi-identifier columns.
    """
    if df.empty or not quasi_identifiers:
        return 0
    counts = df.groupby(quasi_identifiers, dropna=False).size()
    return int(counts.min())


# ─────────────────────────────────────────────────────────────────────────────
# Core anonymization steps
# ─────────────────────────────────────────────────────────────────────────────

def remove_direct_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 – Remove columns that are direct identifiers.
    Columns removed: 'name', 'insurance_id', 'physician_name'
    (physician_name is a quasi-identifier that can re-identify patients
    in small practices and is therefore treated as a direct identifier here).
    """
    direct_ids = ["name", "insurance_id", "physician_name"]
    cols_to_drop = [c for c in direct_ids if c in df.columns]
    anonymized = df.drop(columns=cols_to_drop, errors="ignore").copy()
    return anonymized


def generalize_quasi_identifiers(
    df: pd.DataFrame,
    age_bracket_size: int = 10,
    zip_digits: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Step 2 – Generalise quasi-identifiers.

    * date_of_birth  → age bracket  (e.g. '40-49')
    * zip_code       → first *zip_digits* digits  (e.g. '941**')
    * gender         → kept as-is (already categorical)

    Returns the generalised DataFrame and a dict of per-column information loss.
    """
    result = df.copy()
    info_loss: Dict[str, float] = {}

    # ── date_of_birth → age_bracket ──────────────────────────────────────────
    if "date_of_birth" in result.columns:
        ages = result["date_of_birth"].apply(_compute_age)
        original_dob = result["date_of_birth"].astype(str)
        result["age_bracket"] = ages.apply(
            lambda a: _age_to_bracket(a, bracket_size=age_bracket_size)
        )
        info_loss["date_of_birth"] = _information_loss(original_dob, result["age_bracket"])
        result.drop(columns=["date_of_birth"], inplace=True)

    # ── zip_code → generalised zip ───────────────────────────────────────────
    if "zip_code" in result.columns:
        original_zip = result["zip_code"].astype(str)
        result["zip_code"] = result["zip_code"].apply(
            lambda z: _generalize_zip(z, digits=zip_digits)
        )
        info_loss["zip_code"] = _information_loss(original_zip, result["zip_code"])

    return result, info_loss


def enforce_k_anonymity(
    df: pd.DataFrame,
    quasi_identifiers: list,
    k: int = 5,
    max_suppression_rate: float = 0.20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Step 3 – Verify k-anonymity; apply further generalisation / suppression
    until the target k is reached or the suppression budget is exhausted.

    Strategy (in order):
      1. Widen age brackets (10 → 20 → full suppression to 'Unknown').
      2. Further truncate ZIP to 2 digits.
      3. Suppress (drop) rows in equivalence classes smaller than k.

    Returns the k-anonymous DataFrame and a metrics dict.
    """
    result = df.copy()
    steps_applied: list = []
    original_size = len(result)

    qi_present = [q for q in quasi_identifiers if q in result.columns]

    current_k = _compute_k_anonymity(result, qi_present)
    if current_k >= k:
        return result, {
            "k_achieved": current_k,
            "target_k": k,
            "rows_suppressed": 0,
            "suppression_rate": 0.0,
            "steps_applied": [],
        }

    # ── Pass 1: widen age bracket to 20-year bands ───────────────────────────
    if "age_bracket" in result.columns and current_k < k:
        # Re-derive from the existing bracket lower bound
        def widen_bracket(bracket: str) -> str:
            if bracket == "Unknown":
                return "Unknown"
            try:
                lower = int(bracket.split("-")[0])
                new_lower = (lower // 20) * 20
                return f"{new_lower}-{new_lower + 19}"
            except Exception:
                return "Unknown"

        result["age_bracket"] = result["age_bracket"].apply(widen_bracket)
        current_k = _compute_k_anonymity(result, qi_present)
        steps_applied.append("age_bracket widened to 20-year bands")

    # ── Pass 2: truncate ZIP to 2 digits ─────────────────────────────────────
    if "zip_code" in result.columns and current_k < k:
        result["zip_code"] = result["zip_code"].apply(
            lambda z: str(z)[:2] + "***" if not pd.isna(z) else "***"
        )
        current_k = _compute_k_anonymity(result, qi_present)
        steps_applied.append("zip_code truncated to 2 digits")

    # ── Pass 3: suppress age bracket entirely ────────────────────────────────
    if "age_bracket" in result.columns and current_k < k:
        result["age_bracket"] = "Unknown"
        current_k = _compute_k_anonymity(result, qi_present)
        steps_applied.append("age_bracket fully suppressed")

    # ── Pass 4: row suppression for remaining small equivalence classes ───────
    if current_k < k:
        group_sizes = result.groupby(qi_present, dropna=False).transform("size")
        small_class_mask = group_sizes < k
        rows_to_suppress = small_class_mask.sum()
        suppression_rate = rows_to_suppress / original_size

        if suppression_rate <= max_suppression_rate:
            result = result[~small_class_mask].copy()
            steps_applied.append(
                f"row suppression: {rows_to_suppress} rows removed "
                f"({suppression_rate:.1%})"
            )
            current_k = _compute_k_anonymity(result, qi_present)
        else:
            # Suppress only the smallest classes until budget is used
            counts = result.groupby(qi_present, dropna=False).size().reset_index(
                name="_count"
            )
            counts_sorted = counts.sort_values("_count")
            budget = int(max_suppression_rate * original_size)
            removed = 0
            keys_to_remove = []
            for _, row in counts_sorted.iterrows():
                if row["_count"] + removed > budget:
                    break
                keys_to_remove.append(
                    tuple(row[q] for q in qi_present)
                )
                removed += row["_count"]

            if keys_to_remove:
                mask = result[qi_present].apply(
                    lambda r: tuple(r) in keys_to_remove, axis=1
                )
                result = result[~mask].copy()
                steps_applied.append(
                    f"partial row suppression: {removed} rows removed "
                    f"(budget-constrained)"
                )
                current_k = _compute_k_anonymity(result, qi_present)

    rows_suppressed = original_size - len(result)
    return result, {
        "k_achieved": current_k,
        "target_k": k,
        "rows_suppressed": rows_suppressed,
        "suppression_rate": round(rows_suppressed / original_size, 4),
        "steps_applied": steps_applied,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ML pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _encode_features(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Encode categorical features and return (X, y, feature_names).
    """
    feature_df = df.drop(columns=[target_col], errors="ignore").copy()
    target = df[target_col].copy()

    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(target.astype(str))

    # One-hot encode all remaining columns
    feature_df = pd.get_dummies(feature_df, drop_first=False)
    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(float)

    return X, y, feature_names


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str = "outcome",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a RandomForest classifier on the anonymised data and return
    comprehensive utility metrics.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Drop rows with missing target
    df_clean = df.dropna(subset=[target_col]).copy()
    if len(df_clean) < 20:
        return {"error": "Insufficient data for ML after anonymisation."}

    X, y, feature_names = _encode_features(df_clean, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
        if len(np.unique(y)) > 1 else None
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Probability estimates for AUC (binary or multi-class)
    n_classes = len(np.unique(y))
    try:
        y_prob = clf.predict_proba(X_test)
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = None

    # Cross-validation on full dataset
    cv_scores = cross_val_score(clf, X, y, cv=min(5, len(df_clean) // 10 or 2),
                                scoring="accuracy")

    # Feature importances
    rf_model = clf.named_steps["model"]
    importances = rf_model.feature_importances_
    top_features = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )[:10]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(
            precision_score(y_test, y_pred, average="macro", zero_division=0), 4
        ),
        "recall_macro": round(
            recall_score(y_test, y_pred, average="macro", zero_division=0), 4
        ),
        "f1_macro": round(
            f1_score(y_test, y_pred, average="macro", zero_division=0), 4
        ),
        "roc_auc": round(auc, 4) if auc is not None else "N/A",
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "top_10_features": top_features,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_classes": n_classes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

QUASI_IDENTIFIERS = ["age_bracket", "zip_code", "gender"]


def anonymize_and_train(
    df: pd.DataFrame,
    target_col: str = "outcome",
    k: int = 5,
    age_bracket_size: int = 10,
    zip_digits: int = 3,
    test_size: float = 0.25,
    max_suppression_rate: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Full anonymization + ML pipeline.

    Parameters
    ----------
    df                  : Raw patient DataFrame.
    target_col          : Column to predict (default 'outcome').
    k                   : Desired k-anonymity level (default 5).
    age_bracket_size    : Initial age bracket width in years (default 10).
    zip_digits          : Number of ZIP digits to retain (default 3).
    test_size           : Fraction of data for test split (default 0.25).
    max_suppression_rate: Maximum fraction of rows that may be suppressed.
    random_state        : Random seed for reproducibility.

    Returns
    -------
    anonymized_df   : The k-anonymous DataFrame ready for ML.
    privacy_metrics : Dict with k-anonymity level, suppression stats,
                      information loss, and generalisation steps.
    model_results   : Dict with ML performance metrics.
    """
    print("=" * 65)
    print("  Medical Dataset Anonymization Pipeline")
    print("=" * 65)

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    original_size = len(df)
    print(f"\n[1/5] Input dataset: {original_size} rows, {df.shape[1]} columns")

    # ── Step 1: Remove direct identifiers ────────────────────────────────────
    step1 = remove_direct_identifiers(df)
    removed_cols = set(df.columns) - set(step1.columns)
    print(f"[2/5] Direct identifiers removed: {sorted(removed_cols)}")

    # ── Step 2: Generalise quasi-identifiers ──────────────────────────────────
    step2, info_loss = generalize_quasi_identifiers(
        step1,
        age_bracket_size=age_bracket_size,
        zip_digits=zip_digits,
    )
    print(f"[3/5] Quasi-identifiers generalised. Information loss: {info_loss}")

    # ── Step 3: Enforce k-anonymity ───────────────────────────────────────────
    qi_cols = [q for q in QUASI_IDENTIFIERS if q in step2.columns]
    step3, k_metrics = enforce_k_anonymity(
        step2,
        quasi_identifiers=qi_cols,
        k=k,
        max_suppression_rate=max_suppression_rate,
    )
    print(
        f"[4/5] k-anonymity enforced. "
        f"k_achieved={k_metrics['k_achieved']} (target={k}), "
        f"rows_suppressed={k_metrics['rows_suppressed']}"
    )
    if k_metrics["steps_applied"]:
        for step in k_metrics["steps_applied"]:
            print(f"       → {step}")

    # ── Step 4: Train ML model ────────────────────────────────────────────────
    print(f"[5/5] Training classifier to predict '{target_col}' …")
    model_results = train_and_evaluate(
        step3,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )
    if "error" not in model_results:
        print(
            f"       Accuracy={model_results['accuracy']}, "
            f"F1={model_results['f1_macro']}, "
            f"AUC={model_results['roc_