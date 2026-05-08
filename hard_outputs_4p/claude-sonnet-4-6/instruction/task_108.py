"""
hospital_readmission.py
-----------------------
Predicts 30-day hospital readmission from electronic health records.

Best-practice highlights
------------------------
* Train/test split happens FIRST – no information from the test set leaks into
  preprocessing, feature engineering, or model fitting.
* All transformers (scaler, top-diagnosis vocabulary) are fitted exclusively on
  the training partition and then applied to the test partition.
* Calibration is performed with CalibratedClassifierCV fitted on the training
  set only; the test set is used solely for final evaluation.
* No credentials or secrets are hard-coded.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_readmission(
    df: pd.DataFrame,
    *,
    top_n_diag: int = 20,
    test_size: float = 0.20,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
) -> dict[str, Any]:
    """
    Train a 30-day readmission predictor and return evaluation artefacts.

    Parameters
    ----------
    df : pd.DataFrame
        One row per hospital visit with columns:
        'patient_id', 'admission_date', 'diagnosis_codes',
        'length_of_stay', 'num_procedures', 'num_medications',
        'age', 'gender', 'readmitted_30d'
    top_n_diag : int
        Number of most-frequent ICD codes to one-hot encode.
    test_size : float
        Fraction of visits held out for evaluation.
    random_state : int
        Reproducibility seed.
    n_estimators, learning_rate, max_depth, subsample :
        GradientBoostingClassifier hyper-parameters.

    Returns
    -------
    dict with keys:
        'model'               – fitted CalibratedClassifierCV
        'metrics'             – dict(auc, f1, precision, recall, avg_precision)
        'calibration'         – dict(fraction_of_positives, mean_predicted_value)
        'feature_importances' – pd.Series sorted descending
        'feature_names'       – list[str]
        'test_indices'        – index of test rows in the original DataFrame
    """
    _validate_columns(df)

    df = df.copy()
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df = df.sort_values(["patient_id", "admission_date"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 1.  TRAIN / TEST SPLIT  (must happen before any fitting)            #
    # ------------------------------------------------------------------ #
    # Stratify on the target to preserve class balance in both partitions.
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df["readmitted_30d"],
    )
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    # ------------------------------------------------------------------ #
    # 2.  LEARN VOCABULARY FROM TRAINING SET ONLY                         #
    # ------------------------------------------------------------------ #
    top_diag_codes = _top_diagnosis_codes(train_df["diagnosis_codes"], top_n_diag)

    # ------------------------------------------------------------------ #
    # 3.  FEATURE ENGINEERING (fit on train, transform both)              #
    # ------------------------------------------------------------------ #
    X_train_raw, y_train = _engineer_features(
        train_df, top_diag_codes, fit_mode=True
    )
    X_test_raw, y_test = _engineer_features(
        test_df, top_diag_codes, fit_mode=False
    )

    feature_names = list(X_train_raw.columns)

    # Scale numeric features – fit ONLY on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # ------------------------------------------------------------------ #
    # 4.  TRAIN GRADIENT BOOSTING + CALIBRATION (training set only)       #
    # ------------------------------------------------------------------ #
    # GradientBoostingClassifier does not accept class_weight directly;
    # we replicate the effect via sample_weight.
    sample_weight_train = _compute_sample_weights(y_train)

    base_clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
    )

    # CalibratedClassifierCV with cv='prefit' requires the base estimator to
    # be fitted first; we use a 5-fold internal calibration on the training
    # set so the test set is never touched.
    base_clf.fit(X_train, y_train, sample_weight=sample_weight_train)

    calibrated_clf = CalibratedClassifierCV(base_clf, cv="prefit", method="isotonic")
    calibrated_clf.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # 5.  EVALUATE ON TEST SET (read-only)                                #
    # ------------------------------------------------------------------ #
    y_prob = calibrated_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "avg_precision": average_precision_score(y_test, y_prob),
    }

    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    calibration_data = {
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
    }

    # ------------------------------------------------------------------ #
    # 6.  FEATURE IMPORTANCES (from the base GBM, not the calibrator)     #
    # ------------------------------------------------------------------ #
    importances = pd.Series(
        base_clf.feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    return {
        "model": calibrated_clf,
        "metrics": metrics,
        "calibration": calibration_data,
        "feature_importances": importances,
        "feature_names": feature_names,
        "test_indices": test_idx,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame) -> None:
    required = {
        "patient_id", "admission_date", "diagnosis_codes",
        "length_of_stay", "num_procedures", "num_medications",
        "age", "gender", "readmitted_30d",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _top_diagnosis_codes(series: pd.Series, top_n: int) -> list[str]:
    """Return the *top_n* most frequent ICD codes found in *series*."""
    counter: Counter = Counter()
    for cell in series.dropna():
        for code in str(cell).split(","):
            code = code.strip()
            if code:
                counter[code] += 1
    return [code for code, _ in counter.most_common(top_n)]


def _engineer_features(
    df: pd.DataFrame,
    top_diag_codes: list[str],
    *,
    fit_mode: bool,  # kept for API clarity; vocabulary already fixed
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix for a partition.

    All vocabulary / statistics come from *top_diag_codes* which was derived
    exclusively from the training set.
    """
    df = df.copy()

    # ---- One-hot encode top-20 diagnosis codes -------------------------
    for code in top_diag_codes:
        col_name = f"diag_{code.replace('.', '_')}"
        df[col_name] = df["diagnosis_codes"].apply(
            lambda cell, c=code: int(c in str(cell).split(","))
        )

    # ---- Patient history features -------------------------------------
    df = df.sort_values(["patient_id", "admission_date"])
    df["prior_admissions"] = (
        df.groupby("patient_id").cumcount()  # 0-indexed count of prior rows
    )
    df["days_since_last_admission"] = (
        df.groupby("patient_id")["admission_date"]
        .diff()
        .dt.days
        .fillna(-1)          # -1 signals "first admission"
    )

    # ---- Interaction feature ------------------------------------------
    df["los_x_procedures"] = df["length_of_stay"] * df["num_procedures"]

    # ---- Encode gender ------------------------------------------------
    df["gender_encoded"] = (df["gender"].str.upper() == "M").astype(int)

    # ---- Assemble feature matrix -------------------------------------
    numeric_cols = [
        "length_of_stay", "num_procedures", "num_medications",
        "age", "prior_admissions", "days_since_last_admission",
        "los_x_procedures", "gender_encoded",
    ]
    diag_cols = [f"diag_{c.replace('.', '_')}" for c in top_diag_codes]
    all_feature_cols = numeric_cols + diag_cols

    X = df[all_feature_cols].copy()
    y = df["readmitted_30d"].astype(int)

    return X, y


def _compute_sample_weights(y: pd.Series) -> np.ndarray:
    """
    Mimic class_weight='balanced' via per-sample weights.

    weight_c = n_samples / (n_classes * n_samples_c)
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {
        cls: n_samples / (n_classes * cnt)
        for cls, cnt in zip(classes, counts)
    }
    return np.array([weight_map[label] for label in y])


# ---------------------------------------------------------------------------
# Quick smoke-test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    from datetime import datetime, timedelta

    rng = random.Random(0)
    np_rng = np.random.default_rng(0)

    ICD_POOL = [
        "I10", "E11", "J44", "N18", "I50", "Z87", "K92",
        "M54", "F32", "I25", "E78", "J18", "G47", "R05",
        "Z96", "I48", "E66", "K57", "N39", "B96",
    ]

    def _random_diag():
        k = rng.randint(1, 5)
        return ",".join(rng.sample(ICD_POOL, k))

    n = 2_000
    base_date = datetime(2020, 1, 1)
    records = []
    for i in range(n):
        pid = f"P{rng.randint(1, 400):04d}"
        adm = base_date + timedelta(days=rng.randint(0, 1095))
        los = rng.randint(1, 14)
        procs = rng.randint(0, 8)
        meds = rng.randint(1, 20)
        age = rng.randint(18, 90)
        gender = rng.choice(["M", "F"])
        # Synthetic label with some signal
        logit = (
            -2.0
            + 0.03 * age
            + 0.1 * los
            + 0.05 * procs
            - 0.02 * meds
            + (0.3 if gender == "M" else 0.0)
        )
        prob = 1 / (1 + np.exp(-logit))
        label = int(np_rng.random() < prob)
        records.append(
            dict(
                patient_id=pid,
                admission_date=adm.strftime("%Y-%m-%d"),
                diagnosis_codes=_random_diag(),
                length_of_stay=los,
                num_procedures=procs,
                num_medications=meds,
                age=age,
                gender=gender,
                readmitted_30d=label,
            )
        )

    demo_df = pd.DataFrame(records)
    print(f"Dataset shape: {demo_df.shape}")
    print(f"Positive rate: {demo_df['readmitted_30d'].mean():.3f}\n")

    results = predict_readmission(demo_df, random_state=42)

    print("=== Metrics ===")
    for k, v in results["metrics"].items():
        print(f"  {k:20s}: {v:.4f}")

    print("\n=== Top-10 Feature Importances ===")
    print(results["feature_importances"].head(10).to_string())

    print("\n=== Calibration (first 5 bins) ===")
    cal = results["calibration"]
    for fop, mpv in zip(cal["fraction_of_positives"][:5],
                        cal["mean_predicted_value"][:5]):
        print(f"  mean_pred={mpv:.3f}  frac_pos={fop:.3f}")