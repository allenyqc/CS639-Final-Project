"""
Hospital Readmission Prediction Module

Predicts 30-day hospital readmission from electronic health records using
Gradient Boosting with proper data splitting, feature engineering, and evaluation.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_diagnosis_codes(series: pd.Series) -> pd.Series:
    """Split comma-separated ICD codes into lists, stripping whitespace."""
    return series.fillna("").apply(
        lambda x: [c.strip() for c in x.split(",") if c.strip()]
    )


def _get_top_k_codes(code_lists: pd.Series, k: int = 20) -> list[str]:
    """Return the k most frequent ICD codes across all visits."""
    from collections import Counter

    counter: Counter = Counter()
    for codes in code_lists:
        counter.update(codes)
    return [code for code, _ in counter.most_common(k)]


def _one_hot_codes(
    code_lists: pd.Series, top_codes: list[str]
) -> pd.DataFrame:
    """Binary indicator columns for each of the top ICD codes."""
    data = {
        f"dx_{code}": code_lists.apply(lambda lst: int(code in lst))
        for code in top_codes
    }
    return pd.DataFrame(data, index=code_lists.index)


def _build_patient_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-visit patient history features using only information
    available *before* the current admission (no data leakage).

    Features:
        - prior_admissions : number of admissions before this one
        - days_since_last  : days since the previous admission (NaN → -1)
    """
    df = df.copy()
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df = df.sort_values(["patient_id", "admission_date"])

    prior_admissions = []
    days_since_last = []

    for _, group in df.groupby("patient_id", sort=False):
        dates = group["admission_date"].values
        for i in range(len(dates)):
            prior_admissions.append(i)  # visits before current
            if i == 0:
                days_since_last.append(-1.0)  # sentinel: no prior visit
            else:
                delta = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D")
                days_since_last.append(float(delta))

    df["prior_admissions"] = prior_admissions
    df["days_since_last"] = days_since_last
    return df


# ---------------------------------------------------------------------------
# Chronological / group-aware split
# ---------------------------------------------------------------------------

def _chronological_group_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data so that:
      1. The split is chronological (earlier visits → train, later → val/test).
      2. No patient appears in more than one partition (group-aware).

    Strategy:
      - Sort patients by their *earliest* admission date.
      - Assign the earliest patients to train, next to val, latest to test.
    """
    df = df.copy()
    df["admission_date"] = pd.to_datetime(df["admission_date"])

    # Earliest admission per patient
    patient_first = (
        df.groupby("patient_id")["admission_date"]
        .min()
        .sort_values()
        .reset_index()
    )
    n_patients = len(patient_first)

    n_test = max(1, int(n_patients * test_size))
    n_val = max(1, int(n_patients * val_size))
    n_train = n_patients - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Not enough patients ({n_patients}) for the requested splits."
        )

    train_patients = set(patient_first.iloc[:n_train]["patient_id"])
    val_patients = set(patient_first.iloc[n_train : n_train + n_val]["patient_id"])
    test_patients = set(patient_first.iloc[n_train + n_val :]["patient_id"])

    train_df = df[df["patient_id"].isin(train_patients)].copy()
    val_df = df[df["patient_id"].isin(val_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Feature engineering (fit on train, transform all)
# ---------------------------------------------------------------------------

class ReadmissionFeatureEngineer:
    """Stateful transformer that learns from training data only."""

    def __init__(self, top_k_codes: int = 20) -> None:
        self.top_k_codes = top_k_codes
        self._top_codes: list[str] = []
        self._scaler = StandardScaler()
        self._numeric_cols: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *df* and return transformed features."""
        df = _build_patient_history(df)
        code_lists = _parse_diagnosis_codes(df["diagnosis_codes"])
        self._top_codes = _get_top_k_codes(code_lists, k=self.top_k_codes)
        return self._transform_internal(df, code_lists, fit=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform *df* using parameters learned during fit_transform."""
        if not self._top_codes:
            raise RuntimeError("Call fit_transform before transform.")
        df = _build_patient_history(df)
        code_lists = _parse_diagnosis_codes(df["diagnosis_codes"])
        return self._transform_internal(df, code_lists, fit=False)

    def _transform_internal(
        self,
        df: pd.DataFrame,
        code_lists: pd.Series,
        fit: bool,
    ) -> pd.DataFrame:
        # One-hot diagnosis codes
        dx_df = _one_hot_codes(code_lists, self._top_codes)

        # Gender encoding
        gender_encoded = (df["gender"].str.lower() == "m").astype(int)

        # Interaction feature
        interaction = df["length_of_stay"] * df["num_procedures"]

        # Assemble raw numeric features
        numeric_df = pd.DataFrame(
            {
                "length_of_stay": df["length_of_stay"].values,
                "num_procedures": df["num_procedures"].values,
                "num_medications": df["num_medications"].values,
                "age": df["age"].values,
                "gender_male": gender_encoded.values,
                "prior_admissions": df["prior_admissions"].values,
                "days_since_last": df["days_since_last"].values,
                "los_x_procedures": interaction.values,
            },
            index=df.index,
        )

        if fit:
            self._numeric_cols = list(numeric_df.columns)
            scaled_values = self._scaler.fit_transform(numeric_df)
        else:
            scaled_values = self._scaler.transform(numeric_df[self._numeric_cols])

        scaled_df = pd.DataFrame(
            scaled_values, columns=self._numeric_cols, index=df.index
        )

        features = pd.concat([scaled_df, dx_df], axis=1)
        return features


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> tuple[Any, Any]:
    """
    Train a GradientBoostingClassifier and calibrate probabilities on the
    validation set (isotonic regression).  The test set is never touched here.
    """
    base_clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=random_state,
    )

    # Compute sample weights to handle class imbalance (mirrors class_weight='balanced')
    classes, counts = np.unique(y_train, return_counts=True)
    weight_map = {c: len(y_train) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = y_train.map(weight_map).values

    base_clf.fit(X_train, y_train, sample_weight=sample_weights)

    # Calibrate on validation set — test set remains untouched
    calibrated_clf = CalibratedClassifierCV(
        base_clf, method="isotonic", cv="prefit"
    )
    calibrated_clf.fit(X_val, y_val)

    return base_clf, calibrated_clf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute AUC, F1, precision, recall on the provided partition."""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
    }


def _tune_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """
    Select the decision threshold that maximises F1 on the *validation* set.
    The test set is never used for threshold selection.
    """
    proba = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.linspace(0.1, 0.9, 81):
        preds = (proba >= thresh).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return float(best_thresh)


def _get_calibration_data(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Return fraction_of_positives and mean_predicted_value for a reliability diagram."""
    proba = model.predict_proba(X)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="uniform")
    return {
        "fraction_of_positives": fraction_pos,
        "mean_predicted_value": mean_pred,
    }


def _get_feature_importances(
    base_clf: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances from the base (uncalibrated) model."""
    importances = base_clf.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_readmission(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    top_k_codes: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train and evaluate a 30-day hospital readmission prediction model.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns: patient_id, admission_date, diagnosis_codes,
        length_of_stay, num_procedures, num_medications, age, gender,
        readmitted_30d.
    test_size : float
        Fraction of *patients* reserved for the held-out test set.
    val_size : float
        Fraction of *patients* reserved for the validation set.
    top_k_codes : int
        Number of most-frequent ICD codes to one-hot encode.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    dict with keys:
        model            – calibrated GradientBoostingClassifier
        metrics          – dict of AUC, F1, precision, recall on test set
        calibration_data – dict with fraction_of_positives & mean_predicted_value
        feature_importances – pd.DataFrame sorted by importance
        threshold        – decision threshold tuned on validation set
    """
    required_cols = {
        "patient_id", "admission_date", "diagnosis_codes",
        "length_of_stay", "num_procedures", "num_medications",
        "age", "gender", "readmitted_30d",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    # ------------------------------------------------------------------ #
    # 1. Chronological, group-aware split BEFORE any preprocessing        #
    # ------------------------------------------------------------------ #
    train_df, val_df, test_df = _chronological_group_split(
        df,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    y_train = train_df["readmitted_30d"].astype(int)
    y_val = val_df["readmitted_30d"].astype(int)
    y_test = test_df["readmitted_30d"].astype(int)

    # ------------------------------------------------------------------ #
    # 2. Feature engineering — fit ONLY on training data                  #
    # ------------------------------------------------------------------ #
    engineer = ReadmissionFeatureEngineer(top_k_codes=top_k_codes)
    X_train = engineer.fit_transform(train_df)
    X_val = engineer.transform(val_df)
    X_test = engineer.transform(test_df)

    feature_names = list(X_train.columns)

    # ------------------------------------------------------------------ #
    # 3. Train model; calibrate on validation set                         #
    # ------------------------------------------------------------------ #
    base_clf, calibrated_clf = _train_model(
        X_train, y_train, X_val, y_val, random_state=random_state
    )

    # ------------------------------------------------------------------ #
    # 4. Tune decision threshold on validation set                        #
    # ------------------------------------------------------------------ #
    best_threshold = _tune_threshold(calibrated_clf, X_val, y_val)

    # ------------------------------------------------------------------ #
    # 5. Final evaluation on held-out test set (single, untouched pass)   #
    # ------------------------------------------------------------------ #
    test_metrics = _evaluate(calibrated_clf, X_test, y_test, threshold=best_threshold)
    calibration_data = _get_calibration_data(calibrated_clf, X_test, y_test)
    feature_importances = _get_feature_importances(base_clf, feature_names)

    return {
        "model": calibrated_clf,
        "metrics": test_metrics,
        "calibration_data": calibration_data,
        "feature_importances": feature_importances,
        "threshold": best_threshold,
    }


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_patients: int = 400,
    visits_per_patient: int = 3,
    random_state: int = 0,
) -> pd.DataFrame:
    """Generate a small synthetic EHR dataset for testing."""
    rng = np.random.default_rng(random_state)
    icd_pool = [f"E{i:02d}" for i in range(30)]

    rows = []
    base_date = pd.Timestamp("2018-01-01")

    for pid in range(n_patients):
        n_visits = rng.integers(1, visits_per_patient + 1)
        current_date = base_date + pd.Timedelta(days=int(rng.integers(0, 365)))
        age = int(rng.integers(18, 90))
        gender = rng.choice(["M", "F"])

        for _ in range(n_visits):
            n_codes = rng.integers(1, 6)
            codes = ",".join(rng.choice(icd_pool, size=n_codes, replace=False).tolist())
            los = int(rng.integers(1, 15))
            n_proc = int(rng.integers(0, 10))
            n_med = int(rng.integers(1, 20))
            readmitted = int(rng.random() < 0.25)

            rows.append(
                {
                    "patient_id": pid,
                    "admission_date": current_date.strftime("%Y-%m-%d"),
                    "diagnosis_codes": codes,
                    "length_of_stay": los,
                    "num_procedures": n_proc,
                    "num_medications": n_med,
                    "age": age,
                    "gender": gender,
                    "readmitted_30d": readmitted,
                }
            )
            current_date += pd.Timedelta(days=int(rng.integers(30, 180)))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating synthetic EHR data …")
    synthetic_df = _generate_synthetic_data(n_patients=500, visits_per_patient=4)
    print(f"Dataset shape: {synthetic_df.shape}")

    print("\nTraining readmission model …")
    results = predict_readmission(synthetic_df, top_k_codes=20)

    print("\n=== Test-set Metrics ===")
    for metric, value in results["metrics"].items():
        print(f"  {metric:12s}: {value:.4f}")

    print(f"\nOptimal threshold (tuned on validation set): {results['threshold']:.3f}")

    print("\n=== Top-10 Feature Importances ===")
    print(results["feature_importances"].head(10).to_string(index=False))

    print("\n=== Calibration Data (first 5 bins) ===")
    cal = results["calibration_data"]
    for fop, mpv in zip(
        cal["fraction_of_positives"][:5], cal["mean_predicted_value"][:5]
    ):
        print(f"  mean_pred={mpv:.3f}  fraction_pos={fop:.3f}")