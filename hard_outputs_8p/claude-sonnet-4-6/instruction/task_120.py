```python
"""
Network Intrusion Detection using Anomaly Detection.

This module builds an anomaly detection system for network intrusion detection
using Isolation Forest or One-Class SVM, following ML best practices.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = ["protocol_type", "flag", "service"]
NUMERIC_FEATURES = ["duration", "bytes_sent", "bytes_received", "num_packets"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABEL_COLUMN = "label"


@dataclass
class PRCurveData:
    """Stores precision-recall curve data."""
    precisions: np.ndarray
    recalls: np.ndarray
    thresholds: np.ndarray
    pr_auc: float


@dataclass
class DetectionMetrics:
    """Stores evaluation metrics for the anomaly detector."""
    precision: float
    recall: float
    f1: float
    pr_auc: float
    threshold: float


@dataclass
class AnomalyDetectionResult:
    """Full result bundle returned by the pipeline."""
    model: object
    scaler: StandardScaler
    label_encoders: dict
    threshold: float
    metrics: DetectionMetrics
    pr_curve: PRCurveData


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    required = set(ALL_FEATURES + [LABEL_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _fit_preprocessors(
    X_train: pd.DataFrame,
) -> Tuple[StandardScaler, dict]:
    """
    Fit scaler and label encoders on training data only.

    Returns
    -------
    scaler : StandardScaler fitted on numeric features.
    label_encoders : dict mapping categorical column name -> fitted LabelEncoder.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[NUMERIC_FEATURES])

    label_encoders: dict = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        le.fit(X_train[col].astype(str))
        label_encoders[col] = le

    return scaler, label_encoders


def _apply_preprocessors(
    X: pd.DataFrame,
    scaler: StandardScaler,
    label_encoders: dict,
) -> np.ndarray:
    """
    Apply already-fitted preprocessors to a DataFrame.

    Unseen categorical values are mapped to a special 'unknown' index.
    """
    X = X.copy()

    # Encode categoricals
    for col, le in label_encoders.items():
        known_classes = set(le.classes_)
        X[col] = X[col].astype(str).apply(
            lambda v: v if v in known_classes else le.classes_[0]
        )
        X[col] = le.transform(X[col])

    # Scale numerics
    X[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])

    return X[ALL_FEATURES].values.astype(np.float64)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _build_model(
    model_type: Literal["isolation_forest", "one_class_svm"],
    contamination: float,
    random_state: int,
) -> object:
    """Instantiate the chosen anomaly detection model."""
    if model_type == "isolation_forest":
        return IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "one_class_svm":
        return OneClassSVM(
            kernel="rbf",
            nu=contamination,
            gamma="scale",
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose 'isolation_forest' or 'one_class_svm'."
        )


def _fit_model(model: object, X_normal_train: np.ndarray) -> object:
    """Fit the anomaly detector on normal (non-anomalous) training records."""
    logger.info("Fitting anomaly detector on %d normal training samples.", len(X_normal_train))
    model.fit(X_normal_train)
    return model


# ---------------------------------------------------------------------------
# Threshold selection on validation set
# ---------------------------------------------------------------------------

def _select_threshold(
    model: object,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: Literal["f1", "precision", "recall"] = "f1",
) -> float:
    """
    Select the decision threshold that maximises `metric` on the validation set.

    The model's decision_function / score_samples returns higher values for
    inliers and lower values for outliers.  We negate the scores so that
    higher values correspond to anomalies (consistent with probability-like
    interpretation).

    Parameters
    ----------
    model   : Fitted anomaly detector.
    X_val   : Preprocessed validation features.
    y_val   : True binary labels (0=normal, 1=anomaly).
    metric  : Optimisation target.

    Returns
    -------
    best_threshold : float
    """
    scores = _anomaly_scores(model, X_val)  # higher → more anomalous

    precisions, recalls, thresholds = precision_recall_curve(y_val, scores)

    if metric == "f1":
        # Avoid division by zero
        denom = precisions[:-1] + recalls[:-1]
        denom = np.where(denom == 0, 1e-9, denom)
        f1_scores = 2 * precisions[:-1] * recalls[:-1] / denom
        best_idx = int(np.argmax(f1_scores))
    elif metric == "precision":
        best_idx = int(np.argmax(precisions[:-1]))
    elif metric == "recall":
        best_idx = int(np.argmax(recalls[:-1]))
    else:
        raise ValueError(f"Unknown metric '{metric}'.")

    best_threshold = float(thresholds[best_idx])
    logger.info(
        "Selected threshold=%.4f (optimised for %s on validation set).",
        best_threshold,
        metric,
    )
    return best_threshold


def _anomaly_scores(model: object, X: np.ndarray) -> np.ndarray:
    """
    Return anomaly scores where higher values indicate more anomalous records.

    Both IsolationForest and OneClassSVM expose `decision_function` which
    returns negative scores for outliers; we negate so outliers score high.
    """
    raw_scores = model.decision_function(X)
    return -raw_scores  # invert: higher → more anomalous


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> Tuple[DetectionMetrics, PRCurveData]:
    """
    Evaluate the model on the held-out test set.

    Parameters
    ----------
    model     : Fitted anomaly detector.
    X_test    : Preprocessed test features.
    y_test    : True binary labels.
    threshold : Decision threshold (from validation set).

    Returns
    -------
    metrics   : DetectionMetrics
    pr_curve  : PRCurveData
    """
    scores = _anomaly_scores(model, X_test)
    y_pred = (scores >= threshold).astype(int)

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    precisions, recalls, thresholds = precision_recall_curve(y_test, scores)
    pr_auc = float(auc(recalls, precisions))

    metrics = DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
        threshold=threshold,
    )

    pr_curve = PRCurveData(
        precisions=precisions,
        recalls=recalls,
        thresholds=thresholds,
        pr_auc=pr_auc,
    )

    logger.info(
        "Test metrics — Precision: %.4f | Recall: %.4f | F1: %.4f | PR-AUC: %.4f",
        precision,
        recall,
        f1,
        pr_auc,
    )
    return metrics, pr_curve


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_precision_recall_curve(
    pr_curve: PRCurveData,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the precision-recall curve.

    Parameters
    ----------
    pr_curve  : PRCurveData instance.
    save_path : If provided, save the figure to this path.

    Returns
    -------
    fig : matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        pr_curve.recalls,
        pr_curve.precisions,
        color="steelblue",
        lw=2,
        label=f"PR curve (AUC = {pr_curve.pr_auc:.4f})",
    )
    ax.fill_between(pr_curve.recalls, pr_curve.precisions, alpha=0.15, color="steelblue")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve — Network Intrusion Detection", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        logger.info("PR curve saved to '%s'.", save_path)

    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_anomaly_detector(
    df: pd.DataFrame,
    model_type: Literal["isolation_forest", "one_class_svm"] = "isolation_forest",
    test_size: float = 0.20,
    val_size: float = 0.15,
    threshold_metric: Literal["f1", "precision", "recall"] = "f1",
    contamination: float = 0.05,
    random_state: int = 42,
    plot_curve: bool = True,
    save_plot_path: Optional[str] = None,
) -> AnomalyDetectionResult:
    """
    Build, train, and evaluate a network intrusion anomaly detector.

    Data splits (performed BEFORE any preprocessing):
        - Train  : (1 - test_size - val_size) fraction, normal records only for fitting.
        - Val    : val_size fraction, used for threshold selection.
        - Test   : test_size fraction, used ONLY for final evaluation.

    Parameters
    ----------
    df               : DataFrame with columns in ALL_FEATURES + [LABEL_COLUMN].
    model_type       : 'isolation_forest' or 'one_class_svm'.
    test_size        : Fraction of data reserved for final testing.
    val_size         : Fraction of data reserved for threshold tuning.
    threshold_metric : Metric to optimise when selecting the decision threshold.
    contamination    : Expected fraction of anomalies (used by the model).
    random_state     : Reproducibility seed.
    plot_curve       : Whether to display the PR curve.
    save_plot_path   : Optional file path to save the PR curve figure.

    Returns
    -------
    AnomalyDetectionResult
    """
    # ------------------------------------------------------------------
    # 1. Validate input
    # ------------------------------------------------------------------
    _validate_dataframe(df)
    df = df.copy().reset_index(drop=True)

    X = df[ALL_FEATURES]
    y = df[LABEL_COLUMN].astype(int).values

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Label column must contain only 0 (normal) and 1 (anomaly).")

    # ------------------------------------------------------------------
    # 2. Split BEFORE any preprocessing
    #    train+val | test
    #    train     | val   (from train+val)
    # ------------------------------------------------------------------
    trainval_size = 1.0 - test_size
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    relative_val_size = val_size / trainval_size
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    logger.info(
        "Split sizes — Train: %d | Val: %d | Test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # ------------------------------------------------------------------
    # 3. Fit preprocessors on training data ONLY
    # ------------------------------------------------------------------
    scaler, label_encoders = _fit_preprocessors(X_train)

    X_train_proc = _apply_preprocessors(X_train, scaler, label_encoders)
    X_val_proc   = _apply_preprocessors(X_val,   scaler, label_encoders)
    X_test_proc  = _apply_preprocessors(X_test,  scaler, label_encoders)

    # ------------------------------------------------------------------
    # 4. Fit model on NORMAL training records only
    # ------------------------------------------------------------------
    normal_mask_train = y_train == 0
    X_normal_train = X_train_proc[normal_mask_train]

    if len(X_normal_train) == 0:
        raise ValueError("No normal records found in the training split.")

    model = _build_model(model_type, contamination, random_state)
    model = _fit_model(model, X_normal_train)

    # ------------------------------------------------------------------
    # 5. Select threshold on VALIDATION set (test set untouched)
    # ------------------------------------------------------------------
    threshold = _select_threshold(model, X_val_proc, y_val, metric=threshold_metric)

    # ------------------------------------------------------------------
    # 6. Evaluate on TEST set (final, one-time evaluation)
    # ------------------------------------------------------------------
    metrics, pr_curve = _evaluate(model, X_test_proc, y_test, threshold)

    # ------------------------------------------------------------------
    # 7. Plot precision-recall curve
    # ------------------------------------------------------------------
    if plot_curve:
        fig = plot_precision_recall_curve(pr_curve, save_path=save_plot_path)
        plt.show()

    return AnomalyDetectionResult(
        model=model,
        scaler=scaler,
        label_encoders=label_encoders,
        threshold=threshold,
        metrics=metrics,
        pr_curve=pr_curve,
    )


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict(
    result: AnomalyDetectionResult,
    df_new: pd.DataFrame,
) -> np.ndarray:
    """
    Apply a trained anomaly detector to new records.

    Parameters
    ----------
    result : AnomalyDetectionResult from build_anomaly_detector.
    df_new : DataFrame with the same feature columns (label not required).

    Returns
    -------
    predictions : np.ndarray of shape (n,) with values 0 (normal) or 1 (anomaly).
    """
    required_features = set(ALL_FEATURES)
    missing = required_features - set(df_new.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    X_proc = _apply_preprocessors(df_new[ALL_FEATURES], result.scaler, result.label_encoders)
    scores = _anomaly_scores(result.model, X_proc)
    return (scores >= result.threshold).astype(int)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_normal: int = 2000,
    n_anomaly: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a small synthetic dataset for demonstration purposes."""
    rng = np.random.default_rng(random_state)

    protocols = ["tcp", "udp", "icmp"]
    flags = ["SF", "S0", "REJ", "RSTO"]
    services = ["http", "ftp", "smtp", "ssh", "dns"]

    def _make_records(n: int, anomaly: bool) -> pd.DataFrame:
        if anomaly:
            duration      = rng.exponential(scale=500, size=n)
            bytes_sent    = rng.integers(100_000, 10_000_000, size=n)
            bytes_received= rng.integers(0, 500, size=n)
            num_packets   = rng.integers(5000, 100_000, size=n)
        else:
            duration      = rng.exponential(scale=10, size=n)
            bytes_sent    = rng.integers(100, 50_000, size=n)
            bytes_received= rng.integers(100, 50_000, size=n)
            num_packets   = rng.integers(1, 500, size=n)

        return pd.DataFrame({
            "duration":       duration,
            "bytes_sent":     bytes_sent.astype(float),
            "bytes_received": bytes_received.astype(float),
            "num_packets":    num_packets.astype(float),
            "protocol_type":  rng.choice(protocols, size=n),
            "flag":           rng.choice(flags, size=n),
            "service":        rng.choice(services, size=n),
            "label":          int(anomaly),
        })

    normal_df  = _make_records(n_normal,  anomaly=False)
    anomaly_df = _make_records(n_anomaly, anomaly=True)
    df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


if __name__ == "__main__":
    logger.info("Generating synthetic network connection data …")
    data = _generate_synthetic_data(n_normal=2000, n_anomaly=200)

    logger.info("Running anomaly detection pipeline …")
    result = build_anomaly_detector(
        df=data,
        model_type="isolation_forest",
        test_size=0.20,
        val_size=0.15,
        threshold_metric="f1",
        contamination=0.05,
        random_state=42,
        plot_curve=True,
    )

    print("\n=== Detection Metrics ===")
    print(f"  Precision : {result.metrics.precision:.4f}")
    print(f"  Recall    : {result.metrics.recall:.4f}")
    print(f"  F1 Score  : {result.metrics.f1:.4f}")
    print(f"