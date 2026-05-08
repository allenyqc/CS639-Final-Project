```python
"""
Anomaly Detection System for Network Intrusion Detection
=========================================================
Supports Isolation Forest and One-Class SVM detectors.
"""

from __future__ import annotations

import warnings
from typing import Literal, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ModelType = Literal["isolation_forest", "one_class_svm"]
PRCurveData = Dict[str, np.ndarray]
Metrics = Dict[str, float]


# ---------------------------------------------------------------------------
# Helper: encode categorical features
# ---------------------------------------------------------------------------
def _encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns; reuse fitted encoders when fit=False."""
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def build_anomaly_detector(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label",
    feature_cols: list[str] | None = None,
    model_type: ModelType = "isolation_forest",
    contamination: float = 0.05,
    threshold_percentile: float = 5.0,
    # Isolation Forest kwargs
    n_estimators: int = 200,
    max_samples: str | int = "auto",
    random_state: int = 42,
    # One-Class SVM kwargs
    kernel: str = "rbf",
    nu: float = 0.05,
    gamma: str | float = "scale",
    # Plot options
    plot: bool = True,
    figsize: Tuple[int, int] = (14, 10),
) -> Dict[str, Any]:
    """
    Build, train, and evaluate an anomaly detection system.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data. Should be predominantly normal (label == 0).
    test_df : pd.DataFrame
        Test data containing both normal and anomalous records.
    label_col : str
        Column name for binary labels (0 = normal, 1 = anomaly).
    feature_cols : list[str] | None
        Feature columns to use. If None, all columns except label_col are used.
    model_type : {"isolation_forest", "one_class_svm"}
        Which anomaly detector to use.
    contamination : float
        Expected fraction of anomalies in training data (used by IsolationForest).
    threshold_percentile : float
        Percentile of training anomaly scores used to set the decision threshold.
        Lower values → more sensitive (more anomalies flagged).
    n_estimators : int
        Number of trees for Isolation Forest.
    max_samples : str | int
        Samples per tree for Isolation Forest.
    random_state : int
        Random seed.
    kernel : str
        Kernel for One-Class SVM.
    nu : float
        Upper bound on fraction of outliers for One-Class SVM.
    gamma : str | float
        Kernel coefficient for One-Class SVM.
    plot : bool
        Whether to display the precision-recall curve and score distribution.
    figsize : tuple
        Figure size for plots.

    Returns
    -------
    dict with keys:
        "model"       – fitted sklearn Pipeline (scaler + detector)
        "threshold"   – float decision threshold on raw anomaly scores
        "encoders"    – dict of fitted LabelEncoders for categorical columns
        "metrics"     – dict {precision, recall, f1, pr_auc, threshold}
        "pr_curve"    – dict {precision, recall, thresholds}
        "predictions" – np.ndarray of binary predictions on test set
        "scores"      – np.ndarray of raw anomaly scores on test set
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    if label_col not in train_df.columns or label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    categorical_cols = ["protocol_type", "flag", "service"]

    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c != label_col]

    # Keep only feature columns + label
    train_df = train_df[feature_cols + [label_col]].copy()
    test_df = test_df[feature_cols + [label_col]].copy()

    # ------------------------------------------------------------------
    # 2. Encode categoricals
    # ------------------------------------------------------------------
    cat_present = [c for c in categorical_cols if c in feature_cols]
    train_df, encoders = _encode_categoricals(train_df, cat_present, fit=True)
    test_df, _ = _encode_categoricals(test_df, cat_present, encoders=encoders, fit=False)

    # ------------------------------------------------------------------
    # 3. Split features / labels
    # ------------------------------------------------------------------
    X_train_all = train_df[feature_cols].values.astype(float)
    y_train = train_df[label_col].values.astype(int)

    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df[label_col].values.astype(int)

    # Train only on normal records
    normal_mask = y_train == 0
    X_train_normal = X_train_all[normal_mask]

    print(f"[INFO] Training samples (normal only): {X_train_normal.shape[0]}")
    print(f"[INFO] Test samples: {X_test.shape[0]}  "
          f"(normal={np.sum(y_test == 0)}, anomaly={np.sum(y_test == 1)})")

    # ------------------------------------------------------------------
    # 4. Build pipeline: scaler + detector
    # ------------------------------------------------------------------
    scaler = StandardScaler()

    if model_type == "isolation_forest":
        detector = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "one_class_svm":
        detector = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         "Choose 'isolation_forest' or 'one_class_svm'.")

    pipeline = Pipeline([("scaler", scaler), ("detector", detector)])

    # ------------------------------------------------------------------
    # 5. Fit on normal training data
    # ------------------------------------------------------------------
    pipeline.fit(X_train_normal)
    print(f"[INFO] Model '{model_type}' fitted on {X_train_normal.shape[0]} normal samples.")

    # ------------------------------------------------------------------
    # 6. Compute anomaly scores
    #    score_samples returns higher = more normal for both models.
    #    We negate so that higher score → more anomalous.
    # ------------------------------------------------------------------
    train_scores_raw = pipeline.score_samples(X_train_normal)   # higher = normal
    anomaly_scores_train = -train_scores_raw                     # higher = anomalous

    test_scores_raw = pipeline.score_samples(X_test)
    anomaly_scores_test = -test_scores_raw

    # ------------------------------------------------------------------
    # 7. Determine decision threshold
    #    Use the (100 - threshold_percentile) percentile of training scores
    #    so that ~threshold_percentile% of normal training data is flagged.
    # ------------------------------------------------------------------
    threshold = float(np.percentile(anomaly_scores_train, 100 - threshold_percentile))
    print(f"[INFO] Decision threshold (percentile={100 - threshold_percentile:.1f}%): "
          f"{threshold:.6f}")

    # ------------------------------------------------------------------
    # 8. Generate binary predictions on test set
    # ------------------------------------------------------------------
    y_pred = (anomaly_scores_test >= threshold).astype(int)

    # ------------------------------------------------------------------
    # 9. Compute metrics
    # ------------------------------------------------------------------
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Precision-recall curve (use anomaly score as confidence)
    pr_precisions, pr_recalls, pr_thresholds = precision_recall_curve(
        y_test, anomaly_scores_test
    )
    pr_auc_score = auc(pr_recalls, pr_precisions)

    metrics: Metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc_score),
        "threshold": float(threshold),
    }

    pr_curve_data: PRCurveData = {
        "precision": pr_precisions,
        "recall": pr_recalls,
        "thresholds": pr_thresholds,
    }

    # ------------------------------------------------------------------
    # 10. Print evaluation summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Model          : {model_type}")
    print(f"  Threshold      : {threshold:.6f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  F1 Score       : {f1:.4f}")
    print(f"  PR AUC         : {pr_auc_score:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
    print("=" * 60)

    # ------------------------------------------------------------------
    # 11. Plot
    # ------------------------------------------------------------------
    if plot:
        _plot_results(
            pr_precisions=pr_precisions,
            pr_recalls=pr_recalls,
            pr_auc_score=pr_auc_score,
            anomaly_scores_train=anomaly_scores_train,
            anomaly_scores_test=anomaly_scores_test,
            y_test=y_test,
            threshold=threshold,
            model_type=model_type,
            figsize=figsize,
        )

    return {
        "model": pipeline,
        "threshold": threshold,
        "encoders": encoders,
        "metrics": metrics,
        "pr_curve": pr_curve_data,
        "predictions": y_pred,
        "scores": anomaly_scores_test,
    }


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def _plot_results(
    pr_precisions: np.ndarray,
    pr_recalls: np.ndarray,
    pr_auc_score: float,
    anomaly_scores_train: np.ndarray,
    anomaly_scores_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    model_type: str,
    figsize: Tuple[int, int],
) -> None:
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ---- (A) Precision-Recall Curve ----
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        pr_recalls, pr_precisions,
        color="steelblue", lw=2,
        label=f"PR Curve (AUC = {pr_auc_score:.4f})",
    )
    ax1.fill_between(pr_recalls, pr_precisions, alpha=0.15, color="steelblue")
    ax1.set_xlabel("Recall", fontsize=12)
    ax1.set_ylabel("Precision", fontsize=12)
    ax1.set_title(
        f"Precision-Recall Curve — {model_type.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold",
    )
    ax1.legend(fontsize=11)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)

    # ---- (B) Score distribution on training (normal) data ----
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(anomaly_scores_train, bins=50, color="seagreen", alpha=0.75,
             edgecolor="white", label="Normal (train)")
    ax2.axvline(threshold, color="crimson", lw=2, linestyle="--",
                label=f"Threshold = {threshold:.4f}")
    ax2.set_xlabel("Anomaly Score", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Score Distribution — Training (Normal)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ---- (C) Score distribution on test data (normal vs anomaly) ----
    ax3 = fig.add_subplot(gs[1, 1])
    normal_scores = anomaly_scores_test[y_test == 0]
    anomaly_scores = anomaly_scores_test[y_test == 1]

    ax3.hist(normal_scores, bins=50, color="seagreen", alpha=0.65,
             edgecolor="white", label=f"Normal (n={len(normal_scores)})")
    ax3.hist(anomaly_scores, bins=50, color="tomato", alpha=0.65,
             edgecolor="white", label=f"Anomaly (n={len(anomaly_scores)})")
    ax3.axvline(threshold, color="navy", lw=2, linestyle="--",
                label=f"Threshold = {threshold:.4f}")
    ax3.set_xlabel("Anomaly Score", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("Score Distribution — Test Set", fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.suptitle(
        "Network Intrusion Anomaly Detection",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Convenience: generate synthetic demo data
# ---------------------------------------------------------------------------
def generate_demo_data(
    n_train: int = 5000,
    n_test_normal: int = 1000,
    n_test_anomaly: int = 200,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic network connection records for demonstration.

    Normal traffic: short durations, moderate bytes, common protocols.
    Anomalous traffic: unusual byte volumes, rare flags, long durations.
    """
    rng = np.random.default_rng(random_state)

    protocols = ["tcp", "udp", "icmp"]
    flags = ["SF", "S0", "REJ", "RSTO", "SH"]
    services = ["http", "ftp", "smtp", "ssh", "dns", "other"]

    def _make_normal(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "duration":        rng.exponential(scale=5, size=n),
            "bytes_sent":      rng.lognormal(mean=7, sigma=1.5, size=n),
            "bytes_received":  rng.lognormal(mean=8, sigma=1.5, size=n),
            "num_packets":     rng.integers(1, 50, size=n).astype(float),
            "protocol_type":   rng.choice(protocols, size=n, p=[0.6, 0.3, 0.1]),
            "flag":            rng.choice(flags, size=n, p=[0.8, 0.1, 0.05, 0.03, 0.02]),
            "service":         rng.choice(services, size=n),
            "label":           np.zeros(n, dtype=int),
        })

    def _make_anomaly(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "duration":        rng.exponential(scale=200, size=n),
            "bytes_sent":      rng.lognormal(mean=12, sigma=2, size=n),
            "bytes_received":  rng.lognormal(mean=4, sigma=3, size=n),
            "num_packets":     rng.integers(200, 5000, size=n).astype(float),
            "protocol_type":   rng.choice(protocols, size=n, p=[0.2, 0.2, 0.6]),
            "flag":            rng.choice(flags, size=n, p=[0.1, 0.5, 0.2, 0.1, 0.1]),
            "service":         rng.choice(services, size=n),
            "label":           np.ones(n, dtype=int),
        })

    # Training: mostly normal + tiny fraction of anomalies
    n_train_anomaly = max(1, int(n_train * 0.02))
    train_df = pd.concat(
        [_make_normal(n_train - n_train_anomaly), _make_anomaly(n_train_anomaly)],
        ignore_index=True,
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Test: balanced mix
    test_df = pd.concat(
        [_make_normal(n_test_normal), _make_anomaly(n_test_anomaly)],
        ignore_index=True,
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df, test_df


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating synthetic network traffic data …")
    train_data, test_data = generate_demo_data(
        n_train=6000,
        n_test_normal=1200,
        n_test_anomaly=300,
        random_state=0,
    )

    print(f"Train shape: {train_data.shape}  "
          f"(anomaly rate: {train_data['label'].mean():.2%})")
    print(f"Test  shape: {test_data.shape}  "
          f"(anomaly rate: {test_data['label'].mean():.2%})\n")

    # ---- Run with Isolation Forest ----
    results_if = build_anomaly_detector(