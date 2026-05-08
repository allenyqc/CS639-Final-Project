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
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ModelType = Literal["isolation_forest", "one_class_svm"]
PRCurveData = Dict[str, np.ndarray]
Metrics = Dict[str, float]
Result = Tuple[Any, float, Metrics, PRCurveData]

# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = ["protocol_type", "flag", "service"]
NUMERIC_COLS = ["duration", "bytes_sent", "bytes_received", "num_packets"]


def _encode_categoricals(
    df: pd.DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns; reuse fitted encoders when fit=False."""
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
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


def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Return a numpy feature matrix from a pre-encoded DataFrame."""
    feature_cols = NUMERIC_COLS + [c for c in CATEGORICAL_COLS if c in df.columns]
    return df[feature_cols].values.astype(float)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_anomaly_detector(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: ModelType = "isolation_forest",
    contamination: float = 0.05,
    threshold_percentile: float = 5.0,
    # Isolation Forest hyper-parameters
    n_estimators: int = 200,
    max_samples: str | int = "auto",
    random_state: int = 42,
    # One-Class SVM hyper-parameters
    nu: float = 0.05,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    # Plotting
    plot: bool = True,
    figsize: Tuple[int, int] = (14, 10),
) -> Result:
    """
    Build, train, and evaluate an anomaly detection system.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.  Must contain feature columns and a ``label`` column
        (0 = normal, 1 = anomaly).  The detector is fitted **only** on the
        normal records.
    test_df : pd.DataFrame
        Test data with the same schema.  Should contain both normal and
        anomalous records.
    model_type : {'isolation_forest', 'one_class_svm'}
        Which unsupervised detector to use.
    contamination : float
        Expected fraction of outliers (used by Isolation Forest).
    threshold_percentile : float
        Percentile of the *normal-training-set* score distribution used to
        set the decision threshold.  Scores below this threshold are flagged
        as anomalies.
    n_estimators : int
        Number of trees for Isolation Forest.
    max_samples : int or 'auto'
        Sub-sample size for Isolation Forest.
    random_state : int
        Random seed.
    nu : float
        Upper bound on the fraction of outliers for One-Class SVM.
    kernel : str
        Kernel for One-Class SVM.
    gamma : str or float
        Kernel coefficient for One-Class SVM.
    plot : bool
        Whether to render the diagnostic plots.
    figsize : tuple
        Figure size for the plots.

    Returns
    -------
    model : fitted sklearn Pipeline (scaler + detector)
    threshold : float
        Decision threshold on the raw anomaly score.
    metrics : dict
        precision, recall, f1, pr_auc, threshold.
    pr_curve_data : dict
        Keys ``precision``, ``recall``, ``thresholds``.
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    required_cols = NUMERIC_COLS + ["label"]
    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Missing required column '{col}' in train_df.")
        if col not in test_df.columns:
            raise ValueError(f"Missing required column '{col}' in test_df.")

    # ------------------------------------------------------------------
    # 2. Encode categoricals
    # ------------------------------------------------------------------
    train_enc, encoders = _encode_categoricals(train_df, fit=True)
    test_enc, _ = _encode_categoricals(test_df, encoders=encoders, fit=False)

    # ------------------------------------------------------------------
    # 3. Split normal / anomaly in training set
    # ------------------------------------------------------------------
    train_normal = train_enc[train_enc["label"] == 0]
    train_anomaly = train_enc[train_enc["label"] == 1]

    print(
        f"[Train] Normal: {len(train_normal):,}  |  "
        f"Anomaly: {len(train_anomaly):,}  |  "
        f"Anomaly ratio: {len(train_anomaly)/len(train_enc):.2%}"
    )
    print(
        f"[Test ] Normal: {(test_enc['label']==0).sum():,}  |  "
        f"Anomaly: {(test_enc['label']==1).sum():,}"
    )

    X_train_normal = _prepare_features(train_normal)
    X_test = _prepare_features(test_enc)
    y_test = test_enc["label"].values

    # ------------------------------------------------------------------
    # 4. Build pipeline (scaler + detector)
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
        detector = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         "Choose 'isolation_forest' or 'one_class_svm'.")

    # Fit scaler on normal training data, then fit detector
    X_train_scaled = scaler.fit_transform(X_train_normal)
    detector.fit(X_train_scaled)

    # ------------------------------------------------------------------
    # 5. Compute anomaly scores on test set
    #    score_samples() returns higher values for *normal* points.
    #    We negate so that higher score → more anomalous.
    # ------------------------------------------------------------------
    X_test_scaled = scaler.transform(X_test)
    raw_scores = detector.score_samples(X_test_scaled)   # higher = more normal
    anomaly_scores = -raw_scores                          # higher = more anomalous

    # ------------------------------------------------------------------
    # 6. Determine decision threshold from normal training scores
    # ------------------------------------------------------------------
    train_scores_normal = -detector.score_samples(X_train_scaled)
    threshold = float(np.percentile(train_scores_normal, 100 - threshold_percentile))
    print(f"\nDecision threshold (at {100-threshold_percentile:.0f}th percentile "
          f"of normal training scores): {threshold:.6f}")

    # ------------------------------------------------------------------
    # 7. Predict labels: score > threshold → anomaly (1)
    # ------------------------------------------------------------------
    y_pred = (anomaly_scores > threshold).astype(int)

    # ------------------------------------------------------------------
    # 8. Compute metrics
    # ------------------------------------------------------------------
    precision_val = precision_score(y_test, y_pred, zero_division=0)
    recall_val = recall_score(y_test, y_pred, zero_division=0)
    f1_val = f1_score(y_test, y_pred, zero_division=0)

    # PR curve (using continuous anomaly scores)
    precisions, recalls, pr_thresholds = precision_recall_curve(
        y_test, anomaly_scores
    )
    pr_auc_val = auc(recalls, precisions)

    metrics: Metrics = {
        "precision": float(precision_val),
        "recall": float(recall_val),
        "f1": float(f1_val),
        "pr_auc": float(pr_auc_val),
        "threshold": float(threshold),
    }

    pr_curve_data: PRCurveData = {
        "precision": precisions,
        "recall": recalls,
        "thresholds": pr_thresholds,
    }

    print("\n── Evaluation Metrics ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print("────────────────────────────────────────────────────\n")

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    if plot:
        _plot_results(
            anomaly_scores=anomaly_scores,
            y_test=y_test,
            y_pred=y_pred,
            threshold=threshold,
            precisions=precisions,
            recalls=recalls,
            pr_auc_val=pr_auc_val,
            train_scores_normal=train_scores_normal,
            model_type=model_type,
            figsize=figsize,
        )

    # ------------------------------------------------------------------
    # 10. Package model as a simple callable wrapper
    # ------------------------------------------------------------------
    model = _AnomalyDetectorWrapper(
        scaler=scaler,
        detector=detector,
        encoders=encoders,
        threshold=threshold,
        model_type=model_type,
    )

    return model, threshold, metrics, pr_curve_data


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _plot_results(
    anomaly_scores: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    precisions: np.ndarray,
    recalls: np.ndarray,
    pr_auc_val: float,
    train_scores_normal: np.ndarray,
    model_type: str,
    figsize: Tuple[int, int],
) -> None:
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Anomaly Detection — {model_type.replace('_', ' ').title()}",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── (A) Precision-Recall Curve ──────────────────────────────────────
    ax_pr = fig.add_subplot(gs[0, 0])
    ax_pr.plot(recalls, precisions, color="steelblue", lw=2,
               label=f"PR AUC = {pr_auc_val:.3f}")
    ax_pr.fill_between(recalls, precisions, alpha=0.15, color="steelblue")
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision-Recall Curve", fontsize=12)
    ax_pr.legend(fontsize=10)
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1.05])
    ax_pr.grid(alpha=0.3)

    # ── (B) Score distribution ──────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[0, 1])
    normal_scores = anomaly_scores[y_test == 0]
    anomaly_scores_pos = anomaly_scores[y_test == 1]

    bins = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 60)
    ax_dist.hist(normal_scores, bins=bins, alpha=0.6, color="royalblue",
                 label="Normal", density=True)
    ax_dist.hist(anomaly_scores_pos, bins=bins, alpha=0.6, color="tomato",
                 label="Anomaly", density=True)
    ax_dist.axvline(threshold, color="black", linestyle="--", lw=1.8,
                    label=f"Threshold = {threshold:.3f}")
    ax_dist.set_xlabel("Anomaly Score", fontsize=11)
    ax_dist.set_ylabel("Density", fontsize=11)
    ax_dist.set_title("Score Distribution (Test Set)", fontsize=12)
    ax_dist.legend(fontsize=9)
    ax_dist.grid(alpha=0.3)

    # ── (C) Confusion Matrix ────────────────────────────────────────────
    ax_cm = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Normal", "Anomaly"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title("Confusion Matrix", fontsize=12)

    # ── (D) Training score distribution + threshold ─────────────────────
    ax_train = fig.add_subplot(gs[1, 1])
    ax_train.hist(train_scores_normal, bins=50, color="mediumseagreen",
                  alpha=0.75, density=True, label="Normal (train)")
    ax_train.axvline(threshold, color="black", linestyle="--", lw=1.8,
                     label=f"Threshold = {threshold:.3f}")
    ax_train.set_xlabel("Anomaly Score", fontsize=11)
    ax_train.set_ylabel("Density", fontsize=11)
    ax_train.set_title("Training Score Distribution (Normal)", fontsize=12)
    ax_train.legend(fontsize=9)
    ax_train.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("anomaly_detection_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to 'anomaly_detection_results.png'.")


# ---------------------------------------------------------------------------
# Wrapper class for inference
# ---------------------------------------------------------------------------

class _AnomalyDetectorWrapper:
    """Thin wrapper that exposes a predict() / score() interface."""

    def __init__(
        self,
        scaler: StandardScaler,
        detector,
        encoders: dict,
        threshold: float,
        model_type: str,
    ) -> None:
        self.scaler = scaler
        self.detector = detector
        self.encoders = encoders
        self.threshold = threshold
        self.model_type = model_type

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        df_enc, _ = _encode_categoricals(df, encoders=self.encoders, fit=False)
        X = _prepare_features(df_enc)
        return self.scaler.transform(X)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        X_scaled = self._preprocess(df)
        return -self.detector.score_samples(X_scaled)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (1 = anomaly, 0 = normal)."""
        scores = self.score(df)
        return (scores > self.threshold).astype(int)

    def __repr__(self) -> str:
        return (
            f"AnomalyDetector(model_type={self.model_type!r}, "
            f"threshold={self.threshold:.4f})"
        )


# ---------------------------------------------------------------------------
# Convenience: generate synthetic demo data
# ---------------------------------------------------------------------------

def generate_demo_data(
    n_normal_train: int = 5000,
    n_normal_test: int = 1000,
    n_anomaly_test: int = 200,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic network connection records for demonstration.

    Normal traffic follows a multivariate Gaussian; anomalies are drawn
    from a shifted / scaled distribution.
    """
    rng = np.random.default_rng(random_state)

    protocols = ["tcp", "udp", "icmp"]
    flags = ["SF", "S0", "REJ", "RSTO", "SH"]
    services = ["http", "ftp", "smtp", "ssh", "dns", "other"]

    def _make_records(n: int, anomaly: bool) -> pd.DataFrame:
        if not anomaly:
            duration = rng.exponential(scale=5, size=n)
            bytes_sent = rng.normal(loc=500, scale=150, size=n).clip(0)
            bytes_received = rng.normal(loc=800, scale=200, size=n).clip(0)
            num_packets = rng.integers(1, 50, size=n)
        else:
            # Anomalies: very high byte counts, unusual durations
            duration = rng.choice(
                [rng.exponential(scale=0.1, size=n),
                 rng.exponential(scale=500, size=n)],
                axis=0,
            ).diagonal()
            bytes_sent = rng.normal(loc=50_000, scale=10_000, size=n).clip(0)
            bytes_received = rng.normal(loc=5, scale=2, size=n).clip(0)
            num_packets = rng.integers(500, 5000, size=n)

        return pd.DataFrame(
            {
                "duration": duration,
                "bytes_sent": bytes_sent,
                "bytes_received": bytes_received,
                "num_packets": num_packets,
                "protocol_type": rng.choice(protocols, size=n),
                "flag": rng.choice(flags, size=n),
                "service": rng.choice(services, size=n),
                "label": int(anomaly),
            }
        )

    train_normal = _make_records(n_normal_train, anomaly=False)
    # Add a tiny fraction of anomalies to training set (realistic scenario)
    n_train_anomaly = max(1, int(n_normal_train * 0.02))
    train_anomaly = _make_records(n_train_anomaly, anomaly=True)
    train_df = (
        pd.concat([train_normal, train_anomaly], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    test_normal = _make_records(n_normal_test, anomaly=False)
    test_anomaly = _make_records(n_anomaly_test, anomaly=True)
    test_df = (
        pd.concat([test_normal, test_anomaly], ignore_index=True)
        .sample(frac=1, random_state=