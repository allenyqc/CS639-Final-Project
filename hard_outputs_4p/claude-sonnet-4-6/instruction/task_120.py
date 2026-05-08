"""
Network Intrusion Detection - Anomaly Detection Module
======================================================
Builds an anomaly detection system using Isolation Forest or One-Class SVM.
Follows strict train/test separation and best practices for ML pipelines.
"""

import warnings
from typing import Dict, Literal, Optional, Tuple

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ModelType = Literal["isolation_forest", "one_class_svm"]

CATEGORICAL_FEATURES = ["protocol_type", "flag", "service"]
NUMERIC_FEATURES = ["duration", "bytes_sent", "bytes_received", "num_packets"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Helper: encode categoricals using mappings fitted ONLY on training data
# ---------------------------------------------------------------------------
class CategoricalEncoder:
    """Fit label encoders on training data; transform train and test."""

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.encoders: Dict[str, LabelEncoder] = {}

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        for col in self.columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.columns:
            le = self.encoders[col]
            known_classes = set(le.classes_)
            # Map unseen categories to a placeholder (last known class index)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known_classes else le.classes_[-1]
            )
            df[col] = le.transform(df[col])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def build_anomaly_detector(
    df: pd.DataFrame,
    model_type: ModelType = "isolation_forest",
    test_size: float = 0.25,
    random_state: int = 42,
    contamination: float = 0.05,
    # Isolation Forest params
    n_estimators: int = 200,
    max_samples: str | int = "auto",
    # One-Class SVM params
    kernel: str = "rbf",
    nu: float = 0.05,
    gamma: str | float = "scale",
    # Threshold strategy: "percentile" uses training score distribution
    threshold_percentile: float = 5.0,
    plot: bool = True,
    figsize: Tuple[int, int] = (8, 6),
) -> Dict:
    """
    Build, train, and evaluate an anomaly detector for network intrusion detection.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: duration, bytes_sent, bytes_received, num_packets,
        protocol_type, flag, service, label (0=normal, 1=anomaly).
    model_type : str
        'isolation_forest' or 'one_class_svm'.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Reproducibility seed.
    contamination : float
        Expected fraction of anomalies (used by Isolation Forest).
    n_estimators : int
        Number of trees for Isolation Forest.
    max_samples : int or 'auto'
        Samples per tree for Isolation Forest.
    kernel : str
        Kernel for One-Class SVM.
    nu : float
        Upper bound on fraction of outliers for One-Class SVM.
    gamma : str or float
        Kernel coefficient for One-Class SVM.
    threshold_percentile : float
        Percentile of training anomaly scores used to set the decision threshold.
        Lower percentile → more sensitive (flags more as anomalies).
    plot : bool
        Whether to display the precision-recall curve.
    figsize : tuple
        Figure size for the PR curve plot.

    Returns
    -------
    dict with keys:
        model          – fitted sklearn estimator
        preprocessor   – dict with fitted scaler and categorical encoder
        threshold      – float decision threshold on raw scores
        metrics        – dict {precision, recall, f1, pr_auc}
        pr_curve       – dict {precisions, recalls, thresholds}
        test_labels    – ground-truth labels for the test set
        test_scores    – raw anomaly scores for the test set
    """
    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------
    required_cols = ALL_FEATURES + ["label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if model_type not in ("isolation_forest", "one_class_svm"):
        raise ValueError("model_type must be 'isolation_forest' or 'one_class_svm'")

    df = df.copy()
    df["label"] = df["label"].astype(int)

    # ------------------------------------------------------------------
    # 1. Train / test split BEFORE any preprocessing
    # ------------------------------------------------------------------
    X = df[ALL_FEATURES]
    y = df["label"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Training set: keep ONLY normal records for fitting the anomaly detector
    normal_mask = y_train == 0
    X_train_normal_raw = X_train_raw[normal_mask]

    print(
        f"[Split] Train total: {len(X_train_raw)} "
        f"(normal: {normal_mask.sum()}, anomaly: {(~normal_mask).sum()}) | "
        f"Test total: {len(X_test_raw)} "
        f"(normal: {(y_test==0).sum()}, anomaly: {(y_test==1).sum()})"
    )

    # ------------------------------------------------------------------
    # 2. Preprocessing — fit ONLY on normal training data
    # ------------------------------------------------------------------
    cat_encoder = CategoricalEncoder(CATEGORICAL_FEATURES)
    X_train_normal_enc = cat_encoder.fit_transform(X_train_normal_raw)

    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal_enc[ALL_FEATURES])

    # Transform full training set and test set using fitted transformers
    X_train_enc = cat_encoder.transform(X_train_raw)
    X_train_scaled = scaler.transform(X_train_enc[ALL_FEATURES])

    X_test_enc = cat_encoder.transform(X_test_raw)
    X_test_scaled = scaler.transform(X_test_enc[ALL_FEATURES])

    # ------------------------------------------------------------------
    # 3. Build and fit the anomaly detector on normal training data
    # ------------------------------------------------------------------
    if model_type == "isolation_forest":
        detector = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
    else:  # one_class_svm
        detector = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma,
        )

    detector.fit(X_train_normal_scaled)
    print(f"[Model] Fitted {model_type} on {len(X_train_normal_scaled)} normal records.")

    # ------------------------------------------------------------------
    # 4. Compute anomaly scores
    #    score_samples() returns higher = more normal for both estimators.
    #    We negate so that higher score → more anomalous.
    # ------------------------------------------------------------------
    train_normal_scores = -detector.score_samples(X_train_normal_scaled)
    test_scores = -detector.score_samples(X_test_scaled)

    # ------------------------------------------------------------------
    # 5. Set decision threshold using TRAINING (normal) score distribution
    #    Threshold = (100 - threshold_percentile)-th percentile of normal
    #    training scores.  Records with score > threshold are flagged.
    # ------------------------------------------------------------------
    threshold = float(np.percentile(train_normal_scores, 100.0 - threshold_percentile))
    print(
        f"[Threshold] {threshold:.6f} "
        f"(set at {100-threshold_percentile:.1f}th percentile of normal train scores)"
    )

    # ------------------------------------------------------------------
    # 6. Evaluate on the test set (never touched during training/threshold)
    # ------------------------------------------------------------------
    y_pred = (test_scores > threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Precision-recall curve (using continuous scores)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, test_scores)
    pr_auc = auc(recalls, precisions)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
    }

    print("\n[Evaluation Metrics on Test Set]")
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v:.4f}")

    # ------------------------------------------------------------------
    # 7. Plot precision-recall curve
    # ------------------------------------------------------------------
    fig = None
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recalls, precisions, lw=2, color="steelblue",
                label=f"PR curve (AUC = {pr_auc:.3f})")
        ax.axhline(y=precision, color="gray", linestyle="--", alpha=0.6,
                   label=f"Threshold precision = {precision:.3f}")
        ax.axvline(x=recall, color="salmon", linestyle="--", alpha=0.6,
                   label=f"Threshold recall = {recall:.3f}")
        ax.scatter([recall], [precision], zorder=5, color="red", s=80,
                   label=f"Operating point (F1={f1:.3f})")
        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title(
            f"Precision-Recall Curve — {model_type.replace('_', ' ').title()}",
            fontsize=14,
        )
        ax.legend(fontsize=10)
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # 8. Return artefacts
    # ------------------------------------------------------------------
    return {
        "model": detector,
        "preprocessor": {
            "categorical_encoder": cat_encoder,
            "scaler": scaler,
        },
        "threshold": threshold,
        "metrics": metrics,
        "pr_curve": {
            "precisions": precisions,
            "recalls": recalls,
            "thresholds": pr_thresholds,
        },
        "test_labels": y_test.values,
        "test_scores": test_scores,
        "figure": fig,
    }


# ---------------------------------------------------------------------------
# Convenience: predict on new data using a fitted artefact bundle
# ---------------------------------------------------------------------------
def predict(
    artefacts: Dict,
    X_new: pd.DataFrame,
) -> np.ndarray:
    """
    Apply a fitted anomaly detector to new records.

    Parameters
    ----------
    artefacts : dict
        Output of build_anomaly_detector().
    X_new : pd.DataFrame
        New records with the same feature columns.

    Returns
    -------
    np.ndarray of int (0=normal, 1=anomaly)
    """
    cat_encoder: CategoricalEncoder = artefacts["preprocessor"]["categorical_encoder"]
    scaler: StandardScaler = artefacts["preprocessor"]["scaler"]
    detector = artefacts["model"]
    threshold: float = artefacts["threshold"]

    X_enc = cat_encoder.transform(X_new[ALL_FEATURES])
    X_scaled = scaler.transform(X_enc[ALL_FEATURES])
    scores = -detector.score_samples(X_scaled)
    return (scores > threshold).astype(int)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------
def _generate_synthetic_data(
    n_normal: int = 4000,
    n_anomaly: int = 400,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic network connection dataset for demonstration."""
    rng = np.random.default_rng(random_state)

    protocols = ["tcp", "udp", "icmp"]
    flags = ["SF", "S0", "REJ", "RSTO", "SH"]
    services = ["http", "ftp", "smtp", "ssh", "dns", "other"]

    def _make_records(n: int, anomaly: bool) -> pd.DataFrame:
        if anomaly:
            duration = rng.exponential(scale=300, size=n)
            bytes_sent = rng.integers(0, 1_000_000, size=n)
            bytes_received = rng.integers(0, 500_000, size=n)
            num_packets = rng.integers(1, 5000, size=n)
            protocol = rng.choice(protocols, size=n, p=[0.2, 0.6, 0.2])
            flag = rng.choice(flags, size=n, p=[0.1, 0.4, 0.3, 0.1, 0.1])
            service = rng.choice(services, size=n)
            label = np.ones(n, dtype=int)
        else:
            duration = rng.exponential(scale=10, size=n)
            bytes_sent = rng.integers(0, 50_000, size=n)
            bytes_received = rng.integers(0, 100_000, size=n)
            num_packets = rng.integers(1, 200, size=n)
            protocol = rng.choice(protocols, size=n, p=[0.6, 0.3, 0.1])
            flag = rng.choice(flags, size=n, p=[0.8, 0.05, 0.05, 0.05, 0.05])
            service = rng.choice(services, size=n)
            label = np.zeros(n, dtype=int)

        return pd.DataFrame({
            "duration": duration,
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "num_packets": num_packets,
            "protocol_type": protocol,
            "flag": flag,
            "service": service,
            "label": label,
        })

    normal_df = _make_records(n_normal, anomaly=False)
    anomaly_df = _make_records(n_anomaly, anomaly=True)
    combined = pd.concat([normal_df, anomaly_df], ignore_index=True)
    return combined.sample(frac=1, random_state=random_state).reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Network Intrusion Detection — Anomaly Detection Demo")
    print("=" * 60)

    df = _generate_synthetic_data(n_normal=4000, n_anomaly=400)
    print(f"\nDataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    # --- Isolation Forest ---
    print("\n--- Isolation Forest ---")
    results_if = build_anomaly_detector(
        df,
        model_type="isolation_forest",
        test_size=0.25,
        contamination=0.05,
        n_estimators=200,
        threshold_percentile=5.0,
        plot=True,
    )

    # --- One-Class SVM ---
    print("\n--- One-Class SVM ---")
    results_svm = build_anomaly_detector(
        df,
        model_type="one_class_svm",
        test_size=0.25,
        nu=0.05,
        kernel="rbf",
        gamma="scale",
        threshold_percentile=5.0,
        plot=True,
    )

    # --- Demonstrate predict() on new records ---
    print("\n--- Predict on 5 new records ---")
    sample = df[ALL_FEATURES].sample(5, random_state=7)
    preds = predict(results_if, sample)
    print(pd.DataFrame({"prediction": preds}, index=sample.index))