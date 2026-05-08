"""
Probability Calibration Pipeline Module

Builds a calibration pipeline that:
1. Splits data into train/validation/test sets
2. Trains a base classifier
3. Calibrates using Platt scaling or isotonic regression (on validation set)
4. Evaluates calibration quality (Brier score, ECE, reliability diagram)
5. Compares calibrated vs uncalibrated predictions
6. Returns calibrated model, metrics, and reliability diagram data
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")  # Non-interactive backend; safe for server environments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityDiagramData:
    """Binned data for a reliability diagram."""
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    fraction_of_positives: np.ndarray
    mean_predicted_value: np.ndarray
    bin_counts: np.ndarray
    label: str = ""


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics for one model variant."""
    brier_score: float
    ece: float
    reliability: ReliabilityDiagramData
    label: str = ""


@dataclass
class CalibrationResult:
    """Full output of the calibration pipeline."""
    calibrated_model: Any
    uncalibrated_metrics: CalibrationMetrics
    calibrated_metrics: CalibrationMetrics
    test_size: int
    n_bins: int
    calibration_method: str
    figure: Optional[plt.Figure] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# ECE calculation (manual implementation)
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, ReliabilityDiagramData]:
    """
    Compute Expected Calibration Error (ECE) manually.

    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True binary labels.
    y_prob : array of shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bins : int
        Number of equal-width bins in [0, 1].

    Returns
    -------
    ece : float
    reliability_data : ReliabilityDiagramData
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise ValueError("y_prob values must be in [0, 1].")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fraction_of_positives = np.zeros(n_bins)
    mean_predicted_value = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Include right edge only for the last bin
        if i < n_bins - 1:
            mask = (y_prob >= low) & (y_prob < high)
        else:
            mask = (y_prob >= low) & (y_prob <= high)

        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            fraction_of_positives[i] = y_true[mask].mean()
            mean_predicted_value[i] = y_prob[mask].mean()

    n_samples = len(y_true)
    ece = float(
        np.sum(
            bin_counts / n_samples * np.abs(fraction_of_positives - mean_predicted_value)
        )
    )

    reliability_data = ReliabilityDiagramData(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        fraction_of_positives=fraction_of_positives,
        mean_predicted_value=mean_predicted_value,
        bin_counts=bin_counts,
    )
    return ece, reliability_data


# ---------------------------------------------------------------------------
# Reliability diagram plotting
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    uncalibrated: ReliabilityDiagramData,
    calibrated: ReliabilityDiagramData,
    uncalibrated_metrics: CalibrationMetrics,
    calibrated_metrics: CalibrationMetrics,
) -> plt.Figure:
    """
    Plot reliability diagrams for uncalibrated and calibrated models side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Reliability Diagrams", fontsize=14, fontweight="bold")

    for ax, rel_data, metrics in zip(
        axes,
        [uncalibrated, calibrated],
        [uncalibrated_metrics, calibrated_metrics],
    ):
        # Perfect calibration reference line
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)

        # Only plot bins that have samples
        mask = rel_data.bin_counts > 0
        ax.plot(
            rel_data.mean_predicted_value[mask],
            rel_data.fraction_of_positives[mask],
            "s-",
            color="steelblue",
            label=f"{metrics.label}\nBrier={metrics.brier_score:.4f}, ECE={metrics.ece:.4f}",
            markersize=6,
        )

        # Bar chart of bin counts (secondary axis)
        ax2 = ax.twinx()
        ax2.bar(
            rel_data.bin_centers,
            rel_data.bin_counts,
            width=rel_data.bin_edges[1] - rel_data.bin_edges[0],
            alpha=0.2,
            color="gray",
            label="Sample count",
        )
        ax2.set_ylabel("Sample count", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_title(metrics.label)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_calibration_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    base_classifier: Literal["gradient_boosting", "svc"] = "gradient_boosting",
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
    test_size: float = 0.20,
    val_size: float = 0.15,
    n_bins: int = 10,
    random_state: int = 42,
    plot: bool = True,
) -> CalibrationResult:
    """
    Build and evaluate a probability calibration pipeline.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Binary target labels (0 / 1).
    base_classifier : {"gradient_boosting", "svc"}
        Which base classifier to use.
    calibration_method : {"sigmoid", "isotonic"}
        Platt scaling ("sigmoid") or isotonic regression ("isotonic").
    test_size : float
        Fraction of data held out as the final test set.
    val_size : float
        Fraction of the *remaining* data used as the calibration/validation set.
    n_bins : int
        Number of bins for ECE and reliability diagram.
    random_state : int
        Random seed for reproducibility.
    plot : bool
        Whether to generate the reliability diagram figure.

    Returns
    -------
    CalibrationResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}.")
    if len(np.unique(y)) != 2:
        raise ValueError("Only binary classification is supported.")

    # ------------------------------------------------------------------
    # Step 1: Split BEFORE any preprocessing
    #   train+val  |  test
    # ------------------------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # Step 2: Further split train+val into train | val
    #   Calibration is fitted on val, NOT on test.
    # ------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    logger.info(
        "Data split — train: %d, val: %d, test: %d",
        len(y_train), len(y_val), len(y_test),
    )

    # ------------------------------------------------------------------
    # Step 3: Build preprocessing + base classifier pipeline
    #   Scaler is fit ONLY on training data.
    # ------------------------------------------------------------------
    if base_classifier == "gradient_boosting":
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
        )
    elif base_classifier == "svc":
        clf = SVC(probability=True, kernel="rbf", random_state=random_state)
    else:
        raise ValueError(f"Unknown base_classifier: {base_classifier!r}")

    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])

    logger.info("Training base classifier (%s) on training set …", base_classifier)
    base_pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Step 4: Calibrate using validation set
    #   CalibratedClassifierCV with cv="prefit" calibrates on the data
    #   we pass to .fit() — we pass the validation set here.
    # ------------------------------------------------------------------
    logger.info(
        "Calibrating with method=%r on validation set …", calibration_method
    )
    calibrated_pipeline = CalibratedClassifierCV(
        estimator=base_pipeline,
        method=calibration_method,
        cv="prefit",
    )
    calibrated_pipeline.fit(X_val, y_val)

    # ------------------------------------------------------------------
    # Step 5: Evaluate on TEST set (never touched before this point)
    # ------------------------------------------------------------------
    logger.info("Evaluating on held-out test set …")

    # Uncalibrated probabilities
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_prob_uncal = base_pipeline.predict_proba(X_test)[:, 1]

    # Calibrated probabilities
    y_prob_cal = calibrated_pipeline.predict_proba(X_test)[:, 1]

    # Brier scores
    brier_uncal = brier_score_loss(y_test, y_prob_uncal)
    brier_cal = brier_score_loss(y_test, y_prob_cal)

    # ECE
    ece_uncal, rel_uncal = compute_ece(y_test, y_prob_uncal, n_bins=n_bins)
    ece_cal, rel_cal = compute_ece(y_test, y_prob_cal, n_bins=n_bins)

    rel_uncal.label = "Uncalibrated"
    rel_cal.label = f"Calibrated ({calibration_method})"

    uncal_metrics = CalibrationMetrics(
        brier_score=brier_uncal,
        ece=ece_uncal,
        reliability=rel_uncal,
        label="Uncalibrated",
    )
    cal_metrics = CalibrationMetrics(
        brier_score=brier_cal,
        ece=ece_cal,
        reliability=rel_cal,
        label=f"Calibrated ({calibration_method})",
    )

    logger.info(
        "Uncalibrated — Brier: %.4f | ECE: %.4f", brier_uncal, ece_uncal
    )
    logger.info(
        "Calibrated   — Brier: %.4f | ECE: %.4f", brier_cal, ece_cal
    )

    # ------------------------------------------------------------------
    # Step 6: Reliability diagram
    # ------------------------------------------------------------------
    fig = None
    if plot:
        fig = plot_reliability_diagram(rel_uncal, rel_cal, uncal_metrics, cal_metrics)

    return CalibrationResult(
        calibrated_model=calibrated_pipeline,
        uncalibrated_metrics=uncal_metrics,
        calibrated_metrics=cal_metrics,
        test_size=len(y_test),
        n_bins=n_bins,
        calibration_method=calibration_method,
        figure=fig,
    )


# ---------------------------------------------------------------------------
# Convenience summary printer
# ---------------------------------------------------------------------------

def print_calibration_summary(result: CalibrationResult) -> None:
    """Print a human-readable summary of calibration results."""
    sep = "-" * 55
    print(sep)
    print(f"  Calibration method : {result.calibration_method}")
    print(f"  Test set size      : {result.test_size} samples")
    print(f"  Reliability bins   : {result.n_bins}")
    print(sep)
    print(f"  {'Metric':<20} {'Uncalibrated':>14} {'Calibrated':>14}")
    print(sep)
    u = result.uncalibrated_metrics
    c = result.calibrated_metrics
    print(f"  {'Brier Score':<20} {u.brier_score:>14.4f} {c.brier_score:>14.4f}")
    print(f"  {'ECE':<20} {u.ece:>14.4f} {c.ece:>14.4f}")
    print(sep)
    improvement_brier = (u.brier_score - c.brier_score) / u.brier_score * 100
    improvement_ece = (u.ece - c.ece) / (u.ece + 1e-12) * 100
    print(f"  Brier improvement  : {improvement_brier:+.1f}%")
    print(f"  ECE improvement    : {improvement_ece:+.1f}%")
    print(sep)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    rng = np.random.default_rng(0)

    X_demo, y_demo = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        flip_y=0.05,
        random_state=42,
    )

    for method in ("sigmoid", "isotonic"):
        print(f"\n{'='*55}")
        print(f"  Running pipeline with calibration_method={method!r}")
        result = build_calibration_pipeline(
            X_demo,
            y_demo,
            base_classifier="gradient_boosting",
            calibration_method=method,  # type: ignore[arg-type]
            test_size=0.20,
            val_size=0.15,
            n_bins=10,
            random_state=42,
            plot=True,
        )
        print_calibration_summary(result)

        if result.figure is not None:
            fname = f"reliability_diagram_{method}.png"
            result.figure.savefig(fname, dpi=120, bbox_inches="tight")
            logger.info("Saved reliability diagram to %s", fname)
            plt.close(result.figure)