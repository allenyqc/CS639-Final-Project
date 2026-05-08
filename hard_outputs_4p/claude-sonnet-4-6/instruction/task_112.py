"""
Probability Calibration Pipeline Module

Builds, calibrates, and evaluates probabilistic classifiers with proper
train/test separation and multiple calibration quality metrics.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.base import BaseEstimator


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityDiagramData:
    """Binned data for a reliability diagram."""
    bin_centers: np.ndarray          # mid-point of each probability bin
    fraction_of_positives: np.ndarray  # observed positive rate per bin
    mean_predicted_prob: np.ndarray  # average predicted probability per bin
    bin_counts: np.ndarray           # number of samples per bin
    n_bins: int


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics for one model variant."""
    brier_score: float
    ece: float
    reliability_diagram: ReliabilityDiagramData


@dataclass
class CalibrationReport:
    """Full comparison report: uncalibrated vs calibrated."""
    uncalibrated_metrics: CalibrationMetrics
    calibrated_metrics: CalibrationMetrics
    calibration_method: str          # 'sigmoid' (Platt) or 'isotonic'
    improvement_brier: float         # positive = calibrated is better
    improvement_ece: float           # positive = calibrated is better
    figure: Optional[plt.Figure] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error) — manual implementation
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, ReliabilityDiagramData]:
    """
    Compute Expected Calibration Error and reliability diagram data.

    ECE = Σ_b (|B_b| / n) * |acc(B_b) − conf(B_b)|

    where B_b is the set of samples whose predicted probability falls in
    bin b, acc(B_b) is the fraction of positives in that bin, and
    conf(B_b) is the mean predicted probability in that bin.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bins : int
        Number of equal-width bins in [0, 1].

    Returns
    -------
    ece : float
    reliability_diagram_data : ReliabilityDiagramData
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fraction_of_positives = np.zeros(n_bins)
    mean_predicted_prob = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        # include right edge only for the last bin
        if b < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        count = mask.sum()
        bin_counts[b] = count

        if count > 0:
            frac_pos = y_true[mask].mean()
            mean_pred = y_prob[mask].mean()
            fraction_of_positives[b] = frac_pos
            mean_predicted_prob[b] = mean_pred
            ece += (count / n) * abs(frac_pos - mean_pred)
        else:
            fraction_of_positives[b] = np.nan
            mean_predicted_prob[b] = np.nan

    diagram_data = ReliabilityDiagramData(
        bin_centers=bin_centers,
        fraction_of_positives=fraction_of_positives,
        mean_predicted_prob=mean_predicted_prob,
        bin_counts=bin_counts,
        n_bins=n_bins,
    )
    return float(ece), diagram_data


# ---------------------------------------------------------------------------
# Reliability diagram plotting
# ---------------------------------------------------------------------------

def plot_reliability_diagrams(
    uncal_data: ReliabilityDiagramData,
    cal_data: ReliabilityDiagramData,
    uncal_metrics: CalibrationMetrics,
    cal_metrics: CalibrationMetrics,
    calibration_method: str,
) -> plt.Figure:
    """
    Plot side-by-side reliability diagrams for uncalibrated and calibrated
    models, including a histogram of predicted probabilities.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Probability Calibration Analysis", fontsize=14, fontweight="bold")

    titles = ["Uncalibrated", f"Calibrated ({calibration_method})"]
    datasets = [(uncal_data, uncal_metrics), (cal_data, cal_metrics)]

    for col, (data, metrics) in enumerate(datasets):
        ax_diag = axes[0, col]
        ax_hist = axes[1, col]

        # --- reliability diagram ---
        valid = ~np.isnan(data.fraction_of_positives)
        ax_diag.plot([0, 1], [0, 1], "k--", label="Perfect calibration", lw=1.5)
        ax_diag.scatter(
            data.mean_predicted_prob[valid],
            data.fraction_of_positives[valid],
            s=data.bin_counts[valid] * 2 + 10,
            color="steelblue",
            edgecolors="navy",
            zorder=3,
            label="Observed",
        )
        ax_diag.plot(
            data.mean_predicted_prob[valid],
            data.fraction_of_positives[valid],
            "o-",
            color="steelblue",
            lw=1.5,
        )
        ax_diag.set_xlim(0, 1)
        ax_diag.set_ylim(0, 1)
        ax_diag.set_xlabel("Mean Predicted Probability")
        ax_diag.set_ylabel("Fraction of Positives")
        ax_diag.set_title(
            f"{titles[col]}\nBrier={metrics.brier_score:.4f}  ECE={metrics.ece:.4f}"
        )
        ax_diag.legend(fontsize=8)
        ax_diag.grid(True, alpha=0.3)

        # --- histogram of predicted probabilities ---
        ax_hist.bar(
            data.bin_centers,
            data.bin_counts,
            width=1.0 / data.n_bins * 0.9,
            color="steelblue",
            edgecolor="navy",
            alpha=0.7,
        )
        ax_hist.set_xlim(0, 1)
        ax_hist.set_xlabel("Predicted Probability")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"{titles[col]} — Predicted Probability Distribution")
        ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_calibration_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    base_estimator: Optional[BaseEstimator] = None,
    calibration_method: str = "sigmoid",   # 'sigmoid' (Platt) or 'isotonic'
    test_size: float = 0.2,
    random_state: int = 42,
    n_bins: int = 10,
    cv: int = 5,
    scale_features: bool = True,
) -> Tuple[Pipeline, CalibrationReport]:
    """
    Build and evaluate a probability calibration pipeline.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Binary target labels.
    base_estimator : sklearn estimator, optional
        Base classifier.  Defaults to GradientBoostingClassifier.
    calibration_method : {'sigmoid', 'isotonic'}
        'sigmoid' → Platt scaling (logistic regression on decision values).
        'isotonic' → isotonic regression.
    test_size : float
        Fraction of data held out for final evaluation.
    random_state : int
        Random seed for reproducibility.
    n_bins : int
        Number of bins for ECE and reliability diagram.
    cv : int
        Number of cross-validation folds used by CalibratedClassifierCV.
    scale_features : bool
        Whether to apply StandardScaler (fitted only on training data).

    Returns
    -------
    calibrated_pipeline : sklearn Pipeline
        The final calibrated model (scaler + calibrated classifier).
    report : CalibrationReport
        Metrics and reliability diagram data for both model variants.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if calibration_method not in ("sigmoid", "isotonic"):
        raise ValueError("calibration_method must be 'sigmoid' or 'isotonic'.")

    # ------------------------------------------------------------------
    # 1. Train / test split — FIRST, before any fitting
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ------------------------------------------------------------------
    # 2. Scaler — fit ONLY on training data
    # ------------------------------------------------------------------
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)   # transform only
    else:
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test

    # ------------------------------------------------------------------
    # 3. Base classifier
    # ------------------------------------------------------------------
    if base_estimator is None:
        base_estimator = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
        )

    # Fit uncalibrated model on training data
    base_estimator.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 4. Uncalibrated evaluation on TEST set (no leakage)
    # ------------------------------------------------------------------
    uncal_probs = base_estimator.predict_proba(X_test_scaled)[:, 1]
    uncal_brier = brier_score_loss(y_test, uncal_probs)
    uncal_ece, uncal_diagram = compute_ece(y_test, uncal_probs, n_bins=n_bins)

    uncal_metrics = CalibrationMetrics(
        brier_score=uncal_brier,
        ece=uncal_ece,
        reliability_diagram=uncal_diagram,
    )

    # ------------------------------------------------------------------
    # 5. Calibration — CalibratedClassifierCV uses internal CV on
    #    training data only; the test set is never touched here.
    # ------------------------------------------------------------------
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_estimator,
        method=calibration_method,
        cv=cv,
    )
    calibrated_clf.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 6. Calibrated evaluation on TEST set
    # ------------------------------------------------------------------
    cal_probs = calibrated_clf.predict_proba(X_test_scaled)[:, 1]
    cal_brier = brier_score_loss(y_test, cal_probs)
    cal_ece, cal_diagram = compute_ece(y_test, cal_probs, n_bins=n_bins)

    cal_metrics = CalibrationMetrics(
        brier_score=cal_brier,
        ece=cal_ece,
        reliability_diagram=cal_diagram,
    )

    # ------------------------------------------------------------------
    # 7. Reliability diagram figure
    # ------------------------------------------------------------------
    fig = plot_reliability_diagrams(
        uncal_diagram, cal_diagram,
        uncal_metrics, cal_metrics,
        calibration_method,
    )

    # ------------------------------------------------------------------
    # 8. Assemble report
    # ------------------------------------------------------------------
    report = CalibrationReport(
        uncalibrated_metrics=uncal_metrics,
        calibrated_metrics=cal_metrics,
        calibration_method=calibration_method,
        improvement_brier=uncal_brier - cal_brier,   # positive = better
        improvement_ece=uncal_ece - cal_ece,
        figure=fig,
    )

    # ------------------------------------------------------------------
    # 9. Build final sklearn Pipeline for deployment
    # ------------------------------------------------------------------
    steps: List[Tuple[str, Any]] = []
    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("calibrated_classifier", calibrated_clf))
    calibrated_pipeline = Pipeline(steps)

    return calibrated_pipeline, report


# ---------------------------------------------------------------------------
# Convenience: print a human-readable summary
# ---------------------------------------------------------------------------

def print_calibration_report(report: CalibrationReport) -> None:
    """Print a formatted summary of the calibration report."""
    sep = "=" * 55
    print(sep)
    print("  PROBABILITY CALIBRATION REPORT")
    print(sep)
    print(f"  Calibration method : {report.calibration_method}")
    print()
    print(f"  {'Metric':<22} {'Uncalibrated':>14} {'Calibrated':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14}")
    print(
        f"  {'Brier Score':<22} "
        f"{report.uncalibrated_metrics.brier_score:>14.6f} "
        f"{report.calibrated_metrics.brier_score:>14.6f}"
    )
    print(
        f"  {'ECE':<22} "
        f"{report.uncalibrated_metrics.ece:>14.6f} "
        f"{report.calibrated_metrics.ece:>14.6f}"
    )
    print()
    print(f"  Brier improvement  : {report.improvement_brier:+.6f}")
    print(f"  ECE   improvement  : {report.improvement_ece:+.6f}")
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
        random_state=42,
    )

    print("Running calibration pipeline demo …")

    for method in ("sigmoid", "isotonic"):
        print(f"\n--- Method: {method} ---")
        pipeline, report = build_calibration_pipeline(
            X_demo,
            y_demo,
            calibration_method=method,
            test_size=0.2,
            random_state=42,
            n_bins=10,
            cv=5,
        )
        print_calibration_report(report)

        # Save figure
        fname = f"reliability_diagram_{method}.png"
        report.figure.savefig(fname, dpi=120, bbox_inches="tight")
        print(f"  Reliability diagram saved → {fname}")

        # Demonstrate the returned pipeline predicts probabilities
        sample = X_demo[:5]
        probs = pipeline.predict_proba(sample)[:, 1]
        print(f"  Sample calibrated probabilities: {probs.round(4)}")