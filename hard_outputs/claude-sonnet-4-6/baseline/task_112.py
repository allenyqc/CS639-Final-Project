"""
Probability Calibration Pipeline Module

Builds, calibrates, and evaluates probability calibration for classifiers.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize
from typing import Optional, Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error) – manual implementation
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Compute Expected Calibration Error (ECE) manually.

    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    y_true : array of shape (n_samples,)
        True binary labels.
    y_prob : array of shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bins : int
        Number of equally-spaced bins in [0, 1].

    Returns
    -------
    ece : float
        Expected Calibration Error.
    bin_data : dict
        Dictionary with bin-level statistics used for the reliability diagram.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1])  # 0 … n_bins-1

    bin_accs   = np.zeros(n_bins)
    bin_confs  = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_accs[b]   = y_true[mask].mean()
            bin_confs[b]  = y_prob[mask].mean()
            bin_counts[b] = mask.sum()

    n = len(y_true)
    ece = float(np.sum(bin_counts / n * np.abs(bin_accs - bin_confs)))

    bin_data = {
        "bin_centers":  bin_centers,
        "bin_accs":     bin_accs,
        "bin_confs":    bin_confs,
        "bin_counts":   bin_counts,
        "bins":         bins,
    }
    return ece, bin_data


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    bin_data_uncal: Dict[str, np.ndarray],
    bin_data_cal:   Dict[str, np.ndarray],
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a reliability diagram comparing calibrated vs uncalibrated predictions.

    Parameters
    ----------
    bin_data_uncal : dict
        Bin statistics for the uncalibrated model (from compute_ece).
    bin_data_cal : dict
        Bin statistics for the calibrated model (from compute_ece).
    title : str
        Figure title.
    save_path : str or None
        If provided, save the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for ax, bd, label in zip(
        axes,
        [bin_data_uncal, bin_data_cal],
        ["Uncalibrated", "Calibrated"],
    ):
        centers = bd["bin_centers"]
        width   = centers[1] - centers[0] if len(centers) > 1 else 0.1

        # Bar: observed frequency per bin
        ax.bar(
            centers, bd["bin_accs"],
            width=width * 0.8,
            alpha=0.7,
            color='steelblue',
            label='Observed frequency',
            edgecolor='black',
            linewidth=0.5,
        )
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
        # Mean confidence per bin
        ax.scatter(
            centers, bd["bin_confs"],
            color='red', zorder=5, s=40, label='Mean confidence',
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", fontsize=11)
        ax.set_ylabel("Fraction of positives", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_calibration_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    base_classifier: str = "gradient_boosting",
    calibration_method: str = "sigmoid",   # "sigmoid" = Platt, "isotonic"
    test_size: float = 0.2,
    random_state: int = 42,
    n_bins_ece: int = 10,
    plot_diagram: bool = True,
    diagram_save_path: Optional[str] = None,
    **classifier_kwargs: Any,
) -> Dict[str, Any]:
    """
    Build a probability calibration pipeline.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Binary target labels (0 / 1).
    base_classifier : str
        One of {"gradient_boosting", "svc"}.
    calibration_method : str
        "sigmoid" for Platt scaling (logistic regression on decision values)
        or "isotonic" for isotonic regression.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Random seed for reproducibility.
    n_bins_ece : int
        Number of bins for ECE computation.
    plot_diagram : bool
        Whether to generate the reliability diagram.
    diagram_save_path : str or None
        Path to save the reliability diagram image.
    **classifier_kwargs
        Extra keyword arguments forwarded to the base classifier constructor.

    Returns
    -------
    result : dict with keys
        "calibrated_model"      – fitted CalibratedClassifierCV
        "uncalibrated_model"    – fitted base classifier
        "metrics"               – dict with Brier scores and ECE values
        "reliability_diagram"   – dict with bin data for both models
        "figure"                – matplotlib Figure (or None)
        "X_test", "y_test"      – held-out data used for evaluation
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # ------------------------------------------------------------------ split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # -------------------------------------------------------- base classifier
    if base_classifier == "gradient_boosting":
        clf_params = dict(n_estimators=100, random_state=random_state)
        clf_params.update(classifier_kwargs)
        base_clf = GradientBoostingClassifier(**clf_params)
    elif base_classifier == "svc":
        clf_params = dict(probability=True, random_state=random_state)
        clf_params.update(classifier_kwargs)
        base_clf = SVC(**clf_params)
    else:
        raise ValueError(
            f"Unknown base_classifier '{base_classifier}'. "
            "Choose 'gradient_boosting' or 'svc'."
        )

    # -------------------------------------------------- train base classifier
    base_clf.fit(X_train, y_train)

    # -------------------------------------------------- calibrate
    if calibration_method not in ("sigmoid", "isotonic"):
        raise ValueError(
            f"Unknown calibration_method '{calibration_method}'. "
            "Choose 'sigmoid' (Platt) or 'isotonic'."
        )

    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method=calibration_method,
        cv="prefit",          # base_clf is already fitted
    )
    calibrated_clf.fit(X_train, y_train)

    # ------------------------------------------ predict probabilities on test
    # Uncalibrated probabilities
    if hasattr(base_clf, "predict_proba"):
        prob_uncal = base_clf.predict_proba(X_test)[:, 1]
    else:
        # Fallback: use decision_function + sigmoid
        df = base_clf.decision_function(X_test)
        prob_uncal = 1.0 / (1.0 + np.exp(-df))

    # Calibrated probabilities
    prob_cal = calibrated_clf.predict_proba(X_test)[:, 1]

    # -------------------------------------------------- Brier score
    brier_uncal = brier_score_loss(y_test, prob_uncal)
    brier_cal   = brier_score_loss(y_test, prob_cal)

    # -------------------------------------------------- ECE
    ece_uncal, bin_data_uncal = compute_ece(y_test, prob_uncal, n_bins=n_bins_ece)
    ece_cal,   bin_data_cal   = compute_ece(y_test, prob_cal,   n_bins=n_bins_ece)

    # -------------------------------------------------- metrics summary
    metrics = {
        "brier_score_uncalibrated": brier_uncal,
        "brier_score_calibrated":   brier_cal,
        "brier_score_improvement":  brier_uncal - brier_cal,
        "ece_uncalibrated":         ece_uncal,
        "ece_calibrated":           ece_cal,
        "ece_improvement":          ece_uncal - ece_cal,
        "calibration_method":       calibration_method,
        "base_classifier":          base_classifier,
        "n_test_samples":           len(y_test),
    }

    # -------------------------------------------------- reliability diagram
    fig = None
    if plot_diagram:
        fig = plot_reliability_diagram(
            bin_data_uncal,
            bin_data_cal,
            title=(
                f"Reliability Diagram – {base_classifier} "
                f"({calibration_method} calibration)"
            ),
            save_path=diagram_save_path,
        )

    reliability_diagram = {
        "uncalibrated": bin_data_uncal,
        "calibrated":   bin_data_cal,
    }

    return {
        "calibrated_model":    calibrated_clf,
        "uncalibrated_model":  base_clf,
        "metrics":             metrics,
        "reliability_diagram": reliability_diagram,
        "figure":              fig,
        "X_test":              X_test,
        "y_test":              y_test,
        "prob_uncalibrated":   prob_uncal,
        "prob_calibrated":     prob_cal,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_calibration_report(result: Dict[str, Any]) -> None:
    """Print a human-readable calibration report from the pipeline result."""
    m = result["metrics"]
    sep = "=" * 55

    print(sep)
    print("  PROBABILITY CALIBRATION REPORT")
    print(sep)
    print(f"  Base classifier   : {m['base_classifier']}")
    print(f"  Calibration method: {m['calibration_method']}")
    print(f"  Test samples      : {m['n_test_samples']}")
    print(sep)
    print(f"  {'Metric':<35} {'Uncal':>8}  {'Cal':>8}  {'Δ':>8}")
    print("-" * 55)
    print(
        f"  {'Brier Score (↓ better)':<35} "
        f"{m['brier_score_uncalibrated']:>8.4f}  "
        f"{m['brier_score_calibrated']:>8.4f}  "
        f"{m['brier_score_improvement']:>+8.4f}"
    )
    print(
        f"  {'ECE (↓ better)':<35} "
        f"{m['ece_uncalibrated']:>8.4f}  "
        f"{m['ece_calibrated']:>8.4f}  "
        f"{m['ece_improvement']:>+8.4f}"
    )
    print(sep)

    rd = result["reliability_diagram"]
    print("\n  Reliability Diagram – Bin Statistics")
    print(f"  {'Bin':>5}  {'Uncal Acc':>10}  {'Uncal Conf':>11}  "
          f"{'Cal Acc':>8}  {'Cal Conf':>9}  {'Count':>6}")
    print("-" * 60)
    n_bins = len(rd["uncalibrated"]["bin_centers"])
    for b in range(n_bins):
        cnt = rd["uncalibrated"]["bin_counts"][b]
        if cnt == 0:
            continue
        print(
            f"  {rd['uncalibrated']['bin_centers'][b]:>5.2f}  "
            f"{rd['uncalibrated']['bin_accs'][b]:>10.4f}  "
            f"{rd['uncalibrated']['bin_confs'][b]:>11.4f}  "
            f"{rd['calibrated']['bin_accs'][b]:>8.4f}  "
            f"{rd['calibrated']['bin_confs'][b]:>9.4f}  "
            f"{cnt:>6d}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    print("Generating synthetic dataset …")
    X_demo, y_demo = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=0,
    )

    # ---- Gradient Boosting + Platt scaling --------------------------------
    print("\n[1] GradientBoosting + Platt (sigmoid) calibration")
    result_gb = build_calibration_pipeline(
        X_demo, y_demo,
        base_classifier="gradient_boosting",
        calibration_method="sigmoid",
        plot_diagram=True,
        diagram_save_path="reliability_gb_sigmoid.png",
    )
    print_calibration_report(result_gb)

    # ---- Gradient Boosting + Isotonic regression --------------------------
    print("\n[2] GradientBoosting + Isotonic calibration")
    result_iso = build_calibration_pipeline(
        X_demo, y_demo,
        base_classifier="gradient_boosting",
        calibration_method="isotonic",
        plot_diagram=True,
        diagram_save_path="reliability_gb_isotonic.png",
    )
    print_calibration_report(result_iso)

    # ---- SVC + Platt scaling ----------------------------------------------
    print("\n[3] SVC + Platt (sigmoid) calibration")
    result_svc = build_calibration_pipeline(
        X_demo, y_demo,
        base_classifier="svc",
        calibration_method="sigmoid",
        plot_diagram=True,
        diagram_save_path="reliability_svc_sigmoid.png",
        kernel="rbf",
        C=1.0,
    )
    print_calibration_report(result_svc)

    print("\nDone. Reliability diagrams saved as PNG files.")