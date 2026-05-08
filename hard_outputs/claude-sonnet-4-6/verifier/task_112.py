"""
Probability Calibration Pipeline Module

Builds, calibrates, and evaluates probability calibration for classifiers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize
from typing import Optional, Tuple, Dict, Any, Union
import warnings


# ---------------------------------------------------------------------------
# ECE calculation
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
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
    bin_data : dict with keys 'bin_centers', 'bin_acc', 'bin_conf', 'bin_counts'
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1])  # 0 … n_bins-1

    bin_acc    = np.zeros(n_bins)
    bin_conf   = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_acc[b]    = y_true[mask].mean()
            bin_conf[b]   = y_prob[mask].mean()
            bin_counts[b] = mask.sum()

    n = len(y_true)
    ece = float(np.sum(bin_counts / n * np.abs(bin_acc - bin_conf)))

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_data = {
        "bin_centers": bin_centers,
        "bin_acc":     bin_acc,
        "bin_conf":    bin_conf,
        "bin_counts":  bin_counts,
    }
    return ece, bin_data


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    bin_data_uncal: Dict[str, np.ndarray],
    bin_data_cal:   Dict[str, np.ndarray],
    brier_uncal: float,
    brier_cal:   float,
    ece_uncal:   float,
    ece_cal:     float,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot reliability diagrams for uncalibrated and calibrated models side-by-side,
    together with a histogram of predicted probabilities.
    """
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.4, wspace=0.35)

    for col, (label, bd, brier, ece) in enumerate(
        [
            ("Uncalibrated", bin_data_uncal, brier_uncal, ece_uncal),
            ("Calibrated",   bin_data_cal,   brier_cal,   ece_cal),
        ]
    ):
        ax_main = fig.add_subplot(gs[0, col])
        ax_hist = fig.add_subplot(gs[1, col])

        centers = bd["bin_centers"]
        acc     = bd["bin_acc"]
        conf    = bd["bin_conf"]
        counts  = bd["bin_counts"]
        width   = centers[1] - centers[0] if len(centers) > 1 else 0.1

        # Perfect calibration line
        ax_main.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")

        # Calibration bars
        ax_main.bar(
            centers,
            acc,
            width=width * 0.9,
            alpha=0.7,
            color="steelblue",
            label="Fraction of positives",
        )
        # Mean confidence dots
        ax_main.scatter(
            conf[counts > 0],
            acc[counts > 0],
            color="darkorange",
            zorder=5,
            s=40,
            label="Mean confidence",
        )

        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        ax_main.set_xlabel("Mean predicted probability")
        ax_main.set_ylabel("Fraction of positives")
        ax_main.set_title(
            f"{label}\nBrier={brier:.4f}  ECE={ece:.4f}"
        )
        ax_main.legend(fontsize=8)

        # Histogram of predicted probabilities
        ax_hist.bar(
            centers,
            counts,
            width=width * 0.9,
            color="steelblue",
            alpha=0.7,
        )
        ax_hist.set_xlim(0, 1)
        ax_hist.set_xlabel("Predicted probability")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Prediction histogram")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_calibration_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    base_classifier: str = "gradient_boosting",
    calibration_method: str = "sigmoid",   # "sigmoid" = Platt, "isotonic"
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    n_bins: int = 10,
    plot: bool = True,
    save_path: Optional[str] = None,
    **classifier_kwargs: Any,
) -> Dict[str, Any]:
    """
    Build a probability calibration pipeline.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    base_classifier : {'gradient_boosting', 'svc'}
    calibration_method : {'sigmoid', 'isotonic'}
        'sigmoid' → Platt scaling (logistic regression on decision values).
        'isotonic' → isotonic regression.
    test_size : float
        Fraction of data held out for final evaluation.
    val_size : float
        Fraction of *training* data used for calibration fitting
        (only relevant when cv='prefit').
    random_state : int
    n_bins : int
        Number of bins for ECE / reliability diagram.
    plot : bool
        Whether to render the reliability diagram.
    save_path : str or None
        If given, save the figure to this path.
    **classifier_kwargs
        Extra keyword arguments forwarded to the base classifier constructor.

    Returns
    -------
    result : dict with keys
        'calibrated_model'   – fitted CalibratedClassifierCV
        'uncalibrated_model' – fitted base classifier
        'metrics'            – dict of Brier scores and ECEs
        'reliability_data'   – dict with 'uncalibrated' and 'calibrated' bin data
        'figure'             – matplotlib Figure (or None if plot=False)
        'X_test', 'y_test'   – held-out evaluation arrays
    """
    X = np.asarray(X)
    y = np.asarray(y)

    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(
            f"Only binary classification is supported; found classes {classes}."
        )

    # ------------------------------------------------------------------
    # 1. Train / test split
    # ------------------------------------------------------------------
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split training data further into fit / calibration sets
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    # ------------------------------------------------------------------
    # 2. Build base classifier
    # ------------------------------------------------------------------
    base_clf: Any
    if base_classifier == "gradient_boosting":
        default_kwargs = dict(n_estimators=100, max_depth=3, random_state=random_state)
        default_kwargs.update(classifier_kwargs)
        base_clf = GradientBoostingClassifier(**default_kwargs)
    elif base_classifier == "svc":
        default_kwargs = dict(kernel="rbf", probability=False, random_state=random_state)
        default_kwargs.update(classifier_kwargs)
        base_clf = SVC(**default_kwargs)
    else:
        raise ValueError(
            f"Unknown base_classifier '{base_classifier}'. "
            "Choose 'gradient_boosting' or 'svc'."
        )

    # ------------------------------------------------------------------
    # 3. Train base classifier on X_fit
    # ------------------------------------------------------------------
    base_clf.fit(X_fit, y_fit)

    # ------------------------------------------------------------------
    # 4. Calibrate using prefit strategy
    # ------------------------------------------------------------------
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method=calibration_method,
        cv="prefit",
    )
    calibrated_clf.fit(X_cal, y_cal)

    # ------------------------------------------------------------------
    # 5. Predict probabilities on test set
    # ------------------------------------------------------------------
    pos_idx = list(calibrated_clf.classes_).index(classes[-1])

    # Uncalibrated probabilities
    if hasattr(base_clf, "predict_proba"):
        prob_uncal = base_clf.predict_proba(X_test)[:, pos_idx]
    elif hasattr(base_clf, "decision_function"):
        # Convert decision values to [0,1] via sigmoid for comparison
        dv = base_clf.decision_function(X_test)
        prob_uncal = 1.0 / (1.0 + np.exp(-dv))
    else:
        raise RuntimeError("Base classifier has neither predict_proba nor decision_function.")

    # Calibrated probabilities
    prob_cal = calibrated_clf.predict_proba(X_test)[:, pos_idx]

    # ------------------------------------------------------------------
    # 6. Evaluate calibration quality
    # ------------------------------------------------------------------
    brier_uncal = brier_score_loss(y_test, prob_uncal, pos_label=classes[-1])
    brier_cal   = brier_score_loss(y_test, prob_cal,   pos_label=classes[-1])

    # Map y_test to 0/1 for ECE
    y_test_bin = (y_test == classes[-1]).astype(int)

    ece_uncal, bin_data_uncal = compute_ece(y_test_bin, prob_uncal, n_bins=n_bins)
    ece_cal,   bin_data_cal   = compute_ece(y_test_bin, prob_cal,   n_bins=n_bins)

    metrics = {
        "brier_score_uncalibrated": brier_uncal,
        "brier_score_calibrated":   brier_cal,
        "brier_improvement":        brier_uncal - brier_cal,
        "ece_uncalibrated":         ece_uncal,
        "ece_calibrated":           ece_cal,
        "ece_improvement":          ece_uncal - ece_cal,
    }

    # ------------------------------------------------------------------
    # 7. Reliability diagram
    # ------------------------------------------------------------------
    fig = None
    if plot:
        fig = plot_reliability_diagram(
            bin_data_uncal=bin_data_uncal,
            bin_data_cal=bin_data_cal,
            brier_uncal=brier_uncal,
            brier_cal=brier_cal,
            ece_uncal=ece_uncal,
            ece_cal=ece_cal,
            title=(
                f"Reliability Diagram — {base_classifier.replace('_', ' ').title()} "
                f"({calibration_method} calibration)"
            ),
            save_path=save_path,
        )

    # ------------------------------------------------------------------
    # 8. Console summary
    # ------------------------------------------------------------------
    _print_summary(metrics, base_classifier, calibration_method)

    return {
        "calibrated_model":   calibrated_clf,
        "uncalibrated_model": base_clf,
        "metrics":            metrics,
        "reliability_data": {
            "uncalibrated": bin_data_uncal,
            "calibrated":   bin_data_cal,
        },
        "figure": fig,
        "X_test": X_test,
        "y_test": y_test,
        "prob_uncalibrated": prob_uncal,
        "prob_calibrated":   prob_cal,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_summary(
    metrics: Dict[str, float],
    base_classifier: str,
    calibration_method: str,
) -> None:
    sep = "=" * 55
    print(sep)
    print(f"  Calibration Pipeline Summary")
    print(f"  Base classifier : {base_classifier}")
    print(f"  Method          : {calibration_method}")
    print(sep)
    print(f"  {'Metric':<35} {'Uncal':>8}  {'Cal':>8}")
    print(f"  {'-'*53}")
    print(
        f"  {'Brier Score':<35} "
        f"{metrics['brier_score_uncalibrated']:>8.4f}  "
        f"{metrics['brier_score_calibrated']:>8.4f}"
    )
    print(
        f"  {'ECE':<35} "
        f"{metrics['ece_uncalibrated']:>8.4f}  "
        f"{metrics['ece_calibrated']:>8.4f}"
    )
    print(sep)
    print(
        f"  Brier improvement : {metrics['brier_improvement']:+.4f}  "
        f"({'better' if metrics['brier_improvement'] > 0 else 'worse'})"
    )
    print(
        f"  ECE   improvement : {metrics['ece_improvement']:+.4f}  "
        f"({'better' if metrics['ece_improvement'] > 0 else 'worse'})"
    )
    print(sep)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    rng = np.random.default_rng(0)

    X_demo, y_demo = make_classification(
        n_samples=3000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )

    print("\n--- Gradient Boosting + Platt (sigmoid) ---")
    result_gb = build_calibration_pipeline(
        X_demo, y_demo,
        base_classifier="gradient_boosting",
        calibration_method="sigmoid",
        n_bins=10,
        plot=True,
    )

    print("\n--- Gradient Boosting + Isotonic ---")
    result_iso = build_calibration_pipeline(
        X_demo, y_demo,
        base_classifier="gradient_boosting",
        calibration_method="isotonic",
        n_bins=10,
        plot=True,
    )

    print("\n--- SVC + Platt (sigmoid) ---")
    result_svc = build_calibration_pipeline(
        X_demo, y_demo,
        base_classifier="svc",
        calibration_method="sigmoid",
        n_bins=10,
        plot=True,
    )

    plt.show()