```python
"""
Bayesian Hyperparameter Optimization Module for GradientBoostingClassifier
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HyperparameterSpace:
    """Defines the search space for each hyperparameter."""
    n_estimators:     Tuple[int, int]   = (50, 500)
    max_depth:        Tuple[int, int]   = (2, 10)
    learning_rate:    Tuple[float, float] = (0.01, 0.5)
    subsample:        Tuple[float, float] = (0.5, 1.0)
    min_samples_leaf: Tuple[int, int]   = (1, 50)


@dataclass
class OptimizationHistory:
    """Stores the full history of the optimization run."""
    hyperparameters: List[Dict[str, Any]] = field(default_factory=list)
    cv_scores:       List[float]          = field(default_factory=list)
    best_so_far:     List[float]          = field(default_factory=list)
    elapsed_times:   List[float]          = field(default_factory=list)
    iteration_idx:   List[int]            = field(default_factory=list)


@dataclass
class OptimizationResult:
    """Final result returned to the caller."""
    best_hyperparameters: Dict[str, Any]
    best_cv_score:        float
    final_model:          GradientBoostingClassifier
    test_metrics:         Dict[str, Any]
    history:              OptimizationHistory


# ─────────────────────────────────────────────────────────────────────────────
# Helper: encode / decode hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

class HyperparameterEncoder:
    """
    Maps hyperparameters to a normalised [0, 1]^d vector and back.
    Integer parameters are rounded on decode.
    """

    PARAM_ORDER = [
        "n_estimators", "max_depth", "learning_rate",
        "subsample", "min_samples_leaf"
    ]
    INTEGER_PARAMS = {"n_estimators", "max_depth", "min_samples_leaf"}

    def __init__(self, space: HyperparameterSpace):
        self.space = space
        self.bounds_raw = {
            "n_estimators":     space.n_estimators,
            "max_depth":        space.max_depth,
            "learning_rate":    space.learning_rate,
            "subsample":        space.subsample,
            "min_samples_leaf": space.min_samples_leaf,
        }
        self.dim = len(self.PARAM_ORDER)

    def encode(self, params: Dict[str, Any]) -> np.ndarray:
        """Dict → normalised numpy vector."""
        vec = []
        for name in self.PARAM_ORDER:
            lo, hi = self.bounds_raw[name]
            vec.append((params[name] - lo) / (hi - lo))
        return np.array(vec)

    def decode(self, vec: np.ndarray) -> Dict[str, Any]:
        """Normalised numpy vector → Dict."""
        params = {}
        for i, name in enumerate(self.PARAM_ORDER):
            lo, hi = self.bounds_raw[name]
            val = lo + vec[i] * (hi - lo)
            if name in self.INTEGER_PARAMS:
                val = int(round(val))
            params[name] = val
        return params

    def random_sample(self, rng: np.random.Generator) -> np.ndarray:
        """Uniform random point in [0, 1]^d."""
        return rng.uniform(0.0, 1.0, size=self.dim)

    @property
    def unit_bounds(self) -> List[Tuple[float, float]]:
        return [(0.0, 1.0)] * self.dim


# ─────────────────────────────────────────────────────────────────────────────
# Acquisition function
# ─────────────────────────────────────────────────────────────────────────────

def expected_improvement(
    X_candidate: np.ndarray,
    gp: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Expected Improvement acquisition function.

    Parameters
    ----------
    X_candidate : (n, d) array of candidate points
    gp          : fitted Gaussian Process
    y_best      : best observed (negated) score so far
    xi          : exploration–exploitation trade-off

    Returns
    -------
    ei : (n,) array of EI values (higher is better)
    """
    mu, sigma = gp.predict(X_candidate, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-9] = 0.0
    return ei


def propose_next_point(
    gp: GaussianProcessRegressor,
    y_best: float,
    encoder: HyperparameterEncoder,
    rng: np.random.Generator,
    n_restarts: int = 10,
) -> np.ndarray:
    """
    Maximise EI via multi-start L-BFGS-B to propose the next evaluation point.
    """
    best_x, best_ei = None, -np.inf

    def neg_ei(x):
        return -expected_improvement(x.reshape(1, -1), gp, y_best)[0]

    for _ in range(n_restarts):
        x0 = encoder.random_sample(rng)
        result = minimize(
            neg_ei,
            x0,
            bounds=encoder.unit_bounds,
            method="L-BFGS-B",
        )
        if -result.fun > best_ei:
            best_ei = -result.fun
            best_x = result.x

    # Clip to [0, 1] for numerical safety
    return np.clip(best_x, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Objective function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_hyperparameters(
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 3,
    random_state: int = 42,
) -> float:
    """
    Train a GradientBoostingClassifier with `params` and return mean CV accuracy.
    """
    model = GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=random_state,
    )
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Main optimisation loop
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_optimize(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 50,
    n_initial_points: int = 5,
    test_size: float = 0.2,
    cv: int = 3,
    random_state: int = 42,
    space: Optional[HyperparameterSpace] = None,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Bayesian hyperparameter optimisation for GradientBoostingClassifier.

    Parameters
    ----------
    X               : Feature matrix (n_samples, n_features)
    y               : Target vector  (n_samples,)
    n_iterations    : Total number of objective evaluations
    n_initial_points: Random evaluations before GP is fitted
    test_size       : Fraction of data held out for final evaluation
    cv              : Cross-validation folds used inside the objective
    random_state    : Global random seed
    space           : HyperparameterSpace (uses defaults if None)
    verbose         : Print progress

    Returns
    -------
    OptimizationResult
    """
    rng = np.random.default_rng(random_state)

    if space is None:
        space = HyperparameterSpace()

    encoder = HyperparameterEncoder(space)

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if verbose:
        print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
        print(f"Running Bayesian optimisation for {n_iterations} iterations …\n")

    history = OptimizationHistory()

    # Collected observations (in normalised space)
    X_obs: List[np.ndarray] = []
    y_obs: List[float]      = []   # we maximise accuracy → store as-is

    best_score  = -np.inf
    best_params: Dict[str, Any] = {}
    t0 = time.time()

    # ── Gaussian Process surrogate ────────────────────────────────────────────
    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
    )

    # ── Optimisation loop ─────────────────────────────────────────────────────
    for iteration in range(1, n_iterations + 1):
        iter_start = time.time()

        # Phase 1: random exploration
        if iteration <= n_initial_points:
            x_next = encoder.random_sample(rng)
        else:
            # Phase 2: GP-guided exploitation / exploration
            gp.fit(np.array(X_obs), np.array(y_obs))
            x_next = propose_next_point(gp, best_score, encoder, rng)

        params = encoder.decode(x_next)
        score  = evaluate_hyperparameters(params, X_train, y_train, cv=cv, random_state=random_state)

        X_obs.append(x_next)
        y_obs.append(score)

        if score > best_score:
            best_score  = score
            best_params = params.copy()

        elapsed = time.time() - iter_start

        # Record history
        history.hyperparameters.append(params.copy())
        history.cv_scores.append(score)
        history.best_so_far.append(best_score)
        history.elapsed_times.append(elapsed)
        history.iteration_idx.append(iteration)

        if verbose:
            print(
                f"Iter {iteration:3d}/{n_iterations} | "
                f"CV acc: {score:.4f} | "
                f"Best: {best_score:.4f} | "
                f"n_est={params['n_estimators']:4d} "
                f"depth={params['max_depth']:2d} "
                f"lr={params['learning_rate']:.4f} "
                f"sub={params['subsample']:.2f} "
                f"msl={params['min_samples_leaf']:3d} | "
                f"{elapsed:.1f}s"
            )

    total_time = time.time() - t0
    if verbose:
        print(f"\nOptimisation finished in {total_time:.1f}s")
        print(f"Best CV accuracy : {best_score:.4f}")
        print(f"Best params      : {best_params}")

    # ── Retrain on full training set with best params ─────────────────────────
    final_model = GradientBoostingClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=random_state,
    )
    final_model.fit(X_train, y_train)

    # ── Test-set evaluation ───────────────────────────────────────────────────
    test_metrics = compute_test_metrics(final_model, X_test, y_test)

    if verbose:
        print("\n── Test-set metrics ──────────────────────────────────────────")
        for k, v in test_metrics.items():
            if k not in ("confusion_matrix", "classification_report"):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print("\nClassification report:\n", test_metrics["classification_report"])

    return OptimizationResult(
        best_hyperparameters=best_params,
        best_cv_score=best_score,
        final_model=final_model,
        test_metrics=test_metrics,
        history=history,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test-set metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_test_metrics(
    model: GradientBoostingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Compute a comprehensive set of test-set metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    classes = model.classes_
    n_classes = len(classes)

    metrics: Dict[str, Any] = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1_macro":  f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix":      confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }

    # ROC-AUC (handles binary and multi-class)
    try:
        if n_classes == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            y_bin = label_binarize(y_test, classes=classes)
            metrics["roc_auc"] = roc_auc_score(
                y_bin, y_proba, multi_class="ovr", average="macro"
            )
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Convergence plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    result: OptimizationResult,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a multi-panel convergence / diagnostics plot.

    Panels
    ------
    1. CV score per iteration + best-so-far curve
    2. Hyperparameter traces over iterations
    3. Score distribution (histogram)
    4. Improvement per iteration (delta best)
    """
    hist = result.history
    iters = hist.iteration_idx

    fig = plt.figure(figsize=figsize)
    fig.suptitle("Bayesian Hyperparameter Optimisation – Convergence Report", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: CV score + best-so-far ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(iters, hist.cv_scores, alpha=0.55, s=30, color="steelblue", label="CV score (each iter)")
    ax1.plot(iters, hist.best_so_far, color="crimson", linewidth=2.0, label="Best so far")
    ax1.axhline(result.test_metrics["accuracy"], color="green", linestyle="--",
                linewidth=1.5, label=f"Test accuracy ({result.test_metrics['accuracy']:.4f})")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Optimisation Convergence")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Hyperparameter traces ───────────────────────────────────────
    param_names = ["n_estimators", "max_depth", "learning_rate", "subsample", "min_samples_leaf"]
    colours     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    ax2 = fig.add_subplot(gs[1, :])
    ax2_twin = ax2.twinx()

    # Plot normalised values so all params fit on one axis
    for pname, col in zip(param_names, colours):
        vals = np.array([h[pname] for h in hist.hyperparameters], dtype=float)
        lo   = min(vals)
        hi   = max(vals)
        norm_vals = (vals - lo) / (hi - lo + 1e-12)
        ax2.plot(iters, norm_vals, alpha=0.6, linewidth=1.2, color=col, label=pname)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Normalised hyperparameter value")
    ax2.set_title("Hyperparameter Traces (normalised)")
    ax2.legend(loc="upper right", fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2_twin.set_yticks([])

    # ── Panel 3: Score distribution ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(hist.cv_scores, bins=15, color="steelblue", edgecolor="white", alpha=0.8)
    ax3.axvline(result.best_cv_score, color="crimson", linestyle="--",
                linewidth=1.5, label=f"Best CV ({result.best_cv_score:.4f})")
    ax3.set_xlabel("CV Accuracy")
    ax3.set_ylabel("Count")
    ax3.set_title("Score Distribution")
    ax3.legend()
    ax3.grid