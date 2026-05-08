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
    """Defines the search space for GradientBoostingClassifier."""
    n_estimators:     Tuple[int, int]   = (50, 500)
    max_depth:        Tuple[int, int]   = (2, 10)
    learning_rate:    Tuple[float, float] = (0.01, 0.5)
    subsample:        Tuple[float, float] = (0.5, 1.0)
    min_samples_leaf: Tuple[int, int]   = (1, 50)

    def bounds(self) -> List[Tuple]:
        return [
            self.n_estimators,
            self.max_depth,
            self.learning_rate,
            self.subsample,
            self.min_samples_leaf,
        ]

    @property
    def param_names(self) -> List[str]:
        return [
            "n_estimators", "max_depth", "learning_rate",
            "subsample", "min_samples_leaf"
        ]

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert a raw vector to a dict of typed hyperparameters."""
        return {
            "n_estimators":     int(round(x[0])),
            "max_depth":        int(round(x[1])),
            "learning_rate":    float(x[2]),
            "subsample":        float(x[3]),
            "min_samples_leaf": int(round(x[4])),
        }


@dataclass
class OptimizationHistory:
    """Stores the full optimization trajectory."""
    hyperparams:  List[Dict[str, Any]] = field(default_factory=list)
    scores:       List[float]          = field(default_factory=list)
    best_so_far:  List[float]          = field(default_factory=list)
    elapsed_time: List[float]          = field(default_factory=list)
    iteration:    List[int]            = field(default_factory=list)

    def record(self, params: Dict, score: float, elapsed: float, it: int):
        self.hyperparams.append(params)
        self.scores.append(score)
        best = max(self.best_so_far[-1], score) if self.best_so_far else score
        self.best_so_far.append(best)
        self.elapsed_time.append(elapsed)
        self.iteration.append(it)

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.scores))

    @property
    def best_score(self) -> float:
        return max(self.scores)

    @property
    def best_params(self) -> Dict[str, Any]:
        return self.hyperparams[self.best_index]


# ─────────────────────────────────────────────────────────────────────────────
# Acquisition functions
# ─────────────────────────────────────────────────────────────────────────────

def expected_improvement(
    X_candidates: np.ndarray,
    gp: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement acquisition function."""
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = sigma.reshape(-1, 1)
    mu    = mu.reshape(-1, 1)

    with np.errstate(divide="ignore"):
        Z  = (mu - y_best - xi) / (sigma + 1e-9)
        ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-10] = 0.0
    return ei.ravel()


def upper_confidence_bound(
    X_candidates: np.ndarray,
    gp: GaussianProcessRegressor,
    kappa: float = 2.576,
) -> np.ndarray:
    """Upper Confidence Bound acquisition function."""
    mu, sigma = gp.predict(X_candidates, return_std=True)
    return mu + kappa * sigma


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class BayesianOptimizer:
    """
    Manual Gaussian-Process-based Bayesian optimizer.

    Parameters
    ----------
    space          : HyperparameterSpace
    n_iterations   : total BO iterations (including warm-up)
    n_initial      : random warm-up evaluations before GP kicks in
    acquisition    : 'ei' | 'ucb'
    cv_folds       : cross-validation folds used for scoring
    random_state   : reproducibility seed
    verbose        : print progress
    """

    def __init__(
        self,
        space: HyperparameterSpace = None,
        n_iterations: int = 50,
        n_initial: int = 10,
        acquisition: str = "ei",
        cv_folds: int = 3,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.space         = space or HyperparameterSpace()
        self.n_iterations  = n_iterations
        self.n_initial     = n_initial
        self.acquisition   = acquisition
        self.cv_folds      = cv_folds
        self.random_state  = random_state
        self.verbose       = verbose
        self.history       = OptimizationHistory()
        self._rng          = np.random.RandomState(random_state)

        # GP surrogate
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state,
        )

        self._X_obs: List[np.ndarray] = []
        self._y_obs: List[float]      = []

    # ── internal helpers ──────────────────────────────────────────────────────

    def _random_sample(self) -> np.ndarray:
        """Draw a random point from the search space."""
        bounds = self.space.bounds()
        x = np.array([
            self._rng.uniform(lo, hi) for lo, hi in bounds
        ])
        return x

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] per dimension."""
        bounds = np.array(self.space.bounds(), dtype=float)
        lo, hi = bounds[:, 0], bounds[:, 1]
        return (X - lo) / (hi - lo + 1e-12)

    def _next_candidate(self) -> np.ndarray:
        """Propose the next point via the acquisition function."""
        bounds = np.array(self.space.bounds(), dtype=float)
        lo, hi = bounds[:, 0], bounds[:, 1]

        # Random multi-start optimisation of the acquisition
        best_acq  = -np.inf
        best_x    = None
        n_restarts = 20

        for _ in range(n_restarts):
            x0 = self._random_sample()
            x0_norm = self._normalize(x0)

            def neg_acq(x_norm):
                x_full = np.array(x_norm).reshape(1, -1)
                if self.acquisition == "ei":
                    val = expected_improvement(
                        x_full, self.gp, max(self._y_obs)
                    )
                else:
                    val = upper_confidence_bound(x_full, self.gp)
                return -val[0]

            bounds_norm = [(0.0, 1.0)] * len(lo)
            res = minimize(
                neg_acq, x0_norm,
                bounds=bounds_norm,
                method="L-BFGS-B",
            )
            if -res.fun > best_acq:
                best_acq = -res.fun
                best_x   = res.x

        # Denormalize
        x_denorm = best_x * (hi - lo) + lo
        return np.clip(x_denorm, lo, hi)

    def _evaluate(
        self,
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> float:
        """Train + cross-validate; return mean CV accuracy."""
        model = GradientBoostingClassifier(
            random_state=self.random_state, **params
        )
        scores = cross_val_score(
            model, X_train, y_train,
            cv=self.cv_folds, scoring="accuracy", n_jobs=-1
        )
        return float(scores.mean())

    # ── public API ────────────────────────────────────────────────────────────

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> OptimizationHistory:
        """
        Run the Bayesian optimization loop.

        Returns
        -------
        OptimizationHistory
        """
        t0 = time.time()

        for it in range(1, self.n_iterations + 1):
            iter_start = time.time()

            # ── propose ──────────────────────────────────────────────────────
            if it <= self.n_initial or len(self._X_obs) < 2:
                x_raw = self._random_sample()
            else:
                X_norm = self._normalize(np.array(self._X_obs))
                self.gp.fit(X_norm, np.array(self._y_obs))
                x_raw = self._next_candidate()

            params = self.space.decode(x_raw)

            # ── evaluate ─────────────────────────────────────────────────────
            score = self._evaluate(params, X_train, y_train)

            # ── record ───────────────────────────────────────────────────────
            elapsed = time.time() - iter_start
            self.history.record(params, score, elapsed, it)
            self._X_obs.append(x_raw)
            self._y_obs.append(score)

            if self.verbose:
                print(
                    f"[{it:3d}/{self.n_iterations}] "
                    f"score={score:.4f}  "
                    f"best={self.history.best_score:.4f}  "
                    f"({elapsed:.1f}s)  "
                    f"params={params}"
                )

        total = time.time() - t0
        if self.verbose:
            print(f"\nOptimization finished in {total:.1f}s")
            print(f"Best score : {self.history.best_score:.4f}")
            print(f"Best params: {self.history.best_params}")

        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    history: OptimizationHistory,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Comprehensive convergence dashboard.

    Panels
    ------
    1. Score per iteration + best-so-far envelope
    2. Improvement over best-so-far
    3. Hyperparameter trajectories
    4. Score distribution (histogram)
    5. Cumulative time
    6. Parallel-coordinates of top-10 configurations
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Bayesian Hyperparameter Optimization – Convergence Report",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    iters  = history.iteration
    scores = history.scores
    best   = history.best_so_far
    params = history.hyperparams
    times  = history.elapsed_time
    names  = list(params[0].keys())

    # ── 1. Score + best-so-far ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(iters, scores, s=30, alpha=0.6, color="steelblue",
                label="CV score", zorder=3)
    ax1.plot(iters, best, color="crimson", lw=2, label="Best so far")
    ax1.fill_between(iters, scores, best, alpha=0.08, color="crimson")
    ax1.axvline(history.best_index + 1, ls="--", color="green",
                alpha=0.7, label=f"Best iter={history.best_index+1}")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("CV Accuracy")
    ax1.set_title("Optimization Trajectory")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 2. Improvement per step ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    improvements = [0.0] + [
        max(0.0, best[i] - best[i - 1]) for i in range(1, len(best))
    ]
    ax2.bar(iters, improvements, color="darkorange", alpha=0.7)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Δ Best Score")
    ax2.set_title("Incremental Improvement")
    ax2.grid(True, alpha=0.3, axis="y")

    # ── 3. Hyperparameter trajectories ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    param_arrays = {n: [p[n] for p in params] for n in names}
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    ax3_twin = ax3.twinx()

    # n_estimators, max_depth, min_samples_leaf on left axis
    left_params  = ["n_estimators", "max_depth", "min_samples_leaf"]
    right_params = ["learning_rate", "subsample"]

    for i, n in enumerate(left_params):
        ax3.plot(iters, param_arrays[n], alpha=0.6,
                 label=n, color=colors[i], lw=1.2)
    for i, n in enumerate(right_params):
        ax3_twin.plot(iters, param_arrays[n], alpha=0.6,
                      label=n, color=colors[len(left_params) + i],
                      lw=1.2, ls="--")

    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Integer params")
    ax3_twin.set_ylabel("Float params")
    ax3.set_title("Hyperparameter Trajectories")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2,
               fontsize=7, loc="upper right")
    ax3.grid(True, alpha=0.3)

    # ── 4. Score distribution ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(scores, bins=15, color="mediumpurple", edgecolor="white",
             alpha=0.8)
    ax4.axvline(history.best_score, color="crimson", lw=2,
                label=f"Best={history.best_score:.4f}")
    ax4.axvline(np.mean(scores), color="navy", lw=1.5, ls="--",
                label=f"Mean={np.mean(scores):.4f}")
    ax4.set_xlabel("CV Accuracy")
    ax4.set_ylabel("Count")
    ax4.set_title("Score Distribution")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    # ── 5. Cumulative time ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    cum_time = np.cumsum(times)
    ax5.plot(iters, cum_time, color="teal", lw=2)
    ax5.fill_between(iters, cum_time, alpha=0.15, color="teal")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Cumulative Time (s)")
    ax5.set_title("Cumulative Evaluation Time")
    ax5.grid(True, alpha=0.3)

    # ── 6. Parallel coordinates of top-10 ───────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1:])
    top_n = 10
    sorted_idx = np.argsort(scores)[-top_n:][::-1]
    top_params = [params[i] for i in sorted_idx]
    top_scores = [scores[i] for i in sorted_idx]

    # Normalize each param to [0,1] for display
    space = HyperparameterSpace()
    bounds = np.array(space.bounds(), dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]

    x_ticks = np.arange(len(names))
    cmap = plt.cm.RdYlGn
    norm_scores = (np.array(top_scores) - min(top_scores)) / (
        max(top_scores) - min(top_scores) + 1e-12
    )

    for rank, (p, ns) in enumerate(zip(top_params, norm_scores)):
        vals = np.array([p[n] for n in names], dtype=float)
        vals_norm = (vals - lo) / (hi - lo + 1e-12)
        ax6.plot(x_ticks, vals_norm, marker="o", ms=5,
                 color=cmap(ns), alpha=0.8, lw=1.5)

    ax6.set_xticks(x_ticks)
    ax6.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax6.set_ylabel("Normalized Value")
    ax6.set_title(f"Parallel Coordinates – Top {top_n} Configs\n"
                  "(colour: green=best)")
    ax6.grid(True, alpha=0.3, axis="y")

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(min(top_scores),
                                                   max(top_scores)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax6, label="CV Accuracy",