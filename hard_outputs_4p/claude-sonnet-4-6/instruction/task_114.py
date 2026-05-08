```python
"""
Bayesian Hyperparameter Optimization Module for GradientBoostingClassifier.

This module implements a complete Bayesian optimization pipeline using
scikit-optimize's gp_minimize, following ML best practices for data splitting,
evaluation, and reproducibility.
"""

import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; switch to "TkAgg" / "Qt5Agg" for GUI
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn(
        "scikit-optimize not found. Falling back to manual GP-based optimization. "
        "Install with: pip install scikit-optimize",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Helper: Manual GP-based Bayesian Optimisation (fallback)
# ===========================================================================

class _ManualBayesianOptimizer:
    """
    Minimal Gaussian-Process-based Bayesian optimiser used when
    scikit-optimize is not available.

    Uses Expected Improvement (EI) as the acquisition function.
    """

    def __init__(self, bounds: List[Tuple[float, float]], n_random_starts: int = 10):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        self.bounds = np.array(bounds, dtype=float)
        self.n_random_starts = n_random_starts
        self.gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
        )
        self.X_obs: List[np.ndarray] = []
        self.y_obs: List[float] = []

    def _random_sample(self) -> np.ndarray:
        return np.array(
            [np.random.uniform(lo, hi) for lo, hi in self.bounds]
        )

    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        from scipy.stats import norm

        mu, sigma = self.gpr.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu = mu.reshape(-1, 1)

        mu_sample_opt = np.min(self.y_obs)  # minimising negative score
        with np.errstate(divide="warn"):
            imp = mu_sample_opt - mu - xi
            Z = imp / (sigma + 1e-9)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma < 1e-10] = 0.0
        return ei.flatten()

    def suggest(self) -> np.ndarray:
        if len(self.X_obs) < self.n_random_starts:
            return self._random_sample()

        X_obs = np.array(self.X_obs)
        y_obs = np.array(self.y_obs)
        self.gpr.fit(X_obs, y_obs)

        # Random search over acquisition function
        n_candidates = 10_000
        candidates = np.array(
            [self._random_sample() for _ in range(n_candidates)]
        )
        ei = self._expected_improvement(candidates)
        return candidates[np.argmax(ei)]

    def register(self, x: np.ndarray, y: float) -> None:
        self.X_obs.append(x)
        self.y_obs.append(y)


# ===========================================================================
# Core optimisation module
# ===========================================================================

class BayesianHPOptimizer:
    """
    Bayesian hyperparameter optimiser for GradientBoostingClassifier.

    Parameters
    ----------
    n_iter : int
        Number of optimisation iterations (default 50).
    cv_folds : int
        Inner cross-validation folds used to estimate generalisation during
        the search (default 5).  The test set is NEVER touched during search.
    test_size : float
        Fraction of data held out as the final test set (default 0.2).
    random_state : int
        Global random seed for reproducibility.
    scoring : str
        Sklearn scoring string used during CV (default "roc_auc").
    n_jobs : int
        Parallel jobs for cross_val_score (default -1).
    """

    # ------------------------------------------------------------------
    # Hyperparameter search space definition
    # ------------------------------------------------------------------
    _PARAM_NAMES = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "min_samples_leaf",
    ]

    # (low, high) — continuous representation; integers are rounded later
    _BOUNDS_CONTINUOUS = [
        (50, 500),      # n_estimators
        (2, 10),        # max_depth
        (0.01, 0.5),    # learning_rate
        (0.5, 1.0),     # subsample
        (1, 50),        # min_samples_leaf
    ]

    def __init__(
        self,
        n_iter: int = 50,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
    ) -> None:
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.scoring = scoring
        self.n_jobs = n_jobs

        # Will be populated during fit()
        self.history_: List[Dict[str, Any]] = []
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: float = -np.inf
        self.best_model_: Optional[Pipeline] = None
        self.test_metrics_: Optional[Dict[str, Any]] = None

        self._X_train: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "BayesianHPOptimizer":
        """
        Run the full optimisation pipeline.

        Steps
        -----
        1. Train / test split (stratified).
        2. Fit a StandardScaler on the TRAINING partition only.
        3. Run Bayesian optimisation for ``n_iter`` iterations using
           inner CV on the training partition.
        4. Retrain the best model on the full training set.
        5. Evaluate on the held-out test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # ------------------------------------------------------------------ #
        # 1. Train / test split — BEFORE any preprocessing                    #
        # ------------------------------------------------------------------ #
        logger.info("Splitting data into train (%.0f%%) / test (%.0f%%) sets.",
                    (1 - self.test_size) * 100, self.test_size * 100)
        (
            self._X_train,
            self._X_test,
            self._y_train,
            self._y_test,
        ) = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # ------------------------------------------------------------------ #
        # 2. Fit scaler on training data only                                 #
        # ------------------------------------------------------------------ #
        self._scaler = StandardScaler()
        self._scaler.fit(self._X_train)  # fit ONLY on train

        logger.info(
            "Train size: %d | Test size: %d | Features: %d",
            len(self._y_train),
            len(self._y_test),
            X.shape[1],
        )

        # ------------------------------------------------------------------ #
        # 3. Bayesian optimisation loop                                        #
        # ------------------------------------------------------------------ #
        logger.info("Starting Bayesian optimisation (%d iterations).", self.n_iter)
        if SKOPT_AVAILABLE:
            self._run_skopt()
        else:
            self._run_manual_gp()

        # ------------------------------------------------------------------ #
        # 4. Retrain best model on full training set                          #
        # ------------------------------------------------------------------ #
        logger.info("Retraining best model on full training partition.")
        self.best_model_ = self._build_pipeline(self.best_params_)
        self.best_model_.fit(self._X_train, self._y_train)

        # ------------------------------------------------------------------ #
        # 5. Final evaluation on held-out test set                            #
        # ------------------------------------------------------------------ #
        logger.info("Evaluating on held-out test set.")
        self.test_metrics_ = self._evaluate_on_test()

        logger.info("Optimisation complete.")
        logger.info("Best CV %s: %.4f", self.scoring, self.best_score_)
        logger.info("Test metrics: %s", self.test_metrics_)

        return self

    def plot_convergence(
        self,
        save_path: Optional[str] = "convergence_plot.png",
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot the optimisation convergence curve.

        Parameters
        ----------
        save_path : str or None
            If provided, save the figure to this path.
        show : bool
            If True, call plt.show() (requires an interactive backend).

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not self.history_:
            raise RuntimeError("Call fit() before plot_convergence().")

        iterations = [h["iteration"] for h in self.history_]
        scores = [h["cv_score"] for h in self.history_]
        best_so_far = [h["best_so_far"] for h in self.history_]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Bayesian Hyperparameter Optimisation — Convergence", fontsize=14)

        # --- Left: per-iteration score ---
        axes[0].scatter(iterations, scores, alpha=0.6, s=30, label="CV score per trial")
        axes[0].plot(iterations, best_so_far, color="red", linewidth=2, label="Best so far")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel(f"CV {self.scoring}")
        axes[0].set_title("Score per Iteration")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # --- Right: improvement over baseline ---
        baseline = scores[0]
        improvement = [s - baseline for s in best_so_far]
        axes[1].fill_between(iterations, improvement, alpha=0.3, color="green")
        axes[1].plot(iterations, improvement, color="green", linewidth=2)
        axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel(f"Improvement over iteration-1 score")
        axes[1].set_title("Cumulative Improvement")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Convergence plot saved to '%s'.", save_path)

        if show:
            plt.show()

        return fig

    def plot_hyperparameter_history(
        self,
        save_path: Optional[str] = "hp_history_plot.png",
        show: bool = False,
    ) -> plt.Figure:
        """
        Plot how each hyperparameter evolved across iterations.
        """
        if not self.history_:
            raise RuntimeError("Call fit() before plot_hyperparameter_history().")

        iterations = [h["iteration"] for h in self.history_]
        fig, axes = plt.subplots(
            len(self._PARAM_NAMES), 1,
            figsize=(12, 3 * len(self._PARAM_NAMES)),
            sharex=True,
        )
        fig.suptitle("Hyperparameter Values Across Iterations", fontsize=14)

        for ax, param in zip(axes, self._PARAM_NAMES):
            values = [h["params"][param] for h in self.history_]
            scores = [h["cv_score"] for h in self.history_]
            sc = ax.scatter(iterations, values, c=scores, cmap="viridis", s=30, alpha=0.8)
            plt.colorbar(sc, ax=ax, label=f"CV {self.scoring}")
            ax.set_ylabel(param)
            ax.grid(True, linestyle="--", alpha=0.4)

        axes[-1].set_xlabel("Iteration")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("HP history plot saved to '%s'.", save_path)

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        """Build a sklearn Pipeline with scaler + GBC."""
        gbc = GradientBoostingClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=self.random_state,
        )
        # NOTE: We pass a *clone* of the already-fitted scaler so that
        # cross_val_score re-fits it on each fold's training split.
        from sklearn.preprocessing import StandardScaler as _SS
        pipeline = Pipeline([
            ("scaler", _SS()),
            ("gbc", gbc),
        ])
        return pipeline

    def _objective(self, params: Dict[str, Any]) -> float:
        """
        Evaluate a hyperparameter configuration via inner CV on training data.
        Returns NEGATIVE score (we minimise).
        """
        pipeline = self._build_pipeline(params)
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        scores = cross_val_score(
            pipeline,
            self._X_train,
            self._y_train,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
        )
        return -float(np.mean(scores))  # minimise negative score

    def _record(
        self,
        iteration: int,
        params: Dict[str, Any],
        cv_score: float,
    ) -> None:
        """Append a record to the optimisation history."""
        if cv_score > self.best_score_:
            self.best_score_ = cv_score
            self.best_params_ = params.copy()

        self.history_.append(
            {
                "iteration": iteration,
                "params": params.copy(),
                "cv_score": cv_score,
                "best_so_far": self.best_score_,
            }
        )
        logger.info(
            "Iter %3d | score=%.4f | best=%.4f | params=%s",
            iteration,
            cv_score,
            self.best_score_,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()},
        )

    # ------------------------------------------------------------------ #
    # scikit-optimize backend                                              #
    # ------------------------------------------------------------------ #

    def _run_skopt(self) -> None:
        """Run optimisation using scikit-optimize's gp_minimize."""
        space = [
            Integer(50, 500, name="n_estimators"),
            Integer(2, 10, name="max_depth"),
            Real(0.01, 0.5, prior="log-uniform", name="learning_rate"),
            Real(0.5, 1.0, name="subsample"),
            Integer(1, 50, name="min_samples_leaf"),
        ]

        call_counter = [0]

        @use_named_args(space)
        def _skopt_objective(**kwargs):
            call_counter[0] += 1
            neg_score = self._objective(kwargs)
            cv_score = -neg_score
            self._record(call_counter[0], kwargs, cv_score)
            return neg_score

        result = gp_minimize(
            func=_skopt_objective,
            dimensions=space,
            n_calls=self.n_iter,
            n_initial_points=10,
            acq_func="EI",
            random_state=self.random_state,
            verbose=False,
        )

        # best_params_ already set inside _record; double-check consistency
        best_raw = dict(zip(self._PARAM_NAMES, result.x))
        logger.debug("skopt best raw: %s", best_raw)

    # ------------------------------------------------------------------ #
    # Manual GP backend (fallback)                                         #
    # ------------------------------------------------------------------ #

    def _run_manual_gp(self) -> None:
        """Run optimisation using the manual GP-based optimiser."""
        np.random.seed(self.random_state)
        optimizer = _ManualBayesianOptimizer(
            bounds=self._BOUNDS_CONTINUOUS,
            n_random_starts=10,
        )

        for i in range(1, self.n_iter + 1):
            x = optimizer.suggest()
            params = self._vector_to_params(x)
            neg_score = self._objective(params)
            cv_score = -neg_score
            optimizer.register(x, neg_score)
            self._record(i, params, cv_score)

    def _vector_to_params(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert a continuous vector to a named hyperparameter dict."""
        return {
            "n_estimators": int(round(np.clip(x[0], 50, 500))),
            "max_depth": int(round(np.clip(x[1], 2, 10))),
            "learning_rate": float(np.clip(x[2], 0.01, 0.5)),
            "subsample": float(np.clip(x[3], 0.5, 1.0)),
            "min_samples_leaf": int(round(np.clip(x[4], 1, 50))),
        }

    # ------------------------------------------------------------------ #
    # Final evaluation                                                     #
    # ------------------------------------------------------------------ #

    def _evaluate_on_test(self) -> Dict[str, Any]:
        """
        Evaluate the