```python
"""
Bayesian Hyperparameter Optimization Module for GradientBoostingClassifier.

Uses scikit-optimize's gp_minimize for Bayesian optimization with proper
train/validation/test splits and comprehensive tracking.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn(
        "scikit-optimize not installed. Falling back to manual GP-based optimization. "
        "Install with: pip install scikit-optimize",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_hyperparameters: dict[str, Any]
    best_cv_score: float
    final_model: GradientBoostingClassifier
    test_metrics: dict[str, float]
    history: list[dict[str, Any]]
    best_so_far: list[float]


@dataclass
class _IterationRecord:
    """Single iteration record."""
    iteration: int
    hyperparameters: dict[str, Any]
    cv_score: float
    best_so_far: float


# ---------------------------------------------------------------------------
# Fallback: manual Gaussian Process optimizer
# ---------------------------------------------------------------------------

class _ManualGPOptimizer:
    """
    Minimal Gaussian Process-based Bayesian optimizer used when
    scikit-optimize is unavailable.

    Uses a simple RBF kernel GP with Expected Improvement acquisition.
    """

    def __init__(
        self,
        param_bounds: list[tuple[float, float]],
        n_initial: int = 10,
        random_state: int = 42,
    ) -> None:
        self._bounds = np.array(param_bounds, dtype=float)
        self._n_initial = n_initial
        self._rng = np.random.default_rng(random_state)
        self._X_obs: list[np.ndarray] = []
        self._y_obs: list[float] = []

    # ------------------------------------------------------------------
    def _rbf_kernel(
        self, X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0
    ) -> np.ndarray:
        diff = X1[:, None, :] - X2[None, :, :]
        return np.exp(-0.5 * np.sum((diff / length_scale) ** 2, axis=-1))

    def _gp_predict(
        self, X_new: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._X_obs:
            n = len(X_new)
            return np.zeros(n), np.ones(n)

        X_obs = np.array(self._X_obs)
        y_obs = np.array(self._y_obs)
        noise = 1e-6

        K = self._rbf_kernel(X_obs, X_obs) + noise * np.eye(len(X_obs))
        K_s = self._rbf_kernel(X_obs, X_new)
        K_ss = self._rbf_kernel(X_new, X_new)

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
            mu = K_s.T @ alpha
            v = np.linalg.solve(L, K_s)
            var = np.diag(K_ss) - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 0.0)
        except np.linalg.LinAlgError:
            n = len(X_new)
            mu = np.full(n, np.mean(y_obs))
            var = np.ones(n)

        return mu, np.sqrt(var)

    def _expected_improvement(
        self, X_cand: np.ndarray, xi: float = 0.01
    ) -> np.ndarray:
        from scipy.stats import norm  # local import to avoid top-level dep

        mu, sigma = self._gp_predict(X_cand)
        best = max(self._y_obs) if self._y_obs else 0.0
        z = (mu - best - xi) / (sigma + 1e-9)
        ei = (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-10] = 0.0
        return ei

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        lo, hi = self._bounds[:, 0], self._bounds[:, 1]
        return (X - lo) / (hi - lo + 1e-12)

    def suggest(self) -> np.ndarray:
        lo, hi = self._bounds[:, 0], self._bounds[:, 1]
        if len(self._X_obs) < self._n_initial:
            return self._rng.uniform(lo, hi)

        n_cand = 1000
        candidates = self._rng.uniform(lo, hi, size=(n_cand, len(lo)))
        norm_cand = self._normalize(candidates)
        norm_obs = self._normalize(np.array(self._X_obs))

        # Temporarily replace observations with normalised versions
        orig = self._X_obs
        self._X_obs = list(norm_obs)
        ei = self._expected_improvement(norm_cand)
        self._X_obs = orig

        return candidates[np.argmax(ei)]

    def register(self, x: np.ndarray, y: float) -> None:
        self._X_obs.append(np.array(x, dtype=float))
        self._y_obs.append(float(y))


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------

def _compute_test_metrics(
    model: GradientBoostingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute a comprehensive set of test metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    n_classes = len(np.unique(y_test))
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    if n_classes == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        metrics["average_precision"] = average_precision_score(
            y_test, y_proba[:, 1]
        )
    else:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["roc_auc_ovr"] = float("nan")

    return metrics


def _cross_val_score_manual(
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3,
    random_state: int = 42,
) -> float:
    """
    Stratified k-fold CV on the training set only.
    Returns mean ROC-AUC (or balanced accuracy for multi-class).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: list[float] = []
    n_classes = len(np.unique(y_train))

    for fold_train_idx, fold_val_idx in skf.split(X_train, y_train):
        X_fold_tr, X_fold_val = X_train[fold_train_idx], X_train[fold_val_idx]
        y_fold_tr, y_fold_val = y_train[fold_train_idx], y_train[fold_val_idx]

        # Fit scaler ONLY on fold training data
        scaler = StandardScaler()
        X_fold_tr_sc = scaler.fit_transform(X_fold_tr)
        X_fold_val_sc = scaler.transform(X_fold_val)

        clf = GradientBoostingClassifier(**params, random_state=random_state)
        clf.fit(X_fold_tr_sc, y_fold_tr)

        if n_classes == 2:
            proba = clf.predict_proba(X_fold_val_sc)[:, 1]
            score = roc_auc_score(y_fold_val, proba)
        else:
            score = balanced_accuracy_score(y_fold_val, clf.predict(X_fold_val_sc))

        scores.append(score)

    return float(np.mean(scores))


def _build_param_dict(
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    min_samples_leaf: int,
) -> dict[str, Any]:
    return {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "subsample": float(subsample),
        "min_samples_leaf": int(min_samples_leaf),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bayesian_hyperparameter_optimization(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 50,
    test_size: float = 0.15,
    val_size: float = 0.15,
    cv_folds: int = 3,
    random_state: int = 42,
    plot: bool = True,
    plot_path: str | None = None,
) -> OptimizationResult:
    """
    Perform Bayesian hyperparameter optimisation for a GradientBoostingClassifier.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    n_iterations : int
        Number of Bayesian optimisation iterations (default 50).
    test_size : float
        Fraction of data held out as the final test set.
    val_size : float
        Fraction of the *remaining* data used as a validation set
        (for threshold tuning / early stopping — never for HP search).
    cv_folds : int
        Number of stratified CV folds used inside the optimisation loop.
    random_state : int
        Global random seed.
    plot : bool
        Whether to produce convergence plots.
    plot_path : str | None
        If given, save the plot to this path instead of showing it.

    Returns
    -------
    OptimizationResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # ------------------------------------------------------------------
    # 1. Split BEFORE any preprocessing
    # ------------------------------------------------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # val_size is relative to the remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size / (1.0 - test_size),
        stratify=y_temp,
        random_state=random_state,
    )

    print(
        f"Data splits — train: {len(X_train)}, val: {len(X_val)}, "
        f"test: {len(X_test)}"
    )

    # ------------------------------------------------------------------
    # 2. Hyperparameter search space
    # ------------------------------------------------------------------
    param_space_skopt = [
        Integer(50, 500, name="n_estimators"),
        Integer(2, 10, name="max_depth"),
        Real(1e-3, 0.5, prior="log-uniform", name="learning_rate"),
        Real(0.5, 1.0, name="subsample"),
        Integer(1, 50, name="min_samples_leaf"),
    ]

    # Bounds for the manual GP fallback (same ranges, continuous)
    param_bounds_manual = [
        (50.0, 500.0),   # n_estimators
        (2.0, 10.0),     # max_depth
        (1e-3, 0.5),     # learning_rate
        (0.5, 1.0),      # subsample
        (1.0, 50.0),     # min_samples_leaf
    ]

    # ------------------------------------------------------------------
    # 3. Optimisation loop
    # ------------------------------------------------------------------
    history: list[dict[str, Any]] = []
    best_so_far_list: list[float] = []
    best_score = -np.inf
    best_params: dict[str, Any] = {}

    if SKOPT_AVAILABLE:
        # ---- scikit-optimize path ------------------------------------
        @use_named_args(param_space_skopt)
        def _objective(
            n_estimators: int,
            max_depth: int,
            learning_rate: float,
            subsample: float,
            min_samples_leaf: int,
        ) -> float:
            params = _build_param_dict(
                n_estimators, max_depth, learning_rate,
                subsample, min_samples_leaf,
            )
            score = _cross_val_score_manual(
                params, X_train, y_train,
                n_splits=cv_folds, random_state=random_state,
            )
            # gp_minimize minimises, so negate
            return -score

        # Wrap to capture history
        iteration_counter = [0]

        def _objective_with_tracking(*args: Any, **kwargs: Any) -> float:
            val = _objective(*args, **kwargs)
            score = -val
            nonlocal best_score, best_params

            # Reconstruct param dict from positional args
            raw = list(args[0]) if args else []
            if len(raw) == 5:
                params = _build_param_dict(*raw)
            else:
                params = {}

            if score > best_score:
                best_score = score
                best_params = params

            iteration_counter[0] += 1
            best_so_far_list.append(best_score)
            history.append(
                {
                    "iteration": iteration_counter[0],
                    "hyperparameters": params.copy(),
                    "cv_score": score,
                    "best_so_far": best_score,
                }
            )
            print(
                f"  Iter {iteration_counter[0]:3d} | score={score:.4f} | "
                f"best={best_score:.4f} | params={params}"
            )
            return val

        print(f"\nStarting Bayesian optimisation ({n_iterations} iterations) "
              f"using scikit-optimize …\n")
        result = gp_minimize(
            _objective_with_tracking,
            param_space_skopt,
            n_calls=n_iterations,
            n_initial_points=max(10, n_iterations // 5),
            random_state=random_state,
            verbose=False,
        )
        # Extract best from skopt result as well
        best_params = _build_param_dict(*result.x)
        best_score = -result.fun

    else:
        # ---- Manual GP path ------------------------------------------
        optimizer = _ManualGPOptimizer(
            param_bounds=param_bounds_manual,
            n_initial=max(10, n_iterations // 5),
            random_state=random_state,
        )

        print(f"\nStarting Bayesian optimisation ({n_iterations} iterations) "
              f"using manual GP …\n")

        for i in range(1, n_iterations + 1):
            x_raw = optimizer.suggest()
            params = _build_param_dict(
                x_raw[0], x_raw[1], x_raw[2], x_raw[3], x_raw[4]
            )
            score = _cross_val_score_manual(
                params, X_train, y_train,
                n_splits=cv_folds, random_state=random_state,
            )
            optimizer.register(x_raw, score)

            if score > best_score:
                best_score = score
                best_params = params.copy()

            best_so_far_list.append(best_score)
            history.append(
                {
                    "iteration": i,
                    "hyperparameters": params.copy(),
                    "cv_score": score,
                    "best_so_far": best_score,
                }
            )
            print(
                f"  Iter {i:3d} | score={score:.4f} | "
                f"best={best_score:.4f} | params={params}"
            )

    # ------------------------------------------------------------------
    # 4. Retrain final model on train+val with best hyperparameters
    #    (test set is still untouched)
    # ------------------------------------------------------------------
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # Fit scaler on train+val only
    final_scaler = StandardScaler()
    X_trainval_sc = final_scaler.fit_transform(X_trainval)
    X_test_sc = final_scaler.transform(X_test)

    final_model = GradientBoostingClassifier(
        **best_params, random_state=random_state
    )
    final_model.fit(X_trainval_sc, y_trainval)

    # ------------------------------------------------------------------
    # 5. Evaluate on held-out test set (first and only time)
    # ------------------------------------------------------------------
    test_metrics = _compute_test_metrics(final_model, X_test_sc, y_test)

    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"\nBest CV score : {best_score:.4f}")
    print("\nTEST SET METRICS (final evaluation):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nClassification Report (test set):")
    y_pred_test = final_model.predict(X_test_sc)
    print(classification_report(y_test, y_pred_test, zero_division=0))
    print("=" * 60)

    # ------------------------------------------------------------------
    # 6. Convergence plots
    # ------------------------------------------------------------------
    if plot:
        _plot_convergence(history, best_so_far_list, test_metrics, plot_path)

    return OptimizationResult(
        best_hyperparameters=best_params,
        best_cv_score=best_score,
        final_model=final_model,
        test_metrics=test_metrics,
        history=history,
        best_so_far=best_so_far_list,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_convergence(
    history: list[dict[str, Any]],
    best_so_far: list[float],
    test_metrics: dict[str, float],
    save_path: str | None = None,
) -> None