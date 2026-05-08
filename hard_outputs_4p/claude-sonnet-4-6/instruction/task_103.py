"""
walk_forward_cv.py

Walk-forward (expanding window) cross-validation framework for time-series data.
Implements strict temporal splitting to prevent data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Stores results for a single fold."""
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    rmse: float
    r2: float
    y_true: np.ndarray
    y_pred: np.ndarray
    test_index: pd.Index


@dataclass
class WalkForwardResult:
    """Aggregated results across all folds."""
    avg_rmse: float
    avg_r2: float
    std_rmse: float
    std_r2: float
    fold_results: list[FoldResult] = field(default_factory=list)

    # Concatenated predictions in chronological order
    all_predictions: Optional[pd.Series] = None
    all_actuals: Optional[pd.Series] = None

    def summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame summarising per-fold metrics."""
        rows = [
            {
                "fold": fr.fold_index,
                "train_start": fr.train_start,
                "train_end": fr.train_end,
                "test_start": fr.test_start,
                "test_end": fr.test_end,
                "rmse": fr.rmse,
                "r2": fr.r2,
                "n_test": len(fr.y_true),
            }
            for fr in self.fold_results
        ]
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core walk-forward CV class
# ---------------------------------------------------------------------------

class WalkForwardCV:
    """
    Expanding-window (walk-forward) cross-validation for time-series data.

    Parameters
    ----------
    n_folds : int
        Number of out-of-sample evaluation folds.
    min_train_size : int
        Minimum number of observations required in the initial training window.
    alpha : float
        Regularisation strength for Ridge regression.
    fit_intercept : bool
        Whether to fit an intercept in Ridge regression.
    random_state : int | None
        Random seed passed to Ridge (for reproducibility with solvers that
        use randomness).
    """

    def __init__(
        self,
        n_folds: int = 5,
        min_train_size: int = 30,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        if n_folds < 1:
            raise ValueError("`n_folds` must be >= 1.")
        if min_train_size < 1:
            raise ValueError("`min_train_size` must be >= 1.")
        if alpha <= 0:
            raise ValueError("`alpha` must be positive.")

        self.n_folds = n_folds
        self.min_train_size = min_train_size
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> WalkForwardResult:
        """
        Run walk-forward cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with a DatetimeIndex (or any monotonic index).
            Must be sorted in ascending temporal order.
        y : pd.Series
            Target series aligned with X.

        Returns
        -------
        WalkForwardResult
            Contains average scores, per-fold scores, and out-of-sample
            predictions concatenated in chronological order.
        """
        self._validate_inputs(X, y)

        # Ensure chronological order
        X = X.sort_index()
        y = y.loc[X.index]

        n_samples = len(X)
        fold_indices = self._compute_fold_indices(n_samples)

        fold_results: list[FoldResult] = []
        pred_series_list: list[pd.Series] = []
        actual_series_list: list[pd.Series] = []

        for fold_idx, (train_end_pos, test_start_pos, test_end_pos) in enumerate(
            fold_indices, start=1
        ):
            # ----------------------------------------------------------------
            # 1. Temporal split — NO information from test leaks into train
            # ----------------------------------------------------------------
            X_train = X.iloc[:train_end_pos]
            y_train = y.iloc[:train_end_pos]

            X_test = X.iloc[test_start_pos:test_end_pos]
            y_test = y.iloc[test_start_pos:test_end_pos]

            logger.info(
                "Fold %d | train=[%s … %s] (%d obs) | test=[%s … %s] (%d obs)",
                fold_idx,
                X_train.index[0],
                X_train.index[-1],
                len(X_train),
                X_test.index[0],
                X_test.index[-1],
                len(X_test),
            )

            # ----------------------------------------------------------------
            # 2. Fit scaler ONLY on training data
            # ----------------------------------------------------------------
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)   # fit + transform train
            X_test_scaled = scaler.transform(X_test)          # transform test only

            # ----------------------------------------------------------------
            # 3. Fit model ONLY on training data
            # ----------------------------------------------------------------
            model = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state,
            )
            model.fit(X_train_scaled, y_train)

            # ----------------------------------------------------------------
            # 4. Predict on test set and evaluate
            # ----------------------------------------------------------------
            y_pred = model.predict(X_test_scaled)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))

            logger.info("Fold %d | RMSE=%.6f | R²=%.6f", fold_idx, rmse, r2)

            fold_result = FoldResult(
                fold_index=fold_idx,
                train_start=X_train.index[0],
                train_end=X_train.index[-1],
                test_start=X_test.index[0],
                test_end=X_test.index[-1],
                rmse=rmse,
                r2=r2,
                y_true=y_test.to_numpy(),
                y_pred=y_pred,
                test_index=X_test.index,
            )
            fold_results.append(fold_result)

            pred_series_list.append(
                pd.Series(y_pred, index=X_test.index, name="y_pred")
            )
            actual_series_list.append(y_test.rename("y_true"))

        # ------------------------------------------------------------------
        # 5. Aggregate metrics
        # ------------------------------------------------------------------
        rmse_scores = np.array([fr.rmse for fr in fold_results])
        r2_scores = np.array([fr.r2 for fr in fold_results])

        result = WalkForwardResult(
            avg_rmse=float(rmse_scores.mean()),
            avg_r2=float(r2_scores.mean()),
            std_rmse=float(rmse_scores.std(ddof=1)) if len(rmse_scores) > 1 else 0.0,
            std_r2=float(r2_scores.std(ddof=1)) if len(r2_scores) > 1 else 0.0,
            fold_results=fold_results,
            all_predictions=pd.concat(pred_series_list).sort_index(),
            all_actuals=pd.concat(actual_series_list).sort_index(),
        )

        logger.info(
            "Walk-forward CV complete | avg RMSE=%.6f (±%.6f) | avg R²=%.6f (±%.6f)",
            result.avg_rmse,
            result.std_rmse,
            result.avg_r2,
            result.std_r2,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("`X` must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("`y` must be a pandas Series.")
        if len(X) != len(y):
            raise ValueError("`X` and `y` must have the same length.")
        if X.empty:
            raise ValueError("`X` is empty.")
        if not X.index.equals(y.index):
            raise ValueError("`X` and `y` must share the same index.")
        if X.isnull().any().any():
            raise ValueError("`X` contains NaN values. Please impute before calling fit_evaluate.")
        if y.isnull().any():
            raise ValueError("`y` contains NaN values. Please impute before calling fit_evaluate.")

        n_samples = len(X)
        required = self.min_train_size + self.n_folds
        if n_samples < required:
            raise ValueError(
                f"Dataset has {n_samples} samples but requires at least "
                f"min_train_size + n_folds = {required} samples."
            )

    def _compute_fold_indices(
        self, n_samples: int
    ) -> list[tuple[int, int, int]]:
        """
        Compute (train_end, test_start, test_end) position tuples for each fold.

        The remaining observations after the initial training window are split
        evenly across `n_folds` test segments.  Each fold's training window
        expands to include all data up to (but not including) its test segment.
        """
        remaining = n_samples - self.min_train_size
        fold_size = remaining // self.n_folds

        if fold_size < 1:
            raise ValueError(
                "Fold size is < 1.  Increase the dataset size or reduce "
                "`n_folds` / `min_train_size`."
            )

        indices: list[tuple[int, int, int]] = []
        for k in range(self.n_folds):
            test_start = self.min_train_size + k * fold_size
            # Last fold absorbs any remainder
            test_end = (
                test_start + fold_size
                if k < self.n_folds - 1
                else n_samples
            )
            train_end = test_start  # expanding window: train on everything before test
            indices.append((train_end, test_start, test_end))

        return indices


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    min_train_size: int = 30,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    random_state: Optional[int] = 42,
) -> WalkForwardResult:
    """
    Convenience wrapper around :class:`WalkForwardCV`.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (time-indexed, sorted ascending).
    y : pd.Series
        Target series aligned with X.
    n_folds : int
        Number of out-of-sample folds.
    min_train_size : int
        Minimum observations in the initial training window.
    alpha : float
        Ridge regularisation strength.
    fit_intercept : bool
        Whether Ridge fits an intercept.
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    WalkForwardResult
    """
    cv = WalkForwardCV(
        n_folds=n_folds,
        min_train_size=min_train_size,
        alpha=alpha,
        fit_intercept=fit_intercept,
        random_state=random_state,
    )
    return cv.fit_evaluate(X, y)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(0)
    n = 200
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    X_demo = pd.DataFrame(
        rng.standard_normal((n, 4)),
        index=dates,
        columns=["feat_a", "feat_b", "feat_c", "feat_d"],
    )
    # Target: linear combination + noise
    true_coef = np.array([1.5, -2.0, 0.5, 3.0])
    y_demo = pd.Series(
        X_demo.values @ true_coef + rng.standard_normal(n) * 0.5,
        index=dates,
        name="target",
    )

    result = walk_forward_cv(
        X=X_demo,
        y=y_demo,
        n_folds=5,
        min_train_size=50,
        alpha=1.0,
    )

    print("\n=== Walk-Forward CV Summary ===")
    print(result.summary().to_string(index=False))
    print(f"\nAverage RMSE : {result.avg_rmse:.6f} ± {result.std_rmse:.6f}")
    print(f"Average R²   : {result.avg_r2:.6f} ± {result.std_r2:.6f}")