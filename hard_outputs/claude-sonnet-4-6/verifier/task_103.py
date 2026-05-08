"""
Walk-Forward (Expanding Window) Cross-Validation Framework for Time-Series Data.

This module implements a robust walk-forward cross-validation strategy that:
- Uses an expanding training window (no data leakage)
- Fits a fresh StandardScaler and Ridge regression model at each fold
- Collects per-fold RMSE and R² scores
- Returns aggregated results including average scores, per-fold scores, and predictions
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """Container for walk-forward cross-validation results."""

    avg_rmse: float
    avg_r2: float
    per_fold_rmse: list[float] = field(default_factory=list)
    per_fold_r2: list[float] = field(default_factory=list)
    predictions: pd.Series = field(default_factory=pd.Series)
    fold_details: list[dict] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame summarising per-fold metrics."""
        rows = []
        for i, detail in enumerate(self.fold_details, start=1):
            rows.append(
                {
                    "fold": i,
                    "train_start": detail["train_start"],
                    "train_end": detail["train_end"],
                    "test_start": detail["test_start"],
                    "test_end": detail["test_end"],
                    "train_size": detail["train_size"],
                    "test_size": detail["test_size"],
                    "rmse": detail["rmse"],
                    "r2": detail["r2"],
                }
            )
        return pd.DataFrame(rows).set_index("fold")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"WalkForwardResult("
            f"avg_rmse={self.avg_rmse:.6f}, "
            f"avg_r2={self.avg_r2:.6f}, "
            f"n_folds={len(self.per_fold_rmse)})"
        )


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

class WalkForwardCV:
    """
    Walk-forward (expanding window) cross-validator for time-series data.

    Parameters
    ----------
    n_folds : int
        Number of out-of-sample test folds to create.  Must be >= 1.
    min_train_size : int
        Minimum number of observations required in the first training window.
        Must be >= 2 (Ridge needs at least 2 samples).
    alpha : float
        Regularisation strength for Ridge regression (default 1.0).
    fit_intercept : bool
        Whether to fit an intercept in Ridge regression (default True).
    gap : int
        Number of observations to skip between the end of the training window
        and the start of the test window.  Useful when the target is a
        multi-step-ahead forecast (default 0 = no gap).
    verbose : bool
        If True, print progress information for each fold (default False).

    Notes
    -----
    The dataset is split as follows::

        |<--- min_train_size --->|<--- gap --->|<--- fold_size --->| ... |

    The training window *expands* with each fold; the scaler and model are
    re-fitted from scratch on the growing training set so that no future
    information leaks into past folds.
    """

    def __init__(
        self,
        n_folds: int = 5,
        min_train_size: int = 30,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        gap: int = 0,
        verbose: bool = False,
    ) -> None:
        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}.")
        if min_train_size < 2:
            raise ValueError(f"min_train_size must be >= 2, got {min_train_size}.")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}.")
        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}.")

        self.n_folds = n_folds
        self.min_train_size = min_train_size
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.gap = gap
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ridge_kwargs: Optional[dict] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with a DatetimeIndex (or any monotonic index).
            Rows must be sorted in chronological order.
        y : pd.Series
            Target series aligned with *X* (same index).
        ridge_kwargs : dict, optional
            Additional keyword arguments forwarded to ``Ridge()``.

        Returns
        -------
        WalkForwardResult
            Dataclass containing average scores, per-fold scores, and
            out-of-sample predictions for every test observation.

        Raises
        ------
        ValueError
            If the dataset is too small to accommodate the requested folds.
        TypeError
            If *X* is not a DataFrame or *y* is not a Series.
        """
        X, y = self._validate_inputs(X, y)
        ridge_kwargs = ridge_kwargs or {}

        fold_indices = self._compute_fold_indices(len(X))

        all_predictions: dict[int, float] = {}  # position -> predicted value
        per_fold_rmse: list[float] = []
        per_fold_r2: list[float] = []
        fold_details: list[dict] = []

        for fold_num, (train_idx, test_idx) in enumerate(fold_indices, start=1):
            # ---- Slice data (no copies of future data) ----
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # ---- Fit scaler on training data ONLY ----
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # ---- Transform test data with the training scaler ----
            X_test_scaled = scaler.transform(X_test)

            # ---- Fit Ridge on scaled training data ----
            model = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                **ridge_kwargs,
            )
            model.fit(X_train_scaled, y_train)

            # ---- Predict on test fold ----
            y_pred = model.predict(X_test_scaled)

            # ---- Compute metrics ----
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))

            per_fold_rmse.append(rmse)
            per_fold_r2.append(r2)

            # Store predictions keyed by integer position
            for pos, pred_val in zip(test_idx, y_pred):
                all_predictions[pos] = pred_val

            # ---- Record fold metadata ----
            detail = {
                "train_start": X.index[train_idx[0]],
                "train_end": X.index[train_idx[-1]],
                "test_start": X.index[test_idx[0]],
                "test_end": X.index[test_idx[-1]],
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "rmse": rmse,
                "r2": r2,
            }
            fold_details.append(detail)

            if self.verbose:
                print(
                    f"Fold {fold_num:>2d} | "
                    f"train [{detail['train_start']} → {detail['train_end']}] "
                    f"({detail['train_size']} obs) | "
                    f"test  [{detail['test_start']} → {detail['test_end']}] "
                    f"({detail['test_size']} obs) | "
                    f"RMSE={rmse:.6f}  R²={r2:.6f}"
                )

        # ---- Assemble predictions Series (chronological order) ----
        sorted_positions = sorted(all_predictions.keys())
        pred_index = y.index[sorted_positions]
        pred_values = [all_predictions[p] for p in sorted_positions]
        predictions = pd.Series(pred_values, index=pred_index, name="predicted")

        avg_rmse = float(np.mean(per_fold_rmse))
        avg_r2 = float(np.mean(per_fold_r2))

        if self.verbose:
            print(
                f"\n{'='*60}\n"
                f"Average RMSE : {avg_rmse:.6f}\n"
                f"Average R²   : {avg_r2:.6f}\n"
                f"{'='*60}"
            )

        return WalkForwardResult(
            avg_rmse=avg_rmse,
            avg_r2=avg_r2,
            per_fold_rmse=per_fold_rmse,
            per_fold_r2=per_fold_r2,
            predictions=predictions,
            fold_details=fold_details,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Validate and align inputs."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}.")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y).__name__}.")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. "
                f"Got X={len(X)}, y={len(y)}."
            )
        if not X.index.equals(y.index):
            warnings.warn(
                "X and y have different indices; aligning on X.index.",
                UserWarning,
                stacklevel=3,
            )
            y = y.reindex(X.index)

        if X.isnull().any().any():
            warnings.warn(
                "X contains NaN values; Ridge may produce unexpected results.",
                UserWarning,
                stacklevel=3,
            )
        if y.isnull().any():
            warnings.warn(
                "y contains NaN values; Ridge may produce unexpected results.",
                UserWarning,
                stacklevel=3,
            )

        return X, y

    def _compute_fold_indices(
        self, n_samples: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Compute (train_indices, test_indices) pairs for each fold.

        The test folds are of equal size (last fold may be slightly larger
        if the remaining observations do not divide evenly).

        Layout::

            [0 .. min_train_size-1] [gap] [fold_1] [fold_2] ... [fold_n]
        """
        available = n_samples - self.min_train_size - self.gap
        if available <= 0:
            raise ValueError(
                f"Dataset too small: n_samples={n_samples}, "
                f"min_train_size={self.min_train_size}, gap={self.gap}. "
                f"Need at least {self.min_train_size + self.gap + 1} observations."
            )

        fold_size, remainder = divmod(available, self.n_folds)
        if fold_size == 0:
            raise ValueError(
                f"Cannot create {self.n_folds} folds with only {available} "
                f"observations after the initial training window and gap. "
                f"Reduce n_folds or min_train_size."
            )

        fold_indices: list[tuple[np.ndarray, np.ndarray]] = []
        test_start = self.min_train_size + self.gap  # first test observation

        for fold_num in range(self.n_folds):
            # Last fold absorbs any remainder
            extra = remainder if fold_num == self.n_folds - 1 else 0
            test_end = test_start + fold_size + extra  # exclusive

            train_end = test_start - self.gap  # exclusive; respects gap

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            fold_indices.append((train_idx, test_idx))

            # Next fold's test window starts right after this one
            test_start = test_end

        return fold_indices


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    min_train_size: int = 30,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    gap: int = 0,
    verbose: bool = False,
    ridge_kwargs: Optional[dict] = None,
) -> WalkForwardResult:
    """
    Convenience wrapper around :class:`WalkForwardCV`.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (time-indexed, chronologically sorted).
    y : pd.Series
        Target series aligned with *X*.
    n_folds : int
        Number of out-of-sample test folds (default 5).
    min_train_size : int
        Minimum training window size (default 30).
    alpha : float
        Ridge regularisation strength (default 1.0).
    fit_intercept : bool
        Whether Ridge fits an intercept (default True).
    gap : int
        Observations to skip between train end and test start (default 0).
    verbose : bool
        Print per-fold progress (default False).
    ridge_kwargs : dict, optional
        Extra keyword arguments for ``Ridge()``.

    Returns
    -------
    WalkForwardResult
        Aggregated cross-validation results.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> idx = pd.date_range("2020-01-01", periods=200, freq="D")
    >>> X = pd.DataFrame(rng.standard_normal((200, 3)), index=idx,
    ...                  columns=["f1", "f2", "f3"])
    >>> y = pd.Series(rng.standard_normal(200), index=idx, name="target")
    >>> result = walk_forward_validate(X, y, n_folds=5, min_train_size=50,
    ...                                verbose=True)
    >>> print(result.summary())
    """
    cv = WalkForwardCV(
        n_folds=n_folds,
        min_train_size=min_train_size,
        alpha=alpha,
        fit_intercept=fit_intercept,
        gap=gap,
        verbose=verbose,
    )
    return cv.validate(X, y, ridge_kwargs=ridge_kwargs)


# ---------------------------------------------------------------------------
# Self-contained demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2018-01-01", periods=n, freq="B")  # business days

    # Synthetic features
    X_demo = pd.DataFrame(
        {
            "momentum": rng.standard_normal(n).cumsum(),
            "volatility": np.abs(rng.standard_normal(n)),
            "volume": rng.uniform(1_000, 10_000, n),
        },
        index=idx,
    )

    # Target: linear combination of features + noise
    true_coef = np.array([0.5, -1.2, 0.003])
    y_demo = pd.Series(
        X_demo.values @ true_coef + rng.standard_normal(n) * 0.5,
        index=idx,
        name="returns",
    )

    print("=" * 60)
    print("Walk-Forward Cross-Validation Demo")
    print("=" * 60)

    result = walk_forward_validate(
        X_demo,
        y_demo,
        n_folds=5,
        min_train_size=60,
        alpha=0.5,
        gap=1,
        verbose=True,
    )

    print("\nFold Summary:")
    print(result.summary().to_string())

    print(f"\nFirst 10 out-of-sample predictions:\n{result.predictions.head(10)}")
    print(f"\nTotal OOS predictions: {len(result.predictions)}")