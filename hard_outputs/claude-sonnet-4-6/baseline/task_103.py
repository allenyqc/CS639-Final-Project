"""
Walk-Forward (Expanding Window) Cross-Validation Framework for Time-Series Data.

This module implements a rigorous walk-forward cross-validation strategy that:
- Uses an expanding training window (no data leakage)
- Fits a fresh StandardScaler and Ridge model on each fold's training data
- Collects per-fold RMSE and R² scores
- Returns aggregated results and per-fold predictions
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
    fold_rmse: list[float] = field(default_factory=list)
    fold_r2: list[float] = field(default_factory=list)
    predictions: pd.Series = field(default_factory=pd.Series)
    fold_details: list[dict] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame summarising per-fold metrics."""
        rows = []
        for detail in self.fold_details:
            rows.append(
                {
                    "fold": detail["fold"],
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
            f"n_folds={len(self.fold_rmse)})"
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
        Number of out-of-sample test folds to create.
    min_train_size : int
        Minimum number of observations required in the first training window.
    alpha : float
        Regularisation strength for Ridge regression (default 1.0).
    fit_intercept : bool
        Whether to fit an intercept in Ridge (default True).
    random_state : int or None
        Random state passed to Ridge for reproducibility.
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
        if min_train_size < 2:
            raise ValueError("`min_train_size` must be >= 2.")
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

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train_indices, test_indices) pairs for each fold.

        The training window expands with each fold; the test window is the
        next contiguous segment of equal size (last fold may be smaller).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with a DatetimeIndex (or any ordered index).
        y : pd.Series
            Target series aligned with X.

        Returns
        -------
        list of (train_idx, test_idx) tuples (integer positions).
        """
        n = len(X)
        available = n - self.min_train_size
        if available <= 0:
            raise ValueError(
                f"Dataset has {n} rows but `min_train_size` is "
                f"{self.min_train_size}. Need at least "
                f"{self.min_train_size + 1} rows."
            )

        # Determine test-fold size so that n_folds folds fit in `available`
        fold_size = max(1, available // self.n_folds)
        actual_folds = min(self.n_folds, available)

        splits = []
        for fold in range(actual_folds):
            train_end = self.min_train_size + fold * fold_size  # exclusive
            test_start = train_end
            test_end = min(test_start + fold_size, n)  # exclusive

            if test_start >= n:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))

        if not splits:
            raise ValueError("No valid folds could be constructed.")

        return splits

    def fit_predict(
        self, X: pd.DataFrame, y: pd.Series
    ) -> WalkForwardResult:
        """
        Run walk-forward cross-validation.

        For each fold:
          1. Slice the expanding training window.
          2. Fit a **new** StandardScaler on training features only.
          3. Transform both training and test features with that scaler.
          4. Fit a **new** Ridge model on scaled training data.
          5. Predict on scaled test data.
          6. Record RMSE and R².

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (time-indexed, no NaNs).
        y : pd.Series
            Target series aligned with X (no NaNs).

        Returns
        -------
        WalkForwardResult
            Aggregated and per-fold metrics plus out-of-sample predictions.
        """
        self._validate_inputs(X, y)

        splits = self.split(X, y)

        X_arr = X.values
        y_arr = y.values
        index = X.index

        all_predictions: dict[int, float] = {}
        fold_rmse: list[float] = []
        fold_r2: list[float] = []
        fold_details: list[dict] = []

        for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
            # ---- Slice -------------------------------------------------------
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            # ---- Scale (fit ONLY on training data) ---------------------------
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)  # no fit → no leakage

            # ---- Model -------------------------------------------------------
            model = Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state,
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # ---- Metrics -----------------------------------------------------
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))

            fold_rmse.append(rmse)
            fold_r2.append(r2)

            # ---- Store predictions keyed by original integer position --------
            for pos, pred_val in zip(test_idx, y_pred):
                all_predictions[pos] = pred_val

            fold_details.append(
                {
                    "fold": fold_num,
                    "train_start": index[train_idx[0]],
                    "train_end": index[train_idx[-1]],
                    "test_start": index[test_idx[0]],
                    "test_end": index[test_idx[-1]],
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "rmse": rmse,
                    "r2": r2,
                    "model": model,
                    "scaler": scaler,
                }
            )

        # ---- Assemble predictions Series in chronological order --------------
        sorted_positions = sorted(all_predictions.keys())
        pred_values = [all_predictions[p] for p in sorted_positions]
        pred_index = index[sorted_positions]
        predictions = pd.Series(pred_values, index=pred_index, name="predicted")

        return WalkForwardResult(
            avg_rmse=float(np.mean(fold_rmse)),
            avg_r2=float(np.mean(fold_r2)),
            fold_rmse=fold_rmse,
            fold_r2=fold_r2,
            predictions=predictions,
            fold_details=fold_details,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(X: pd.DataFrame, y: pd.Series) -> None:
        """Validate shapes, alignment, and missing values."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("`X` must be a pandas DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("`y` must be a pandas Series.")
        if len(X) != len(y):
            raise ValueError(
                f"`X` has {len(X)} rows but `y` has {len(y)} elements."
            )
        if len(X) == 0:
            raise ValueError("Input data is empty.")
        if X.isnull().any().any():
            raise ValueError("`X` contains NaN values. Please impute or drop them.")
        if y.isnull().any():
            raise ValueError("`y` contains NaN values. Please impute or drop them.")
        if not X.index.equals(y.index):
            warnings.warn(
                "X and y have different indices. Ensure they are aligned.",
                UserWarning,
                stacklevel=3,
            )


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
        Time-indexed feature matrix.
    y : pd.Series
        Target series aligned with X.
    n_folds : int
        Number of out-of-sample folds (default 5).
    min_train_size : int
        Minimum training observations for the first fold (default 30).
    alpha : float
        Ridge regularisation strength (default 1.0).
    fit_intercept : bool
        Whether Ridge fits an intercept (default True).
    random_state : int or None
        Reproducibility seed (default 42).

    Returns
    -------
    WalkForwardResult
        avg_rmse, avg_r2, fold_rmse, fold_r2, predictions, fold_details.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> idx = pd.date_range("2020-01-01", periods=200, freq="D")
    >>> X = pd.DataFrame(rng.standard_normal((200, 3)), index=idx,
    ...                  columns=["f1", "f2", "f3"])
    >>> y = pd.Series(rng.standard_normal(200), index=idx, name="target")
    >>> result = walk_forward_cv(X, y, n_folds=5, min_train_size=50)
    >>> print(result)
    WalkForwardResult(avg_rmse=..., avg_r2=..., n_folds=5)
    """
    validator = WalkForwardCV(
        n_folds=n_folds,
        min_train_size=min_train_size,
        alpha=alpha,
        fit_intercept=fit_intercept,
        random_state=random_state,
    )
    return validator.fit_predict(X, y)


# ---------------------------------------------------------------------------
# Self-contained demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap

    print("=" * 60)
    print("Walk-Forward CV — Demo")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)
    n_samples = 300
    idx = pd.date_range("2018-01-01", periods=n_samples, freq="D")

    # Synthetic features
    X_demo = pd.DataFrame(
        {
            "momentum": rng.standard_normal(n_samples).cumsum(),
            "volatility": np.abs(rng.standard_normal(n_samples)),
            "volume": rng.uniform(1_000, 10_000, n_samples),
        },
        index=idx,
    )

    # Target: linear combination + noise
    true_coefs = np.array([0.5, -1.2, 0.003])
    y_demo = pd.Series(
        X_demo.values @ true_coefs + rng.standard_normal(n_samples) * 2,
        index=idx,
        name="returns",
    )

    result = walk_forward_cv(
        X_demo,
        y_demo,
        n_folds=5,
        min_train_size=60,
        alpha=0.5,
    )

    print(f"\nAverage RMSE : {result.avg_rmse:.4f}")
    print(f"Average R²   : {result.avg_r2:.4f}")
    print("\nPer-fold summary:")
    print(result.summary().to_string())
    print(f"\nPredictions shape : {result.predictions.shape}")
    print(f"Predictions index : {result.predictions.index[0]} → "
          f"{result.predictions.index[-1]}")
    print("\nFirst 5 predictions:")
    print(result.predictions.head())