"""
Walk-forward (expanding window) cross-validation framework for time-series data.

This module implements a rigorous walk-forward CV that:
- Uses chronological splits (no shuffling)
- Fits scalers only on training data at each fold
- Prevents data leakage from future folds
- Collects per-fold RMSE and R² scores
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

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Stores results for a single fold."""

    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    rmse: float
    r2: float
    predictions: np.ndarray
    actuals: np.ndarray
    test_indices: pd.Index


@dataclass
class WalkForwardCVResult:
    """Aggregated results from walk-forward cross-validation."""

    avg_rmse: float
    avg_r2: float
    std_rmse: float
    std_r2: float
    per_fold_results: list[FoldResult] = field(default_factory=list)
    all_predictions: pd.Series = field(default_factory=pd.Series)
    all_actuals: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        lines = [
            "Walk-Forward CV Summary",
            "=" * 40,
            f"Number of folds: {len(self.per_fold_results)}",
            f"Average RMSE:    {self.avg_rmse:.6f} ± {self.std_rmse:.6f}",
            f"Average R²:      {self.avg_r2:.6f} ± {self.std_r2:.6f}",
            "",
            "Per-fold breakdown:",
        ]
        for fr in self.per_fold_results:
            lines.append(
                f"  Fold {fr.fold_index:>2d} | "
                f"train [{fr.train_start}:{fr.train_end}] "
                f"test [{fr.test_start}:{fr.test_end}] | "
                f"RMSE={fr.rmse:.6f}  R²={fr.r2:.6f}"
            )
        return "\n".join(lines)


def _validate_inputs(
    features: pd.DataFrame,
    target: pd.Series,
    n_folds: int,
    min_train_size: int,
) -> None:
    """Validate inputs before running cross-validation."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError(f"features must be a pd.DataFrame, got {type(features)}")
    if not isinstance(target, pd.Series):
        raise TypeError(f"target must be a pd.Series, got {type(target)}")
    if not features.index.equals(target.index):
        raise ValueError("features and target must share the same index")
    if not isinstance(features.index, (pd.DatetimeIndex, pd.RangeIndex, pd.Index)):
        logger.warning(
            "Index is not a DatetimeIndex; ensure data is already sorted chronologically."
        )
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if min_train_size < 1:
        raise ValueError(f"min_train_size must be >= 1, got {min_train_size}")
    n = len(features)
    if min_train_size >= n:
        raise ValueError(
            f"min_train_size ({min_train_size}) must be less than "
            f"the total number of samples ({n})"
        )
    if features.isnull().any().any():
        raise ValueError("features contains NaN values; please handle missing data before CV")
    if target.isnull().any():
        raise ValueError("target contains NaN values; please handle missing data before CV")


def _generate_fold_indices(
    n_samples: int,
    n_folds: int,
    min_train_size: int,
) -> list[tuple[int, int, int, int]]:
    """
    Generate (train_start, train_end, test_start, test_end) index tuples
    for an expanding-window walk-forward split.

    The available data after min_train_size is divided into n_folds segments.
    Each fold uses all data from 0..train_end as training and the next
    segment as the test window.

    Returns
    -------
    List of (train_start, train_end, test_start, test_end) tuples
    where indices follow Python slice semantics [start:end].
    """
    available = n_samples - min_train_size
    if available < n_folds:
        raise ValueError(
            f"Not enough samples for {n_folds} folds with min_train_size={min_train_size}. "
            f"Available samples after min_train_size: {available}. "
            f"Reduce n_folds or min_train_size."
        )

    fold_size = available // n_folds
    if fold_size < 1:
        raise ValueError(
            f"Computed fold_size={fold_size} is too small. "
            "Reduce n_folds or min_train_size."
        )

    splits = []
    for fold_idx in range(n_folds):
        test_start = min_train_size + fold_idx * fold_size
        # Last fold absorbs any remainder
        if fold_idx == n_folds - 1:
            test_end = n_samples
        else:
            test_end = min_train_size + (fold_idx + 1) * fold_size

        train_start = 0
        train_end = test_start  # expanding window: all data before test window

        splits.append((train_start, train_end, test_start, test_end))

    return splits


def walk_forward_cv(
    features: pd.DataFrame,
    target: pd.Series,
    n_folds: int = 5,
    min_train_size: int = 30,
    ridge_alpha: float = 1.0,
    scaler_class: Optional[type] = None,
) -> WalkForwardCVResult:
    """
    Perform walk-forward (expanding window) cross-validation on time-series data.

    Parameters
    ----------
    features : pd.DataFrame
        Time-indexed DataFrame of input features. Must be sorted chronologically
        before passing to this function.
    target : pd.Series
        Target variable aligned with features (same index).
    n_folds : int, default=5
        Number of walk-forward folds.
    min_train_size : int, default=30
        Minimum number of samples in the first training window.
    ridge_alpha : float, default=1.0
        Regularisation strength for Ridge regression.
    scaler_class : type or None, default=None
        Scaler class to use (must follow sklearn API). Defaults to StandardScaler.

    Returns
    -------
    WalkForwardCVResult
        Dataclass containing average scores, per-fold scores, and predictions.

    Raises
    ------
    TypeError
        If features or target are not the expected types.
    ValueError
        If inputs are invalid or there are insufficient samples.
    """
    _validate_inputs(features, target, n_folds, min_train_size)

    if scaler_class is None:
        scaler_class = StandardScaler

    # Ensure chronological order — sort by index if it is a DatetimeIndex
    if isinstance(features.index, pd.DatetimeIndex):
        if not features.index.is_monotonic_increasing:
            logger.info("Sorting features and target by DatetimeIndex (chronological order).")
            features = features.sort_index()
            target = target.loc[features.index]
    else:
        logger.info(
            "Index is not a DatetimeIndex. Assuming data is already in chronological order."
        )

    n_samples = len(features)
    X_array = features.values  # numpy array for efficient slicing
    y_array = target.values
    idx = features.index  # preserve original index for output alignment

    fold_indices = _generate_fold_indices(n_samples, n_folds, min_train_size)

    per_fold_results: list[FoldResult] = []
    all_pred_dict: dict = {}
    all_actual_dict: dict = {}

    for fold_num, (tr_start, tr_end, te_start, te_end) in enumerate(fold_indices):
        logger.debug(
            "Fold %d: train[%d:%d] (%d samples), test[%d:%d] (%d samples)",
            fold_num,
            tr_start,
            tr_end,
            tr_end - tr_start,
            te_start,
            te_end,
            te_end - te_start,
        )

        # ------------------------------------------------------------------ #
        # 1. Slice train and test — NO information from test leaks into train #
        # ------------------------------------------------------------------ #
        X_train = X_array[tr_start:tr_end]
        y_train = y_array[tr_start:tr_end]
        X_test = X_array[te_start:te_end]
        y_test = y_array[te_start:te_end]
        test_idx = idx[te_start:te_end]

        # ------------------------------------------------------------------ #
        # 2. Fit scaler ONLY on training data                                 #
        # ------------------------------------------------------------------ #
        scaler = scaler_class()
        X_train_scaled = scaler.fit_transform(X_train)

        # Transform test data using the scaler fitted on training data only
        X_test_scaled = scaler.transform(X_test)

        # ------------------------------------------------------------------ #
        # 3. Fit model ONLY on training data                                  #
        # ------------------------------------------------------------------ #
        model = Ridge(alpha=ridge_alpha)
        model.fit(X_train_scaled, y_train)

        # ------------------------------------------------------------------ #
        # 4. Predict on out-of-sample test segment                            #
        # ------------------------------------------------------------------ #
        y_pred = model.predict(X_test_scaled)

        # ------------------------------------------------------------------ #
        # 5. Compute fold metrics                                              #
        # ------------------------------------------------------------------ #
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        fold_result = FoldResult(
            fold_index=fold_num,
            train_start=tr_start,
            train_end=tr_end,
            test_start=te_start,
            test_end=te_end,
            rmse=rmse,
            r2=r2,
            predictions=y_pred,
            actuals=y_test,
            test_indices=test_idx,
        )
        per_fold_results.append(fold_result)

        # Accumulate predictions keyed by original index
        for orig_idx, pred_val, actual_val in zip(test_idx, y_pred, y_test):
            all_pred_dict[orig_idx] = pred_val
            all_actual_dict[orig_idx] = actual_val

        logger.info("Fold %d — RMSE: %.6f  R²: %.6f", fold_num, rmse, r2)

    # ---------------------------------------------------------------------- #
    # 6. Aggregate results                                                     #
    # ---------------------------------------------------------------------- #
    rmse_values = np.array([fr.rmse for fr in per_fold_results])
    r2_values = np.array([fr.r2 for fr in per_fold_results])

    all_predictions = pd.Series(all_pred_dict, name="predictions").sort_index()
    all_actuals = pd.Series(all_actual_dict, name="actuals").sort_index()

    result = WalkForwardCVResult(
        avg_rmse=float(rmse_values.mean()),
        avg_r2=float(r2_values.mean()),
        std_rmse=float(rmse_values.std(ddof=1)),
        std_r2=float(r2_values.std(ddof=1)),
        per_fold_results=per_fold_results,
        all_predictions=all_predictions,
        all_actuals=all_actuals,
    )

    logger.info(
        "Walk-forward CV complete — avg RMSE: %.6f ± %.6f  avg R²: %.6f ± %.6f",
        result.avg_rmse,
        result.std_rmse,
        result.avg_r2,
        result.std_r2,
    )

    return result


# --------------------------------------------------------------------------- #
# Convenience helper: generate a synthetic time-series dataset for testing    #
# --------------------------------------------------------------------------- #

def _make_synthetic_ts(
    n_samples: int = 200,
    n_features: int = 5,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a synthetic time-series dataset for demonstration / testing.

    Parameters
    ----------
    n_samples : int
        Number of time steps.
    n_features : int
        Number of feature columns.
    noise_std : float
        Standard deviation of Gaussian noise added to the target.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple of (features DataFrame, target Series) with a DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2018-01-01", periods=n_samples, freq="D")
    X = rng.standard_normal((n_samples, n_features))
    true_coef = rng.standard_normal(n_features)
    y = X @ true_coef + noise_std * rng.standard_normal(n_samples)

    features = pd.DataFrame(
        X,
        index=dates,
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    target = pd.Series(y, index=dates, name="target")
    return features, target


# --------------------------------------------------------------------------- #
# Example usage (runs when module is executed directly)                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    features_df, target_series = _make_synthetic_ts(n_samples=300, n_features=6)

    cv_result = walk_forward_cv(
        features=features_df,
        target=target_series,
        n_folds=5,
        min_train_size=60,
        ridge_alpha=1.0,
    )

    print(cv_result.summary())

    # Demonstrate accessing per-fold predictions
    print("\nFirst 5 out-of-sample predictions vs actuals:")
    comparison = pd.DataFrame(
        {
            "predicted": cv_result.all_predictions,
            "actual": cv_result.all_actuals,
        }
    ).head(5)
    print(comparison.to_string())