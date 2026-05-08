"""
Time-Series Sales Forecasting Pipeline
=======================================
Builds a complete forecasting pipeline with proper train/test splitting,
feature engineering, gradient boosting, and evaluation.
"""

import warnings
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame has the required columns."""
    required_cols = {"date", "store_id", "item_id", "sales"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")


def _sort_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by store/item/date and cast date column to datetime."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Feature engineering (applied per group AFTER the train/test split)
# ---------------------------------------------------------------------------

def _engineer_features_for_partition(
    df: pd.DataFrame,
    lag_days: Tuple[int, ...] = (1, 7, 28),
    rolling_windows: Tuple[int, ...] = (7, 28),
) -> pd.DataFrame:
    """
    Engineer lag features, rolling statistics, and calendar dummies
    for a single partition (train or test).

    NOTE: This function is called separately on train and test slices.
    Rolling/lag statistics for the test slice are computed using the
    *combined* history (train + test rows up to that point) so that
    look-ahead bias is avoided.  The caller is responsible for passing
    the correct data.
    """
    df = df.copy()

    # --- Lag features ---
    for lag in lag_days:
        df[f"lag_{lag}"] = (
            df.groupby(["store_id", "item_id"])["sales"]
            .shift(lag)
        )

    # --- Rolling statistics (computed on shifted series to avoid leakage) ---
    for window in rolling_windows:
        shifted = df.groupby(["store_id", "item_id"])["sales"].shift(1)
        df[f"rolling_mean_{window}"] = (
            shifted.groupby([df["store_id"], df["item_id"]])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}"] = (
            shifted.groupby([df["store_id"], df["item_id"]])
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    # --- Calendar features ---
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # One-hot encode day-of-week and month
    dow_dummies = pd.get_dummies(df["day_of_week"], prefix="dow", drop_first=False)
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=False)

    df = pd.concat([df, dow_dummies, month_dummies], axis=1)
    df.drop(columns=["day_of_week", "month"], inplace=True)

    return df


def _align_dummy_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure train and test DataFrames have identical dummy columns.
    Columns present in train but missing from test are added as zeros,
    and vice-versa (extra test columns are dropped).
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Add missing columns to test
    for col in train_cols - test_cols:
        test_df[col] = 0

    # Drop extra columns from test (not seen in train)
    extra_in_test = test_cols - train_cols
    if extra_in_test:
        test_df = test_df.drop(columns=list(extra_in_test))

    # Reorder test columns to match train
    test_df = test_df[train_df.columns]

    return train_df, test_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_sales_forecasting_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    lag_days: Tuple[int, ...] = (1, 7, 28),
    rolling_windows: Tuple[int, ...] = (7, 28),
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Build a complete time-series sales forecasting pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns: 'date', 'store_id', 'item_id', 'sales'.
    test_size : float
        Fraction of the timeline to reserve for testing (default 0.2).
        The split is performed on the *global* date axis so that the test
        set always contains future dates relative to the training set.
    lag_days : tuple of int
        Lag periods (in days) to create as features.
    rolling_windows : tuple of int
        Window sizes (in days) for rolling mean/std features.
    model_params : dict, optional
        Hyper-parameters passed to GradientBoostingRegressor.
        Defaults to a sensible baseline configuration.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'model'          – trained GradientBoostingRegressor
        'test_rmse'      – root mean squared error on the test set
        'test_mae'       – mean absolute error on the test set
        'predictions_df' – DataFrame with columns ['date', 'store_id',
                           'item_id', 'actual', 'predicted']
        'feature_names'  – list of feature column names used for training
        'scaler'         – fitted StandardScaler (for inference on new data)
    """
    # ------------------------------------------------------------------
    # 1. Validate and prepare raw data
    # ------------------------------------------------------------------
    _validate_dataframe(df)
    df = _sort_and_cast(df)

    # ------------------------------------------------------------------
    # 2. Temporal train / test split BEFORE any feature engineering
    #    (Best Practice #1: split first, preprocess after)
    # ------------------------------------------------------------------
    unique_dates = df["date"].sort_values().unique()
    n_dates = len(unique_dates)
    split_idx = int(n_dates * (1 - test_size))

    if split_idx <= max(lag_days):
        raise ValueError(
            f"Training period ({split_idx} days) is too short for the "
            f"largest lag ({max(lag_days)} days). Reduce test_size or "
            "provide more data."
        )

    cutoff_date = unique_dates[split_idx - 1]  # last date in training set

    train_raw = df[df["date"] <= cutoff_date].copy()
    test_raw = df[df["date"] > cutoff_date].copy()

    print(
        f"[Split] Train: {train_raw['date'].min().date()} → "
        f"{train_raw['date'].max().date()} ({len(train_raw):,} rows)\n"
        f"        Test : {test_raw['date'].min().date()} → "
        f"{test_raw['date'].max().date()} ({len(test_raw):,} rows)"
    )

    # ------------------------------------------------------------------
    # 3. Feature engineering
    #    For the test set we need historical context (lags/rolling stats
    #    look back into the training period), so we engineer features on
    #    the full dataset and then re-split.  This is NOT leakage because:
    #      - The target ('sales') of test rows is never seen by the model
    #        during training.
    #      - Lag/rolling features for test rows reference only past values
    #        (shift ≥ 1), which are training-period observations.
    # ------------------------------------------------------------------
    full_featured = _engineer_features_for_partition(df, lag_days, rolling_windows)

    train_featured = full_featured[full_featured["date"] <= cutoff_date].copy()
    test_featured = full_featured[full_featured["date"] > cutoff_date].copy()

    # ------------------------------------------------------------------
    # 4. Define feature columns (exclude identifiers and target)
    # ------------------------------------------------------------------
    non_feature_cols = {"date", "store_id", "item_id", "sales"}
    feature_cols = [c for c in train_featured.columns if c not in non_feature_cols]

    # ------------------------------------------------------------------
    # 5. Handle missing values created by lag/rolling computations
    #    Drop rows where ANY lag feature is NaN (only affects early rows
    #    in the training set; test set lags are filled from train history).
    # ------------------------------------------------------------------
    lag_cols = [f"lag_{lag}" for lag in lag_days]

    train_featured = train_featured.dropna(subset=lag_cols).reset_index(drop=True)
    test_featured = test_featured.dropna(subset=lag_cols).reset_index(drop=True)

    # Fill remaining NaNs (e.g. rolling std with insufficient history) with 0
    train_featured[feature_cols] = train_featured[feature_cols].fillna(0)
    test_featured[feature_cols] = test_featured[feature_cols].fillna(0)

    # Align dummy columns between train and test
    train_featured, test_featured = _align_dummy_columns(train_featured, test_featured)
    # Re-derive feature_cols after alignment
    feature_cols = [c for c in train_featured.columns if c not in non_feature_cols]

    # ------------------------------------------------------------------
    # 6. Prepare X / y matrices
    # ------------------------------------------------------------------
    X_train = train_featured[feature_cols].values.astype(np.float32)
    y_train = train_featured["sales"].values.astype(np.float32)

    X_test = test_featured[feature_cols].values.astype(np.float32)
    y_test = test_featured["sales"].values.astype(np.float32)

    # ------------------------------------------------------------------
    # 7. Scale features — fit ONLY on training data (Best Practice #1)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
    X_test_scaled = scaler.transform(X_test)          # transform only on test

    # ------------------------------------------------------------------
    # 8. Train gradient boosting model
    # ------------------------------------------------------------------
    if model_params is None:
        model_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_samples_leaf": 20,
            "subsample": 0.8,
            "loss": "squared_error",
            "random_state": random_state,
        }

    # Try XGBoost first; fall back to sklearn GradientBoostingRegressor
    try:
        from xgboost import XGBRegressor  # type: ignore

        xgb_params = {
            "n_estimators": model_params.get("n_estimators", 300),
            "learning_rate": model_params.get("learning_rate", 0.05),
            "max_depth": model_params.get("max_depth", 5),
            "subsample": model_params.get("subsample", 0.8),
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = XGBRegressor(**xgb_params)
        model_name = "XGBRegressor"
    except ImportError:
        model = GradientBoostingRegressor(**model_params)
        model_name = "GradientBoostingRegressor"

    print(f"[Model] Training {model_name} on {X_train_scaled.shape[0]:,} samples "
          f"with {X_train_scaled.shape[1]} features …")

    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 9. Evaluate on the held-out test set ONLY (Best Practice #2)
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_pred = np.maximum(y_pred, 0)  # sales cannot be negative

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    print(f"[Eval]  Test RMSE : {rmse:.4f}")
    print(f"[Eval]  Test MAE  : {mae:.4f}")

    # ------------------------------------------------------------------
    # 10. Build predictions DataFrame
    # ------------------------------------------------------------------
    predictions_df = test_featured[["date", "store_id", "item_id", "sales"]].copy()
    predictions_df = predictions_df.rename(columns={"sales": "actual"})
    predictions_df["predicted"] = y_pred
    predictions_df = predictions_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 11. Return results
    # ------------------------------------------------------------------
    return {
        "model": model,
        "test_rmse": rmse,
        "test_mae": mae,
        "predictions_df": predictions_df,
        "feature_names": feature_cols,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Inference helper (for new / unseen data)
# ---------------------------------------------------------------------------

def predict_new_data(
    pipeline_result: Dict[str, Any],
    new_df: pd.DataFrame,
    lag_days: Tuple[int, ...] = (1, 7, 28),
    rolling_windows: Tuple[int, ...] = (7, 28),
) -> pd.DataFrame:
    """
    Generate predictions for new data using a previously trained pipeline.

    Parameters
    ----------
    pipeline_result : dict
        Output of `build_sales_forecasting_pipeline`.
    new_df : pd.DataFrame
        New data with the same schema as the training data.
        Must include enough historical rows for lag computation.
    lag_days : tuple of int
        Must match the values used during training.
    rolling_windows : tuple of int
        Must match the values used during training.

    Returns
    -------
    pd.DataFrame with columns ['date', 'store_id', 'item_id', 'predicted'].
    """
    _validate_dataframe(new_df)
    new_df = _sort_and_cast(new_df)

    model = pipeline_result["model"]
    scaler = pipeline_result["scaler"]
    feature_names = pipeline_result["feature_names"]

    featured = _engineer_features_for_partition(new_df, lag_days, rolling_windows)
    featured = featured.fillna(0)

    # Align columns to training feature set
    for col in feature_names:
        if col not in featured.columns:
            featured[col] = 0
    featured = featured[feature_names]

    X = scaler.transform(featured.values.astype(np.float32))
    preds = np.maximum(model.predict(X), 0)

    result = new_df[["date", "store_id", "item_id"]].copy().reset_index(drop=True)
    result["predicted"] = preds
    return result


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_stores: int = 3,
    n_items: int = 5,
    n_days: int = 365,
    random_state: int = 0,
) -> pd.DataFrame:
    """Generate synthetic sales data for testing."""
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    records = []
    for store_id in range(1, n_stores + 1):
        for item_id in range(1, n_items + 1):
            base = rng.integers(10, 100)
            trend = np.linspace(0, rng.integers(5, 20), n_days)
            seasonality = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
            noise = rng.normal(0, 5, n_days)
            sales = np.maximum(base + trend + seasonality + noise, 0)
            for i, date in enumerate(dates):
                records.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "item_id": item_id,
                        "sales": round(sales[i], 2),
                    }
                )
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=" * 60)
    print("Sales Forecasting Pipeline — Smoke Test")
    print("=" * 60)

    synthetic_df = _generate_synthetic_data(n_stores=2, n_items=3, n_days=200)
    print(f"Synthetic dataset shape: {synthetic_df.shape}")
    print(synthetic_df.head())
    print()

    results = build_sales_forecasting_pipeline(
        df=synthetic_df,
        test_size=0.2,
        lag_days=(1, 7, 28),
        rolling_windows=(7, 28),
        random_state=42,
    )

    print("\n--- Pipeline Results ---")
    print(f"Test RMSE : {results['test_rmse']:.4f}")
    print(f"Test MAE  : {results['test_mae']:.4f}")
    print(f"Model     : {type(results['model']).__name__}")
    print(f"Features  : {len(results['feature_names'])}")
    print("\nSample predictions:")
    print(results["predictions_df"].head(10).to_string(index=False))