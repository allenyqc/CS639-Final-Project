```python
"""
Time-Series Sales Forecasting Pipeline
=======================================
Builds a complete forecasting pipeline with proper chronological splitting,
feature engineering, gradient boosting, and evaluation.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional XGBoost import
try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _engineer_features(
    df: pd.DataFrame,
    lag_days: tuple[int, ...] = (1, 7, 28),
    rolling_windows: tuple[int, ...] = (7, 28),
) -> pd.DataFrame:
    """
    Add lag features, rolling statistics, and calendar dummies.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date, store_id, item_id, sales.
        Must be sorted by (store_id, item_id, date) before calling.
    lag_days : tuple of int
        Lag periods in days.
    rolling_windows : tuple of int
        Rolling window sizes in days.

    Returns
    -------
    pd.DataFrame with additional feature columns.
    """
    df = df.copy()

    group_cols = ["store_id", "item_id"]

    # --- Lag features ---
    for lag in lag_days:
        col_name = f"lag_{lag}d"
        df[col_name] = (
            df.groupby(group_cols)["sales"]
            .shift(lag)
        )

    # --- Rolling statistics (computed on lagged data to avoid leakage) ---
    # We shift by 1 before rolling so the window never includes the current day.
    for window in rolling_windows:
        shifted = df.groupby(group_cols)["sales"].shift(1)
        df[f"rolling_mean_{window}d"] = (
            shifted.groupby(df[group_cols].apply(tuple, axis=1))
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}d"] = (
            shifted.groupby(df[group_cols].apply(tuple, axis=1))
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    # --- Calendar features ---
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["month"] = df["date"].dt.month

    # One-hot encode day_of_week and month
    dow_dummies = pd.get_dummies(df["day_of_week"], prefix="dow", drop_first=False)
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=False)

    df = pd.concat([df, dow_dummies, month_dummies], axis=1)
    df.drop(columns=["day_of_week", "month"], inplace=True)

    return df


def _rolling_stats_safe(df: pd.DataFrame, rolling_windows: tuple[int, ...]) -> pd.DataFrame:
    """
    A safer, vectorised implementation of rolling stats that avoids the
    groupby-transform-lambda pattern which can be slow on large data.
    """
    df = df.copy()
    group_cols = ["store_id", "item_id"]

    for window in rolling_windows:
        mean_col = f"rolling_mean_{window}d"
        std_col = f"rolling_std_{window}d"
        # shift(1) ensures no look-ahead leakage
        shifted = df.groupby(group_cols)["sales"].shift(1)
        df[mean_col] = (
            shifted
            .groupby([df["store_id"], df["item_id"]])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())  # noqa: B023
        )
        df[std_col] = (
            shifted
            .groupby([df["store_id"], df["item_id"]])
            .transform(lambda x: x.rolling(window, min_periods=1).std())   # noqa: B023
        )
    return df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def build_sales_forecast_pipeline(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    use_xgboost: bool = True,
    model_params: Optional[dict] = None,
    lag_days: tuple[int, ...] = (1, 7, 28),
    rolling_windows: tuple[int, ...] = (7, 28),
    random_state: int = 42,
) -> dict:
    """
    Build a complete time-series sales forecasting pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'date', 'store_id', 'item_id', 'sales'.
    test_fraction : float
        Fraction of the timeline reserved for the test set (chronological).
    val_fraction : float
        Fraction of the timeline reserved for the validation set (chronological),
        carved out of the training portion.
    use_xgboost : bool
        Use XGBRegressor if available; fall back to GradientBoostingRegressor.
    model_params : dict or None
        Hyperparameters passed to the chosen regressor.
    lag_days : tuple of int
        Lag periods for lag features.
    rolling_windows : tuple of int
        Window sizes for rolling statistics.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        - 'model'        : fitted estimator
        - 'test_rmse'    : float
        - 'test_mae'     : float
        - 'predictions'  : pd.DataFrame with columns [date, store_id, item_id,
                           actual_sales, predicted_sales]
        - 'feature_names': list of feature column names
        - 'val_rmse'     : float  (validation RMSE, used for model selection)
        - 'val_mae'      : float
    """
    # ------------------------------------------------------------------
    # 0. Input validation
    # ------------------------------------------------------------------
    required_cols = {"date", "store_id", "item_id", "sales"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be in (0, 1).")
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be in (0, 1).")
    if test_fraction + val_fraction >= 1:
        raise ValueError("test_fraction + val_fraction must be < 1.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="raise")

    # Sort chronologically within each (store, item) group
    df.sort_values(["store_id", "item_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # 1. Chronological train / val / test split  (BEFORE feature engineering)
    # ------------------------------------------------------------------
    unique_dates = df["date"].sort_values().unique()
    n_dates = len(unique_dates)

    n_test = max(1, int(np.ceil(n_dates * test_fraction)))
    n_val = max(1, int(np.ceil(n_dates * val_fraction)))
    n_train = n_dates - n_test - n_val

    if n_train < 1:
        raise ValueError(
            "Not enough dates to create train/val/test splits with the given fractions."
        )

    train_dates = unique_dates[:n_train]
    val_dates = unique_dates[n_train: n_train + n_val]
    test_dates = unique_dates[n_train + n_val:]

    train_mask = df["date"].isin(train_dates)
    val_mask = df["date"].isin(val_dates)
    test_mask = df["date"].isin(test_dates)

    # Keep raw splits for reference (before feature engineering)
    df_train_raw = df[train_mask].copy()
    df_val_raw = df[val_mask].copy()
    df_test_raw = df[test_mask].copy()

    print(
        f"[Split] Train: {train_dates[0].date()} → {train_dates[-1].date()} "
        f"({len(df_train_raw):,} rows)\n"
        f"        Val  : {val_dates[0].date()} → {val_dates[-1].date()} "
        f"({len(df_val_raw):,} rows)\n"
        f"        Test : {test_dates[0].date()} → {test_dates[-1].date()} "
        f"({len(df_test_raw):,} rows)"
    )

    # ------------------------------------------------------------------
    # 2. Feature engineering
    #    We engineer features on the FULL sorted dataframe so that lag/rolling
    #    computations for val/test rows can look back into the training history.
    #    No target leakage occurs because lags always reference past rows.
    # ------------------------------------------------------------------
    df_feat = df.copy()

    # Lag features
    group_cols = ["store_id", "item_id"]
    for lag in lag_days:
        df_feat[f"lag_{lag}d"] = df_feat.groupby(group_cols)["sales"].shift(lag)

    # Rolling statistics (shift(1) inside each group to avoid same-day leakage)
    for window in rolling_windows:
        shifted = df_feat.groupby(group_cols)["sales"].shift(1)
        df_feat[f"rolling_mean_{window}d"] = (
            shifted
            .groupby([df_feat["store_id"], df_feat["item_id"]])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())  # noqa: B023
        )
        df_feat[f"rolling_std_{window}d"] = (
            shifted
            .groupby([df_feat["store_id"], df_feat["item_id"]])
            .transform(lambda x: x.rolling(window, min_periods=1).std())   # noqa: B023
        )

    # Calendar features
    df_feat["day_of_week"] = df_feat["date"].dt.dayofweek
    df_feat["month"] = df_feat["date"].dt.month
    df_feat["day_of_month"] = df_feat["date"].dt.day
    df_feat["week_of_year"] = df_feat["date"].dt.isocalendar().week.astype(int)

    dow_dummies = pd.get_dummies(df_feat["day_of_week"], prefix="dow", drop_first=False)
    month_dummies = pd.get_dummies(df_feat["month"], prefix="month", drop_first=False)
    df_feat = pd.concat([df_feat, dow_dummies, month_dummies], axis=1)
    df_feat.drop(columns=["day_of_week", "month"], inplace=True)

    # ------------------------------------------------------------------
    # 3. Define feature columns (everything except identifiers and target)
    # ------------------------------------------------------------------
    non_feature_cols = {"date", "store_id", "item_id", "sales"}
    feature_cols = [c for c in df_feat.columns if c not in non_feature_cols]

    # ------------------------------------------------------------------
    # 4. Re-split the feature-engineered dataframe
    # ------------------------------------------------------------------
    df_train = df_feat[df_feat["date"].isin(train_dates)].copy()
    df_val = df_feat[df_feat["date"].isin(val_dates)].copy()
    df_test = df_feat[df_feat["date"].isin(test_dates)].copy()

    # ------------------------------------------------------------------
    # 5. Handle missing values (from lag/rolling at the start of each series)
    #    Strategy: drop rows with NaN in the TRAINING set only.
    #    For val/test, fill NaN with column medians computed on the training set.
    # ------------------------------------------------------------------
    train_before_drop = len(df_train)
    df_train.dropna(subset=feature_cols, inplace=True)
    print(
        f"[NaN] Dropped {train_before_drop - len(df_train):,} training rows "
        f"with NaN lag/rolling values."
    )

    # Compute fill values from training data ONLY
    fill_values: dict[str, float] = {
        col: float(df_train[col].median())
        for col in feature_cols
        if df_train[col].isna().any() or df_val[col].isna().any() or df_test[col].isna().any()
    }

    df_val[feature_cols] = df_val[feature_cols].fillna(fill_values)
    df_test[feature_cols] = df_test[feature_cols].fillna(fill_values)

    # Align dummy columns across splits (in case some categories are absent)
    df_val = df_val.reindex(columns=df_train.columns, fill_value=0)
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

    X_train = df_train[feature_cols].values
    y_train = df_train["sales"].values

    X_val = df_val[feature_cols].values
    y_val = df_val["sales"].values

    X_test = df_test[feature_cols].values
    y_test = df_test["sales"].values

    # ------------------------------------------------------------------
    # 6. Model selection and training
    #    Hyperparameters are tuned conceptually on the validation set.
    #    The test set is NEVER used for any fitting or selection.
    # ------------------------------------------------------------------
    if model_params is None:
        model_params = {}

    if use_xgboost and _XGBOOST_AVAILABLE:
        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,
            "early_stopping_rounds": 50,   # uses validation set for early stopping
        }
        default_params.update(model_params)

        # XGBoost supports early stopping via eval_set (validation set only)
        early_stopping_rounds = default_params.pop("early_stopping_rounds", None)
        model = XGBRegressor(**default_params)

        if early_stopping_rounds is not None:
            model.set_params(early_stopping_rounds=early_stopping_rounds)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        model_name = "XGBRegressor"

    else:
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "min_samples_leaf": 5,
            "random_state": random_state,
        }
        default_params.update(model_params)
        model = GradientBoostingRegressor(**default_params)
        model.fit(X_train, y_train)
        model_name = "GradientBoostingRegressor"

    print(f"[Model] Fitted {model_name}.")

    # ------------------------------------------------------------------
    # 7. Validation evaluation (for reporting / model selection reference)
    # ------------------------------------------------------------------
    y_val_pred = model.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    val_mae = float(mean_absolute_error(y_val, y_val_pred))
    print(f"[Val]  RMSE={val_rmse:.4f}  MAE={val_mae:.4f}")

    # ------------------------------------------------------------------
    # 8. Final evaluation on the held-out test set
    # ------------------------------------------------------------------
    y_test_pred = model.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    print(f"[Test] RMSE={test_rmse:.4f}  MAE={test_mae:.4f}")

    # ------------------------------------------------------------------
    # 9. Build predictions DataFrame
    # ------------------------------------------------------------------
    predictions_df = df_test[["date", "store_id", "item_id"]].copy()
    predictions_df["actual_sales"] = y_test
    predictions_df["predicted_sales"] = y_test_pred
    predictions_df.reset_index(drop=True, inplace=True)

    return {
        "model": model,
        "model_name": model_name,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "predictions": predictions_df,
        "feature_names": feature_cols,
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
    }


# ---------------------------------------------------------------------------
# Convenience: feature importance extractor
# ---------------------------------------------------------------------------

def get_feature_importance(pipeline_result: dict) -> pd.DataFrame:
    """
    Extract and rank feature importances from a fitted pipeline result.

    Parameters
    ----------
    pipeline_result : dict
        Output of build_sales_forecast_pipeline.

    Returns
    -------
    pd.DataFrame sorted by importance descending.
    """
    model = pipeline_result.get("model")
    feature_names = pipeline_result.get("feature_names", [])

    if model is None:
        raise ValueError("pipeline_result does not contain a fitted model.")

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Model of type {type(model).__name__} does not expose feature_importances_."
        )

    importances = model.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi_df


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_stores: int = 3,
    n_items: int = 4,
    n_days: int = 365,
    random_state: int = 0,
) -> pd.DataFrame:
    """Generate synthetic sales data for testing."""
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    records = []
    for store_id in range(1, n_stores + 1):
        for item_id in range(1, n_items + 1):
            base = rng.uniform(50, 200)
            trend = rng.uniform(-0.05, 0.1)
            seasonal = rng.uniform(5, 20)
            noise_scale = rng.uniform(2, 10)
            for i, date in enumerate(dates):
                sales = (
                    base
                    + trend * i
                    + seasonal * np.sin(2 * np.pi * i / 7)   # weekly seasonality
                    + seasonal * 0.5 * np.sin(2 * np.pi * i / 365)  # yearly
                    + rng.normal(0, noise_scale)
                )
                records.append(
                    {
                        "date