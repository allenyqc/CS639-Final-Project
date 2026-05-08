import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


def build_sales_forecasting_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    random_state: int = 42,
    use_xgboost: bool = False,
) -> dict:
    """
    Build a complete time-series sales forecasting pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: 'date', 'store_id', 'item_id', 'sales'
    test_size : float
        Fraction of data to use for testing (chronological split)
    n_estimators : int
        Number of boosting rounds
    learning_rate : float
        Learning rate for the gradient boosting model
    max_depth : int
        Maximum tree depth
    random_state : int
        Random seed for reproducibility
    use_xgboost : bool
        If True, attempt to use XGBRegressor; falls back to sklearn if unavailable

    Returns
    -------
    dict with keys:
        'model'        : trained model object
        'rmse'         : test Root Mean Squared Error
        'mae'          : test Mean Absolute Error
        'predictions'  : DataFrame with columns ['date', 'store_id', 'item_id',
                         'actual', 'predicted']
        'feature_names': list of feature column names used for training
    """
    # ------------------------------------------------------------------
    # 1. Validate and prepare the input DataFrame
    # ------------------------------------------------------------------
    required_cols = {"date", "store_id", "item_id", "sales"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Encode categorical identifiers
    # ------------------------------------------------------------------
    le_store = LabelEncoder()
    le_item = LabelEncoder()
    data["store_id_enc"] = le_store.fit_transform(data["store_id"].astype(str))
    data["item_id_enc"] = le_item.fit_transform(data["item_id"].astype(str))

    # ------------------------------------------------------------------
    # 3. Engineer lag features (per store-item group)
    # ------------------------------------------------------------------
    group_keys = ["store_id", "item_id"]

    lag_days = [1, 7, 28]
    for lag in lag_days:
        data[f"lag_{lag}"] = data.groupby(group_keys)["sales"].shift(lag)

    # ------------------------------------------------------------------
    # 4. Rolling-window statistics (7-day and 28-day rolling mean & std)
    #    Use shift(1) to avoid data leakage (window ends the day before)
    # ------------------------------------------------------------------
    rolling_windows = [7, 28]
    for window in rolling_windows:
        shifted = data.groupby(group_keys)["sales"].shift(1)
        data[f"rolling_mean_{window}"] = (
            shifted.groupby([data["store_id"], data["item_id"]])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        data[f"rolling_std_{window}"] = (
            shifted.groupby([data["store_id"], data["item_id"]])
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    # ------------------------------------------------------------------
    # 5. Calendar / date features
    # ------------------------------------------------------------------
    data["day_of_week"] = data["date"].dt.dayofweek          # 0=Mon … 6=Sun
    data["month"] = data["date"].dt.month
    data["day_of_month"] = data["date"].dt.day
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["year"] = data["date"].dt.year
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    # One-hot encode day-of-week and month
    dow_dummies = pd.get_dummies(data["day_of_week"], prefix="dow", drop_first=False)
    month_dummies = pd.get_dummies(data["month"], prefix="month", drop_first=False)
    data = pd.concat([data, dow_dummies, month_dummies], axis=1)

    # ------------------------------------------------------------------
    # 6. Drop rows with NaN values introduced by lag / rolling features
    # ------------------------------------------------------------------
    data = data.dropna().reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7. Define feature columns
    # ------------------------------------------------------------------
    lag_cols = [f"lag_{lag}" for lag in lag_days]
    rolling_cols = [
        f"rolling_mean_{w}" for w in rolling_windows
    ] + [f"rolling_std_{w}" for w in rolling_windows]
    calendar_cols = [
        "day_of_week", "month", "day_of_month",
        "week_of_year", "year", "is_weekend",
    ]
    dow_dummy_cols = [c for c in data.columns if c.startswith("dow_")]
    month_dummy_cols = [c for c in data.columns if c.startswith("month_")]
    id_cols = ["store_id_enc", "item_id_enc"]

    feature_cols = (
        id_cols
        + lag_cols
        + rolling_cols
        + calendar_cols
        + dow_dummy_cols
        + month_dummy_cols
    )

    # ------------------------------------------------------------------
    # 8. Chronological train / test split
    # ------------------------------------------------------------------
    n_total = len(data)
    n_test = max(1, int(n_total * test_size))
    n_train = n_total - n_test

    train_data = data.iloc[:n_train]
    test_data = data.iloc[n_train:]

    X_train = train_data[feature_cols].values
    y_train = train_data["sales"].values
    X_test = test_data[feature_cols].values
    y_test = test_data["sales"].values

    # ------------------------------------------------------------------
    # 9. Select and fit the model
    # ------------------------------------------------------------------
    model = None
    if use_xgboost:
        try:
            from xgboost import XGBRegressor  # type: ignore
            model = XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        except ImportError:
            print("XGBoost not installed; falling back to sklearn GradientBoostingRegressor.")

    if model is None:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            subsample=0.8,
            min_samples_leaf=20,
        )

    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 10. Predict and evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    # Clip predictions to non-negative values (sales cannot be negative)
    y_pred = np.clip(y_pred, 0, None)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    # ------------------------------------------------------------------
    # 11. Build predictions DataFrame
    # ------------------------------------------------------------------
    predictions_df = pd.DataFrame(
        {
            "date": test_data["date"].values,
            "store_id": test_data["store_id"].values,
            "item_id": test_data["item_id"].values,
            "actual": y_test,
            "predicted": y_pred,
            "residual": y_test - y_pred,
        }
    )

    # ------------------------------------------------------------------
    # 12. Return results
    # ------------------------------------------------------------------
    return {
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "predictions": predictions_df,
        "feature_names": feature_cols,
        "train_size": n_train,
        "test_size": n_test,
    }


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------
def _generate_synthetic_data(
    n_stores: int = 2,
    n_items: int = 3,
    n_days: int = 365,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate synthetic daily sales data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    records = []
    for store in range(1, n_stores + 1):
        for item in range(1, n_items + 1):
            base = rng.integers(50, 200)
            trend = np.linspace(0, 20, n_days)
            seasonality = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
            noise = rng.normal(0, 5, n_days)
            sales = np.clip(base + trend + seasonality + noise, 0, None)
            for i, date in enumerate(dates):
                records.append(
                    {
                        "date": date,
                        "store_id": f"store_{store}",
                        "item_id": f"item_{item}",
                        "sales": round(sales[i], 2),
                    }
                )
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating synthetic sales data …")
    synthetic_df = _generate_synthetic_data(n_stores=2, n_items=3, n_days=365)
    print(f"Dataset shape: {synthetic_df.shape}")
    print(synthetic_df.head())

    print("\nRunning forecasting pipeline …")
    results = build_sales_forecasting_pipeline(
        synthetic_df,
        test_size=0.2,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        use_xgboost=False,
    )

    print(f"\n{'='*50}")
    print(f"Model type  : {type(results['model']).__name__}")
    print(f"Train rows  : {results['train_size']}")
    print(f"Test rows   : {results['test_size']}")
    print(f"Test RMSE   : {results['rmse']:.4f}")
    print(f"Test MAE    : {results['mae']:.4f}")
    print(f"\nSample predictions:")
    print(results["predictions"].head(10).to_string(index=False))
    print(f"\nFeatures used ({len(results['feature_names'])}):")
    print(results["feature_names"])