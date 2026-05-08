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
    learning_rate: float = 0.1,
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
        Maximum depth of individual trees
    random_state : int
        Random seed for reproducibility
    use_xgboost : bool
        If True, attempt to use XGBRegressor; falls back to sklearn if unavailable

    Returns
    -------
    dict with keys:
        'model'         : trained model object
        'rmse'          : test Root Mean Squared Error
        'mae'           : test Mean Absolute Error
        'predictions_df': DataFrame with columns ['date', 'store_id', 'item_id',
                          'actual_sales', 'predicted_sales']
        'feature_names' : list of feature column names used for training
    """

    # ------------------------------------------------------------------
    # 1. Validate and prepare the input DataFrame
    # ------------------------------------------------------------------
    required_cols = {"date", "store_id", "item_id", "sales"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

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
    # 3. Engineer time-based features
    # ------------------------------------------------------------------
    data["day_of_week"] = data["date"].dt.dayofweek          # 0=Monday
    data["day_of_month"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["year"] = data["date"].dt.year
    data["quarter"] = data["date"].dt.quarter
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    # Day-of-week dummies (drop first to avoid multicollinearity)
    dow_dummies = pd.get_dummies(
        data["day_of_week"], prefix="dow", drop_first=True
    ).astype(int)

    # Month dummies
    month_dummies = pd.get_dummies(
        data["month"], prefix="month", drop_first=True
    ).astype(int)

    data = pd.concat([data, dow_dummies, month_dummies], axis=1)

    # ------------------------------------------------------------------
    # 4. Engineer lag features and rolling statistics per (store, item)
    # ------------------------------------------------------------------
    group_keys = ["store_id", "item_id"]
    lag_days = [1, 7, 28]
    rolling_windows = [7, 28]

    for lag in lag_days:
        col_name = f"lag_{lag}"
        data[col_name] = data.groupby(group_keys)["sales"].shift(lag)

    for window in rolling_windows:
        # Rolling mean and std computed on the shifted series (no data leakage)
        shifted = data.groupby(group_keys)["sales"].shift(1)
        data[f"rolling_mean_{window}"] = (
            shifted.groupby([data["store_id"], data["item_id"]])
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        data[f"rolling_std_{window}"] = (
            shifted.groupby([data["store_id"], data["item_id"]])
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    # Rolling mean/std on lag-7 to capture weekly seasonality
    data["rolling_mean_7_lag7"] = (
        data.groupby(group_keys)["sales"]
        .shift(7)
        .groupby([data["store_id"], data["item_id"]])
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    # ------------------------------------------------------------------
    # 5. Handle missing values introduced by lags / rolling windows
    # ------------------------------------------------------------------
    # Drop rows where the largest lag (28) would produce NaN for the target
    # We keep rows where at least lag_1 is available; remaining NaNs in
    # rolling/lag columns are filled with the column median.
    data = data.dropna(subset=["lag_1"]).reset_index(drop=True)

    # Fill remaining NaNs (e.g., rolling_std at the start) with column medians
    feature_cols_for_fill = [c for c in data.columns if c.startswith(
        ("lag_", "rolling_", "dow_", "month_")
    )]
    for col in feature_cols_for_fill:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    # ------------------------------------------------------------------
    # 6. Define feature matrix
    # ------------------------------------------------------------------
    feature_columns = (
        ["store_id_enc", "item_id_enc",
         "day_of_week", "day_of_month", "month", "week_of_year",
         "year", "quarter", "is_weekend"]
        + [f"lag_{lag}" for lag in lag_days]
        + [f"rolling_mean_{w}" for w in rolling_windows]
        + [f"rolling_std_{w}" for w in rolling_windows]
        + ["rolling_mean_7_lag7"]
        + [c for c in data.columns if c.startswith("dow_")]
        + [c for c in data.columns if c.startswith("month_")]
    )

    # Keep only columns that actually exist in data
    feature_columns = [c for c in feature_columns if c in data.columns]
    # Remove duplicates while preserving order
    seen = set()
    feature_columns = [
        c for c in feature_columns if not (c in seen or seen.add(c))
    ]

    target_column = "sales"

    # ------------------------------------------------------------------
    # 7. Chronological train / test split
    # ------------------------------------------------------------------
    n_total = len(data)
    n_test = max(1, int(n_total * test_size))
    n_train = n_total - n_test

    train_data = data.iloc[:n_train].copy()
    test_data = data.iloc[n_train:].copy()

    X_train = train_data[feature_columns].values
    y_train = train_data[target_column].values

    X_test = test_data[feature_columns].values
    y_test = test_data[target_column].values

    # ------------------------------------------------------------------
    # 8. Select and fit the model
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
            print("Using XGBRegressor.")
        except ImportError:
            print("XGBoost not available; falling back to sklearn GradientBoostingRegressor.")

    if model is None:
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            subsample=0.8,
            min_samples_leaf=10,
        )
        print("Using sklearn GradientBoostingRegressor.")

    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 9. Predict and evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    # Clip predictions to non-negative values (sales cannot be negative)
    y_pred = np.clip(y_pred, 0, None)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    print(f"\n{'='*50}")
    print(f"Test RMSE : {rmse:.4f}")
    print(f"Test MAE  : {mae:.4f}")
    print(f"{'='*50}\n")

    # ------------------------------------------------------------------
    # 10. Build predictions DataFrame
    # ------------------------------------------------------------------
    predictions_df = pd.DataFrame(
        {
            "date": test_data["date"].values,
            "store_id": test_data["store_id"].values,
            "item_id": test_data["item_id"].values,
            "actual_sales": y_test,
            "predicted_sales": y_pred,
            "residual": y_test - y_pred,
        }
    )

    # ------------------------------------------------------------------
    # 11. Feature importance summary (if available)
    # ------------------------------------------------------------------
    importance_df = None
    if hasattr(model, "feature_importances_"):
        importance_df = (
            pd.DataFrame(
                {"feature": feature_columns,
                 "importance": model.feature_importances_}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return {
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "predictions_df": predictions_df,
        "feature_names": feature_columns,
        "feature_importance": importance_df,
        "train_size": n_train,
        "test_size": n_test,
    }


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # Generate synthetic sales data
    dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
    stores = ["S1", "S2"]
    items = ["I1", "I2", "I3"]

    records = []
    for store in stores:
        for item in items:
            base = np.random.randint(50, 200)
            trend = np.linspace(0, 20, len(dates))
            seasonality = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
            noise = np.random.normal(0, 5, len(dates))
            sales = np.clip(base + trend + seasonality + noise, 0, None)
            for i, d in enumerate(dates):
                records.append(
                    {"date": d, "store_id": store, "item_id": item,
                     "sales": round(sales[i], 2)}
                )

    df_demo = pd.DataFrame(records)
    print(f"Demo DataFrame shape: {df_demo.shape}")
    print(df_demo.head())

    results = build_sales_forecasting_pipeline(
        df_demo,
        test_size=0.2,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        use_xgboost=False,
    )

    print("\nPredictions sample:")
    print(results["predictions_df"].head(10).to_string(index=False))

    if results["feature_importance"] is not None:
        print("\nTop-10 Feature Importances:")
        print(results["feature_importance"].head(10).to_string(index=False))