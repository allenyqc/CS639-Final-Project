"""
Customer Lifetime Value (CLV) Prediction Module

Predicts CLV from transaction history using RFM and extended features.
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CLVResult:
    model: object
    metrics: dict
    feature_importances: pd.Series
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    feature_matrix: pd.DataFrame
    target: pd.Series
    train_idx: np.ndarray
    test_idx: np.ndarray


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_purchase_trend(amounts: pd.Series, dates: pd.Series) -> float:
    """Return the slope of a linear regression of amount over time (days)."""
    if len(amounts) < 2:
        return 0.0
    days = (dates - dates.min()).dt.days.values.astype(float)
    slope, *_ = stats.linregress(days, amounts.values)
    return float(slope) if np.isfinite(slope) else 0.0


def compute_rfm_features(
    df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
    prediction_window_days: int = 90,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute RFM + extended features and the CLV target for each customer.

    Parameters
    ----------
    df : DataFrame with columns ['customer_id', 'transaction_date', 'amount']
    snapshot_date : reference date for recency; defaults to max date in df
    prediction_window_days : future window for CLV target

    Returns
    -------
    features : DataFrame indexed by customer_id
    target   : Series (total spend in next `prediction_window_days` days)
    """
    required = {"customer_id", "transaction_date", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df = df.dropna(subset=["customer_id", "transaction_date", "amount"])
    df = df[df["amount"] >= 0]  # remove refunds / negative amounts

    if snapshot_date is None:
        snapshot_date = df["transaction_date"].max()
    else:
        snapshot_date = pd.Timestamp(snapshot_date)

    future_cutoff = snapshot_date + pd.Timedelta(days=prediction_window_days)

    # Split into history (≤ snapshot) and future (> snapshot, ≤ future_cutoff)
    history = df[df["transaction_date"] <= snapshot_date].copy()
    future = df[
        (df["transaction_date"] > snapshot_date)
        & (df["transaction_date"] <= future_cutoff)
    ].copy()

    if history.empty:
        raise ValueError("No historical transactions found before snapshot_date.")

    # -----------------------------------------------------------------------
    # Build feature matrix from history
    # -----------------------------------------------------------------------
    records = []
    for cid, grp in history.groupby("customer_id"):
        grp = grp.sort_values("transaction_date")
        dates = grp["transaction_date"]
        amounts = grp["amount"]

        recency = (snapshot_date - dates.max()).days
        frequency = len(grp)
        monetary_total = amounts.sum()
        monetary_avg = amounts.mean()
        monetary_std = amounts.std(ddof=0) if frequency > 1 else 0.0
        monetary_max = amounts.max()
        monetary_min = amounts.min()

        # Inter-purchase intervals
        if frequency > 1:
            intervals = dates.diff().dt.days.dropna()
            ipi_mean = intervals.mean()
            ipi_std = intervals.std(ddof=0) if len(intervals) > 1 else 0.0
        else:
            ipi_mean = np.nan
            ipi_std = 0.0

        # Purchase trend (slope of amount over time)
        trend = _compute_purchase_trend(amounts, dates)

        # Customer age (days from first to last purchase)
        customer_age = (dates.max() - dates.min()).days

        # Average spend per day (avoid division by zero)
        spend_per_day = monetary_total / max(customer_age, 1)

        records.append(
            {
                "customer_id": cid,
                "recency": recency,
                "frequency": frequency,
                "monetary_total": monetary_total,
                "monetary_avg": monetary_avg,
                "monetary_std": monetary_std,
                "monetary_max": monetary_max,
                "monetary_min": monetary_min,
                "ipi_mean": ipi_mean,
                "ipi_std": ipi_std,
                "purchase_trend": trend,
                "customer_age_days": customer_age,
                "spend_per_day": spend_per_day,
            }
        )

    features = pd.DataFrame(records).set_index("customer_id")

    # Fill NaN inter-purchase interval mean with recency (single-purchase customers)
    features["ipi_mean"] = features["ipi_mean"].fillna(features["recency"])

    # -----------------------------------------------------------------------
    # Build target: total spend in next `prediction_window_days` days
    # -----------------------------------------------------------------------
    future_spend = (
        future.groupby("customer_id")["amount"].sum().rename("clv_target")
    )
    # Customers with no future purchases get 0
    target = features.index.to_series().map(future_spend).fillna(0.0)
    target.name = "clv_target"

    return features, target


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

def train_clv_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_type: str = "gbr",
    test_size: float = 0.2,
    random_state: int = 42,
    gbr_params: Optional[dict] = None,
    ridge_alpha: float = 1.0,
) -> CLVResult:
    """
    Train a regression model to predict CLV.

    Parameters
    ----------
    features    : feature DataFrame (output of compute_rfm_features)
    target      : CLV target Series
    model_type  : 'gbr' (GradientBoostingRegressor) or 'ridge'
    test_size   : fraction of customers for test set
    random_state: reproducibility seed
    gbr_params  : optional hyperparameters for GradientBoostingRegressor
    ridge_alpha : regularisation strength for Ridge

    Returns
    -------
    CLVResult dataclass
    """
    X = features.copy()
    y = target.reindex(X.index).fillna(0.0)

    X_arr = X.values
    y_arr = y.values
    customer_ids = X.index.values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_arr, y_arr, np.arange(len(y_arr)),
        test_size=test_size,
        random_state=random_state,
    )

    # -----------------------------------------------------------------------
    # Build pipeline
    # -----------------------------------------------------------------------
    if model_type == "gbr":
        default_gbr = dict(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=random_state,
        )
        if gbr_params:
            default_gbr.update(gbr_params)
        regressor = GradientBoostingRegressor(**default_gbr)
        # GBR handles scale internally; StandardScaler still helps Ridge
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", regressor)])
    elif model_type == "ridge":
        regressor = Ridge(alpha=ridge_alpha)
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", regressor)])
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'gbr' or 'ridge'.")

    pipeline.fit(X_train, y_train)

    # -----------------------------------------------------------------------
    # Predictions & metrics
    # -----------------------------------------------------------------------
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    # Clip negative predictions (CLV cannot be negative)
    y_pred_train = np.clip(y_pred_train, 0, None)
    y_pred_test = np.clip(y_pred_test, 0, None)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    metrics = {
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    # -----------------------------------------------------------------------
    # Feature importances
    # -----------------------------------------------------------------------
    feature_names = X.columns.tolist()
    model_step = pipeline.named_steps["model"]

    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
    elif hasattr(model_step, "coef_"):
        importances = np.abs(model_step.coef_)
    else:
        importances = np.ones(len(feature_names))

    feature_importances = pd.Series(
        importances, index=feature_names, name="importance"
    ).sort_values(ascending=False)

    return CLVResult(
        model=pipeline,
        metrics=metrics,
        feature_importances=feature_importances,
        train_predictions=y_pred_train,
        test_predictions=y_pred_test,
        feature_matrix=X,
        target=y,
        train_idx=idx_train,
        test_idx=idx_test,
    )


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------

def predict_clv(
    df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
    prediction_window_days: int = 90,
    model_type: str = "gbr",
    test_size: float = 0.2,
    random_state: int = 42,
    gbr_params: Optional[dict] = None,
    ridge_alpha: float = 1.0,
    verbose: bool = True,
) -> CLVResult:
    """
    End-to-end CLV prediction pipeline.

    Parameters
    ----------
    df                      : raw transaction DataFrame
    snapshot_date           : reference date (defaults to max date in df)
    prediction_window_days  : future window for CLV target (default 90)
    model_type              : 'gbr' or 'ridge'
    test_size               : test split fraction
    random_state            : seed
    gbr_params              : optional GBR hyperparameters
    ridge_alpha             : Ridge regularisation
    verbose                 : print summary

    Returns
    -------
    CLVResult
    """
    features, target = compute_rfm_features(
        df,
        snapshot_date=snapshot_date,
        prediction_window_days=prediction_window_days,
    )

    result = train_clv_model(
        features,
        target,
        model_type=model_type,
        test_size=test_size,
        random_state=random_state,
        gbr_params=gbr_params,
        ridge_alpha=ridge_alpha,
    )

    if verbose:
        print("=" * 55)
        print("  Customer Lifetime Value Prediction Summary")
        print("=" * 55)
        print(f"  Customers (total)  : {len(features)}")
        print(f"  Train / Test split : {result.metrics['n_train']} / {result.metrics['n_test']}")
        print(f"  Model              : {model_type.upper()}")
        print(f"  Prediction window  : {prediction_window_days} days")
        print("-" * 55)
        print(f"  RMSE  (train)      : {result.metrics['rmse_train']:.4f}")
        print(f"  RMSE  (test)       : {result.metrics['rmse_test']:.4f}")
        print(f"  R²    (train)      : {result.metrics['r2_train']:.4f}")
        print(f"  R²    (test)       : {result.metrics['r2_test']:.4f}")
        print("-" * 55)
        print("  Top feature importances:")
        for feat, imp in result.feature_importances.head(5).items():
            print(f"    {feat:<25s}: {imp:.4f}")
        print("=" * 55)

    return result


# ---------------------------------------------------------------------------
# Prediction on new / unseen customers
# ---------------------------------------------------------------------------

def predict_new_customers(
    result: CLVResult,
    new_df: pd.DataFrame,
    snapshot_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Predict CLV for new customers using a trained CLVResult.

    Parameters
    ----------
    result        : trained CLVResult
    new_df        : transaction DataFrame for new customers
    snapshot_date : reference date

    Returns
    -------
    DataFrame with customer_id and predicted_clv
    """
    features, _ = compute_rfm_features(new_df, snapshot_date=snapshot_date)
    predictions = result.model.predict(features.values)
    predictions = np.clip(predictions, 0, None)
    return pd.DataFrame(
        {"customer_id": features.index, "predicted_clv": predictions}
    ).set_index("customer_id")


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_customers: int = 500,
    n_transactions: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customer_ids = rng.integers(1, n_customers + 1, size=n_transactions)
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 540, size=n_transactions), unit="D"
    )
    amounts = rng.exponential(scale=50, size=n_transactions).round(2)
    return pd.DataFrame(
        {"customer_id": customer_ids, "transaction_date": dates, "amount": amounts}
    )


if __name__ == "__main__":
    df = _generate_synthetic_data(n_customers=300, n_transactions=3000)
    snapshot = pd.Timestamp("2023-06-01")

    print("\n--- GradientBoostingRegressor ---")
    result_gbr = predict_clv(df, snapshot_date=snapshot, model_type="gbr")

    print("\n--- Ridge Regression ---")
    result_ridge = predict_clv(df, snapshot_date=snapshot, model_type="ridge")

    # Predict on a small new batch
    new_batch = _generate_synthetic_data(n_customers=20, n_transactions=200, seed=99)
    preds = predict_new_customers(result_gbr, new_batch, snapshot_date=snapshot)
    print("\nSample predictions for new customers:")
    print(preds.head(10))