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
    """Container for CLV model outputs."""
    model: object
    metrics: dict
    feature_importances: pd.Series
    train_predictions: pd.Series
    test_predictions: pd.Series
    feature_matrix: pd.DataFrame
    target: pd.Series
    train_index: np.ndarray
    test_index: np.ndarray


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _compute_purchase_trend(dates: pd.Series, amounts: pd.Series) -> float:
    """
    Compute the slope of purchase amounts over time (days from first purchase).
    Returns 0.0 if fewer than 2 transactions exist.
    """
    if len(dates) < 2:
        return 0.0
    days = (dates - dates.min()).dt.days.values.astype(float)
    if days.std() == 0:
        return 0.0
    slope, *_ = stats.linregress(days, amounts.values.astype(float))
    return float(slope)


def _compute_inter_purchase_stats(dates: pd.Series) -> tuple[float, float]:
    """
    Compute mean and std of inter-purchase intervals in days.
    Returns (0.0, 0.0) if fewer than 2 transactions.
    """
    if len(dates) < 2:
        return 0.0, 0.0
    sorted_dates = dates.sort_values()
    gaps = sorted_dates.diff().dropna().dt.days.values.astype(float)
    return float(gaps.mean()), float(gaps.std(ddof=0))


def compute_rfm_features(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    window_days: int = 90,
) -> pd.DataFrame:
    """
    Compute RFM + extended features for each customer using transactions
    that occurred *before* snapshot_date.

    Parameters
    ----------
    df : DataFrame with columns ['customer_id', 'transaction_date', 'amount']
    snapshot_date : reference date for recency calculation
    window_days : look-back window in days for feature computation

    Returns
    -------
    DataFrame indexed by customer_id with feature columns.
    """
    history = df[df["transaction_date"] < snapshot_date].copy()

    features = []
    for cid, grp in history.groupby("customer_id"):
        grp = grp.sort_values("transaction_date")
        amounts = grp["amount"]
        dates = grp["transaction_date"]

        recency = (snapshot_date - dates.max()).days
        frequency = len(grp)
        monetary_total = amounts.sum()
        monetary_avg = amounts.mean()
        monetary_std = amounts.std(ddof=0) if frequency > 1 else 0.0
        monetary_max = amounts.max()
        monetary_min = amounts.min()

        ipi_mean, ipi_std = _compute_inter_purchase_stats(dates)
        trend = _compute_purchase_trend(dates, amounts)

        # Coefficient of variation of amounts
        cv_amount = monetary_std / monetary_avg if monetary_avg != 0 else 0.0

        # Days active (span of purchase history)
        days_active = (dates.max() - dates.min()).days if frequency > 1 else 0

        features.append(
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
                "cv_amount": cv_amount,
                "days_active": days_active,
            }
        )

    return pd.DataFrame(features).set_index("customer_id")


def compute_clv_target(
    df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    horizon_days: int = 90,
) -> pd.Series:
    """
    Compute total spend per customer in [snapshot_date, snapshot_date + horizon].

    Returns
    -------
    pd.Series indexed by customer_id (customers with zero future spend included).
    """
    end_date = snapshot_date + pd.Timedelta(days=horizon_days)
    future = df[
        (df["transaction_date"] >= snapshot_date)
        & (df["transaction_date"] < end_date)
    ]
    target = future.groupby("customer_id")["amount"].sum()
    target.name = "clv_target"
    return target


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {"customer_id", "transaction_date", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["transaction_date", "amount"])
    n_after = len(df)
    if n_before != n_after:
        warnings.warn(
            f"Dropped {n_before - n_after} rows with NaN in "
            "'transaction_date' or 'amount'."
        )

    if (df["amount"] < 0).any():
        warnings.warn("Negative 'amount' values detected; they will be kept as-is.")

    return df


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def _build_pipeline(model_type: str = "gbr") -> Pipeline:
    """
    Build a sklearn Pipeline with optional scaling + regressor.

    Parameters
    ----------
    model_type : 'gbr' for GradientBoostingRegressor, 'ridge' for Ridge
    """
    if model_type == "ridge":
        regressor = Ridge(alpha=1.0)
        return Pipeline([("scaler", StandardScaler()), ("model", regressor)])
    elif model_type == "gbr":
        regressor = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )
        return Pipeline([("model", regressor)])
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'gbr' or 'ridge'.")


def _extract_feature_importances(
    pipeline: Pipeline, feature_names: list[str], model_type: str
) -> pd.Series:
    """Extract feature importances or coefficients from the fitted pipeline."""
    model = pipeline.named_steps["model"]
    if model_type == "gbr":
        importances = model.feature_importances_
    else:  # ridge
        importances = np.abs(model.coef_)

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def predict_clv(
    df: pd.DataFrame,
    snapshot_date: Optional[str | pd.Timestamp] = None,
    horizon_days: int = 90,
    test_size: float = 0.2,
    model_type: str = "gbr",
    random_state: int = 42,
) -> CLVResult:
    """
    Train a CLV prediction model from transaction history.

    Parameters
    ----------
    df : DataFrame with columns ['customer_id', 'transaction_date', 'amount']
    snapshot_date : date that separates historical features from future target.
                    Defaults to (max_date - horizon_days).
    horizon_days : number of days in the future to predict spend for (default 90).
    test_size : fraction of customers held out for evaluation (default 0.2).
    model_type : 'gbr' (GradientBoostingRegressor) or 'ridge' (Ridge regression).
    random_state : random seed for reproducibility.

    Returns
    -------
    CLVResult dataclass with model, metrics, feature importances, predictions, etc.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> df = pd.DataFrame({
    ...     'customer_id': rng.integers(1, 51, n),
    ...     'transaction_date': pd.date_range('2022-01-01', periods=n, freq='12h'),
    ...     'amount': rng.exponential(50, n),
    ... })
    >>> result = predict_clv(df)
    >>> print(result.metrics)
    """
    # ---- Validate ----
    df = _validate_dataframe(df)

    # ---- Determine snapshot date ----
    max_date = df["transaction_date"].max()
    if snapshot_date is None:
        snapshot_date = max_date - pd.Timedelta(days=horizon_days)
    else:
        snapshot_date = pd.Timestamp(snapshot_date)

    if snapshot_date <= df["transaction_date"].min():
        raise ValueError(
            "snapshot_date is before or at the earliest transaction. "
            "No historical data available for feature computation."
        )

    print(f"[CLV] Snapshot date : {snapshot_date.date()}")
    print(f"[CLV] Prediction horizon : {horizon_days} days")
    print(f"[CLV] Max transaction date : {max_date.date()}")

    # ---- Feature engineering ----
    features_df = compute_rfm_features(df, snapshot_date)
    target_series = compute_clv_target(df, snapshot_date, horizon_days)

    # Align: only customers with historical data; fill missing future spend with 0
    target_series = target_series.reindex(features_df.index).fillna(0.0)

    print(f"[CLV] Customers with history : {len(features_df)}")
    print(
        f"[CLV] Customers with future purchases : "
        f"{(target_series > 0).sum()} "
        f"({100*(target_series > 0).mean():.1f}%)"
    )

    X = features_df.copy()
    y = target_series.copy()

    feature_names = X.columns.tolist()

    # ---- Train / test split (by customer) ----
    customer_ids = X.index.values
    train_ids, test_ids = train_test_split(
        customer_ids, test_size=test_size, random_state=random_state
    )

    X_train, X_test = X.loc[train_ids], X.loc[test_ids]
    y_train, y_test = y.loc[train_ids], y.loc[test_ids]

    print(f"[CLV] Train customers : {len(X_train)} | Test customers : {len(X_test)}")

    # ---- Build & train model ----
    pipeline = _build_pipeline(model_type)
    pipeline.fit(X_train.values, y_train.values)

    # ---- Predictions ----
    y_pred_train = pipeline.predict(X_train.values)
    y_pred_test = pipeline.predict(X_test.values)

    # Clip negative predictions (spend cannot be negative)
    y_pred_train = np.clip(y_pred_train, 0, None)
    y_pred_test = np.clip(y_pred_test, 0, None)

    # ---- Metrics ----
    rmse_train = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    r2_train = float(r2_score(y_train, y_pred_train))
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2_test = float(r2_score(y_test, y_pred_test))

    metrics = {
        "train_rmse": rmse_train,
        "train_r2": r2_train,
        "test_rmse": rmse_test,
        "test_r2": r2_test,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "horizon_days": horizon_days,
        "snapshot_date": str(snapshot_date.date()),
        "model_type": model_type,
    }

    print(
        f"[CLV] Train  RMSE={rmse_train:.2f}  R²={r2_train:.4f}\n"
        f"[CLV] Test   RMSE={rmse_test:.2f}  R²={r2_test:.4f}"
    )

    # ---- Feature importances ----
    importances = _extract_feature_importances(pipeline, feature_names, model_type)

    print("\n[CLV] Feature importances:")
    print(importances.to_string())

    return CLVResult(
        model=pipeline,
        metrics=metrics,
        feature_importances=importances,
        train_predictions=pd.Series(y_pred_train, index=train_ids, name="predicted_clv"),
        test_predictions=pd.Series(y_pred_test, index=test_ids, name="predicted_clv"),
        feature_matrix=X,
        target=y,
        train_index=train_ids,
        test_index=test_ids,
    )


def predict_new_customers(
    result: CLVResult,
    new_df: pd.DataFrame,
    snapshot_date: str | pd.Timestamp,
) -> pd.Series:
    """
    Apply a trained CLV model to new / unseen transaction data.

    Parameters
    ----------
    result : CLVResult returned by predict_clv()
    new_df : DataFrame with columns ['customer_id', 'transaction_date', 'amount']
    snapshot_date : reference date for feature computation

    Returns
    -------
    pd.Series of predicted CLV indexed by customer_id
    """
    new_df = _validate_dataframe(new_df)
    snapshot_date = pd.Timestamp(snapshot_date)
    features_df = compute_rfm_features(new_df, snapshot_date)
    preds = result.model.predict(features_df.values)
    preds = np.clip(preds, 0, None)
    return pd.Series(preds, index=features_df.index, name="predicted_clv")


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    N_TRANSACTIONS = 2000
    N_CUSTOMERS = 100

    customer_ids = rng.integers(1, N_CUSTOMERS + 1, N_TRANSACTIONS)
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 540, N_TRANSACTIONS), unit="D"
    )
    amounts = rng.exponential(scale=75, size=N_TRANSACTIONS).round(2)

    demo_df = pd.DataFrame(
        {"customer_id": customer_ids, "transaction_date": dates, "amount": amounts}
    )

    print("=" * 60)
    print("CLV Prediction Demo")
    print("=" * 60)
    print(f"Total transactions : {len(demo_df)}")
    print(f"Unique customers   : {demo_df['customer_id'].nunique()}")
    print(f"Date range         : {demo_df['transaction_date'].min().date()} "
          f"→ {demo_df['transaction_date'].max().date()}")
    print()

    result = predict_clv(demo_df, model_type="gbr", horizon_days=90)

    print("\n--- Metrics ---")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Top 5 test predictions ---")
    top5 = result.test_predictions.sort_values(ascending=False).head(5)
    print(top5)