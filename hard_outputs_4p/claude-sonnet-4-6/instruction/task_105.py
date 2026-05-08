"""
Customer Lifetime Value (CLV) Prediction Module

Predicts CLV from transaction history using RFM and additional features.
Follows strict train/test separation best practices.
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
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
    feature_names: list
    scaler: StandardScaler
    snapshot_date: pd.Timestamp
    clv_horizon_days: int = 90


# ---------------------------------------------------------------------------
# Feature engineering helpers (operate only on a given partition)
# ---------------------------------------------------------------------------

def _compute_rfm_features(
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute RFM + extended features for each customer.

    Parameters
    ----------
    transactions : DataFrame with columns customer_id, transaction_date, amount
    snapshot_date : reference date for recency calculation

    Returns
    -------
    DataFrame indexed by customer_id with feature columns
    """
    df = transactions.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df = df.sort_values(["customer_id", "transaction_date"])

    features_list = []

    for cid, grp in df.groupby("customer_id"):
        grp = grp.sort_values("transaction_date")
        dates = grp["transaction_date"]
        amounts = grp["amount"].values

        # --- RFM ---
        recency = (snapshot_date - dates.max()).days
        frequency = len(grp)
        monetary_total = amounts.sum()
        monetary_avg = amounts.mean()

        # --- Inter-purchase interval ---
        if frequency > 1:
            deltas = dates.diff().dropna().dt.days.values
            ipi_mean = deltas.mean()
            ipi_std = deltas.std(ddof=0)
        else:
            ipi_mean = np.nan
            ipi_std = np.nan

        # --- Purchase trend (slope of amount over time) ---
        if frequency > 1:
            t = (dates - dates.min()).dt.days.values.astype(float)
            # simple linear regression slope
            t_mean = t.mean()
            a_mean = amounts.mean()
            denom = ((t - t_mean) ** 2).sum()
            slope = ((t - t_mean) * (amounts - a_mean)).sum() / denom if denom != 0 else 0.0
        else:
            slope = 0.0

        # --- Additional features ---
        monetary_max = amounts.max()
        monetary_min = amounts.min()
        monetary_std = amounts.std(ddof=0) if frequency > 1 else 0.0
        days_active = (dates.max() - dates.min()).days if frequency > 1 else 0

        features_list.append(
            {
                "customer_id": cid,
                "recency": recency,
                "frequency": frequency,
                "monetary_total": monetary_total,
                "monetary_avg": monetary_avg,
                "monetary_max": monetary_max,
                "monetary_min": monetary_min,
                "monetary_std": monetary_std,
                "ipi_mean": ipi_mean,
                "ipi_std": ipi_std,
                "purchase_trend_slope": slope,
                "days_active": days_active,
            }
        )

    feat_df = pd.DataFrame(features_list).set_index("customer_id")
    return feat_df


def _compute_clv_target(
    transactions: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    horizon_days: int = 90,
) -> pd.Series:
    """
    Compute total spend per customer in the [snapshot_date, snapshot_date + horizon_days] window.
    """
    df = transactions.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    end_date = snapshot_date + pd.Timedelta(days=horizon_days)
    future = df[
        (df["transaction_date"] > snapshot_date)
        & (df["transaction_date"] <= end_date)
    ]
    target = future.groupby("customer_id")["amount"].sum().rename("clv_target")
    return target


# ---------------------------------------------------------------------------
# Train / test split at the customer level
# ---------------------------------------------------------------------------

def _split_customers(
    customer_ids: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    train_ids, test_ids = train_test_split(
        customer_ids, test_size=test_size, random_state=random_state
    )
    return train_ids, test_ids


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _build_model(model_type: str = "gradient_boosting") -> object:
    if model_type == "ridge":
        return Ridge()
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'ridge' or 'gradient_boosting'.")


def _get_param_grid(model_type: str) -> dict:
    if model_type == "ridge":
        return {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    else:  # gradient_boosting
        return {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
            "model__subsample": [0.8, 1.0],
        }


def _extract_feature_importances(
    pipeline: Pipeline,
    feature_names: list,
    model_type: str,
) -> pd.Series:
    model = pipeline.named_steps["model"]
    if model_type == "gradient_boosting":
        importances = model.feature_importances_
    elif model_type == "ridge":
        importances = np.abs(model.coef_)
    else:
        importances = np.ones(len(feature_names))

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def predict_clv(
    transactions: pd.DataFrame,
    snapshot_date: Optional[str] = None,
    horizon_days: int = 90,
    test_size: float = 0.2,
    random_state: int = 42,
    model_type: str = "gradient_boosting",
    cv_folds: int = 5,
) -> CLVResult:
    """
    Predict Customer Lifetime Value from transaction history.

    Parameters
    ----------
    transactions : pd.DataFrame
        Must contain columns: 'customer_id', 'transaction_date', 'amount'
    snapshot_date : str or None
        Reference date (YYYY-MM-DD). If None, uses the median date of all
        transactions, leaving the latter half as the future window.
    horizon_days : int
        Number of days into the future to define CLV target (default 90).
    test_size : float
        Fraction of customers held out for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.
    model_type : str
        'ridge' or 'gradient_boosting'
    cv_folds : int
        Number of cross-validation folds for hyperparameter search.

    Returns
    -------
    CLVResult
        Contains model, metrics, feature_importances, feature_names,
        scaler, snapshot_date, clv_horizon_days.
    """
    # ------------------------------------------------------------------
    # 0. Validate input
    # ------------------------------------------------------------------
    required_cols = {"customer_id", "transaction_date", "amount"}
    missing = required_cols - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = transactions.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="raise")

    if df["amount"].lt(0).any():
        warnings.warn("Negative amounts detected; they will be included as-is.")

    # ------------------------------------------------------------------
    # 1. Determine snapshot date
    # ------------------------------------------------------------------
    if snapshot_date is None:
        all_dates = df["transaction_date"].sort_values()
        snap = all_dates.quantile(0.5)  # median date
    else:
        snap = pd.Timestamp(snapshot_date)

    # Keep only transactions up to snapshot for feature engineering
    history = df[df["transaction_date"] <= snap].copy()
    if history.empty:
        raise ValueError("No transactions found on or before the snapshot date.")

    # ------------------------------------------------------------------
    # 2. Compute targets for ALL customers (uses full df)
    # ------------------------------------------------------------------
    target_series = _compute_clv_target(df, snap, horizon_days)

    # ------------------------------------------------------------------
    # 3. Compute features for ALL customers (uses only history)
    # ------------------------------------------------------------------
    all_features = _compute_rfm_features(history, snap)

    # Align features and targets; fill missing target with 0 (no future purchases)
    all_features = all_features.join(target_series, how="left")
    all_features["clv_target"] = all_features["clv_target"].fillna(0.0)

    feature_cols = [c for c in all_features.columns if c != "clv_target"]
    X_all = all_features[feature_cols]
    y_all = all_features["clv_target"]

    customer_ids = all_features.index.values

    # ------------------------------------------------------------------
    # 4. Split customers into train / test BEFORE any fitting
    # ------------------------------------------------------------------
    train_ids, test_ids = _split_customers(customer_ids, test_size, random_state)

    X_train = X_all.loc[train_ids]
    y_train = y_all.loc[train_ids]
    X_test = X_all.loc[test_ids]
    y_test = y_all.loc[test_ids]

    # ------------------------------------------------------------------
    # 5. Impute missing values using TRAIN statistics only
    # ------------------------------------------------------------------
    train_medians = X_train.median()
    X_train_imp = X_train.fillna(train_medians)
    X_test_imp = X_test.fillna(train_medians)  # use train medians on test

    # ------------------------------------------------------------------
    # 6. Build pipeline (scaler fitted only on train via GridSearchCV)
    # ------------------------------------------------------------------
    base_model = _build_model(model_type)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", base_model),
        ]
    )

    param_grid = _get_param_grid(model_type)

    # GridSearchCV uses only training data (inner CV)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,  # refit best model on full training set
    )
    grid_search.fit(X_train_imp, y_train)

    best_pipeline = grid_search.best_estimator_

    # ------------------------------------------------------------------
    # 7. Evaluate on held-out TEST set (never touched during training)
    # ------------------------------------------------------------------
    y_pred = best_pipeline.predict(X_test_imp)
    y_pred_clipped = np.clip(y_pred, 0, None)  # CLV cannot be negative

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
    r2 = r2_score(y_test, y_pred_clipped)

    metrics = {
        "test_rmse": rmse,
        "test_r2": r2,
        "best_cv_params": grid_search.best_params_,
        "best_cv_neg_rmse": grid_search.best_score_,
        "n_train_customers": len(train_ids),
        "n_test_customers": len(test_ids),
        "snapshot_date": str(snap.date()),
        "horizon_days": horizon_days,
    }

    # ------------------------------------------------------------------
    # 8. Feature importances
    # ------------------------------------------------------------------
    importances = _extract_feature_importances(best_pipeline, feature_cols, model_type)

    # ------------------------------------------------------------------
    # 9. Return results
    # ------------------------------------------------------------------
    return CLVResult(
        model=best_pipeline,
        metrics=metrics,
        feature_importances=importances,
        feature_names=feature_cols,
        scaler=best_pipeline.named_steps["scaler"],
        snapshot_date=snap,
        clv_horizon_days=horizon_days,
    )


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_new_customers(
    result: CLVResult,
    new_transactions: pd.DataFrame,
) -> pd.Series:
    """
    Predict CLV for new customers using a trained CLVResult.

    Parameters
    ----------
    result : CLVResult
        Output of predict_clv().
    new_transactions : pd.DataFrame
        Transaction history for new customers (same schema as training data).

    Returns
    -------
    pd.Series indexed by customer_id with predicted CLV.
    """
    df = new_transactions.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    history = df[df["transaction_date"] <= result.snapshot_date]
    if history.empty:
        raise ValueError("No transactions found on or before the snapshot date.")

    features = _compute_rfm_features(history, result.snapshot_date)

    # Impute using the scaler's mean_ (which was fit on training data)
    # We use a simple approach: fill NaN with 0 then rely on scaler
    X = features[result.feature_names]

    # Impute with column means from the fitted scaler
    train_means = pd.Series(result.scaler.mean_, index=result.feature_names)
    X_imp = X.fillna(train_means)

    preds = result.model.predict(X_imp)
    preds_clipped = np.clip(preds, 0, None)

    return pd.Series(preds_clipped, index=features.index, name="predicted_clv")


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _generate_synthetic_transactions(
    n_customers: int = 300,
    n_transactions: int = 3000,
    random_state: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    customer_ids = rng.integers(1, n_customers + 1, size=n_transactions)
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 2, size=n_transactions), unit="D"
    )
    amounts = rng.exponential(scale=100, size=n_transactions).round(2)
    return pd.DataFrame(
        {"customer_id": customer_ids, "transaction_date": dates, "amount": amounts}
    )


if __name__ == "__main__":
    print("Generating synthetic transaction data...")
    txn_df = _generate_synthetic_transactions(n_customers=300, n_transactions=3000)

    print("Training CLV model...")
    result = predict_clv(
        transactions=txn_df,
        snapshot_date="2023-06-01",
        horizon_days=90,
        test_size=0.2,
        random_state=42,
        model_type="gradient_boosting",
        cv_folds=3,
    )

    print("\n=== CLV Prediction Results ===")
    print(f"Snapshot date   : {result.metrics['snapshot_date']}")
    print(f"Horizon         : {result.metrics['horizon_days']} days")
    print(f"Train customers : {result.metrics['n_train_customers']}")
    print(f"Test customers  : {result.metrics['n_test_customers']}")
    print(f"Test RMSE       : {result.metrics['test_rmse']:.4f}")
    print(f"Test R²         : {result.metrics['test_r2']:.4f}")
    print(f"Best CV params  : {result.metrics['best_cv_params']}")

    print("\n=== Feature Importances ===")
    print(result.feature_importances.to_string())