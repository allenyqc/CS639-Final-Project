```python
"""
Customer Lifetime Value (CLV) Prediction Module

Predicts CLV from transaction history using RFM and extended features.
Follows best practices for data leakage prevention, chronological splitting,
and proper evaluation.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CLVResult:
    """Container for CLV model outputs."""
    model: Any
    metrics: dict[str, float]
    feature_importances: pd.Series
    train_customers: np.ndarray
    test_customers: np.ndarray
    feature_names: list[str]
    predictions: pd.DataFrame  # customer_id, y_true, y_pred


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _compute_purchase_trend(dates: pd.Series, amounts: pd.Series) -> float:
    """Return slope of OLS regression of amount on days-since-first-purchase."""
    if len(dates) < 2:
        return 0.0
    days = (dates - dates.min()).dt.days.values.astype(float)
    if days.std() == 0:
        return 0.0
    slope, *_ = stats.linregress(days, amounts.values.astype(float))
    return float(slope)


def _compute_rfm_features(
    history: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute RFM + extended features for each customer.

    Parameters
    ----------
    history : DataFrame with columns customer_id, transaction_date, amount
    reference_date : the cutoff date (last day of the training window)

    Returns
    -------
    DataFrame indexed by customer_id with feature columns
    """
    history = history.copy()
    history["transaction_date"] = pd.to_datetime(history["transaction_date"])

    records: list[dict] = []
    for cid, grp in history.groupby("customer_id"):
        grp = grp.sort_values("transaction_date")
        dates = grp["transaction_date"]
        amounts = grp["amount"]

        recency = (reference_date - dates.max()).days
        frequency = len(grp)
        monetary_total = amounts.sum()
        monetary_avg = amounts.mean()
        monetary_std = amounts.std(ddof=0) if frequency > 1 else 0.0

        # Inter-purchase intervals
        if frequency > 1:
            intervals = dates.diff().dt.days.dropna()
            ipi_mean = intervals.mean()
            ipi_std = intervals.std(ddof=0) if len(intervals) > 1 else 0.0
        else:
            ipi_mean = np.nan
            ipi_std = np.nan

        trend = _compute_purchase_trend(dates, amounts)

        # Tenure in days
        tenure = (dates.max() - dates.min()).days

        records.append(
            {
                "customer_id": cid,
                "recency": recency,
                "frequency": frequency,
                "monetary_total": monetary_total,
                "monetary_avg": monetary_avg,
                "monetary_std": monetary_std,
                "ipi_mean": ipi_mean,
                "ipi_std": ipi_std,
                "purchase_trend": trend,
                "tenure_days": tenure,
            }
        )

    features = pd.DataFrame(records).set_index("customer_id")
    return features


def _compute_clv_target(
    future: pd.DataFrame,
    customer_ids: np.ndarray,
    window_days: int = 90,
) -> pd.Series:
    """
    Compute total spend in the next `window_days` for each customer.

    Customers with no future transactions get 0.
    """
    future = future.copy()
    future["transaction_date"] = pd.to_datetime(future["transaction_date"])
    spend = (
        future.groupby("customer_id")["amount"]
        .sum()
        .reindex(customer_ids, fill_value=0.0)
    )
    spend.name = "clv_target"
    return spend


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def _chronological_customer_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split customers chronologically by their *last* transaction date.

    Customers whose last purchase falls in the earliest (1 - test_fraction - val_fraction)
    quantile go to train; the next val_fraction quantile to validation; the rest to test.

    This prevents data leakage: a customer's future behaviour cannot influence
    the model fitted on earlier customers.
    """
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    last_tx = (
        df.groupby("customer_id")["transaction_date"]
        .max()
        .sort_values()
    )

    n = len(last_tx)
    train_end = int(n * (1 - test_fraction - val_fraction))
    val_end = int(n * (1 - test_fraction))

    train_customers = last_tx.iloc[:train_end].index.values
    val_customers = last_tx.iloc[train_end:val_end].index.values
    test_customers = last_tx.iloc[val_end:].index.values

    logger.info(
        "Split: train=%d, val=%d, test=%d customers",
        len(train_customers),
        len(val_customers),
        len(test_customers),
    )
    return train_customers, val_customers, test_customers


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def _build_pipeline(model_type: str = "gbr") -> Pipeline:
    """Return a sklearn Pipeline with scaler + regressor."""
    if model_type == "ridge":
        regressor = Ridge()
    elif model_type == "gbr":
        regressor = GradientBoostingRegressor(random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Choose 'ridge' or 'gbr'.")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )


def _get_param_grid(model_type: str) -> dict:
    if model_type == "ridge":
        return {"regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    return {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__max_depth": [3, 4, 5],
        "regressor__learning_rate": [0.05, 0.1, 0.2],
        "regressor__subsample": [0.7, 0.9, 1.0],
        "regressor__min_samples_leaf": [5, 10, 20],
    }


def _extract_feature_importances(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pd.Series:
    """Extract feature importances or coefficients from the fitted pipeline."""
    regressor = pipeline.named_steps["regressor"]
    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        importances = np.abs(regressor.coef_)
    else:
        importances = np.zeros(len(feature_names))

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def predict_clv(
    transactions: pd.DataFrame,
    model_type: str = "gbr",
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    future_window_days: int = 90,
    n_iter_search: int = 20,
    random_state: int = 42,
) -> CLVResult:
    """
    Train a CLV prediction model from transaction history.

    Parameters
    ----------
    transactions : pd.DataFrame
        Must contain columns: 'customer_id', 'transaction_date', 'amount'.
    model_type : str
        'gbr' (GradientBoostingRegressor) or 'ridge'.
    test_fraction : float
        Fraction of customers (by last-purchase date) held out for testing.
    val_fraction : float
        Fraction of customers used for validation / hyperparameter search.
    future_window_days : int
        Number of days ahead to define the CLV target (default 90).
    n_iter_search : int
        Number of iterations for RandomizedSearchCV on the validation set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    CLVResult
    """
    # ------------------------------------------------------------------
    # 0. Validate input
    # ------------------------------------------------------------------
    required_cols = {"customer_id", "transaction_date", "amount"}
    missing = required_cols - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    transactions = transactions.copy()
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="raise")

    if transactions["amount"].lt(0).any():
        logger.warning("Negative transaction amounts detected; they will be included as-is.")

    # ------------------------------------------------------------------
    # 1. Chronological customer split (BEFORE any feature engineering)
    # ------------------------------------------------------------------
    train_customers, val_customers, test_customers = _chronological_customer_split(
        transactions,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
    )

    # ------------------------------------------------------------------
    # 2. Define observation window and prediction window per split
    #    Reference date = last transaction date of the *training* customers
    # ------------------------------------------------------------------
    train_tx = transactions[transactions["customer_id"].isin(train_customers)]
    val_tx = transactions[transactions["customer_id"].isin(val_customers)]
    test_tx = transactions[transactions["customer_id"].isin(test_customers)]

    # Reference date for feature computation = last date in training set
    train_ref_date = train_tx["transaction_date"].max()
    val_ref_date = val_tx["transaction_date"].max()
    test_ref_date = test_tx["transaction_date"].max()

    logger.info(
        "Reference dates — train: %s, val: %s, test: %s",
        train_ref_date.date(),
        val_ref_date.date(),
        test_ref_date.date(),
    )

    # ------------------------------------------------------------------
    # 3. Feature engineering — fit only on training data
    # ------------------------------------------------------------------
    # For each split, "history" = transactions up to ref_date,
    # "future" = transactions in the next future_window_days.

    def _split_history_future(
        tx: pd.DataFrame, ref_date: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        history = tx[tx["transaction_date"] <= ref_date]
        future_end = ref_date + pd.Timedelta(days=future_window_days)
        future = tx[
            (tx["transaction_date"] > ref_date)
            & (tx["transaction_date"] <= future_end)
        ]
        return history, future

    train_history, train_future = _split_history_future(train_tx, train_ref_date)
    val_history, val_future = _split_history_future(val_tx, val_ref_date)
    test_history, test_future = _split_history_future(test_tx, test_ref_date)

    # Compute features (no scaler fitted yet — that happens inside the pipeline)
    X_train_raw = _compute_rfm_features(train_history, train_ref_date)
    X_val_raw = _compute_rfm_features(val_history, val_ref_date)
    X_test_raw = _compute_rfm_features(test_history, test_ref_date)

    # Targets
    y_train = _compute_clv_target(train_future, train_customers, future_window_days)
    y_val = _compute_clv_target(val_future, val_customers, future_window_days)
    y_test = _compute_clv_target(test_future, test_customers, future_window_days)

    # Align indices
    X_train_raw = X_train_raw.reindex(train_customers)
    X_val_raw = X_val_raw.reindex(val_customers)
    X_test_raw = X_test_raw.reindex(test_customers)

    # ------------------------------------------------------------------
    # 4. Impute missing values using TRAINING statistics only
    # ------------------------------------------------------------------
    feature_names = X_train_raw.columns.tolist()

    train_medians = X_train_raw.median()  # computed on train only
    X_train_raw = X_train_raw.fillna(train_medians)
    X_val_raw = X_val_raw.fillna(train_medians)   # use train medians for val/test
    X_test_raw = X_test_raw.fillna(train_medians)

    X_train = X_train_raw.values
    X_val = X_val_raw.values
    X_test = X_test_raw.values

    y_train_arr = y_train.values.astype(float)
    y_val_arr = y_val.values.astype(float)
    y_test_arr = y_test.values.astype(float)

    # ------------------------------------------------------------------
    # 5. Hyperparameter search on VALIDATION set only
    #    (test set remains untouched)
    # ------------------------------------------------------------------
    logger.info("Starting hyperparameter search on validation set …")

    base_pipeline = _build_pipeline(model_type)
    param_grid = _get_param_grid(model_type)

    # Combine train + val for CV, using GroupKFold so that train customers
    # are always in one fold and val customers in another.
    X_search = np.vstack([X_train, X_val])
    y_search = np.concatenate([y_train_arr, y_val_arr])
    groups_search = np.concatenate(
        [np.zeros(len(X_train), dtype=int), np.ones(len(X_val), dtype=int)]
    )

    # GroupKFold with 2 splits: fold 0 = train, fold 1 = val
    gkf = GroupKFold(n_splits=2)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_grid,
        n_iter=n_iter_search,
        cv=gkf,
        scoring="neg_root_mean_squared_error",
        refit=False,  # we will refit manually on train only
        n_jobs=-1,
        random_state=random_state,
        error_score="raise",
    )
    search.fit(X_search, y_search, groups=groups_search)

    best_params = search.best_params_
    logger.info("Best hyperparameters (from val CV): %s", best_params)

    # ------------------------------------------------------------------
    # 6. Refit on TRAINING data only with best hyperparameters
    # ------------------------------------------------------------------
    final_pipeline = _build_pipeline(model_type)
    final_pipeline.set_params(**best_params)
    final_pipeline.fit(X_train, y_train_arr)

    # ------------------------------------------------------------------
    # 7. Final evaluation on TEST set (first and only time)
    # ------------------------------------------------------------------
    y_pred_test = final_pipeline.predict(X_test)
    y_pred_test = np.clip(y_pred_test, 0.0, None)  # CLV cannot be negative

    rmse = float(np.sqrt(mean_squared_error(y_test_arr, y_pred_test)))
    r2 = float(r2_score(y_test_arr, y_pred_test))
    mae = float(np.mean(np.abs(y_test_arr - y_pred_test)))

    # Median absolute percentage error (robust to zero targets)
    nonzero_mask = y_test_arr > 0
    if nonzero_mask.sum() > 0:
        mape = float(
            np.median(
                np.abs(y_test_arr[nonzero_mask] - y_pred_test[nonzero_mask])
                / y_test_arr[nonzero_mask]
            )
            * 100
        )
    else:
        mape = float("nan")

    metrics = {
        "test_rmse": rmse,
        "test_r2": r2,
        "test_mae": mae,
        "test_median_ape_pct": mape,
    }
    logger.info("Test metrics: %s", metrics)

    # ------------------------------------------------------------------
    # 8. Feature importances
    # ------------------------------------------------------------------
    importances = _extract_feature_importances(final_pipeline, feature_names)

    # ------------------------------------------------------------------
    # 9. Predictions DataFrame
    # ------------------------------------------------------------------
    predictions_df = pd.DataFrame(
        {
            "customer_id": test_customers,
            "y_true": y_test_arr,
            "y_pred": y_pred_test,
        }
    )

    return CLVResult(
        model=final_pipeline,
        metrics=metrics,
        feature_importances=importances,
        train_customers=train_customers,
        test_customers=test_customers,
        feature_names=feature_names,
        predictions=predictions_df,
    )


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_new_customers(
    result: CLVResult,
    new_transactions: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Apply a trained CLV model to new / unseen customers.

    Parameters
    ----------
    result : CLVResult returned by predict_clv.
    new_transactions : DataFrame with customer_id, transaction_date, amount.
    reference_date : cutoff date for feature computation. Defaults to max date.

    Returns
    -------
    DataFrame with customer_id and predicted_clv columns.
    """
    new_transactions = new_transactions.copy()
    new_transactions["transaction_date"] = pd.to_datetime(
        new_transactions["transaction_date"]
    )

    if reference_date is None:
        reference_date = new_transactions["transaction_date"].max()

    features = _compute_rfm_features(new_transactions, reference_date)

    # Impute using training medians stored implicitly in the pipeline's scaler
    # (we re-use the same column order)
    features = features.reindex(columns=result.feature_names)

    # Fill NaN with 0 as a safe fallback for inference
    features = features.fillna(0.0)

    preds = result.model.predict(features.values)
    preds = np.clip(preds, 0.0, None)

    return pd.DataFrame(
        {"customer_id": features.index, "predicted_clv": preds}
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _generate_synthetic_transactions(
    n_customers: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate synthetic transaction data for testing."""
    rng = np.random.default_rng(seed)
    records = []
    base_date = pd.Timestamp("2021-01-01")

    for cid in range(n_customers):
        n_tx = int(rng.integers(1, 30))
        days_offsets = np.sort(rng.integers(0, 730, size=n_tx))