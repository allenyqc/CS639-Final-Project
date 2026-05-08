"""
Stock Price Direction Predictor Module

Builds a binary classifier to predict whether the next day's closing price
will be higher than today's closing price, using technical indicators as features.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Technical Indicator Computation
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi_14")


def _compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_hist": histogram,
        },
        index=close.index,
    )


def _compute_bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Compute Bollinger Bands: upper, lower, and bandwidth."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma.replace(0, np.nan)

    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_pct_b": (close - lower) / (upper - lower).replace(0, np.nan),
        },
        index=close.index,
    )


def _compute_sma_crossover(
    close: pd.Series, short: int = 5, long: int = 20
) -> pd.DataFrame:
    """Compute SMA crossover signal (1 if short SMA > long SMA, else 0)."""
    sma_short = close.rolling(window=short).mean()
    sma_long = close.rolling(window=long).mean()
    crossover_signal = (sma_short > sma_long).astype(int)

    return pd.DataFrame(
        {
            f"sma_{short}": sma_short,
            f"sma_{long}": sma_long,
            "sma_crossover": crossover_signal,
        },
        index=close.index,
    )


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators and the binary target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date' and 'close' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with features and target, NaN rows dropped.
    """
    required_cols = {"date", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    data = df[["date", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    close = data["close"].astype(float)

    # --- Technical Indicators ---
    rsi = _compute_rsi(close)
    macd = _compute_macd(close)
    bb = _compute_bollinger_bands(close)
    sma_cross = _compute_sma_crossover(close)

    # --- Binary Target: 1 if next-day close > today's close ---
    target = (close.shift(-1) > close).astype(int).rename("target")

    # --- Assemble feature matrix ---
    features = pd.concat([data, rsi, macd, bb, sma_cross, target], axis=1)

    # Drop last row (no next-day target) and NaN rows from warm-up periods
    features = features.iloc[:-1]
    features = features.dropna().reset_index(drop=True)

    return features


# ---------------------------------------------------------------------------
# Chronological Train / Validation / Test Split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame chronologically into train, validation,
    and test sets.  No shuffling is performed.

    Parameters
    ----------
    df : pd.DataFrame
        Time-ordered feature DataFrame.
    train_frac : float
        Fraction of data for training.
    val_frac : float
        Fraction of data for validation (threshold tuning / early stopping).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.0.")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Model Training and Evaluation
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "bb_bandwidth",
    "bb_pct_b",
    "sma_5",
    "sma_20",
    "sma_crossover",
]

TARGET_COL = "target"


def build_predictor(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    gb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build and evaluate a stock price direction predictor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' and 'close' columns.
    train_frac : float
        Fraction of data for training (chronological).
    val_frac : float
        Fraction of data for validation.
    gb_params : dict or None
        Optional hyperparameters for GradientBoostingClassifier.

    Returns
    -------
    dict with keys:
        - 'model'       : fitted GradientBoostingClassifier
        - 'scaler'      : fitted StandardScaler
        - 'metrics'     : dict with accuracy, f1, classification_report
        - 'predictions' : DataFrame with date, actual, predicted, probability
        - 'train_df'    : training partition (for reference)
        - 'val_df'      : validation partition (for reference)
        - 'test_df'     : test partition (for reference)
    """
    # --- Step 1: Compute features on the full dataset ---
    full_features = compute_features(df)

    if len(full_features) < 50:
        raise ValueError(
            "Insufficient data after indicator warm-up. "
            "Provide at least ~100 rows of price data."
        )

    # --- Step 2: Chronological split BEFORE any fitting ---
    train_df, val_df, test_df = chronological_split(
        full_features, train_frac=train_frac, val_frac=val_frac
    )

    print(
        f"Split sizes — Train: {len(train_df)}, "
        f"Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # --- Step 3: Separate features and targets ---
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values

    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[TARGET_COL].values

    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values

    # --- Step 4: Fit scaler ONLY on training data ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Step 5: Train Gradient Boosting classifier ---
    default_params: dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "subsample": 0.8,
        "random_state": 42,
    }
    if gb_params is not None:
        default_params.update(gb_params)

    model = GradientBoostingClassifier(**default_params)
    model.fit(X_train_scaled, y_train)

    # --- Step 6: Validation set — monitor performance (not used for test eval) ---
    y_val_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    print(f"Validation Accuracy: {val_accuracy:.4f} | Validation F1: {val_f1:.4f}")

    # --- Step 7: Final evaluation on held-out test set ---
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_f1_weighted = f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_report = classification_report(
        y_test, y_test_pred, target_names=["Down/Flat", "Up"], zero_division=0
    )

    metrics = {
        "test_accuracy": test_accuracy,
        "test_f1_binary": test_f1,
        "test_f1_weighted": test_f1_weighted,
        "test_classification_report": test_report,
        "val_accuracy": val_accuracy,
        "val_f1": val_f1,
    }

    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"F1 (binary)   : {test_f1:.4f}")
    print(f"F1 (weighted) : {test_f1_weighted:.4f}")
    print("\nClassification Report:")
    print(test_report)

    # --- Step 8: Build predictions DataFrame ---
    predictions_df = pd.DataFrame(
        {
            "date": test_df["date"].values,
            "close": test_df["close"].values,
            "actual": y_test,
            "predicted": y_test_pred,
            "probability_up": y_test_proba,
        }
    )

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "predictions": predictions_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


# ---------------------------------------------------------------------------
# Inference Helper
# ---------------------------------------------------------------------------

def predict_direction(
    model: GradientBoostingClassifier,
    scaler: StandardScaler,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions for new price data using a fitted model and scaler.

    Parameters
    ----------
    model : GradientBoostingClassifier
        Fitted classifier returned by build_predictor.
    scaler : StandardScaler
        Fitted scaler returned by build_predictor.
    new_df : pd.DataFrame
        DataFrame with 'date' and 'close' columns (must have enough history
        for indicator warm-up, typically ≥ 30 rows).

    Returns
    -------
    pd.DataFrame
        DataFrame with date, close, predicted direction, and probability.
    """
    features = compute_features(new_df)

    if features.empty:
        raise ValueError(
            "No valid rows after computing indicators. "
            "Provide more historical data."
        )

    X = features[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    return pd.DataFrame(
        {
            "date": features["date"].values,
            "close": features["close"].values,
            "predicted_direction": predictions,
            "probability_up": probabilities,
        }
    )


# ---------------------------------------------------------------------------
# Demo / Smoke Test
# ---------------------------------------------------------------------------

def _generate_synthetic_prices(
    n: int = 500, seed: int = 0
) -> pd.DataFrame:
    """Generate synthetic daily closing prices for testing."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=0.0003, scale=0.015, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"date": dates, "close": prices})


if __name__ == "__main__":
    synthetic_df = _generate_synthetic_prices(n=600)
    result = build_predictor(synthetic_df)

    print("\nSample Predictions:")
    print(result["predictions"].head(10).to_string(index=False))

    print("\nFeature Importances:")
    importances = pd.Series(
        result["model"].feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print(importances.to_string())