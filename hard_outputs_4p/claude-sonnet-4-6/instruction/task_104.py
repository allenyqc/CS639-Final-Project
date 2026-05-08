"""
stock_direction_predictor.py

A module for predicting stock price direction using technical indicators
and a Gradient Boosting classifier.
"""

import warnings
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Technical Indicator Computation
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands: upper, lower, and bandwidth."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma.replace(0, np.nan)
    return upper, lower, bandwidth


def compute_sma_crossover(
    close: pd.Series, short: int = 5, long_: int = 20
) -> pd.Series:
    """
    Compute SMA crossover signal.
    Returns 1 when short SMA > long SMA, else 0.
    """
    sma_short = close.rolling(window=short).mean()
    sma_long = close.rolling(window=long_).mean()
    signal = (sma_short > sma_long).astype(int)
    return signal


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with 'date' and 'close', compute all technical
    indicators and the binary target variable.

    Returns a DataFrame with features and target, NaN rows dropped.
    """
    required_cols = {"date", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    data = df[["date", "close"]].copy()
    data = data.sort_values("date").reset_index(drop=True)

    close = data["close"]

    # RSI
    data["rsi_14"] = compute_rsi(close, period=14)

    # MACD
    data["macd"], data["macd_signal"], data["macd_hist"] = compute_macd(
        close, fast=12, slow=26, signal=9
    )

    # Bollinger Bands
    data["bb_upper"], data["bb_lower"], data["bb_bandwidth"] = compute_bollinger_bands(
        close, period=20
    )

    # SMA crossover signal
    data["sma_crossover"] = compute_sma_crossover(close, short=5, long_=20)

    # Additional derived features
    data["sma_5"] = close.rolling(window=5).mean()
    data["sma_20"] = close.rolling(window=20).mean()
    data["close_to_bb_upper"] = (close - data["bb_upper"]) / close
    data["close_to_bb_lower"] = (close - data["bb_lower"]) / close

    # Binary target: 1 if next-day close > today's close
    data["target"] = (close.shift(-1) > close).astype(int)

    # Drop the last row (no next-day target) and any NaN rows from warm-up
    data = data.dropna().reset_index(drop=True)

    return data


# ---------------------------------------------------------------------------
# Train / Test Split  (time-series aware — no shuffling)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "bb_bandwidth",
    "sma_crossover",
    "sma_5",
    "sma_20",
    "close_to_bb_upper",
    "close_to_bb_lower",
]


def time_series_split(
    data: pd.DataFrame, test_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train and test sets.
    The split is purely chronological — no shuffling.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1 (exclusive).")

    n = len(data)
    split_idx = int(n * (1 - test_fraction))
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    return train, test


# ---------------------------------------------------------------------------
# Model Training & Evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    gb_params: Dict[str, Any] | None = None,
) -> Tuple[GradientBoostingClassifier, StandardScaler, Dict[str, Any], pd.DataFrame]:
    """
    Full pipeline:
      1. Build features from raw price data.
      2. Split chronologically into train / test.
      3. Fit scaler ONLY on training data; transform both sets.
      4. Train a GradientBoostingClassifier on training data.
      5. Evaluate on the held-out test set.
      6. Return model, scaler, metrics dict, and predictions DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date' and 'close' columns.
    test_fraction : float
        Fraction of data reserved for testing (default 0.2).
    gb_params : dict, optional
        Hyperparameters for GradientBoostingClassifier.

    Returns
    -------
    model : GradientBoostingClassifier
    scaler : StandardScaler  (fitted on training data only)
    metrics : dict  with accuracy, f1, and classification_report
    predictions_df : pd.DataFrame  with test-set predictions
    """
    # ------------------------------------------------------------------ #
    # Step 1: Build features (all rows, no leakage yet)
    # ------------------------------------------------------------------ #
    data = build_features(df)

    if len(data) < 50:
        raise ValueError(
            "Not enough data after indicator warm-up. "
            "Provide at least ~80 rows of daily prices."
        )

    # ------------------------------------------------------------------ #
    # Step 2: Chronological train / test split  ← BEFORE any fitting
    # ------------------------------------------------------------------ #
    train_data, test_data = time_series_split(data, test_fraction=test_fraction)

    X_train = train_data[FEATURE_COLS].values
    y_train = train_data["target"].values
    X_test = test_data[FEATURE_COLS].values
    y_test = test_data["target"].values

    # ------------------------------------------------------------------ #
    # Step 3: Fit scaler on TRAINING data only, then transform both sets
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform train
    X_test_scaled = scaler.transform(X_test)          # transform only (no fit)

    # ------------------------------------------------------------------ #
    # Step 4: Train the classifier on training data only
    # ------------------------------------------------------------------ #
    default_params: Dict[str, Any] = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "subsample": 0.8,
        "random_state": 42,
    }
    if gb_params:
        default_params.update(gb_params)

    model = GradientBoostingClassifier(**default_params)
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------ #
    # Step 5: Evaluate on the held-out test set ONLY
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    clf_report = classification_report(y_test, y_pred, zero_division=0)

    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": clf_report,
        "n_train": len(train_data),
        "n_test": len(test_data),
    }

    # ------------------------------------------------------------------ #
    # Step 6: Build predictions DataFrame
    # ------------------------------------------------------------------ #
    predictions_df = test_data[["date", "close", "target"]].copy().reset_index(drop=True)
    predictions_df["predicted"] = y_pred
    predictions_df["predicted_prob"] = y_prob
    predictions_df["correct"] = (predictions_df["predicted"] == predictions_df["target"]).astype(int)

    return model, scaler, metrics, predictions_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_direction(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    gb_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    High-level entry point.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 'date' (str or datetime) and 'close' (float).
    test_fraction : float
        Proportion of data used for testing (default 0.2).
    gb_params : dict, optional
        Override default GradientBoostingClassifier hyperparameters.

    Returns
    -------
    result : dict
        {
            'model'          : trained GradientBoostingClassifier,
            'scaler'         : fitted StandardScaler,
            'metrics'        : dict of accuracy, f1, classification_report,
            'predictions_df' : DataFrame with test-set predictions,
            'feature_cols'   : list of feature column names,
        }
    """
    # Coerce date column
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)

    model, scaler, metrics, predictions_df = train_and_evaluate(
        df, test_fraction=test_fraction, gb_params=gb_params
    )

    print("=" * 60)
    print("Stock Direction Predictor — Evaluation Results")
    print("=" * 60)
    print(f"Training samples : {metrics['n_train']}")
    print(f"Test samples     : {metrics['n_test']}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"F1-Score         : {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "predictions_df": predictions_df,
        "feature_cols": FEATURE_COLS,
    }


# ---------------------------------------------------------------------------
# Inference helper (for new, unseen data)
# ---------------------------------------------------------------------------

def predict_new(
    model: GradientBoostingClassifier,
    scaler: StandardScaler,
    df_new: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate predictions for new data using a previously trained model and scaler.

    Parameters
    ----------
    model : GradientBoostingClassifier
        Trained model returned by predict_direction().
    scaler : StandardScaler
        Fitted scaler returned by predict_direction().
    df_new : pd.DataFrame
        New price data with 'date' and 'close' columns.

    Returns
    -------
    pd.DataFrame with date, close, predicted direction, and probability.
    """
    df_new = df_new.copy()
    df_new["date"] = pd.to_datetime(df_new["date"])
    df_new["close"] = df_new["close"].astype(float)

    data = build_features(df_new)

    # Use scaler fitted on training data — no re-fitting
    X = scaler.transform(data[FEATURE_COLS].values)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    result = data[["date", "close"]].copy().reset_index(drop=True)
    result["predicted"] = preds
    result["predicted_prob"] = probs
    return result


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate synthetic price data for demonstration
    np.random.seed(0)
    n_days = 500
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="B")
    returns = np.random.normal(loc=0.0003, scale=0.015, size=n_days)
    prices = 100 * np.cumprod(1 + returns)

    demo_df = pd.DataFrame({"date": dates, "close": prices})

    result = predict_direction(demo_df, test_fraction=0.2)

    print("\nSample predictions (first 10 rows):")
    print(result["predictions_df"].head(10).to_string(index=False))