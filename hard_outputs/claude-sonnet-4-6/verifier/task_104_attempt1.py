import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Tuple, Dict, Any


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands: upper, lower, and bandwidth."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    bandwidth = (upper - lower) / sma.replace(0, np.nan)
    return upper, lower, bandwidth


def compute_sma_crossover(
    series: pd.Series, short: int = 5, long: int = 20
) -> pd.Series:
    """Compute SMA crossover signal: 1 if short SMA > long SMA, else 0."""
    sma_short = series.rolling(window=short).mean()
    sma_long = series.rolling(window=long).mean()
    signal = (sma_short > sma_long).astype(int)
    return signal


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build technical indicator features from a DataFrame with 'date' and 'close'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 'date' and 'close' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with computed features and binary target column.
    """
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("DataFrame must contain 'date' and 'close' columns.")

    data = df[["date", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    close = data["close"]

    # RSI
    data["rsi_14"] = compute_rsi(close, period=14)

    # MACD
    data["macd_line"], data["macd_signal"], data["macd_hist"] = compute_macd(
        close, fast=12, slow=26, signal=9
    )

    # Bollinger Bands
    data["bb_upper"], data["bb_lower"], data["bb_bandwidth"] = compute_bollinger_bands(
        close, period=20
    )
    data["bb_pct_b"] = (close - data["bb_lower"]) / (
        data["bb_upper"] - data["bb_lower"]
    ).replace(0, np.nan)

    # SMA Crossover
    data["sma_5"] = close.rolling(window=5).mean()
    data["sma_20"] = close.rolling(window=20).mean()
    data["sma_crossover"] = compute_sma_crossover(close, short=5, long=20)

    # Price-based features
    data["returns_1d"] = close.pct_change(1)
    data["returns_5d"] = close.pct_change(5)

    # Binary target: 1 if next-day close > today's close
    data["target"] = (close.shift(-1) > close).astype(int)

    # Drop last row (no target available) and NaN rows from warm-up
    data = data.iloc[:-1]
    data = data.dropna()
    data = data.reset_index(drop=True)

    return data


def train_stock_predictor(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    gb_params: Dict[str, Any] = None,
) -> Tuple[GradientBoostingClassifier, Dict[str, Any], pd.DataFrame]:
    """
    Build and evaluate a stock price direction predictor.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' and 'close' columns.
    test_size : float
        Fraction of data to use for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.
    gb_params : dict, optional
        Custom parameters for GradientBoostingClassifier.

    Returns
    -------
    model : GradientBoostingClassifier
        Trained gradient boosting model.
    metrics : dict
        Dictionary containing accuracy, F1-score, and classification report.
    predictions_df : pd.DataFrame
        DataFrame with dates, actual targets, and predicted labels/probabilities.
    """
    # Build features
    data = build_features(df)

    feature_cols = [
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
        "returns_1d",
        "returns_5d",
    ]

    X = data[feature_cols].values
    y = data["target"].values
    dates = data["date"].values
    close_prices = data["close"].values

    # Temporal split (preserve time order)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]
    close_test = close_prices[split_idx:]

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Class distribution (train) - Up: {y_train.sum()}, Down: {(1-y_train).sum()}")

    # Default GB parameters
    default_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "subsample": 0.8,
        "random_state": random_state,
    }
    if gb_params:
        default_params.update(gb_params)

    # Train model
    model = GradientBoostingClassifier(**default_params)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    report = classification_report(y_test, y_pred, target_names=["Down (0)", "Up (1)"])

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "classification_report": report,
    }

    print("\n=== Model Evaluation ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(report)

    # Feature importances
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("Feature Importances (top 5):")
    print(importances.sort_values(ascending=False).head(5).to_string())

    # Predictions DataFrame
    predictions_df = pd.DataFrame(
        {
            "date": dates_test,
            "close": close_test,
            "actual": y_test,
            "predicted": y_pred,
            "prob_up": y_prob,
            "correct": (y_pred == y_test).astype(int),
        }
    )

    return model, metrics, predictions_df


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    n = 800
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
    # Simulate a random-walk price series
    returns = np.random.normal(0.0003, 0.015, size=n)
    close = 100 * np.exp(np.cumsum(returns))

    sample_df = pd.DataFrame({"date": dates, "close": close})

    model, metrics, preds = train_stock_predictor(sample_df, test_size=0.2)

    print("\nSample predictions (first 10 rows):")
    print(preds.head(10).to_string(index=False))