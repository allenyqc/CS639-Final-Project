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
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"]

    # RSI
    df["rsi_14"] = compute_rsi(close, period=14)

    # MACD
    df["macd_line"], df["macd_signal"], df["macd_hist"] = compute_macd(
        close, fast=12, slow=26, signal=9
    )

    # Bollinger Bands
    df["bb_upper"], df["bb_lower"], df["bb_bandwidth"] = compute_bollinger_bands(
        close, period=20
    )
    df["bb_pct_b"] = (close - df["bb_lower"]) / (
        df["bb_upper"] - df["bb_lower"]
    ).replace(0, np.nan)

    # SMA Crossover
    df["sma_5"] = close.rolling(window=5).mean()
    df["sma_20"] = close.rolling(window=20).mean()
    df["sma_crossover"] = compute_sma_crossover(close, short=5, long=20)

    # Additional price-derived features
    df["returns_1d"] = close.pct_change(1)
    df["returns_5d"] = close.pct_change(5)
    df["volatility_10d"] = close.pct_change().rolling(window=10).std()

    # Binary target: 1 if next-day close > today's close
    df["target"] = (close.shift(-1) > close).astype(int)

    # Drop last row (no target) and NaN rows from warm-up
    df = df.iloc[:-1]
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df


def build_stock_predictor(
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
        DataFrame with dates, actual targets, predicted labels, and probabilities.

    Raises
    ------
    ValueError
        If required columns are missing or insufficient data after preprocessing.
    """
    required_cols = {"date", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Build features
    feature_df = build_features(df)

    if len(feature_df) < 50:
        raise ValueError(
            f"Insufficient data after preprocessing: {len(feature_df)} rows. "
            "Need at least 50 rows."
        )

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
        "volatility_10d",
    ]

    X = feature_df[feature_cols].values
    y = feature_df["target"].values
    dates = feature_df["date"].values
    close_vals = feature_df["close"].values

    # Time-series aware split (no shuffling to avoid look-ahead bias)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]
    close_test = close_vals[split_idx:]

    # Default GradientBoosting parameters
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
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(
        y_test, y_pred, target_names=["Down/Flat", "Up"], zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "classification_report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_importances": dict(
            zip(feature_cols, model.feature_importances_)
        ),
    }

    # Predictions DataFrame
    predictions_df = pd.DataFrame(
        {
            "date": dates_test,
            "close": close_test,
            "actual": y_test,
            "predicted": y_pred,
            "probability_up": y_prob,
        }
    )
    predictions_df["correct"] = (
        predictions_df["actual"] == predictions_df["predicted"]
    )

    return model, metrics, predictions_df


def print_summary(metrics: Dict[str, Any]) -> None:
    """Print a formatted summary of model metrics."""
    print("=" * 60)
    print("STOCK PRICE DIRECTION PREDICTOR — RESULTS")
    print("=" * 60)
    print(f"Training samples : {metrics['train_size']}")
    print(f"Test samples     : {metrics['test_size']}")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"F1-Score         : {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("\nTop Feature Importances:")
    sorted_fi = sorted(
        metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True
    )
    for feat, imp in sorted_fi[:5]:
        print(f"  {feat:<20} {imp:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Example usage / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    n = 500
    dates = pd.date_range(start="2020-01-01", periods=n, freq="B")
    # Simulate a random walk for closing prices
    returns = np.random.normal(0.0005, 0.015, size=n)
    close = 100 * np.exp(np.cumsum(returns))

    sample_df = pd.DataFrame({"date": dates, "close": close})

    model, metrics, predictions_df = build_stock_predictor(
        sample_df, test_size=0.2, random_state=42
    )

    print_summary(metrics)
    print("\nSample predictions (first 10 rows):")
    print(predictions_df.head(10).to_string(index=False))