"""
E-commerce Session Purchase Prediction Module
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Any


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _build_session_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw events into one row per session with session-level features.

    Features produced
    -----------------
    session_duration_sec   : seconds between first and last event in session
    n_clicks               : number of click events
    n_add_to_cart          : number of add_to_cart events
    n_distinct_categories  : number of distinct page_category values visited
    has_add_to_cart        : binary flag (1 if any add_to_cart event)
    target                 : 1 if session contains a purchase event, else 0
    """
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    agg = (
        events.groupby(["user_id", "session_id"])
        .apply(
            lambda g: pd.Series(
                {
                    "session_duration_sec": (
                        g["timestamp"].max() - g["timestamp"].min()
                    ).total_seconds(),
                    "n_clicks": (g["event_type"] == "click").sum(),
                    "n_add_to_cart": (g["event_type"] == "add_to_cart").sum(),
                    "n_distinct_categories": g["page_category"].nunique(),
                    "has_add_to_cart": int((g["event_type"] == "add_to_cart").any()),
                    "target": int((g["event_type"] == "purchase").any()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return agg


def _build_user_history_features(
    session_df: pd.DataFrame,
    train_session_ids: pd.Index,
) -> pd.DataFrame:
    """
    Compute user-level historical features using ONLY training sessions to
    avoid data leakage.  For users that appear in the test set but not in
    training, global averages are used as fallback.

    Features produced
    -----------------
    user_historical_purchase_rate : fraction of the user's *training* sessions
                                    that contained a purchase
    user_avg_session_length_sec   : mean session duration across the user's
                                    *training* sessions
    """
    train_sessions = session_df[session_df["session_id"].isin(train_session_ids)]

    user_stats = (
        train_sessions.groupby("user_id")
        .agg(
            user_historical_purchase_rate=("target", "mean"),
            user_avg_session_length_sec=("session_duration_sec", "mean"),
        )
        .reset_index()
    )

    # Global fallbacks for unseen users
    global_purchase_rate = train_sessions["target"].mean()
    global_avg_length = train_sessions["session_duration_sec"].mean()

    merged = session_df.merge(user_stats, on="user_id", how="left")
    merged["user_historical_purchase_rate"] = merged[
        "user_historical_purchase_rate"
    ].fillna(global_purchase_rate)
    merged["user_avg_session_length_sec"] = merged[
        "user_avg_session_length_sec"
    ].fillna(global_avg_length)

    return merged


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "session_duration_sec",
    "n_clicks",
    "n_add_to_cart",
    "n_distinct_categories",
    "has_add_to_cart",
    "user_historical_purchase_rate",
    "user_avg_session_length_sec",
]


def train_purchase_predictor(
    events: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    rf_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    End-to-end pipeline: feature engineering → train/test split →
    Random Forest training → evaluation.

    Parameters
    ----------
    events : pd.DataFrame
        Raw event log with columns:
        ['user_id', 'session_id', 'timestamp', 'event_type', 'page_category']
    test_size : float
        Fraction of sessions to hold out for testing (default 0.20).
    random_state : int
        Seed for reproducibility.
    rf_params : dict, optional
        Extra keyword arguments forwarded to RandomForestClassifier.

    Returns
    -------
    dict with keys:
        model               : fitted RandomForestClassifier
        f1_score            : F1 on the test set (binary, positive class = 1)
        auc                 : ROC-AUC on the test set
        feature_importances : pd.Series mapping feature name → importance
        X_test              : test feature matrix (pd.DataFrame)
        y_test              : test labels (pd.Series)
        session_df          : full session-level feature DataFrame
    """
    # ------------------------------------------------------------------ #
    # 0. Validate input
    # ------------------------------------------------------------------ #
    required_cols = {"user_id", "session_id", "timestamp", "event_type", "page_category"}
    missing = required_cols - set(events.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    if events.empty:
        raise ValueError("Input DataFrame is empty.")

    # ------------------------------------------------------------------ #
    # 1. Session-level feature engineering
    # ------------------------------------------------------------------ #
    session_df = _build_session_features(events)

    # ------------------------------------------------------------------ #
    # 2. Train / test split (session-level, stratified by target)
    # ------------------------------------------------------------------ #
    train_sessions, test_sessions = train_test_split(
        session_df,
        test_size=test_size,
        random_state=random_state,
        stratify=session_df["target"],
    )

    # ------------------------------------------------------------------ #
    # 3. User-level historical features (leak-free)
    # ------------------------------------------------------------------ #
    session_df = _build_user_history_features(
        session_df, train_session_ids=train_sessions["session_id"]
    )

    # Re-align train/test splits after adding user features
    train_idx = session_df["session_id"].isin(train_sessions["session_id"])
    test_idx = session_df["session_id"].isin(test_sessions["session_id"])

    train_df = session_df[train_idx].copy()
    test_df = session_df[test_idx].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"]

    # ------------------------------------------------------------------ #
    # 4. Train Random Forest
    # ------------------------------------------------------------------ #
    default_rf_params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
    }
    if rf_params:
        default_rf_params.update(rf_params)

    model = RandomForestClassifier(**default_rf_params)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # 5. Evaluate
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred, zero_division=0)

    # AUC requires at least two classes in y_test
    if len(y_test.unique()) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_test, y_prob)

    feature_importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    return {
        "model": model,
        "f1_score": f1,
        "auc": auc,
        "feature_importances": feature_importances,
        "X_test": X_test,
        "y_test": y_test,
        "session_df": session_df,
    }


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_sessions(
    model: RandomForestClassifier,
    new_events: pd.DataFrame,
    historical_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Predict purchase probability for sessions in *new_events*.

    Parameters
    ----------
    model : fitted RandomForestClassifier returned by train_purchase_predictor
    new_events : pd.DataFrame
        New raw event log (same schema as training data).
    historical_events : pd.DataFrame, optional
        Past events used to compute user-level historical features.
        If None, global averages of 0 are used for user features.

    Returns
    -------
    pd.DataFrame with columns:
        user_id, session_id, purchase_probability, predicted_purchase
    """
    session_df = _build_session_features(new_events)

    if historical_events is not None:
        hist_sessions = _build_session_features(historical_events)
        user_stats = (
            hist_sessions.groupby("user_id")
            .agg(
                user_historical_purchase_rate=("target", "mean"),
                user_avg_session_length_sec=("session_duration_sec", "mean"),
            )
            .reset_index()
        )
        global_purchase_rate = hist_sessions["target"].mean()
        global_avg_length = hist_sessions["session_duration_sec"].mean()
        session_df = session_df.merge(user_stats, on="user_id", how="left")
        session_df["user_historical_purchase_rate"] = session_df[
            "user_historical_purchase_rate"
        ].fillna(global_purchase_rate)
        session_df["user_avg_session_length_sec"] = session_df[
            "user_avg_session_length_sec"
        ].fillna(global_avg_length)
    else:
        session_df["user_historical_purchase_rate"] = 0.0
        session_df["user_avg_session_length_sec"] = 0.0

    X = session_df[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    return pd.DataFrame(
        {
            "user_id": session_df["user_id"].values,
            "session_id": session_df["session_id"].values,
            "purchase_probability": probs,
            "predicted_purchase": preds,
        }
    )


# ---------------------------------------------------------------------------
# Quick demo / smoke-test
# ---------------------------------------------------------------------------

def _generate_synthetic_events(
    n_users: int = 200,
    n_sessions: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Generate a small synthetic event log for testing."""
    rng = np.random.default_rng(random_state)
    categories = ["electronics", "clothing", "books", "home", "sports"]
    event_types = ["click", "add_to_cart", "purchase"]

    rows = []
    base_time = pd.Timestamp("2024-01-01")

    for sid in range(n_sessions):
        uid = rng.integers(0, n_users)
        n_events = rng.integers(2, 20)
        start = base_time + pd.Timedelta(seconds=int(rng.integers(0, 86400 * 30)))
        has_purchase = rng.random() < 0.15  # ~15 % purchase rate

        for i in range(n_events):
            ts = start + pd.Timedelta(seconds=int(rng.integers(0, 600) * i))
            if has_purchase and i == n_events - 1:
                etype = "purchase"
            elif rng.random() < 0.2:
                etype = "add_to_cart"
            else:
                etype = "click"
            rows.append(
                {
                    "user_id": f"u{uid:04d}",
                    "session_id": f"s{sid:05d}",
                    "timestamp": ts,
                    "event_type": etype,
                    "page_category": rng.choice(categories),
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating synthetic data …")
    events_df = _generate_synthetic_events(n_users=200, n_sessions=1000)
    print(f"  {len(events_df):,} events | {events_df['session_id'].nunique()} sessions "
          f"| {events_df['user_id'].nunique()} users")

    print("\nTraining model …")
    results = train_purchase_predictor(events_df, test_size=0.2, random_state=42)

    print(f"\n  Test F1-score : {results['f1_score']:.4f}")
    print(f"  Test AUC      : {results['auc']:.4f}")
    print("\nFeature importances:")
    print(results["feature_importances"].to_string())