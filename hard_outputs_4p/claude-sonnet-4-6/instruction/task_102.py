"""
E-commerce Session Purchase Prediction Module

Predicts whether a user session will result in a purchase using
session-level and user-level historical features.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper: session-level feature extraction
# ---------------------------------------------------------------------------

def _extract_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-session features from raw event rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw events with columns:
        ['user_id', 'session_id', 'timestamp', 'event_type', 'page_category']

    Returns
    -------
    pd.DataFrame
        One row per session with engineered features and binary target.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    agg = (
        df.groupby(["user_id", "session_id"])
        .apply(
            lambda g: pd.Series(
                {
                    "session_start": g["timestamp"].min(),
                    "session_end": g["timestamp"].max(),
                    "session_duration_sec": (
                        g["timestamp"].max() - g["timestamp"].min()
                    ).total_seconds(),
                    "num_clicks": (g["event_type"] == "click").sum(),
                    "num_add_to_cart": (g["event_type"] == "add_to_cart").sum(),
                    "num_distinct_categories": g["page_category"].nunique(),
                    "add_to_cart_occurred": int(
                        (g["event_type"] == "add_to_cart").any()
                    ),
                    "target": int((g["event_type"] == "purchase").any()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    return agg


# ---------------------------------------------------------------------------
# Helper: user-level historical features (computed ONLY from training data)
# ---------------------------------------------------------------------------

def _compute_user_history(train_sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user-level statistics from the training session DataFrame.

    Parameters
    ----------
    train_sessions : pd.DataFrame
        Session-level DataFrame (output of _extract_session_features) for
        training sessions only.

    Returns
    -------
    pd.DataFrame
        One row per user with columns:
        ['user_id', 'hist_purchase_rate', 'hist_avg_session_duration',
         'hist_avg_clicks', 'hist_num_sessions']
    """
    user_hist = (
        train_sessions.groupby("user_id")
        .agg(
            hist_purchase_rate=("target", "mean"),
            hist_avg_session_duration=("session_duration_sec", "mean"),
            hist_avg_clicks=("num_clicks", "mean"),
            hist_num_sessions=("session_id", "count"),
        )
        .reset_index()
    )
    return user_hist


# ---------------------------------------------------------------------------
# Helper: merge user history into session frame
# ---------------------------------------------------------------------------

def _merge_user_history(
    sessions: pd.DataFrame,
    user_history: pd.DataFrame,
    global_defaults: dict[str, float],
) -> pd.DataFrame:
    """
    Left-join user history onto sessions; fill missing users with global
    defaults computed from training data.
    """
    merged = sessions.merge(user_history, on="user_id", how="left")

    for col, default_val in global_defaults.items():
        merged[col] = merged[col].fillna(default_val)

    return merged


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def train_and_evaluate(
    events_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """
    Train a Random Forest to predict session-level purchase probability.

    Parameters
    ----------
    events_df : pd.DataFrame
        Raw event log with columns:
        ['user_id', 'session_id', 'timestamp', 'event_type', 'page_category']
    test_size : float
        Fraction of sessions to hold out for testing.
    random_state : int
        Reproducibility seed.
    n_estimators : int
        Number of trees in the Random Forest.
    max_depth : int or None
        Maximum tree depth (None = unlimited).

    Returns
    -------
    dict with keys:
        'model'               – fitted RandomForestClassifier
        'scaler'              – fitted StandardScaler
        'f1_score'            – F1 on test set
        'auc'                 – ROC-AUC on test set
        'feature_importances' – pd.Series sorted descending
        'feature_names'       – list of feature column names
        'test_sessions'       – session-level test DataFrame with predictions
    """
    # ------------------------------------------------------------------
    # 0. Validate input
    # ------------------------------------------------------------------
    required_cols = {"user_id", "session_id", "timestamp", "event_type", "page_category"}
    missing = required_cols - set(events_df.columns)
    if missing:
        raise ValueError(f"events_df is missing required columns: {missing}")

    # ------------------------------------------------------------------
    # 1. Build session-level frame (no leakage: just aggregation)
    # ------------------------------------------------------------------
    sessions = _extract_session_features(events_df)

    # ------------------------------------------------------------------
    # 2. Train / test split on SESSIONS before any feature engineering
    #    that depends on labels or cross-session statistics.
    #    We split by session_id to avoid data leakage.
    # ------------------------------------------------------------------
    session_ids = sessions["session_id"].unique()
    train_ids, test_ids = train_test_split(
        session_ids, test_size=test_size, random_state=random_state
    )

    train_sessions = sessions[sessions["session_id"].isin(train_ids)].copy()
    test_sessions = sessions[sessions["session_id"].isin(test_ids)].copy()

    # ------------------------------------------------------------------
    # 3. Compute user-level historical features ONLY from training data
    # ------------------------------------------------------------------
    user_history = _compute_user_history(train_sessions)

    # Global defaults (from training data) for users unseen in training
    global_defaults = {
        "hist_purchase_rate": float(train_sessions["target"].mean()),
        "hist_avg_session_duration": float(
            train_sessions["session_duration_sec"].mean()
        ),
        "hist_avg_clicks": float(train_sessions["num_clicks"].mean()),
        "hist_num_sessions": 1.0,
    }

    # ------------------------------------------------------------------
    # 4. Merge user history into both splits
    #    Test users who also appear in training get their TRAINING-derived
    #    statistics (no leakage). New test-only users get global defaults.
    # ------------------------------------------------------------------
    train_full = _merge_user_history(train_sessions, user_history, global_defaults)
    test_full = _merge_user_history(test_sessions, user_history, global_defaults)

    # ------------------------------------------------------------------
    # 5. Define feature columns
    # ------------------------------------------------------------------
    feature_cols = [
        "session_duration_sec",
        "num_clicks",
        "num_add_to_cart",
        "num_distinct_categories",
        "add_to_cart_occurred",
        "hist_purchase_rate",
        "hist_avg_session_duration",
        "hist_avg_clicks",
        "hist_num_sessions",
    ]

    X_train = train_full[feature_cols].values
    y_train = train_full["target"].values

    X_test = test_full[feature_cols].values
    y_test = test_full["target"].values

    # ------------------------------------------------------------------
    # 6. Scale features — fit ONLY on training data
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform only, never fit

    # ------------------------------------------------------------------
    # 7. Train Random Forest
    # ------------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",  # handles class imbalance
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 8. Evaluate on held-out test set ONLY
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")

    # ------------------------------------------------------------------
    # 9. Feature importances
    # ------------------------------------------------------------------
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # 10. Attach predictions to test frame for downstream inspection
    # ------------------------------------------------------------------
    test_full = test_full.copy()
    test_full["predicted_label"] = y_pred
    test_full["predicted_prob"] = y_prob

    return {
        "model": model,
        "scaler": scaler,
        "f1_score": test_f1,
        "auc": test_auc,
        "feature_importances": importances,
        "feature_names": feature_cols,
        "test_sessions": test_full,
    }


# ---------------------------------------------------------------------------
# Inference helper for new, unseen sessions
# ---------------------------------------------------------------------------

def predict_sessions(
    new_events_df: pd.DataFrame,
    model: RandomForestClassifier,
    scaler: StandardScaler,
    user_history: pd.DataFrame,
    global_defaults: dict[str, float],
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Score new sessions using a previously trained model.

    Parameters
    ----------
    new_events_df : pd.DataFrame
        Raw events for new sessions (same schema as training data).
    model : RandomForestClassifier
        Fitted model returned by train_and_evaluate.
    scaler : StandardScaler
        Fitted scaler returned by train_and_evaluate.
    user_history : pd.DataFrame
        User-level history computed from training data.
    global_defaults : dict
        Fallback values for unseen users.
    feature_cols : list[str]
        Feature column names (same order used during training).

    Returns
    -------
    pd.DataFrame
        Session-level DataFrame with 'predicted_label' and 'predicted_prob'.
    """
    sessions = _extract_session_features(new_events_df)
    sessions = _merge_user_history(sessions, user_history, global_defaults)

    X = scaler.transform(sessions[feature_cols].values)
    sessions["predicted_label"] = model.predict(X)
    sessions["predicted_prob"] = model.predict_proba(X)[:, 1]

    return sessions


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_users: int = 200,
    n_sessions: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Generate a small synthetic event log for testing."""
    rng = np.random.default_rng(random_state)

    user_ids = rng.integers(1, n_users + 1, size=n_sessions)
    session_ids = np.arange(1, n_sessions + 1)
    categories = ["electronics", "clothing", "books", "home", "sports"]
    event_types = ["click", "add_to_cart", "purchase"]

    rows = []
    base_time = pd.Timestamp("2024-01-01")

    for i in range(n_sessions):
        uid = user_ids[i]
        sid = session_ids[i]
        n_events = rng.integers(2, 15)
        start = base_time + pd.Timedelta(seconds=int(rng.integers(0, 86400 * 30)))

        for j in range(n_events):
            ts = start + pd.Timedelta(seconds=int(rng.integers(0, 600)))
            # Bias toward clicks; rare purchases
            et_weights = [0.70, 0.20, 0.10]
            et = rng.choice(event_types, p=et_weights)
            cat = rng.choice(categories)
            rows.append(
                {
                    "user_id": uid,
                    "session_id": sid,
                    "timestamp": ts,
                    "event_type": et,
                    "page_category": cat,
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating synthetic data …")
    df = _generate_synthetic_data(n_users=300, n_sessions=2000, random_state=42)
    print(f"  Events: {len(df):,}  |  Sessions: {df['session_id'].nunique():,}  "
          f"|  Users: {df['user_id'].nunique():,}")

    print("\nTraining model …")
    results = train_and_evaluate(df, test_size=0.2, random_state=42)

    print(f"\n{'='*50}")
    print(f"  Test F1-Score : {results['f1_score']:.4f}")
    print(f"  Test ROC-AUC  : {results['auc']:.4f}")
    print(f"\nFeature Importances:")
    for feat, imp in results["feature_importances"].items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<35s} {imp:.4f}  {bar}")
    print(f"{'='*50}")