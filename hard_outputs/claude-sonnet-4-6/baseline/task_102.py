import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer session-level and user-level features from raw event data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw events with columns: user_id, session_id, timestamp,
        event_type, page_category.

    Returns
    -------
    pd.DataFrame
        One row per session with engineered features and binary target.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ------------------------------------------------------------------ #
    # Session-level features
    # ------------------------------------------------------------------ #
    session_grp = df.groupby(["user_id", "session_id"])

    session_duration = (
        session_grp["timestamp"]
        .agg(lambda x: (x.max() - x.min()).total_seconds())
        .rename("session_duration_sec")
    )

    num_clicks = (
        session_grp["event_type"]
        .apply(lambda x: (x == "click").sum())
        .rename("num_clicks")
    )

    num_add_to_cart = (
        session_grp["event_type"]
        .apply(lambda x: (x == "add_to_cart").sum())
        .rename("num_add_to_cart")
    )

    has_add_to_cart = (num_add_to_cart > 0).astype(int).rename("has_add_to_cart")

    num_distinct_categories = (
        session_grp["page_category"]
        .nunique()
        .rename("num_distinct_categories")
    )

    num_events = session_grp.size().rename("num_events")

    # Binary target: 1 if session contains at least one purchase
    has_purchase = (
        session_grp["event_type"]
        .apply(lambda x: int((x == "purchase").any()))
        .rename("target")
    )

    session_features = pd.concat(
        [
            session_duration,
            num_clicks,
            num_add_to_cart,
            has_add_to_cart,
            num_distinct_categories,
            num_events,
            has_purchase,
        ],
        axis=1,
    ).reset_index()

    # ------------------------------------------------------------------ #
    # User-level historical features
    # (computed over ALL sessions to avoid leakage at this stage;
    #  leakage is handled later by computing history only on train data
    #  and joining to test)
    # ------------------------------------------------------------------ #
    user_stats = (
        session_features.groupby("user_id")
        .agg(
            user_total_sessions=("session_id", "count"),
            user_purchase_rate=("target", "mean"),
            user_avg_session_length=("session_duration_sec", "mean"),
            user_avg_clicks=("num_clicks", "mean"),
            user_avg_add_to_cart=("num_add_to_cart", "mean"),
        )
        .reset_index()
    )

    session_features = session_features.merge(user_stats, on="user_id", how="left")

    return session_features


def compute_user_history_from_train(
    train_sessions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute user-level statistics strictly from training sessions.
    This prevents target leakage when the same user appears in test.

    Parameters
    ----------
    train_sessions : pd.DataFrame
        Session-level data for training sessions only.

    Returns
    -------
    pd.DataFrame
        User-level historical statistics derived from training data.
    """
    user_history = (
        train_sessions.groupby("user_id")
        .agg(
            user_total_sessions=("session_id", "count"),
            user_purchase_rate=("target", "mean"),
            user_avg_session_length=("session_duration_sec", "mean"),
            user_avg_clicks=("num_clicks", "mean"),
            user_avg_add_to_cart=("num_add_to_cart", "mean"),
        )
        .reset_index()
    )
    return user_history


def attach_user_history(
    sessions: pd.DataFrame,
    user_history: pd.DataFrame,
    global_defaults: Dict[str, float],
) -> pd.DataFrame:
    """
    Attach user-level history to a sessions DataFrame.
    Users not present in history receive global default values.

    Parameters
    ----------
    sessions : pd.DataFrame
        Session-level data (may be train or test).
    user_history : pd.DataFrame
        User-level statistics computed from training data.
    global_defaults : dict
        Default values for users unseen in training.

    Returns
    -------
    pd.DataFrame
        Sessions with user-level features attached.
    """
    user_cols = [
        "user_total_sessions",
        "user_purchase_rate",
        "user_avg_session_length",
        "user_avg_clicks",
        "user_avg_add_to_cart",
    ]
    # Drop stale user columns if present
    sessions = sessions.drop(columns=[c for c in user_cols if c in sessions.columns])
    sessions = sessions.merge(user_history, on="user_id", how="left")

    for col in user_cols:
        sessions[col] = sessions[col].fillna(global_defaults.get(col, 0.0))

    return sessions


def predict_purchase(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = None,
) -> Tuple[RandomForestClassifier, float, float, pd.Series]:
    """
    Full pipeline: feature engineering → train/test split →
    leak-free user history → Random Forest → evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw event log with columns:
        user_id, session_id, timestamp, event_type, page_category.
    test_size : float
        Fraction of sessions to hold out for testing.
    random_state : int
        Random seed for reproducibility.
    n_estimators : int
        Number of trees in the Random Forest.
    max_depth : int or None
        Maximum depth of each tree.

    Returns
    -------
    model : RandomForestClassifier
        Trained classifier.
    f1 : float
        F1-score on the test set.
    auc : float
        ROC-AUC on the test set.
    feature_importances : pd.Series
        Feature importances sorted in descending order.
    """
    # ------------------------------------------------------------------ #
    # Validate input
    # ------------------------------------------------------------------ #
    required_cols = {"user_id", "session_id", "timestamp", "event_type", "page_category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # ------------------------------------------------------------------ #
    # Step 1: Build session-level features (user history computed globally
    #         here; will be recomputed leak-free after splitting)
    # ------------------------------------------------------------------ #
    session_df = engineer_features(df)

    # ------------------------------------------------------------------ #
    # Step 2: Train / test split at the session level
    # ------------------------------------------------------------------ #
    train_sessions, test_sessions = train_test_split(
        session_df,
        test_size=test_size,
        random_state=random_state,
        stratify=session_df["target"],
    )

    # ------------------------------------------------------------------ #
    # Step 3: Recompute user history strictly from training sessions
    #         to avoid leakage for users who appear in both splits
    # ------------------------------------------------------------------ #
    user_history_train = compute_user_history_from_train(train_sessions)

    # Global defaults = training-set averages (used for cold-start users)
    global_defaults = {
        "user_total_sessions": user_history_train["user_total_sessions"].mean(),
        "user_purchase_rate": user_history_train["user_purchase_rate"].mean(),
        "user_avg_session_length": user_history_train["user_avg_session_length"].mean(),
        "user_avg_clicks": user_history_train["user_avg_clicks"].mean(),
        "user_avg_add_to_cart": user_history_train["user_avg_add_to_cart"].mean(),
    }

    train_sessions = attach_user_history(train_sessions, user_history_train, global_defaults)
    test_sessions = attach_user_history(test_sessions, user_history_train, global_defaults)

    # ------------------------------------------------------------------ #
    # Step 4: Define feature matrix
    # ------------------------------------------------------------------ #
    feature_cols = [
        "session_duration_sec",
        "num_clicks",
        "num_add_to_cart",
        "has_add_to_cart",
        "num_distinct_categories",
        "num_events",
        "user_total_sessions",
        "user_purchase_rate",
        "user_avg_session_length",
        "user_avg_clicks",
        "user_avg_add_to_cart",
    ]

    X_train = train_sessions[feature_cols].values
    y_train = train_sessions["target"].values
    X_test = test_sessions[feature_cols].values
    y_test = test_sessions["target"].values

    # ------------------------------------------------------------------ #
    # Step 5: Train Random Forest
    # ------------------------------------------------------------------ #
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",  # handles class imbalance
    )
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # Step 6: Evaluate on test set
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")

    feature_importances = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )

    return model, f1, auc, feature_importances


# --------------------------------------------------------------------------- #
# Convenience wrapper that also prints a summary report
# --------------------------------------------------------------------------- #
def run_pipeline(
    df: pd.DataFrame,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run the full purchase-prediction pipeline and return a results dict.

    Parameters
    ----------
    df : pd.DataFrame
        Raw event log.
    **kwargs
        Additional keyword arguments forwarded to predict_purchase.

    Returns
    -------
    dict with keys: model, f1, auc, feature_importances.
    """
    model, f1, auc, importances = predict_purchase(df, **kwargs)

    print("=" * 50)
    print("Purchase Prediction Pipeline Results")
    print("=" * 50)
    print(f"Test F1-Score : {f1:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print("\nFeature Importances:")
    print(importances.to_string())
    print("=" * 50)

    return {
        "model": model,
        "f1": f1,
        "auc": auc,
        "feature_importances": importances,
    }


# --------------------------------------------------------------------------- #
# Quick smoke-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n_users = 200
    n_sessions = 1_000
    n_events = 8_000

    user_ids = rng.integers(1, n_users + 1, size=n_events)
    session_ids = rng.integers(1, n_sessions + 1, size=n_events)
    base_time = pd.Timestamp("2024-01-01")
    timestamps = [
        base_time + pd.Timedelta(seconds=int(s))
        for s in rng.integers(0, 3_600 * 24 * 30, size=n_events)
    ]
    event_types = rng.choice(
        ["click", "add_to_cart", "purchase"],
        size=n_events,
        p=[0.75, 0.20, 0.05],
    )
    page_categories = rng.choice(
        ["electronics", "clothing", "books", "home", "sports"],
        size=n_events,
    )

    sample_df = pd.DataFrame(
        {
            "user_id": user_ids,
            "session_id": session_ids,
            "timestamp": timestamps,
            "event_type": event_types,
            "page_category": page_categories,
        }
    )

    results = run_pipeline(sample_df, test_size=0.2, random_state=42)