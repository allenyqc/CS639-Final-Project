"""
E-commerce Session Purchase Prediction Module

Predicts whether a user session will result in a purchase using session-level
and user-level historical features with a Random Forest classifier.
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_input_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame has the required columns and types."""
    required_columns = {"user_id", "session_id", "timestamp", "event_type", "page_category"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    valid_event_types = {"click", "add_to_cart", "purchase"}
    invalid_events = set(df["event_type"].unique()) - valid_event_types
    if invalid_events:
        raise ValueError(f"Invalid event_type values found: {invalid_events}")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if df["timestamp"].dtype == object:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as exc:
            raise ValueError(f"Cannot parse 'timestamp' column: {exc}") from exc


def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build session-level features from raw event data.

    Features:
    - session_duration_seconds: time from first to last event in session
    - num_clicks: number of click events
    - num_distinct_categories: number of distinct page categories visited
    - add_to_cart_occurred: binary flag for add_to_cart event
    - session_start: timestamp of first event (used for chronological splitting)
    - user_id: for group-aware operations
    - has_purchase: binary target (1 if session contains a purchase)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    session_groups = df.groupby("session_id")

    session_features = pd.DataFrame({
        "session_id": list(session_groups.groups.keys()),
    }).set_index("session_id")

    # Session duration in seconds
    session_features["session_duration_seconds"] = session_groups["timestamp"].apply(
        lambda x: (x.max() - x.min()).total_seconds()
    )

    # Number of clicks
    session_features["num_clicks"] = session_groups["event_type"].apply(
        lambda x: (x == "click").sum()
    )

    # Number of distinct page categories
    session_features["num_distinct_categories"] = session_groups["page_category"].nunique()

    # Whether add_to_cart occurred
    session_features["add_to_cart_occurred"] = session_groups["event_type"].apply(
        lambda x: int((x == "add_to_cart").any())
    )

    # Session start time (for chronological splitting)
    session_features["session_start"] = session_groups["timestamp"].min()

    # User ID (for group-aware operations)
    session_features["user_id"] = session_groups["user_id"].first()

    # Binary target: 1 if session contains a purchase
    session_features["has_purchase"] = session_groups["event_type"].apply(
        lambda x: int((x == "purchase").any())
    )

    session_features = session_features.reset_index()
    return session_features


def compute_user_historical_features(
    train_sessions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute user-level historical features from TRAINING data only.

    Features:
    - user_historical_purchase_rate: fraction of sessions with a purchase
    - user_avg_session_length: average session duration in seconds
    """
    user_stats = train_sessions.groupby("user_id").agg(
        user_historical_purchase_rate=("has_purchase", "mean"),
        user_avg_session_length=("session_duration_seconds", "mean"),
    ).reset_index()
    return user_stats


def merge_user_features(
    sessions: pd.DataFrame,
    user_stats: pd.DataFrame,
    global_purchase_rate: float,
    global_avg_session_length: float,
) -> pd.DataFrame:
    """
    Merge user-level historical features into session DataFrame.
    Users not seen in training get global averages (cold-start handling).
    """
    merged = sessions.merge(user_stats, on="user_id", how="left")
    merged["user_historical_purchase_rate"] = merged["user_historical_purchase_rate"].fillna(
        global_purchase_rate
    )
    merged["user_avg_session_length"] = merged["user_avg_session_length"].fillna(
        global_avg_session_length
    )
    return merged


def get_feature_columns() -> list:
    """Return the list of feature column names used for modeling."""
    return [
        "session_duration_seconds",
        "num_clicks",
        "num_distinct_categories",
        "add_to_cart_occurred",
        "user_historical_purchase_rate",
        "user_avg_session_length",
    ]


def chronological_train_test_split(
    sessions: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split sessions chronologically: earlier sessions go to train,
    later sessions go to test. This respects temporal ordering.

    Note: A user may appear in both train and test (which is realistic —
    we train on their early sessions and predict on later ones).
    User-level historical features are computed only from training sessions.
    """
    sessions_sorted = sessions.sort_values("session_start").reset_index(drop=True)
    split_idx = int(len(sessions_sorted) * (1 - test_fraction))
    train = sessions_sorted.iloc[:split_idx].copy()
    test = sessions_sorted.iloc[split_idx:].copy()
    logger.info(
        "Chronological split: %d train sessions, %d test sessions",
        len(train),
        len(test),
    )
    return train, test


def train_val_split_from_train(
    train_sessions: pd.DataFrame,
    val_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Further split training data into train and validation sets chronologically.
    Validation set is used for threshold tuning only — never for model selection
    or hyperparameter tuning (which uses inner CV).
    """
    train_sorted = train_sessions.sort_values("session_start").reset_index(drop=True)
    split_idx = int(len(train_sorted) * (1 - val_fraction))
    train_core = train_sorted.iloc[:split_idx].copy()
    val = train_sorted.iloc[split_idx:].copy()
    logger.info(
        "Train/val split: %d core train sessions, %d validation sessions",
        len(train_core),
        len(val),
    )
    return train_core, val


def tune_classification_threshold(
    model: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Tune the classification threshold on the validation set to maximize F1.
    This uses ONLY the validation set, never the test set.
    """
    proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_f1 = -1.0

    for thresh in thresholds:
        preds = (proba >= thresh).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = thresh

    logger.info(
        "Best threshold from validation set: %.3f (val F1=%.4f)",
        best_threshold,
        best_f1,
    )
    return float(best_threshold)


def predict_session_purchases(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    val_fraction: float = 0.15,
    n_estimators: int = 200,
    random_state: int = 42,
    class_weight: Optional[str] = "balanced",
) -> dict:
    """
    Full pipeline: feature engineering → chronological split → train RF → evaluate.

    Parameters
    ----------
    df : pd.DataFrame
        Raw event-level DataFrame with columns:
        'user_id', 'session_id', 'timestamp', 'event_type', 'page_category'
    test_fraction : float
        Fraction of sessions (by time) to hold out as test set.
    val_fraction : float
        Fraction of training sessions (by time) to use as validation set
        for threshold tuning.
    n_estimators : int
        Number of trees in the Random Forest.
    random_state : int
        Random seed for reproducibility.
    class_weight : str or None
        Class weight strategy for the Random Forest (e.g., 'balanced').

    Returns
    -------
    dict with keys:
        'model': trained RandomForestClassifier
        'test_f1': F1-score on test set
        'test_auc': ROC-AUC on test set
        'feature_importances': dict mapping feature name → importance
        'threshold': classification threshold tuned on validation set
        'scaler': fitted StandardScaler
    """
    # ------------------------------------------------------------------ #
    # 1. Validate input
    # ------------------------------------------------------------------ #
    validate_input_dataframe(df)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ------------------------------------------------------------------ #
    # 2. Build session-level features (no leakage: uses raw events only)
    # ------------------------------------------------------------------ #
    logger.info("Building session-level features...")
    sessions = build_session_features(df)

    if len(sessions) < 10:
        raise ValueError(
            f"Too few sessions ({len(sessions)}) to build a meaningful model."
        )

    # ------------------------------------------------------------------ #
    # 3. Chronological train/test split BEFORE any user-level feature
    #    computation. This is the critical leakage-prevention step.
    # ------------------------------------------------------------------ #
    logger.info("Performing chronological train/test split...")
    train_sessions, test_sessions = chronological_train_test_split(
        sessions, test_fraction=test_fraction
    )

    # Further split training into core train + validation
    train_core, val_sessions = train_val_split_from_train(
        train_sessions, val_fraction=val_fraction
    )

    # ------------------------------------------------------------------ #
    # 4. Compute user-level historical features ONLY from core training data
    # ------------------------------------------------------------------ #
    logger.info("Computing user-level historical features from training data only...")
    user_stats = compute_user_historical_features(train_core)

    # Global fallback statistics (computed from training only)
    global_purchase_rate = float(train_core["has_purchase"].mean())
    global_avg_session_length = float(train_core["session_duration_seconds"].mean())

    logger.info(
        "Global training purchase rate: %.4f, avg session length: %.2f s",
        global_purchase_rate,
        global_avg_session_length,
    )

    # ------------------------------------------------------------------ #
    # 5. Merge user features into each split
    # ------------------------------------------------------------------ #
    train_core = merge_user_features(
        train_core, user_stats, global_purchase_rate, global_avg_session_length
    )
    val_sessions = merge_user_features(
        val_sessions, user_stats, global_purchase_rate, global_avg_session_length
    )
    test_sessions = merge_user_features(
        test_sessions, user_stats, global_purchase_rate, global_avg_session_length
    )

    feature_cols = get_feature_columns()

    X_train = train_core[feature_cols].values.astype(np.float64)
    y_train = train_core["has_purchase"].values.astype(int)

    X_val = val_sessions[feature_cols].values.astype(np.float64)
    y_val = val_sessions["has_purchase"].values.astype(int)

    X_test = test_sessions[feature_cols].values.astype(np.float64)
    y_test = test_sessions["has_purchase"].values.astype(int)

    # ------------------------------------------------------------------ #
    # 6. Fit scaler on training data only
    # ------------------------------------------------------------------ #
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------ #
    # 7. Train Random Forest
    #    Hyperparameters are fixed here; for tuning use GridSearchCV with
    #    inner CV on training data only, then report on the held-out test set.
    # ------------------------------------------------------------------ #
    logger.info("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------ #
    # 8. Tune classification threshold on VALIDATION set only
    # ------------------------------------------------------------------ #
    logger.info("Tuning classification threshold on validation set...")
    if len(np.unique(y_val)) < 2:
        logger.warning(
            "Validation set has only one class; using default threshold 0.5."
        )
        best_threshold = 0.5
    else:
        best_threshold = tune_classification_threshold(model, X_val_scaled, y_val)

    # ------------------------------------------------------------------ #
    # 9. Final evaluation on TEST set (untouched until this point)
    # ------------------------------------------------------------------ #
    logger.info("Evaluating on test set...")
    test_proba = model.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_proba >= best_threshold).astype(int)

    if len(np.unique(y_test)) < 2:
        logger.warning(
            "Test set has only one class present; AUC is undefined. Returning NaN."
        )
        test_auc = float("nan")
    else:
        test_auc = float(roc_auc_score(y_test, test_proba))

    test_f1 = float(f1_score(y_test, test_preds, zero_division=0))

    logger.info("Test F1: %.4f | Test AUC: %.4f", test_f1, test_auc)

    # ------------------------------------------------------------------ #
    # 10. Feature importances
    # ------------------------------------------------------------------ #
    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    logger.info("Feature importances: %s", importances)

    # ------------------------------------------------------------------ #
    # 11. Log class distribution for transparency
    # ------------------------------------------------------------------ #
    train_purchase_rate = float(y_train.mean())
    test_purchase_rate = float(y_test.mean())
    logger.info(
        "Purchase rate — train: %.4f, test: %.4f",
        train_purchase_rate,
        test_purchase_rate,
    )

    return {
        "model": model,
        "test_f1": test_f1,
        "test_auc": test_auc,
        "feature_importances": importances,
        "threshold": best_threshold,
        "scaler": scaler,
    }


def predict_new_sessions(
    new_df: pd.DataFrame,
    model: RandomForestClassifier,
    scaler: StandardScaler,
    user_stats: pd.DataFrame,
    global_purchase_rate: float,
    global_avg_session_length: float,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Score new (unseen) sessions using a previously trained model and scaler.

    Parameters
    ----------
    new_df : pd.DataFrame
        Raw event-level DataFrame for new sessions.
    model : RandomForestClassifier
        Trained model returned by predict_session_purchases.
    scaler : StandardScaler
        Fitted scaler returned by predict_session_purchases.
    user_stats : pd.DataFrame
        User-level historical features computed from training data.
    global_purchase_rate : float
        Fallback purchase rate for unseen users.
    global_avg_session_length : float
        Fallback average session length for unseen users.
    threshold : float
        Classification threshold (tuned on validation set).

    Returns
    -------
    pd.DataFrame with columns: session_id, purchase_probability, predicted_purchase
    """
    validate_input_dataframe(new_df)
    sessions = build_session_features(new_df)
    sessions = merge_user_features(
        sessions, user_stats, global_purchase_rate, global_avg_session_length
    )

    feature_cols = get_feature_columns()
    X = sessions[feature_cols].values.astype(np.float64)
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[:, 1]
    preds = (proba >= threshold).astype(int)

    result = pd.DataFrame({
        "session_id": sessions["session_id"],
        "purchase_probability": proba,
        "predicted_purchase": preds,
    })
    return result


# --------------------------------------------------------------------------- #
# Example usage / smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import random

    rng = np.random.default_rng(42)
    random.seed(42)

    n_users = 50
    n_sessions = 500
    n_events = 3000

    user_ids = [f"u{i:03d}" for i in range(n_users)]
    session_ids = [f"s{i:04d}" for i in range(n_sessions)]

    # Assign each session to a user
    session_user_map = {
        sid: random.choice(user_ids) for sid in session_ids
    }

    base_time = pd.Timestamp("2024-01-01")
    event_types = ["click", "add_to_cart", "purchase"]
    event_weights = [0.70, 0.20, 0.10]
    categories = ["electronics", "clothing", "books", "home", "sports"]

    rows = []
    for i in range(n_events):
        sid = random.choice(session_ids)
        uid = session_user_map[sid]
        ts = base_time + pd.Timedelta(seconds=int(rng.integers(0, 86400 * 60)))
        etype = rng.choice(event_types, p=event_weights)
        cat = random.choice(categories)
        rows.append({
            "user_id": uid,
            "session_id": sid,
            "timestamp": ts,
            "event_type": etype,
            "page_category": cat,
        })

    sample_df = pd.DataFrame(rows)
    logger.info("Sample DataFrame shape: %s", sample_df.shape)

    results = predict_session_purchases(
        df=sample_df,
        test_fraction=0.2,
        val_fraction=0.15,
        n_estimators=100,
        random_state=42,
    )

    print("\n=== Results ===")
    print(f"Test F1-Score:  {results['test_f1']:.4f}")
    print(f"Test AUC:       {results['test_auc']:.4f}")
    print(f"Threshold:      {results['threshold']:.3f}")
    print("Feature Importances:")
    for feat, imp in sorted(
        results["feature_importances"].items(), key=lambda x: -x[1]
    ):
        print(f"  {feat:<35s}: {imp:.4f}")