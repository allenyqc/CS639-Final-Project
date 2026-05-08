import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def mean_target_encode_and_score(
    X_train,
    X_test,
    y_train,
    y_test,
    cat_features=None,
    n_splits=5,
    smoothing=10.0,
    random_state=42,
):
    """
    Apply out-of-fold mean target encoding to categorical features (fit on train only),
    train a GradientBoostingClassifier, and return test ROC-AUC.
    """
    X_train = pd.DataFrame(X_train).copy().reset_index(drop=True)
    X_test = pd.DataFrame(X_test).copy().reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    # Auto-detect categorical features if not specified
    if cat_features is None:
        cat_features = [
            c for c in X_train.columns
            if X_train[c].dtype == "object" or str(X_train[c].dtype).startswith("category")
        ]

    global_mean = y_train.mean()

    def smoothed_mean(stats_sum, stats_count, prior, m):
        return (stats_sum + m * prior) / (stats_count + m)

    # Out-of-fold target encoding on the training set to prevent leakage
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for col in cat_features:
        encoded_col = np.full(len(X_train), global_mean, dtype=float)
        for train_idx, val_idx in kf.split(X_train):
            fold_train = X_train.iloc[train_idx]
            fold_y = y_train.iloc[train_idx]
            agg = fold_y.groupby(fold_train[col]).agg(["sum", "count"])
            mapping = smoothed_mean(agg["sum"], agg["count"], global_mean, smoothing)
            encoded_col[val_idx] = (
                X_train.iloc[val_idx][col].map(mapping).fillna(global_mean).values
            )
        # Compute final mapping using full training data for test transform
        full_agg = y_train.groupby(X_train[col]).agg(["sum", "count"])
        full_mapping = smoothed_mean(full_agg["sum"], full_agg["count"], global_mean, smoothing)

        X_train[col] = encoded_col
        X_test[col] = X_test[col].map(full_mapping).fillna(global_mean).values

    # Ensure numeric dtypes
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(global_mean)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(global_mean)

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_proba)