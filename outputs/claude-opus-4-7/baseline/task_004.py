import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def mean_target_encode_and_score(X_train, X_test, y_train, y_test, smoothing=10.0, random_state=42):
    """
    Apply mean target encoding to categorical features, train a gradient boosting
    classifier, and return the test ROC-AUC score.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and test feature sets.
    y_train, y_test : pd.Series or np.ndarray
        Training and test target values (binary).
    smoothing : float
        Smoothing factor for target encoding to handle low-frequency categories.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    float
        ROC-AUC score on the test set.
    """
    X_train = pd.DataFrame(X_train).copy()
    X_test = pd.DataFrame(X_test).copy()
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # Identify categorical columns
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    global_mean = y_train.mean()

    # Apply mean target encoding with smoothing for each categorical column
    for col in cat_cols:
        agg = y_train.groupby(X_train[col]).agg(["mean", "count"])
        counts = agg["count"]
        means = agg["mean"]
        smoothed = (counts * means + smoothing * global_mean) / (counts + smoothing)

        X_train[col] = X_train[col].map(smoothed).fillna(global_mean).astype(float)
        X_test[col] = X_test[col].map(smoothed).fillna(global_mean).astype(float)

    # Train gradient boosting classifier
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Predict probabilities and compute ROC-AUC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    return auc