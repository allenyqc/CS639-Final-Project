import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def mean_target_encode_and_score(X_train, X_test, y_train, y_test, cat_col=None, smoothing=10.0, random_state=42):
    """
    Apply mean target encoding to a categorical feature, train a gradient boosting
    classifier, and return the test ROC-AUC score.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Training and test feature sets.
    y_train, y_test : array-like
        Training and test target labels (binary).
    cat_col : str, optional
        Name of the categorical column to encode. If None, the first
        object/category dtype column is used.
    smoothing : float
        Smoothing factor for blending category mean with global mean.
    random_state : int
        Random state for the classifier.

    Returns
    -------
    float
        ROC-AUC score on the test set.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    # Identify categorical column if not specified
    if cat_col is None:
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) == 0:
            raise ValueError("No categorical column found; please specify `cat_col`.")
        cat_col = cat_cols[0]

    # Compute smoothed mean target encoding from training data
    global_mean = y_train.mean()
    train_cat = X_train[cat_col].astype(str).reset_index(drop=True)
    stats = y_train.groupby(train_cat).agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    encoding_map = smoothed.to_dict()

    # Apply encoding
    X_train[cat_col] = train_cat.map(encoding_map).fillna(global_mean).values
    X_test[cat_col] = X_test[cat_col].astype(str).map(encoding_map).fillna(global_mean).values

    # Keep only numeric columns for the model
    X_train_num = X_train.select_dtypes(include=[np.number])
    X_test_num = X_test[X_train_num.columns]

    # Train gradient boosting classifier
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train_num, y_train)

    # Predict and compute ROC-AUC
    y_proba = model.predict_proba(X_test_num)[:, 1]
    return roc_auc_score(y_test, y_proba)