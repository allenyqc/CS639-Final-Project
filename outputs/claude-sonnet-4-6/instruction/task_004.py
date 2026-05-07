import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def mean_target_encoding_and_gradient_boosting(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical_cols: list = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """
    Applies mean target encoding to categorical features, trains a gradient
    boosting classifier, and returns the test ROC-AUC score.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.
    categorical_cols : list, optional
        List of categorical column names. If None, auto-detected from dtypes.
    n_splits : int
        Number of folds for out-of-fold target encoding on training data.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    float
        ROC-AUC score on the test set.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()

    # Auto-detect categorical columns if not provided
    if categorical_cols is None:
        categorical_cols = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    # ------------------------------------------------------------------ #
    # Mean Target Encoding                                                 #
    # ------------------------------------------------------------------ #
    # Training data: out-of-fold encoding to prevent target leakage
    # Test data   : encoding derived solely from full training set statistics

    # Global mean used as fallback for unseen categories
    global_mean = y_train.mean()

    # Dictionary to store per-column encoding maps (fitted on full train set)
    encoding_maps: dict = {}

    # Initialise encoded training columns with NaN
    train_encoded = X_train.copy()
    for col in categorical_cols:
        train_encoded[col] = np.nan

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for col in categorical_cols:
        oof_encoded = np.full(len(X_train), np.nan)

        for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
            # Compute mean target per category using only the fold's training rows
            fold_map = (
                y_train.iloc[tr_idx]
                .groupby(X_train[col].iloc[tr_idx])
                .mean()
            )
            # Map validation rows; unseen categories get global mean
            oof_encoded[val_idx] = (
                X_train[col].iloc[val_idx].map(fold_map).fillna(global_mean).values
            )

        train_encoded[col] = oof_encoded

        # Full-training encoding map for test set (fitted on ALL training data)
        full_map = y_train.groupby(X_train[col]).mean()
        encoding_maps[col] = full_map

    # Encode test set using the full-training encoding maps
    test_encoded = X_test.copy()
    for col in categorical_cols:
        test_encoded[col] = (
            X_test[col].map(encoding_maps[col]).fillna(global_mean)
        )

    # Ensure all columns are numeric (drop any remaining non-numeric columns)
    train_encoded = train_encoded.select_dtypes(include=[np.number])
    test_encoded = test_encoded[train_encoded.columns]

    # Fill any remaining NaNs with column medians computed on training data
    col_medians = train_encoded.median()
    train_encoded = train_encoded.fillna(col_medians)
    test_encoded = test_encoded.fillna(col_medians)

    # ------------------------------------------------------------------ #
    # Gradient Boosting Classifier                                         #
    # ------------------------------------------------------------------ #
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=random_state,
    )
    clf.fit(train_encoded, y_train)

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    y_prob = clf.predict_proba(test_encoded)

    # Handle binary and multiclass scenarios
    if y_prob.shape[1] == 2:
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        roc_auc = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="macro"
        )

    print(f"Test ROC-AUC Score: {roc_auc:.4f}")
    return roc_auc


# ------------------------------------------------------------------ #
# Example usage                                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Synthetic dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=6,
        weights=[0.8, 0.2],   # imbalanced
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"num_{i}" for i in range(10)])
    y = pd.Series(y, name="target")

    # Add a synthetic categorical column
    categories = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    X["cat_feature"] = rng.choice(categories, size=len(X))

    # Split BEFORE any feature engineering
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    score = mean_target_encoding_and_gradient_boosting(
        X_train, X_test, y_train, y_test,
        categorical_cols=["cat_feature"],
    )
    print(f"Returned ROC-AUC: {score:.4f}")