from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def smote_logistic_regression_f1(
    X,
    y,
    test_size=0.2,
    random_state=42,
    smote_k_neighbors=5,
    logreg_kwargs=None,
):
    """
    Oversamples the minority class using SMOTE, trains a logistic regression
    classifier, and reports the test F1-score.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target labels.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    smote_k_neighbors : int, default=5
        Number of nearest neighbors to use in SMOTE.
    logreg_kwargs : dict, optional
        Extra keyword arguments to pass to LogisticRegression.

    Returns
    -------
    f1 : float
        F1-score on the test set.
    model : LogisticRegression
        The trained logistic regression model.
    """
    if logreg_kwargs is None:
        logreg_kwargs = {"max_iter": 1000, "solver": "lbfgs"}

    # Stratified train/test split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Apply SMOTE only to the training data to avoid leakage
    smote = SMOTE(random_state=random_state, k_neighbors=smote_k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train logistic regression on the resampled data
    model = LogisticRegression(random_state=random_state, **logreg_kwargs)
    model.fit(X_train_res, y_train_res)

    # Evaluate on the untouched test set
    y_pred = model.predict(X_test)

    # Use binary if there are exactly two classes, otherwise weighted
    average = "binary" if len(set(y)) == 2 else "weighted"
    f1 = f1_score(y_test, y_pred, average=average)

    print(f"Test F1-score ({average}): {f1:.4f}")
    return f1, model


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        weights=[0.92, 0.08],
        random_state=42,
    )

    smote_logistic_regression_f1(X, y)