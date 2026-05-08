from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def smote_logistic_f1(
    X,
    y,
    test_size=0.2,
    random_state=42,
    smote_kwargs=None,
    logreg_kwargs=None,
):
    """
    Oversample the minority class with SMOTE, train a logistic regression
    classifier, and report the test F1-score.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target labels.
    test_size : float, default=0.2
        Proportion of the dataset reserved for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    smote_kwargs : dict, optional
        Extra keyword arguments forwarded to SMOTE.
    logreg_kwargs : dict, optional
        Extra keyword arguments forwarded to LogisticRegression.

    Returns
    -------
    f1 : float
        F1-score on the held-out test set.
    model : LogisticRegression
        The fitted logistic regression model.
    """
    smote_kwargs = smote_kwargs or {}
    logreg_kwargs = logreg_kwargs or {}

    # Split first to avoid leaking synthetic samples into the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Oversample the minority class on the training set only
    smote = SMOTE(random_state=random_state, **smote_kwargs)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train logistic regression
    model = LogisticRegression(
        max_iter=1000, random_state=random_state, **logreg_kwargs
    )
    model.fit(X_train_res, y_train_res)

    # Evaluate on the original (untouched) test set
    y_pred = model.predict(X_test)
    average = "binary" if len(set(y)) == 2 else "macro"
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
        weights=[0.9, 0.1],
        flip_y=0.01,
        random_state=42,
    )
    smote_logistic_f1(X, y)