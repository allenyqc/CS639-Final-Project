import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def impute_and_knn_accuracy(
    X,
    y,
    n_neighbors=5,
    test_size=0.2,
    random_state=42,
    scale=True,
):
    """
    Impute missing values with column means, train a KNN classifier, and
    return test accuracy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix, possibly containing NaNs.
    y : array-like of shape (n_samples,)
        Target labels.
    n_neighbors : int, default=5
        Number of neighbors for KNN.
    test_size : float, default=0.2
        Proportion of data used for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    scale : bool, default=True
        Whether to standardize features before KNN.

    Returns
    -------
    accuracy : float
        Test set classification accuracy.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    rng = np.random.default_rng(0)
    data = load_iris()
    X, y = data.data.copy(), data.target

    mask = rng.random(X.shape) < 0.1
    X[mask] = np.nan

    acc = impute_and_knn_accuracy(X, y)
    print(f"Test accuracy: {acc:.4f}")