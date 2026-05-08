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
    Impute missing values with column means, train a KNN classifier,
    and return the test accuracy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix, possibly containing NaN values.
    y : array-like of shape (n_samples,)
        Target labels.
    n_neighbors : int, default=5
        Number of neighbors for KNN.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed for reproducibility.
    scale : bool, default=True
        Whether to standardize features before fitting KNN.

    Returns
    -------
    accuracy : float
        Test accuracy of the trained KNN classifier.
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

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    return model.score(X_test, y_test)