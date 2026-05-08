from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def pca_svm_accuracy(X, y, n_components=10, test_size=0.2, random_state=42):
    """
    Reduce dataset to n_components principal components, train an SVM,
    and return the test accuracy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Raw feature matrix.
    y : array-like of shape (n_samples,)
        Target labels.
    n_components : int, default=10
        Number of principal components to keep.
    test_size : float, default=0.2
        Fraction of data used for testing.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    float
        Test accuracy of the SVM classifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        ("svm", SVC()),
    ])

    pipeline.fit(X_train, y_train)
    return pipeline.score(X_test, y_test)