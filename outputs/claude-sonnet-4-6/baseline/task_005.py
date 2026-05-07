import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def pca_svm_classifier(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> float:
    """
    Reduces a dataset to 10 principal components using PCA, trains an SVM classifier
    on the reduced features, and returns the test accuracy.

    Parameters
    ----------
    X : np.ndarray
        Raw feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target labels of shape (n_samples,).
    test_size : float, optional
        Proportion of the dataset to include in the test split (default 0.2).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    float
        Test accuracy of the SVM classifier trained on PCA-reduced features.

    Raises
    ------
    ValueError
        If the number of features is less than 10 (after checking n_components).
    """
    n_components = min(10, X.shape[1], X.shape[0])
    if n_components < 1:
        raise ValueError("Dataset must have at least 1 sample and 1 feature.")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features before PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA to reduce to n_components principal components
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Train SVM classifier on the PCA-reduced training data
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)
    svm.fit(X_train_pca, y_train)

    # Predict on the test set and compute accuracy
    y_pred = svm.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


# ---------------------------------------------------------------------------
# Quick smoke-test (runs only when the script is executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_digits, load_iris, make_classification

    # Test 1: digits dataset (64 features → 10 PCs)
    digits = load_digits()
    acc = pca_svm_classifier(digits.data, digits.target)
    print(f"Digits dataset  – Test accuracy: {acc:.4f}")

    # Test 2: iris dataset (4 features → min(10,4)=4 PCs)
    iris = load_iris()
    acc = pca_svm_classifier(iris.data, iris.target)
    print(f"Iris dataset    – Test accuracy: {acc:.4f}")

    # Test 3: synthetic dataset (20 features → 10 PCs)
    X_syn, y_syn = make_classification(
        n_samples=500, n_features=20, n_informative=15, random_state=0
    )
    acc = pca_svm_classifier(X_syn, y_syn)
    print(f"Synthetic data  – Test accuracy: {acc:.4f}")