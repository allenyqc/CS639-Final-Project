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
        Proportion of the dataset to include in the test split (default: 0.2).
    random_state : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    float
        Test accuracy of the SVM classifier trained on PCA-reduced features.

    Raises
    ------
    ValueError
        If the number of features is less than 10 or if the dataset is too small.
    """
    n_samples, n_features = X.shape
    n_components = 10

    if n_features < n_components:
        raise ValueError(
            f"Number of features ({n_features}) must be >= n_components ({n_components})."
        )

    if n_samples < 2:
        raise ValueError("Dataset must have at least 2 samples.")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 1: Standardize the features (important before PCA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 2: Apply PCA to reduce to 10 components
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Explained variance by {n_components} components: {explained_variance:.2f}%")

    # Step 3: Train an SVM classifier on the reduced features
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)
    svm.fit(X_train_pca, y_train)

    # Step 4: Evaluate on the test set
    y_pred = svm.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {test_accuracy:.4f}")

    return test_accuracy


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_digits

    # Load a sample dataset with more than 10 features
    digits = load_digits()
    X_raw, y_raw = digits.data, digits.target  # shape: (1797, 64)

    accuracy = pca_svm_classifier(X_raw, y_raw)
    print(f"Final test accuracy: {accuracy:.4f}")