import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def pca_svm_classifier(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> float:
    """
    Reduces dataset to 10 principal components using PCA, trains an SVM classifier,
    and returns the test accuracy.

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
        Test accuracy of the trained SVM classifier.

    Notes
    -----
    - Train/test split is performed BEFORE any preprocessing.
    - StandardScaler and PCA are fit ONLY on training data, then applied to test data.
    - n_components is capped at min(10, n_features, n_train_samples) to avoid errors.
    """
    # Step 1: Split BEFORE any preprocessing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 2: Determine safe number of PCA components
    n_components = min(10, X_train.shape[1], X_train.shape[0])

    # Step 3: Build a pipeline — ensures all steps are fit only on training data
    pipeline = Pipeline([
        ("scaler", StandardScaler()),          # Normalize features (fit on train only)
        ("pca", PCA(n_components=n_components, random_state=random_state)),  # Reduce dimensions
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state))  # SVM classifier
    ])

    # Step 4: Fit the entire pipeline on training data only
    pipeline.fit(X_train, y_train)

    # Step 5: Evaluate on held-out test data
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    return test_accuracy


# ---------------------------------------------------------------------------
# Example usage / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X, y = data.data, data.target

    accuracy = pca_svm_classifier(X, y)
    print(f"Test Accuracy: {accuracy:.4f}")