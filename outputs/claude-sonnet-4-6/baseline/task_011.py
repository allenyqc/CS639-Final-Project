import numpy as np
from collections import Counter


def impute_and_classify(X_train, y_train, X_test, y_test, k=5):
    """
    Imputes missing values using column means from the training set,
    trains a k-nearest-neighbours classifier, and returns the test accuracy.

    Parameters
    ----------
    X_train : array-like of shape (n_train_samples, n_features)
        Training feature matrix, may contain np.nan values.
    y_train : array-like of shape (n_train_samples,)
        Training labels.
    X_test : array-like of shape (n_test_samples, n_features)
        Test feature matrix, may contain np.nan values.
    y_test : array-like of shape (n_test_samples,)
        True test labels.
    k : int, optional (default=5)
        Number of nearest neighbours to use.

    Returns
    -------
    accuracy : float
        Classification accuracy on the test set.
    """
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Compute column means from training data (ignoring NaNs)
    col_means = np.nanmean(X_train, axis=0)

    # Impute missing values in training set
    train_nan_mask = np.isnan(X_train)
    X_train_imputed = X_train.copy()
    for col_idx in range(X_train.shape[1]):
        X_train_imputed[train_nan_mask[:, col_idx], col_idx] = col_means[col_idx]

    # Impute missing values in test set using training column means
    test_nan_mask = np.isnan(X_test)
    X_test_imputed = X_test.copy()
    for col_idx in range(X_test.shape[1]):
        X_test_imputed[test_nan_mask[:, col_idx], col_idx] = col_means[col_idx]

    # KNN prediction
    def predict_knn(X_tr, y_tr, X_te, k):
        predictions = []
        for test_point in X_te:
            # Euclidean distances to all training points
            distances = np.sqrt(np.sum((X_tr - test_point) ** 2, axis=1))
            # Indices of k nearest neighbours
            nn_indices = np.argsort(distances)[:k]
            # Majority vote
            nn_labels = y_tr[nn_indices]
            most_common = Counter(nn_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

    y_pred = predict_knn(X_train_imputed, y_train, X_test_imputed, k)

    accuracy = np.mean(y_pred == y_test)
    return accuracy


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)

    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Introduce ~15 % missing values artificially
    X_train_missing = X_train.copy().astype(float)
    X_test_missing = X_test.copy().astype(float)

    mask_train = rng.random(X_train_missing.shape) < 0.15
    mask_test = rng.random(X_test_missing.shape) < 0.15
    X_train_missing[mask_train] = np.nan
    X_test_missing[mask_test] = np.nan

    acc = impute_and_classify(X_train_missing, y_train, X_test_missing, y_test, k=5)
    print(f"Test accuracy: {acc:.4f}")