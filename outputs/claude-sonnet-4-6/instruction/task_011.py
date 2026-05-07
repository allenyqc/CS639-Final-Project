import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from typing import Optional


def impute_and_train_knn(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
) -> dict:
    """
    Impute missing values using column means (fit on train only),
    optionally scale features, train a KNN classifier, and return metrics.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix, may contain NaN values.
    y : np.ndarray, shape (n_samples,)
        Target labels.
    n_neighbors : int
        Number of neighbours for KNN.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility.
    scale_features : bool
        Whether to standardise features after imputation.

    Returns
    -------
    dict with keys:
        'accuracy'   – test accuracy
        'model'      – fitted KNeighborsClassifier
        'imputer'    – fitted SimpleImputer  (use on new data)
        'scaler'     – fitted StandardScaler or None
    """
    # ------------------------------------------------------------------ #
    # 1. Train / test split FIRST — before any fitting                    #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ------------------------------------------------------------------ #
    # 2. Imputation — fit ONLY on training data                           #
    # ------------------------------------------------------------------ #
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)   # fit + transform train
    X_test_imputed  = imputer.transform(X_test)        # transform test only

    # ------------------------------------------------------------------ #
    # 3. (Optional) Scaling — fit ONLY on training data                   #
    # ------------------------------------------------------------------ #
    scaler: Optional[StandardScaler] = None
    if scale_features:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train_imputed)
        X_test_processed  = scaler.transform(X_test_imputed)
    else:
        X_train_processed = X_train_imputed
        X_test_processed  = X_test_imputed

    # ------------------------------------------------------------------ #
    # 4. Train KNN classifier                                             #
    # ------------------------------------------------------------------ #
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_processed, y_train)

    # ------------------------------------------------------------------ #
    # 5. Evaluate on held-out test set                                    #
    # ------------------------------------------------------------------ #
    y_pred   = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "model":    model,
        "imputer":  imputer,
        "scaler":   scaler,
    }


# ---------------------------------------------------------------------- #
# Quick smoke-test                                                        #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    rng = np.random.default_rng(0)
    data = load_iris()
    X, y = data.data.astype(float), data.target

    # Inject ~15 % missing values at random
    mask = rng.random(X.shape) < 0.15
    X[mask] = np.nan

    results = impute_and_train_knn(X, y, n_neighbors=5)
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Imputer means (train only): {results['imputer'].statistics_}")