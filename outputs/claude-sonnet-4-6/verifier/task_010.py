import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError("Please install imbalanced-learn: pip install imbalanced-learn")


def smote_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    smote_k_neighbors: int = 5,
    lr_max_iter: int = 1000,
    average: str = "binary",
) -> dict:
    """
    Oversample the minority class using SMOTE, train a Logistic Regression
    classifier, and report the test F1-score.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.
    smote_k_neighbors : int
        Number of nearest neighbours used in SMOTE.
    lr_max_iter : int
        Maximum number of iterations for Logistic Regression.
    average : str
        Averaging strategy for F1-score ('binary', 'macro', 'weighted', etc.).

    Returns
    -------
    dict
        Dictionary containing:
        - 'f1_score'          : F1-score on the test set.
        - 'class_distribution_before': class counts before SMOTE (train split).
        - 'class_distribution_after' : class counts after SMOTE.
        - 'model'             : trained LogisticRegression instance.
    """
    # ------------------------------------------------------------------ #
    # 1. Split BEFORE applying SMOTE (avoids data leakage)               #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    dist_before = {cls: int(np.sum(y_train == cls)) for cls in np.unique(y_train)}
    print(f"Class distribution BEFORE SMOTE (train): {dist_before}")

    # ------------------------------------------------------------------ #
    # 2. Scale features                                                   #
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------ #
    # 3. Apply SMOTE only on the training set                             #
    # ------------------------------------------------------------------ #
    smote = SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    dist_after = {
        cls: int(np.sum(y_train_resampled == cls))
        for cls in np.unique(y_train_resampled)
    }
    print(f"Class distribution AFTER  SMOTE (train): {dist_after}")

    # ------------------------------------------------------------------ #
    # 4. Train Logistic Regression                                        #
    # ------------------------------------------------------------------ #
    model = LogisticRegression(max_iter=lr_max_iter, random_state=random_state)
    model.fit(X_train_resampled, y_train_resampled)

    # ------------------------------------------------------------------ #
    # 5. Evaluate on the (untouched) test set                             #
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test_scaled)
    score = f1_score(y_test, y_pred, average=average)
    print(f"Test F1-score ({average}): {score:.4f}")

    return {
        "f1_score": score,
        "class_distribution_before": dist_before,
        "class_distribution_after": dist_after,
        "model": model,
        "scaler": scaler,
    }


# ------------------------------------------------------------------ #
# Demo / self-test                                                    #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Create a highly imbalanced binary dataset
    X_demo, y_demo = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.90, 0.10],   # 90 % majority, 10 % minority
        flip_y=0.01,
        random_state=42,
    )

    print("=" * 55)
    print("Binary classification demo (imbalance ratio 9:1)")
    print("=" * 55)
    results = smote_logistic_regression(X_demo, y_demo, average="binary")
    print(f"\nFinal F1-score: {results['f1_score']:.4f}")

    # Multi-class imbalanced example
    print("\n" + "=" * 55)
    print("Multi-class demo")
    print("=" * 55)
    X_multi, y_multi = make_classification(
        n_samples=1500,
        n_features=15,
        n_informative=8,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.70, 0.20, 0.10],
        random_state=0,
    )
    results_multi = smote_logistic_regression(
        X_multi, y_multi, average="macro", smote_k_neighbors=3
    )
    print(f"\nFinal macro F1-score: {results_multi['f1_score']:.4f}")