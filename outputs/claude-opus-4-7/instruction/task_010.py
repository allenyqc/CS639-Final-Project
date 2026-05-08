from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE


def smote_logreg_f1(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> float:
    """
    Train a logistic regression classifier on a SMOTE-oversampled training set
    and return the F1-score on the held-out test set.

    Best practices applied:
    - Train/test split is performed BEFORE any preprocessing or resampling.
    - StandardScaler is fit ONLY on training data, then applied to test data.
    - SMOTE is applied ONLY to training data (never to test data).
    - F1-score is reported (appropriate for imbalanced classification).
    """
    # 1) Split first — no peeking at test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 2) Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3) Apply SMOTE ONLY on training data
    # k_neighbors must be < number of minority samples
    minority_count = np.min(np.bincount(y_train))
    k_neighbors = max(1, min(5, minority_count - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # 4) Train logistic regression on resampled training data
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train_res, y_train_res)

    # 5) Evaluate on the untouched test set
    y_pred = clf.predict(X_test_scaled)
    average = "binary" if len(np.unique(y)) == 2 else "macro"
    f1 = f1_score(y_test, y_pred, average=average)

    print(f"Test F1-score ({average}): {f1:.4f}")
    return f1


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        weights=[0.92, 0.08],
        random_state=42,
    )
    smote_logreg_f1(X, y)