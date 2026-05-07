import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


def train_with_smote(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    smote_k_neighbors: int = 5,
    lr_max_iter: int = 1000,
    f1_average: str = "binary",
) -> dict:
    """
    Handle class imbalance with SMOTE, train logistic regression, and report F1.

    Pipeline (best-practice order):
        1. Train/test split  (no data leakage from test set)
        2. Fit scaler on TRAIN only, transform both splits
        3. Apply SMOTE on TRAIN only  (never oversample test data)
        4. Train logistic regression on the resampled training set
        5. Evaluate on the original (unmodified) test set

    Parameters
    ----------
    X              : Feature matrix (n_samples, n_features)
    y              : Target vector (n_samples,)
    test_size      : Fraction of data held out for testing
    random_state   : Reproducibility seed
    smote_k_neighbors : Number of nearest neighbours used by SMOTE
    lr_max_iter    : Max iterations for logistic regression solver
    f1_average     : Averaging strategy for F1 ('binary', 'macro', 'weighted')

    Returns
    -------
    dict with keys: f1_score, classification_report, confusion_matrix,
                    class_distribution_before, class_distribution_after
    """

    # ------------------------------------------------------------------ #
    # 1. Train / test split — FIRST, before any preprocessing             #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("=== Class distribution (train, before SMOTE) ===")
    unique, counts = np.unique(y_train, return_counts=True)
    dist_before = dict(zip(unique.tolist(), counts.tolist()))
    for cls, cnt in dist_before.items():
        print(f"  Class {cls}: {cnt} samples ({cnt / len(y_train):.2%})")

    # ------------------------------------------------------------------ #
    # 2. Scale features — fit ONLY on training data                       #
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform train
    X_test_scaled  = scaler.transform(X_test)        # transform only test

    # ------------------------------------------------------------------ #
    # 3. SMOTE — applied ONLY to training data                            #
    # ------------------------------------------------------------------ #
    smote = SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled, y_train
    )

    print("\n=== Class distribution (train, after SMOTE) ===")
    unique_r, counts_r = np.unique(y_train_resampled, return_counts=True)
    dist_after = dict(zip(unique_r.tolist(), counts_r.tolist()))
    for cls, cnt in dist_after.items():
        print(f"  Class {cls}: {cnt} samples ({cnt / len(y_train_resampled):.2%})")

    # ------------------------------------------------------------------ #
    # 4. Train logistic regression on the resampled training set          #
    # ------------------------------------------------------------------ #
    clf = LogisticRegression(
        max_iter=lr_max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    clf.fit(X_train_resampled, y_train_resampled)

    # ------------------------------------------------------------------ #
    # 5. Evaluate on the ORIGINAL (untouched) test set                    #
    # ------------------------------------------------------------------ #
    y_pred = clf.predict(X_test_scaled)

    test_f1   = f1_score(y_test, y_pred, average=f1_average)
    report    = classification_report(y_test, y_pred)
    conf_mat  = confusion_matrix(y_test, y_pred)

    print(f"\n=== Test F1-score ({f1_average}) ===")
    print(f"  F1 = {test_f1:.4f}")
    print("\n=== Classification Report ===")
    print(report)
    print("=== Confusion Matrix ===")
    print(conf_mat)

    return {
        "f1_score": test_f1,
        "classification_report": report,
        "confusion_matrix": conf_mat,
        "class_distribution_before": dist_before,
        "class_distribution_after": dist_after,
    }


# ---------------------------------------------------------------------- #
# Demo — synthetic imbalanced dataset                                     #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    X_demo, y_demo = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.90, 0.10],   # 90 % majority / 10 % minority
        flip_y=0.01,
        random_state=42,
    )

    results = train_with_smote(
        X_demo,
        y_demo,
        test_size=0.2,
        random_state=42,
        smote_k_neighbors=5,
        lr_max_iter=1000,
        f1_average="binary",
    )

    print(f"\nFinal test F1-score: {results['f1_score']:.4f}")