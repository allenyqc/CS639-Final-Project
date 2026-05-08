import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
)


def train_imbalanced_classifier(
    X=None,
    y=None,
    test_size=0.2,
    random_state=42,
    threshold=None,
):
    """
    Train a binary classifier on an imbalanced dataset and report performance.

    Parameters
    ----------
    X : array-like, optional
        Feature matrix. If None, a synthetic imbalanced dataset is generated.
    y : array-like, optional
        Binary target vector. If None, a synthetic imbalanced dataset is generated.
    test_size : float
        Fraction of data to use for the test set.
    random_state : int
        Random seed for reproducibility.
    threshold : float, optional
        Custom decision threshold. If None, the threshold maximizing F1 on
        the training set is selected.

    Returns
    -------
    dict
        Dictionary containing the fitted model, chosen threshold, and metrics.
    """
    # Generate a synthetic imbalanced dataset if none is provided
    if X is None or y is None:
        X, y = make_classification(
            n_samples=10_000,
            n_features=20,
            n_informative=5,
            n_redundant=2,
            n_clusters_per_class=2,
            weights=[0.95, 0.05],  # ~5% positives -> imbalanced
            flip_y=0.01,
            random_state=random_state,
        )

    # Stratified split to preserve the class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Report class distribution
    pos_train = int(np.sum(y_train == 1))
    neg_train = int(np.sum(y_train == 0))
    print(f"Training class distribution -> 0: {neg_train}, 1: {pos_train} "
          f"(positive rate: {pos_train / len(y_train):.2%})")

    # Pipeline: scaling + classifier with class_weight='balanced' to handle imbalance
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # Predict probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Choose decision threshold by maximizing F1 on the training set
    if threshold is None:
        candidate_thresholds = np.linspace(0.01, 0.99, 99)
        f1_scores = [
            f1_score(y_train, (y_train_proba >= t).astype(int), zero_division=0)
            for t in candidate_thresholds
        ]
        best_idx = int(np.argmax(f1_scores))
        threshold = float(candidate_thresholds[best_idx])

    y_test_pred = (y_test_proba >= threshold).astype(int)

    # Threshold-independent metrics
    roc_auc = roc_auc_score(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)

    print("\n=== Model Performance on Test Set ===")
    print(f"Chosen decision threshold: {threshold:.3f}")
    print(f"ROC AUC:              {roc_auc:.4f}")
    print(f"PR AUC (Avg. Prec.):  {pr_auc:.4f}")
    print(f"Balanced Accuracy:    {bal_acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    return {
        "model": model,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": bal_acc,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "y_test_proba": y_test_proba,
    }


if __name__ == "__main__":
    results = train_imbalanced_classifier()