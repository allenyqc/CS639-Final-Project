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
    test_size=0.25,
    random_state=42,
    threshold=0.5,
    verbose=True,
):
    """
    Train a binary classifier on an imbalanced dataset and report performance.

    Parameters
    ----------
    X : array-like, optional
        Feature matrix. If None, a synthetic imbalanced dataset is generated.
    y : array-like, optional
        Binary target vector. If None, generated alongside X.
    test_size : float
        Fraction of data held out for testing.
    random_state : int
        Reproducibility seed.
    threshold : float
        Probability threshold for converting probabilities to class labels.
    verbose : bool
        Whether to print the performance report.

    Returns
    -------
    model : fitted sklearn Pipeline
    metrics : dict of evaluation metrics
    """
    # Generate a synthetic imbalanced dataset if none is provided
    if X is None or y is None:
        X, y = make_classification(
            n_samples=10000,
            n_features=20,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            weights=[0.95, 0.05],  # ~5% positive class
            flip_y=0.01,
            random_state=random_state,
        )

    # Stratified train/test split preserves class balance in both subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Use class_weight='balanced' to handle class imbalance internally
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Predict probabilities and apply threshold
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Collect metrics suited for imbalanced classification
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, digits=4
        ),
    }

    if verbose:
        class_counts = np.bincount(y)
        print("Class distribution (full dataset):")
        for cls, count in enumerate(class_counts):
            print(f"  Class {cls}: {count} ({count / len(y):.2%})")
        print(f"\nDecision threshold: {threshold}")
        print(f"ROC AUC:           {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        print(f"F1 (positive):     {metrics['f1']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print("\nConfusion matrix:")
        print(metrics["confusion_matrix"])
        print("\nClassification report:")
        print(metrics["classification_report"])

    return pipeline, metrics


if __name__ == "__main__":
    model, results = train_imbalanced_classifier()