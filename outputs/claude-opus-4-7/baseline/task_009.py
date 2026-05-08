import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


def train_and_evaluate_multiclass_classifier(
    X=None,
    y=None,
    test_size=0.2,
    random_state=42,
):
    """
    Train a multi-class classifier on a 10-class dataset and evaluate it.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), optional
        Feature matrix. If None, the sklearn `digits` dataset (10 classes) is used.
    y : array-like of shape (n_samples,), optional
        Target vector with 10 classes.
    test_size : float, default=0.2
        Fraction of samples to use for the test split.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary with the trained model, predictions, and evaluation metrics.
    """
    # Load default 10-class dataset if data not provided
    if X is None or y is None:
        data = load_digits()
        X, y = data.data, data.target

    X = np.asarray(X)
    y = np.asarray(y)

    n_classes = len(np.unique(y))
    if n_classes != 10:
        raise ValueError(
            f"Expected a 10-class dataset, but got {n_classes} classes."
        )

    # Stratified split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pipeline: scale features then fit a multinomial logistic regression
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # For multi-class classification, accuracy and macro-F1 are appropriate.
    # Macro-F1 is robust to class imbalance.
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    train_and_evaluate_multiclass_classifier()