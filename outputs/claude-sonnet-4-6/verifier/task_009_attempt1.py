```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


def train_multiclass_classifier(
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    plot_results: bool = True,
) -> dict:
    """
    Train a multi-class classifier on a 10-class dataset and evaluate its performance.

    Args:
        model_type: Type of classifier ('random_forest', 'logistic_regression', 'svm')
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        plot_results: Whether to plot confusion matrix and metrics

    Returns:
        Dictionary containing model, metrics, and evaluation results
    """
    # -------------------------------------------------------------------------
    # 1. Load and prepare the dataset (digits dataset: 10 classes, 0-9)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("MULTI-CLASS CLASSIFIER TRAINING & EVALUATION")
    print("=" * 60)

    digits = load_digits()
    X, y = digits.data, digits.target
    class_names = [str(i) for i in range(10)]

    print(f"\nDataset: Digits (0-9)")
    print(f"Total samples  : {X.shape[0]}")
    print(f"Features       : {X.shape[1]}")
    print(f"Classes        : {len(np.unique(y))} ({np.unique(y).tolist()})")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # -------------------------------------------------------------------------
    # 2. Split data
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain samples  : {X_train.shape[0]}")
    print(f"Test  samples  : {X_test.shape[0]}")

    # -------------------------------------------------------------------------
    # 3. Preprocessing
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------------------
    # 4. Select and train model
    # -------------------------------------------------------------------------
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, multi_class="multinomial", solver="lbfgs",
            random_state=random_state, C=1.0
        ),
        "svm": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=random_state
        ),
    }

    if model_type not in models:
        raise ValueError(f"model_type must be one of {list(models.keys())}")

    model = models[model_type]
    print(f"\nModel          : {model_type.replace('_', ' ').title()}")
    print(f"Parameters     : {model.get_params()}")

    print("\nTraining model...")
    model.fit(X_train_scaled, y_train)

    # -------------------------------------------------------------------------
    # 5. Predictions
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)  # shape: (n_samples, 10)

    # -------------------------------------------------------------------------
    # 6. Evaluation metrics
    # -------------------------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_per_class = f1_score(y_test, y_pred, average=None)

    # ROC-AUC (one-vs-rest, macro average)
    y_test_bin = label_binarize(y_test, classes=list(range(10)))
    roc_auc_macro = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
    roc_auc_weighted = roc_auc_score(
        y_test_bin, y_prob, multi_class="ovr", average="weighted"
    )

    # Cross-validation accuracy (5-fold)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTest Accuracy          : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score (Macro)       : {f1_macro:.4f}")
    print(f"F1-Score (Weighted)    : {f1_weighted:.4f}")
    print(f"ROC-AUC (Macro OvR)    : {roc_auc_macro:.4f}")
    print(f"ROC-AUC (Weighted OvR) : {roc_auc_weighted:.4f}")
    print(f"CV Accuracy (5-fold)   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\nPer-Class F1-Scores:")
    for cls, score in enumerate(f1_per_class):
        print(f"  Class {cls}: {score:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # -------------------------------------------------------------------------
    # 7. Confusion matrix
    # -------------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    if plot_results:
        _plot_results(cm, class_names, f1_per_class, cv_scores, model_type)

    # -------------------------------------------------------------------------
    # 8. Return results
    # -------------------------------------------------------------------------
    results = {
        "model": model,
        "scaler": scaler,
        "metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_per_class": f1_per_class.tolist(),
            "roc_auc_macro": roc_auc_macro,
            "roc_auc_weighted": roc_auc_weighted,
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
        },
        "confusion_matrix": cm,
        "predictions": y_pred,
        "probabilities": y_prob,
        "data": {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
        },
    }
    return results


def _plot_results(
    cm: np.ndarray,
    class_names: list,
    f1_per_class: np.ndarray,
    cv_scores: np.ndarray,
    model_type: str,
) -> None