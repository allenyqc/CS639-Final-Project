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
        test_size: Proportion of dataset to include in the test split
        random_state: Random state for reproducibility
        plot_results: Whether to plot confusion matrix and metrics

    Returns:
        Dictionary containing trained model, metrics, and evaluation results
    """
    # -------------------------------------------------------------------------
    # 1. Load and prepare the dataset (digits dataset has 10 classes: 0-9)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("MULTI-CLASS CLASSIFIER TRAINING & EVALUATION")
    print("=" * 60)

    digits = load_digits()
    X, y = digits.data, digits.target
    class_names = [str(i) for i in range(10)]

    print(f"\nDataset: Digits (0-9)")
    print(f"Total samples: {X.shape[0]}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # -------------------------------------------------------------------------
    # 2. Split data into training and testing sets
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")

    # -------------------------------------------------------------------------
    # 3. Preprocess features
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------------------
    # 4. Select and configure the model
    # -------------------------------------------------------------------------
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
            random_state=random_state,
        ),
        "svm": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=random_state,
        ),
    }

    if model_type not in models:
        raise ValueError(f"model_type must be one of {list(models.keys())}")

    model = models[model_type]
    print(f"\nModel: {model_type.replace('_', ' ').title()}")
    print(f"Parameters: {model.get_params()}")

    # -------------------------------------------------------------------------
    # 5. Train the model
    # -------------------------------------------------------------------------
    print("\nTraining model...")
    model.fit(X_train_scaled, y_train)
    print("Training complete!")

    # -------------------------------------------------------------------------
    # 6. Make predictions
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    # -------------------------------------------------------------------------
    # 7. Evaluate performance using multiple metrics
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy:                {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # F1 Score (macro, micro, weighted)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_micro = f1_score(y_test, y_pred, average="micro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    print(f"F1 Score (Macro):        {f1_macro:.4f}")
    print(f"F1 Score (Micro):        {f1_micro:.4f}")
    print(f"F1 Score (Weighted):     {f1_weighted:.4f}")

    # ROC-AUC (One-vs-Rest for multi-class)
    y_test_binarized = label_binarize(y_test, classes=list(range(10)))
    roc_auc_macro = roc_auc_score(
        y_test_binarized, y_prob, multi_class="ovr", average="macro"
    )
    roc_auc_weighted = roc_auc_score(
        y_test_binarized, y_prob, multi_class="ovr", average="weighted"
    )
    print(f"ROC-AUC (Macro OvR):     {roc_auc_macro:.4f}")
    print(f"ROC-AUC (Weighted OvR):  {roc_auc_weighted:.4f}")

    # Cross-validation score
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )
    print(f"\nCross-Validation (5-fold):")
    print(f"  Scores: {cv_scores.round(4)}")
    print(f"  Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Detailed classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Per Class)")
    print("=" * 60)
    print(
        classification_report(
            y_test, y_pred, target_names=[f"Digit {c}" for c in class_names]
        )
    )

    # -------------------------------------------------------------------------
    # 8. Confusion Matrix
    # -------------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    if plot_results:
        _plot_results(cm, class_names, y_test, y_pred, y_prob, model_type)

    # -------------------------------------------------------------------------
    # 9. Compile and return results
    # -------------------------------------------------------------------------
    results = {
        "model": model,
        "scaler": scaler,
        "metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "roc_auc_macro": roc_auc_macro,
            "roc_auc_weighted": roc_auc_weighted,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores,
        },
        "confusion_matrix": cm,