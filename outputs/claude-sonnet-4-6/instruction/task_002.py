```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. Generate an imbalanced binary dataset
# ──────────────────────────────────────────────
def make_imbalanced_dataset(
    n_samples: int = 5_000,
    minority_fraction: float = 0.05,
    n_features: int = 20,
    random_state: int = 42,
):
    """Return X, y with a severe class imbalance."""
    weights = [1 - minority_fraction, minority_fraction]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=4,
        weights=weights,
        flip_y=0.01,
        random_state=random_state,
    )
    return X, y


# ──────────────────────────────────────────────
# 2. Build a pipeline (scaler + classifier)
# ──────────────────────────────────────────────
def build_pipeline(random_state: int = 42) -> Pipeline:
    """
    GradientBoostingClassifier with class-weight-aware sub-sampling.
    All preprocessing is encapsulated so it is fitted only on training data.
    """
    clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=random_state,
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),  # fitted on train only (Pipeline guarantees this)
            ("classifier", clf),
        ]
    )
    return pipeline


# ──────────────────────────────────────────────
# 3. Threshold selection on VALIDATION data only
# ──────────────────────────────────────────────
def select_threshold_on_validation(
    pipeline: Pipeline,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Choose the decision threshold that maximises `metric` on the
    held-out validation set.  Test data is never touched here.
    """
    proba = pipeline.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 200)
    best_score, best_thresh = -np.inf, 0.5

    for t in thresholds:
        preds = (proba >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_val, preds, zero_division=0)
        elif metric == "mcc":
            score = matthews_corrcoef(y_val, preds)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score, best_thresh = score, t

    print(f"\n[Threshold selection] Best {metric.upper()} on validation = "
          f"{best_score:.4f}  →  threshold = {best_thresh:.3f}")
    return best_thresh


# ──────────────────────────────────────────────
# 4. Evaluation helper
# ──────────────────────────────────────────────
def evaluate(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    split_name: str = "Test",
) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    roc_auc  = roc_auc_score(y_test, proba)
    pr_auc   = average_precision_score(y_test, proba)
    f1       = f1_score(y_test, preds, zero_division=0)
    mcc      = matthews_corrcoef(y_test, preds)

    print(f"\n{'='*55}")
    print(f"  {split_name} Results  (threshold = {threshold:.3f})")
    print(f"{'='*55}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  MCC      : {mcc:.4f}")
    print(f"\n{classification_report(y_test, preds, digits=4)}")

    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:")
    print(cm)

    return dict(roc_auc=roc_auc, pr_auc=pr_auc, f1=f1, mcc=mcc,
                proba=proba, preds=preds)


# ──────────────────────────────────────────────
# 5. Plotting
# ──────────────────────────────────────────────
def plot_results(metrics: dict, y_test: np.ndarray) -> None:
    from sklearn.metrics import roc_curve, precision_recall_curve

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, metrics["proba"])
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[0].legend()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, metrics["proba"])
    axes[1].plot(rec, prec, lw=2, label=f"PR-AUC = {metrics['pr_auc']:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend()

    # Confusion matrix
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, metrics["preds"])
    ).plot(ax=axes[2], colorbar=False)
    axes[2].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig("imbalanced_classifier_results.png", dpi=120)
    plt.show()
    print("\nPlot saved to imbalanced_classifier_results.png")


# ──────────────────────────────────────────────
# 6. Main training function
# ──────────────────────────────────────────────
def train_imbalanced_classifier(
    n_samples: int = 5_000,
    minority_fraction: float = 0.05,
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = 42,
    threshold_metric: str = "f1",
) -> dict:
    """
    End-to-end training and evaluation on an imbalanced binary dataset.

    Returns a dict with test metrics.
    """