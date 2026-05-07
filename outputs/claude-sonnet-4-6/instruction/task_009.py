```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


def generate_multiclass_dataset(
    n_samples: int = 5000,
    n_features: int = 20,
    n_classes: int = 10,
    imbalanced: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic 10-class dataset, optionally imbalanced."""
    if imbalanced:
        # Create imbalanced weights so some classes are rarer
        weights = np.array([0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04, 0.03, 0.02])
        weights /= weights.sum()
    else:
        weights = None

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=weights,
        random_state=random_state,
    )
    return X, y


def build_pipeline(model_name: str = "random_forest") -> Pipeline:
    """Build a sklearn Pipeline with scaler + classifier."""
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            class_weight="balanced",   # handles imbalance
            random_state=42,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            C=1.0,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(models.keys())}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),   # fitted ONLY on train data via pipeline
        ("classifier", models[model_name]),
    ])
    return pipeline


def evaluate_multiclass_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> dict:
    """
    Full training + evaluation pipeline for a 10-class classifier.

    Best-practice checklist:
      ✔ Train/test split BEFORE any preprocessing.
      ✔ Scaler fitted ONLY on training data (via Pipeline).
      ✔ Hyperparameters NOT tuned on test data.
      ✔ Reports macro-F1, MCC, and OvR-AUC (robust to class imbalance).

    Returns
    -------
    dict with all evaluation metrics.
    """
    # ------------------------------------------------------------------ #
    # 1. Split FIRST — no preprocessing has touched the data yet          #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,          # preserve class distribution
        random_state=random_state,
    )
    print(f"Dataset split: {len(X_train)} train / {len(X_test)} test samples")
    print(f"Classes: {np.unique(y)} | Class counts (train): {np.bincount(y_train)}\n")

    # ------------------------------------------------------------------ #
    # 2. Build pipeline (scaler + model)                                  #
    # ------------------------------------------------------------------ #
    pipeline = build_pipeline(model_name)

    # ------------------------------------------------------------------ #
    # 3. Cross-validation on TRAINING data only (no test leakage)        #
    # ------------------------------------------------------------------ #
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_f1_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    print(f"Cross-validation macro-F1 ({cv_folds}-fold): "
          f"{cv_f1_scores.mean():.4f} ± {cv_f1_scores.std():.4f}")

    # ------------------------------------------------------------------ #
    # 4. Final training on full training set                              #
    # ------------------------------------------------------------------ #
    pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # 5. Evaluation on held-out test set                                  #
    # ------------------------------------------------------------------ #
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)   # shape (n_test, n_classes)

    # --- Primary metrics (robust to imbalance) ---
    macro_f1   = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    mcc        = matthews_corrcoef(y_test, y_pred)
    auc_ovr    = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    auc_ovo    = roc_auc_score(y_test, y_proba, multi_class="ovo", average="macro")
    accuracy   = accuracy_score(y_test, y_pred)

    # --- Per-class F1 ---
    per_class_f1 = f1_score(y_test, y_pred, average=None)

    print("\n" + "=" * 60)
    print(f"  Model          : {model_name}")
    print("=" * 60)
    print(f"  Accuracy       : {accuracy:.4f}  (less reliable for imbalanced data)")
    print(f"  Macro-F1       : {macro_f1:.4f}  ← primary metric")
    print(f"  Weighted-F1    : {weighted_f1:.4f}")
    print(f"  MCC            : {mcc:.4f}  (range -1 to +1)")
    print(f"  AUC-OvR (macro): {auc_ovr:.4f}")
    print(f"  AUC-OvO (macro): {auc_ovo:.4f}")
    print("=" * 60)

    print("\nPer-class F1 scores:")
    for cls, f1 in enumerate(per_class_f1):
        support = np.sum(y_test == cls)
        print(f"  Class {cls:2d}: F1 = {f1:.4f}  (test support = {support})")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # ------------------------------------------------------------------ #
    # 6. Confusion matrix                                                 #
    # ------------------------------------------------------------------ #
    cm