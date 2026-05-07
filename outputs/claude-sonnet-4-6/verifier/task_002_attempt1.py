```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings("ignore")


def train_imbalanced_classifier(
    X=None,
    y=None,
    imbalance_ratio: float = 0.05,
    n_samples: int = 5000,
    n_features: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train a binary classifier on an imbalanced dataset and report performance.

    Parameters
    ----------
    X : array-like, optional
        Feature matrix. If None, a synthetic dataset is generated.
    y : array-like, optional
        Target vector. If None, a synthetic dataset is generated.
    imbalance_ratio : float
        Fraction of positive (minority) samples when generating synthetic data.
    n_samples : int
        Number of samples for synthetic data generation.
    n_features : int
        Number of features for synthetic data generation.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing the trained model, scaler, and performance metrics.
    """

    # ------------------------------------------------------------------ #
    # 1. Data preparation                                                  #
    # ------------------------------------------------------------------ #
    if X is None or y is None:
        print("No dataset provided – generating a synthetic imbalanced dataset …\n")
        weights = [1 - imbalance_ratio, imbalance_ratio]
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_redundant=5,
            weights=weights,
            flip_y=0.01,
            random_state=random_state,
        )

    X = np.asarray(X)
    y = np.asarray(y)

    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print("=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    for cls, cnt in class_dist.items():
        pct = cnt / len(y) * 100
        print(f"  Class {cls}: {cnt:>6} samples  ({pct:.2f} %)")
    minority_ratio = counts.min() / counts.max()
    print(f"\n  Imbalance ratio (minority/majority): {minority_ratio:.4f}")

    # ------------------------------------------------------------------ #
    # 2. Train / test split (stratified)                                   #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # ------------------------------------------------------------------ #
    # 3. Feature scaling                                                   #
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------ #
    # 4. Compute class weights to handle imbalance                         #
    # ------------------------------------------------------------------ #
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print(f"\n  Computed class weights: {class_weight_dict}")

    # ------------------------------------------------------------------ #
    # 5. Model definition                                                  #
    # ------------------------------------------------------------------ #
    # GradientBoostingClassifier with scale_pos_weight-like tuning via
    # sample_weight passed at fit time.
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=random_state,
    )

    # Build per-sample weights from class weights
    sample_weights = np.array([class_weight_dict[label] for label in y_train])

    # ------------------------------------------------------------------ #
    # 6. Cross-validation (stratified, on training set)                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-fold, training set)")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for metric_name, scoring in [
        ("ROC-AUC", "roc_auc"),
        ("Average Precision", "average_precision"),
        ("F1 (macro)", "f1_macro"),
        ("Balanced Accuracy", "balanced_accuracy"),
    ]:
        scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=cv,
            scoring=scoring,
            fit_params={"sample_weight": sample_weights},
        )
        print(f"  {metric_name:<22}: {scores.mean():.4f} ± {scores.std():.4f}")

    # ------------------------------------------------------------------ #
    # 7. Final training on the full training set                           #
    # ------------------------------------------------------------------ #
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    # ------------------------------------------------------------------ #
    # 8. Predictions                                                       #
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # ------------------------------------------------------------------ #
    # 9. Performance report                                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("TEST SET PERFORMANCE")
    print("=" * 60)

    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_minority = f1_score(y_test, y_pred, pos_label=1)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"  ROC-AUC              : {roc_auc:.4f}")
    print(f"  Average Precision    : {avg_precision:.4f}")
    print(f"  F1 (macro)           : {f1_macro:.4f}")
    print(f"  F1 (minority class)  : {f1_minority:.4f}")
    print(f"  Balanced Accuracy    : {bal_acc:.4f}")

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in classes],
        columns=[f"Predicted {c}" for c in classes],
    )
    print(cm_df.to_string(index=True))

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(