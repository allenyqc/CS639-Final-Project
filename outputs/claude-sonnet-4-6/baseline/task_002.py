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
    matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings("ignore")


def train_imbalanced_classifier(
    X=None,
    y=None,
    test_size: float = 0.2,
    random_state: int = 42,
    imbalance_ratio: float = 0.05,
    n_samples: int = 5000,
    n_features: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Train a binary classifier on an imbalanced dataset and report performance.

    Parameters
    ----------
    X : array-like, optional
        Feature matrix. If None, a synthetic dataset is generated.
    y : array-like, optional
        Target vector. If None, a synthetic dataset is generated.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.
    imbalance_ratio : float
        Ratio of minority to majority class when generating synthetic data.
    n_samples : int
        Number of samples for synthetic data generation.
    n_features : int
        Number of features for synthetic data generation.
    verbose : bool
        Whether to print detailed performance reports.

    Returns
    -------
    dict
        Dictionary containing model, metrics, and predictions.
    """

    # ------------------------------------------------------------------ #
    # 1. Data preparation                                                  #
    # ------------------------------------------------------------------ #
    if X is None or y is None:
        minority_samples = int(n_samples * imbalance_ratio)
        majority_samples = n_samples - minority_samples
        weights = [majority_samples / n_samples, minority_samples / n_samples]

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.6),
            n_redundant=int(n_features * 0.2),
            weights=weights,
            flip_y=0.01,
            random_state=random_state,
        )
        if verbose:
            print("=" * 60)
            print("Synthetic imbalanced dataset generated.")

    X = np.asarray(X)
    y = np.asarray(y)

    unique, counts = np.unique(y, return_counts=True)
    class_distribution = dict(zip(unique.tolist(), counts.tolist()))

    if verbose:
        print(f"\nClass distribution  : {class_distribution}")
        minority_pct = counts.min() / counts.sum() * 100
        print(f"Minority class      : {minority_pct:.2f}% of total samples")

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
    class_weights = compute_class_weight(
        class_weight="balanced", classes=unique, y=y_train
    )
    weight_dict = dict(zip(unique.tolist(), class_weights.tolist()))
    if verbose:
        print(f"\nComputed class weights: {weight_dict}")

    # ------------------------------------------------------------------ #
    # 5. Model definition                                                  #
    # ------------------------------------------------------------------ #
    # GradientBoostingClassifier with scale_pos_weight-like tuning via
    # sample_weight inside fit().
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=random_state,
    )

    # Build per-sample weights for the training set
    sample_weights = np.where(y_train == 1, weight_dict[1], weight_dict[0])

    # ------------------------------------------------------------------ #
    # 6. Cross-validation (stratified, weighted)                           #
    # ------------------------------------------------------------------ #
    if verbose:
        print("\n" + "=" * 60)
        print("Cross-validation (5-fold, stratified) …")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_roc_auc = cross_val_score(
        model, X_train_scaled, y_train, cv=cv, scoring="roc_auc"
    )
    cv_f1 = cross_val_score(
        model, X_train_scaled, y_train, cv=cv, scoring="f1"
    )
    cv_ap = cross_val_score(
        model, X_train_scaled, y_train, cv=cv, scoring="average_precision"
    )

    if verbose:
        print(
            f"  ROC-AUC : {cv_roc_auc.mean():.4f} ± {cv_roc_auc.std():.4f}"
        )
        print(f"  F1      : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        print(f"  Avg-Prec: {cv_ap.mean():.4f} ± {cv_ap.std():.4f}")

    # ------------------------------------------------------------------ #
    # 7. Final training on full training set                               #
    # ------------------------------------------------------------------ #
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    # ------------------------------------------------------------------ #
    # 8. Predictions                                                       #
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # ------------------------------------------------------------------ #
    # 9. Threshold tuning (maximise F1 on test set)                        #
    # ------------------------------------------------------------------ #
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_prob)
    f1_scores_thresh = (
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    )
    best_threshold_idx = np.argmax(f1_scores_thresh)
    best_threshold = thresholds_pr[best_threshold_idx]
    y_pred_tuned = (y_prob >= best_threshold).astype(int)

    # ------------------------------------------------------------------ #
    # 10. Metrics                                                          #
    # ------------------------------------------------------------------ #
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    f1_default = f1_score(y_test, y_pred)
    f1_tuned = f1_score(y_test, y_pred_tuned)
    mcc = matthews_corrcoef(y_test, y_pred_tuned)
    cm = confusion_matrix(y_test, y_pred_tuned)

    tn, fp, fn