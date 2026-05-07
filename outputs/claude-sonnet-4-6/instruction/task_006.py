import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import warnings

warnings.filterwarnings("ignore")


def train_logistic_with_optimal_threshold(
    X,
    y,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    class_weight: str = "balanced",
    n_threshold_steps: int = 100,
    verbose: bool = True,
):
    """
    Train a logistic regression classifier and find the classification threshold
    that maximises F1-score on a validation set, then report final test F1.

    Best-practice guarantees
    ------------------------
    * Train/val/test split is performed FIRST, before any preprocessing.
    * The StandardScaler is fitted ONLY on the training split.
    * The optimal threshold is selected on the VALIDATION split (never on test).
    * Final evaluation is performed ONCE on the held-out test split.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    test_size  : fraction of data reserved for the final test set.
    val_size   : fraction of the *remaining* data used for threshold selection.
    random_state : reproducibility seed.
    class_weight : passed to LogisticRegression; 'balanced' helps with imbalance.
    n_threshold_steps : number of candidate thresholds to evaluate.
    verbose    : print progress and results.

    Returns
    -------
    results : dict with keys
        'optimal_threshold', 'val_f1_at_threshold',
        'test_f1_at_threshold', 'test_f1_default',
        'test_auc', 'model', 'scaler'
    """

    # ------------------------------------------------------------------ #
    # 1. Split FIRST — test set is locked away immediately                 #
    # ------------------------------------------------------------------ #
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    if verbose:
        print(
            f"Split sizes  →  train: {len(X_train)}, "
            f"val: {len(X_val)}, test: {len(X_test)}"
        )
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Train class distribution: {dict(zip(unique, counts))}")

    # ------------------------------------------------------------------ #
    # 2. Preprocessing — fit ONLY on training data                        #
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
    X_val_scaled   = scaler.transform(X_val)          # transform only
    X_test_scaled  = scaler.transform(X_test)         # transform only

    # ------------------------------------------------------------------ #
    # 3. Train logistic regression                                         #
    # ------------------------------------------------------------------ #
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------ #
    # 4. Find optimal threshold on VALIDATION set                         #
    # ------------------------------------------------------------------ #
    val_probs = model.predict_proba(X_val_scaled)[:, 1]

    thresholds = np.linspace(0.01, 0.99, n_threshold_steps)
    val_f1_scores = []

    for thresh in thresholds:
        preds = (val_probs >= thresh).astype(int)
        # zero_division=0 avoids warnings when a class is never predicted
        f1 = f1_score(y_val, preds, zero_division=0)
        val_f1_scores.append(f1)

    val_f1_scores = np.array(val_f1_scores)
    best_idx       = int(np.argmax(val_f1_scores))
    optimal_threshold = float(thresholds[best_idx])
    val_f1_at_threshold = float(val_f1_scores[best_idx])

    if verbose:
        print(f"\nThreshold search on validation set")
        print(f"  Optimal threshold : {optimal_threshold:.4f}")
        print(f"  Val F1 at threshold: {val_f1_at_threshold:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Final evaluation on TEST set (done exactly once)                 #
    # ------------------------------------------------------------------ #
    test_probs = model.predict_proba(X_test_scaled)[:, 1]

    # Default threshold (0.5)
    test_preds_default = model.predict(X_test_scaled)
    test_f1_default    = f1_score(y_test, test_preds_default, zero_division=0)

    # Optimal threshold found on validation set
    test_preds_optimal = (test_probs >= optimal_threshold).astype(int)
    test_f1_optimal    = f1_score(y_test, test_preds_optimal, zero_division=0)

    test_auc = roc_auc_score(y_test, test_probs)

    if verbose:
        print(f"\n{'='*50}")
        print("FINAL TEST SET RESULTS")
        print(f"{'='*50}")
        print(f"  AUC-ROC                    : {test_auc:.4f}")
        print(f"  F1 @ default threshold 0.5 : {test_f1_default:.4f}")
        print(f"  F1 @ optimal threshold {optimal_threshold:.2f}  : {test_f1_optimal:.4f}")
        print(f"\nClassification report (optimal threshold):")
        print(classification_report(y_test, test_preds_optimal, zero_division=0))

    return {
        "optimal_threshold":      optimal_threshold,
        "val_f1_at_threshold":    val_f1_at_threshold,
        "test_f1_at_threshold":   test_f1_optimal,
        "test_f1_default":        test_f1_default,
        "test_auc":               test_auc,
        "model":                  model,
        "scaler":                 scaler,
        # Expose internals for inspection / further use
        "val_thresholds":         thresholds,
        "val_f1_curve":           val_f1_scores,
    }


# ------------------------------------------------------------------ #
# Demo                                                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Imbalanced binary classification dataset
    X, y = make_classification(
        n_samples=5_000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.85, 0.15],   # ~15 % positive class → imbalanced
        flip_y=0.02,
        random_state=0,
    )

    results = train_logistic_with_optimal_threshold(
        X, y,
        test_size=0.20,
        val_size=0.20,
        random_state=42,
        class_weight="balanced",
        n_threshold_steps=200,
        verbose=True,
    )