import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler


def train_and_optimize_threshold(
    X=None,
    y=None,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    lr_params=None,
):
    """
    Train a logistic regression classifier, find the threshold that maximizes
    F1-score on a validation set, and report the final test F1 at that threshold.

    Parameters
    ----------
    X : array-like, optional
        Feature matrix. If None, a synthetic dataset is generated.
    y : array-like, optional
        Target vector. If None, a synthetic dataset is generated.
    test_size : float
        Fraction of data reserved for the final test set.
    val_size : float
        Fraction of the remaining (non-test) data reserved for threshold tuning.
    random_state : int
        Random seed for reproducibility.
    lr_params : dict, optional
        Extra keyword arguments forwarded to LogisticRegression.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'model'            : trained LogisticRegression
        - 'best_threshold'   : threshold that maximised validation F1
        - 'val_f1'           : F1 on the validation set at best_threshold
        - 'test_f1_default'  : F1 on the test set at the default threshold (0.5)
        - 'test_f1_optimal'  : F1 on the test set at best_threshold
        - 'threshold_candidates' : array of evaluated thresholds
        - 'val_f1_scores'    : corresponding validation F1 scores
    """
    # ------------------------------------------------------------------ #
    # 1. Data preparation
    # ------------------------------------------------------------------ #
    if X is None or y is None:
        print("No data provided – generating a synthetic binary classification dataset.")
        X, y = make_classification(
            n_samples=5_000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            weights=[0.7, 0.3],   # slight class imbalance
            random_state=random_state,
        )

    X = np.asarray(X)
    y = np.asarray(y)

    # Split: train+val | test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split: train | val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    print(f"Dataset sizes  →  train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

    # ------------------------------------------------------------------ #
    # 2. Feature scaling
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # ------------------------------------------------------------------ #
    # 3. Train logistic regression
    # ------------------------------------------------------------------ #
    if lr_params is None:
        lr_params = {}
    lr_params.setdefault("max_iter", 1_000)
    lr_params.setdefault("random_state", random_state)

    model = LogisticRegression(**lr_params)
    model.fit(X_train_sc, y_train)
    print("Logistic regression trained.")

    # ------------------------------------------------------------------ #
    # 4. Find optimal threshold on the validation set
    # ------------------------------------------------------------------ #
    val_probs = model.predict_proba(X_val_sc)[:, 1]

    # precision_recall_curve returns thresholds for every unique probability
    # value, giving us a dense, principled search grid.
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

    # F1 = 2 * P * R / (P + R);  guard against division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precisions[:-1] + recalls[:-1]) > 0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
            0.0,
        )

    best_idx       = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_val_f1    = f1_scores[best_idx]

    print(f"\nThreshold search over {len(thresholds)} candidates (from precision-recall curve)")
    print(f"Best threshold : {best_threshold:.4f}")
    print(f"Validation F1  : {best_val_f1:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Evaluate on the test set
    # ------------------------------------------------------------------ #
    test_probs = model.predict_proba(X_test_sc)[:, 1]

    # Default threshold (0.5)
    y_pred_default = (test_probs >= 0.5).astype(int)
    test_f1_default = f1_score(y_test, y_pred_default)

    # Optimal threshold
    y_pred_optimal = (test_probs >= best_threshold).astype(int)
    test_f1_optimal = f1_score(y_test, y_pred_optimal)

    print(f"\nTest F1 @ default threshold (0.50) : {test_f1_default:.4f}")
    print(f"Test F1 @ optimal threshold ({best_threshold:.4f}) : {test_f1_optimal:.4f}")
    print(f"Improvement : {test_f1_optimal - test_f1_default:+.4f}")

    return {
        "model":               model,
        "scaler":              scaler,
        "best_threshold":      best_threshold,
        "val_f1":              best_val_f1,
        "test_f1_default":     test_f1_default,
        "test_f1_optimal":     test_f1_optimal,
        "threshold_candidates": thresholds,
        "val_f1_scores":       f1_scores,
    }


# ------------------------------------------------------------------ #
# Quick demo
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    results = train_and_optimize_threshold()

    # Optional: plot validation F1 vs threshold
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(results["threshold_candidates"], results["val_f1_scores"],
                 linewidth=1.5, label="Validation F1")
        plt.axvline(results["best_threshold"], color="red", linestyle="--",
                    label=f"Best threshold = {results['best_threshold']:.3f}")
        plt.xlabel("Classification threshold")
        plt.ylabel("F1-score")
        plt.title("Validation F1 vs Classification Threshold")
        plt.legend()
        plt.tight_layout()
        plt.savefig("threshold_vs_f1.png", dpi=150)
        print("\nPlot saved to threshold_vs_f1.png")
    except ImportError:
        print("matplotlib not available – skipping plot.")