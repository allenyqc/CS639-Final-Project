import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def train_logreg_optimal_f1(
    X,
    y,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    n_thresholds=101,
    **logreg_kwargs,
):
    """
    Train a logistic regression classifier, find the threshold that maximizes
    F1-score on a validation set, and report final test F1 at that threshold.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Binary target labels (0/1).
    test_size : float, default=0.2
        Fraction of data held out for the test set.
    val_size : float, default=0.2
        Fraction of the remaining data used as a validation set
        (for threshold tuning).
    random_state : int, default=42
        Random seed for reproducibility.
    n_thresholds : int, default=101
        Number of candidate thresholds to evaluate in [0, 1].
    **logreg_kwargs :
        Additional keyword args forwarded to ``LogisticRegression``.

    Returns
    -------
    results : dict
        Dictionary with the trained model, the best threshold, validation F1
        at that threshold, default-threshold (0.5) test F1, and the final
        test F1 at the tuned threshold.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    # Train / Val / Test split
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

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, **logreg_kwargs)
    model.fit(X_train, y_train)

    # Probabilities on validation set
    val_probs = model.predict_proba(X_val)[:, 1]

    # Sweep thresholds and pick the best by F1 on validation
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_threshold, best_val_f1 = 0.5, -1.0
    for thr in thresholds:
        preds = (val_probs >= thr).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_threshold = float(thr)

    # Evaluate on test set
    test_probs = model.predict_proba(X_test)[:, 1]
    default_test_f1 = f1_score(y_test, (test_probs >= 0.5).astype(int), zero_division=0)
    tuned_test_f1 = f1_score(
        y_test, (test_probs >= best_threshold).astype(int), zero_division=0
    )

    results = {
        "model": model,
        "best_threshold": best_threshold,
        "val_f1_at_best_threshold": float(best_val_f1),
        "test_f1_at_default_threshold": float(default_test_f1),
        "test_f1_at_best_threshold": float(tuned_test_f1),
    }

    print(f"Best threshold (val): {best_threshold:.3f}")
    print(f"Validation F1 @ best threshold: {best_val_f1:.4f}")
    print(f"Test F1 @ default 0.5 threshold: {default_test_f1:.4f}")
    print(f"Test F1 @ tuned threshold: {tuned_test_f1:.4f}")

    return results


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=5,
        weights=[0.85, 0.15],
        random_state=0,
    )
    train_logreg_optimal_f1(X, y)