import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def train_logistic_with_best_threshold(
    X,
    y,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    n_thresholds=101,
    **lr_kwargs,
):
    """
    Train a logistic regression classifier, find the classification threshold
    that maximizes F1-score on a validation set, and report the final F1 on
    the test set at that threshold.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Binary target labels (0 / 1).
    test_size : float
        Fraction of data to reserve for the test set.
    val_size : float
        Fraction of the (non-test) data used for threshold tuning.
    random_state : int
        Seed for reproducibility.
    n_thresholds : int
        Number of candidate thresholds to evaluate in [0, 1].
    **lr_kwargs :
        Additional keyword arguments forwarded to LogisticRegression.

    Returns
    -------
    dict with keys:
        'model'           : fitted LogisticRegression
        'best_threshold'  : threshold maximizing validation F1
        'val_f1'          : F1 on the validation set at that threshold
        'test_f1'         : F1 on the test set at that threshold
        'default_test_f1' : F1 on the test set using threshold 0.5
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    # Split off the test set first.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Carve a validation set out of the training pool.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    # Train the model.
    model = LogisticRegression(max_iter=1000, **lr_kwargs)
    model.fit(X_train, y_train)

    # Predicted probabilities for the positive class.
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Sweep thresholds and pick the one with the best validation F1.
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_threshold, best_val_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_val_f1:
            best_val_f1, best_threshold = f1, t

    # Evaluate on the test set with the chosen threshold.
    test_preds = (test_probs >= best_threshold).astype(int)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    default_test_f1 = f1_score(y_test, (test_probs >= 0.5).astype(int), zero_division=0)

    print(f"Best threshold (val F1): {best_threshold:.3f}")
    print(f"Validation F1 @ best threshold: {best_val_f1:.4f}")
    print(f"Test F1 @ best threshold:       {test_f1:.4f}")
    print(f"Test F1 @ 0.5 (default):        {default_test_f1:.4f}")

    return {
        "model": model,
        "best_threshold": float(best_threshold),
        "val_f1": float(best_val_f1),
        "test_f1": float(test_f1),
        "default_test_f1": float(default_test_f1),
    }


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=5,
        weights=[0.85, 0.15],
        random_state=0,
    )
    train_logistic_with_best_threshold(X, y)