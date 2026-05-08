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
    F1-score on a held-out validation set (NOT the test set), and report the
    final test F1 at that tuned threshold.

    The threshold is selected purely from validation data so that the test
    set remains an unbiased estimate of generalization performance.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    # 1) Split off the test set first; it must NOT influence threshold choice.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2) Split the remainder into train and validation sets.
    #    Validation is used exclusively for threshold tuning.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    # 3) Fit logistic regression on the training set only.
    model = LogisticRegression(max_iter=1000, **logreg_kwargs)
    model.fit(X_train, y_train)

    # 4) Tune threshold on validation probabilities/labels.
    #    The test set is never inspected during threshold selection.
    val_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    best_threshold, best_val_f1 = 0.5, -1.0
    for thr in thresholds:
        val_preds = (val_probs >= thr).astype(int)
        f1 = f1_score(y_val, val_preds, zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_threshold = float(thr)

    # 5) Final, single evaluation on the untouched test set using the
    #    threshold chosen on the validation set.
    test_probs = model.predict_proba(X_test)[:, 1]
    tuned_test_f1 = f1_score(
        y_test, (test_probs >= best_threshold).astype(int), zero_division=0
    )

    results = {
        "model": model,
        "best_threshold": best_threshold,
        "val_f1_at_best_threshold": float(best_val_f1),
        "test_f1_at_best_threshold": float(tuned_test_f1),
    }

    print(f"Best threshold (selected on validation): {best_threshold:.3f}")
    print(f"Validation F1 @ best threshold: {best_val_f1:.4f}")
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