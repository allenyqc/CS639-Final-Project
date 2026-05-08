import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve


def train_logreg_with_best_threshold(
    X,
    y,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    class_weight="balanced",
):
    """
    Train a logistic regression classifier, select the probability threshold
    that maximizes F1 on a validation split (carved out of training data),
    and report the final F1 on the held-out test set at that threshold.

    Best practices enforced:
      - Test set is split off BEFORE any preprocessing.
      - Threshold is tuned on a validation split, NOT on the test set.
      - Scaler is fit ONLY on training data.
      - Uses F1 (suitable for imbalanced binary classification).
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    # 1) Train/test split BEFORE any feature engineering.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2) Carve out a validation set from the training data for threshold tuning.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    # 3) Fit preprocessor on training data only.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 4) Fit logistic regression.
    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train_s, y_train)

    # 5) Find threshold that maximizes F1 on the VALIDATION set.
    val_probs = model.predict_proba(X_val_s)[:, 1]
    precision, recall, pr_thresholds = precision_recall_curve(y_val, val_probs)
    # precision_recall_curve returns arrays of length n+1 for P/R, length n for thresholds.
    # Compute F1 for each threshold.
    f1_scores = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12),
        0.0,
    )

    if len(f1_scores) == 0:
        best_threshold = 0.5
        best_val_f1 = f1_score(y_val, (val_probs >= 0.5).astype(int))
    else:
        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(pr_thresholds[best_idx])
        best_val_f1 = float(f1_scores[best_idx])

    # 6) Evaluate on TEST set using the threshold chosen on validation.
    test_probs = model.predict_proba(X_test_s)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    test_f1 = f1_score(y_test, test_preds)
    default_test_f1 = f1_score(y_test, (test_probs >= 0.5).astype(int))

    results = {
        "model": model,
        "scaler": scaler,
        "best_threshold": best_threshold,
        "val_f1_at_best_threshold": best_val_f1,
        "test_f1_at_best_threshold": float(test_f1),
        "test_f1_at_0.5": float(default_test_f1),
    }

    print(f"Best threshold (chosen on validation): {best_threshold:.4f}")
    print(f"Validation F1 at best threshold:       {best_val_f1:.4f}")
    print(f"Test F1 at best threshold:             {test_f1:.4f}")
    print(f"Test F1 at default 0.5 threshold:      {default_test_f1:.4f}")

    return results


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=5,
        weights=[0.9, 0.1],
        random_state=0,
    )
    train_logreg_with_best_threshold(X, y)