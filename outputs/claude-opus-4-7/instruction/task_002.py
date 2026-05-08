import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    average_precision_score,
    classification_report,
    confusion_matrix,
)


def train_imbalanced_classifier(X, y, random_state=42):
    """
    Train a binary classifier on an imbalanced dataset following ML best practices.

    - Splits data BEFORE any feature engineering.
    - Fits preprocessors ONLY on training data (via Pipeline).
    - Selects hyperparameters and decision threshold using CV on training data only.
    - Reports F1, ROC-AUC, PR-AUC, and MCC (better than accuracy for imbalance).
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    # 1) Split BEFORE any feature engineering, stratify to preserve class ratio.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # 2) Pipeline ensures scaler is fit only on training folds.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # 3) Hyperparameter selection using CV on training data only.
    best_C, best_score = None, -np.inf
    for C in [0.01, 0.1, 1.0, 10.0]:
        pipe.set_params(clf__C=C)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score, best_C = mean_score, C

    pipe.set_params(clf__C=best_C)

    # 4) Threshold tuning using CV on TRAINING data only (never the test set).
    from sklearn.model_selection import cross_val_predict
    train_proba = cross_val_predict(
        pipe, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]

    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr, best_thr_f1 = 0.5, -np.inf
    for thr in thresholds:
        preds = (train_proba >= thr).astype(int)
        f1 = f1_score(y_train, preds, zero_division=0)
        if f1 > best_thr_f1:
            best_thr_f1, best_thr = f1, thr

    # 5) Refit on full training set, evaluate ONCE on the held-out test set.
    pipe.fit(X_train, y_train)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_thr).astype(int)

    metrics = {
        "best_C": best_C,
        "best_threshold": float(best_thr),
        "cv_f1_train": float(best_score),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_proba)),
        "test_pr_auc": float(average_precision_score(y_test, test_proba)),
        "test_mcc": float(matthews_corrcoef(y_test, test_pred)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "classification_report": classification_report(
            y_test, test_pred, zero_division=0
        ),
    }

    print(f"Best C: {metrics['best_C']}")
    print(f"Best threshold (from training CV): {metrics['best_threshold']:.2f}")
    print(f"CV F1 (train): {metrics['cv_f1_train']:.4f}")
    print(f"Test F1:       {metrics['test_f1']:.4f}")
    print(f"Test ROC-AUC:  {metrics['test_roc_auc']:.4f}")
    print(f"Test PR-AUC:   {metrics['test_pr_auc']:.4f}")
    print(f"Test MCC:      {metrics['test_mcc']:.4f}")
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print(metrics["classification_report"])

    return pipe, metrics


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        weights=[0.95, 0.05],  # imbalanced
        random_state=42,
    )
    model, results = train_imbalanced_classifier(X, y)