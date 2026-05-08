import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)


def train_knn_with_mean_imputation(
    X,
    y,
    test_size=0.2,
    random_state=42,
    k_grid=(1, 3, 5, 7, 9, 11, 15),
    imbalance_threshold=0.2,
):
    """
    Imputes missing values with the column mean (fit on training data only),
    scales features, tunes k via cross-validation on the training set,
    and evaluates on a held-out test set.

    Returns a dict of metrics. For imbalanced datasets, F1/AUC/MCC are
    reported alongside accuracy.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # 1. Train/test split BEFORE any feature engineering or imputation.
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # 2. Build a pipeline so imputer/scaler are fit ONLY on training folds.
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )

    # 3. Detect class imbalance and pick a CV scoring metric accordingly.
    classes, counts = np.unique(y_train, return_counts=True)
    class_freq = counts / counts.sum()
    is_binary = len(classes) == 2
    is_imbalanced = class_freq.min() < imbalance_threshold

    if is_imbalanced:
        scoring = "roc_auc" if is_binary else "f1_macro"
    else:
        scoring = "accuracy"

    # 4. Hyperparameter selection using cross-validation on TRAINING data only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    param_grid = {
        "knn__n_neighbors": [k for k in k_grid if k <= len(X_train)],
        "knn__weights": ["uniform", "distance"],
    }
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # 5. Evaluate on the untouched test set.
    y_pred = best_model.predict(X_test)
    metrics = {
        "best_params": grid.best_params_,
        "cv_score": grid.best_score_,
        "cv_scoring": scoring,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "is_imbalanced": bool(is_imbalanced),
    }

    if is_binary:
        try:
            y_proba = best_model.predict_proba(X_test)[:, 1]
            metrics["test_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            pass
        metrics["test_f1"] = f1_score(y_test, y_pred)
        metrics["test_mcc"] = matthews_corrcoef(y_test, y_pred)
    else:
        metrics["test_f1_macro"] = f1_score(y_test, y_pred, average="macro")
        metrics["test_mcc"] = matthews_corrcoef(y_test, y_pred)

    return metrics


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = data.data.astype(float).copy()

    # Inject some missing values to demonstrate mean imputation.
    rng = np.random.default_rng(0)
    mask = rng.random(X.shape) < 0.05
    X[mask] = np.nan

    results = train_knn_with_mean_imputation(X, data.target)
    for k, v in results.items():
        print(f"{k}: {v}")