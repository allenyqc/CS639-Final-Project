import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    accuracy_score,
)


def tune_and_evaluate_random_forest(
    X,
    y,
    test_size=0.2,
    random_state=42,
    cv_splits=5,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
):
    """
    Tune a Random Forest classifier with GridSearchCV and evaluate on a held-out test set.

    Best practices applied:
      - Train/test split is performed BEFORE any preprocessing or feature engineering.
      - Preprocessing (StandardScaler) is fit ONLY on training data via a Pipeline,
        which ensures no leakage during cross-validation.
      - Hyperparameters are selected purely from CV on the training set; the test set
        is never used for model/threshold selection.
      - Default scoring is `f1_weighted` (robust to class imbalance); AUC and MCC are
        also reported alongside accuracy on the held-out test set.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    test_size : float, fraction of data reserved for the held-out test set.
    random_state : int, reproducibility seed.
    cv_splits : int, number of stratified CV folds.
    scoring : str, sklearn scoring metric used by GridSearchCV.
    n_jobs : int, parallel jobs.
    verbose : int, verbosity level.

    Returns
    -------
    results : dict containing the fitted GridSearchCV, best estimator,
              best params, and held-out test metrics.
    """
    # ---- 1) Split BEFORE any feature engineering / preprocessing ----
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # ---- 2) Build a pipeline so preprocessors are fit ONLY on training folds ----
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    random_state=random_state, class_weight="balanced", n_jobs=n_jobs
                ),
            ),
        ]
    )

    # ---- 3) Hyperparameter grid (over the classifier step only) ----
    param_grid = {
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # ---- 4) Tune on training data ONLY ----
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=False,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # ---- 5) Evaluate ONCE on the held-out test set ----
    y_pred = best_model.predict(X_test)

    classes = np.unique(y_train)
    is_binary = len(classes) == 2

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }

    # AUC requires probability scores
    try:
        y_proba = best_model.predict_proba(X_test)
        if is_binary:
            test_metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            test_metrics["roc_auc_ovr_weighted"] = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted"
            )
    except Exception:
        pass

    if verbose:
        print("\nBest CV score ({}): {:.4f}".format(scoring, grid.best_score_))
        print("Best params:", grid.best_params_)
        print("\nHeld-out test metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification report (test):")
        print(classification_report(y_test, y_pred, digits=4))
        print("Confusion matrix (test):")
        print(confusion_matrix(y_test, y_pred))

    return {
        "grid_search": grid,
        "best_estimator": best_model,
        "best_params": grid.best_params_,
        "best_cv_score": grid.best_score_,
        "test_metrics": test_metrics,
        "split": {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
    }


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        weights=[0.85, 0.15],  # imbalanced
        random_state=42,
    )
    results = tune_and_evaluate_random_forest(X, y, verbose=1)