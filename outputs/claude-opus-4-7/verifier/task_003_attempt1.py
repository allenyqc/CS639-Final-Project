import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    mean_squared_error,
    r2_score,
)


def tune_random_forest(
    X,
    y,
    task="classification",
    param_grid=None,
    test_size=0.2,
    cv=5,
    scoring=None,
    random_state=42,
    n_jobs=-1,
    verbose=1,
):
    """
    Tune hyperparameters for a Random Forest model using GridSearchCV
    and evaluate the best model on a held-out test set.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    task : {"classification", "regression"}, default="classification"
        Type of supervised learning task.
    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV. A sensible default is used if None.
    test_size : float, default=0.2
        Proportion of the data to set aside as the held-out test set.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, optional
        Scoring metric for GridSearchCV. Defaults to "f1_weighted" for
        classification and "r2" for regression.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs.
    verbose : int, default=1
        Verbosity level.

    Returns
    -------
    results : dict
        Dictionary with the best estimator, best params, CV score, and
        test-set evaluation metrics.
    """
    # Default hyperparameter grid
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }

    # Choose model and default scoring
    if task == "classification":
        model = RandomForestClassifier(random_state=random_state)
        scoring = scoring or "f1_weighted"
        stratify = y
    elif task == "regression":
        model = RandomForestRegressor(random_state=random_state)
        scoring = scoring or "r2"
        stratify = None
    else:
        raise ValueError("task must be 'classification' or 'regression'")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate on the held-out test set
    if task == "classification":
        test_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "classification_report": classification_report(y_test, y_pred),
        }
    else:
        test_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": r2_score(y_test, y_pred),
        }

    results = {
        "best_estimator": best_model,
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "test_metrics": test_metrics,
    }

    if verbose:
        print("Best CV params:", results["best_params"])
        print(f"Best CV {scoring}: {results['best_cv_score']:.4f}")
        print("Test metrics:")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}:\n{v}")

    return results


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    tune_random_forest(data.data, data.target, task="classification")