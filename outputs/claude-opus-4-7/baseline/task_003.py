from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def tune_and_evaluate_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: dict | None = None,
    test_size: float = 0.2,
    cv: int = 5,
    scoring: str = "f1_weighted",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
) -> dict[str, Any]:
    """
    Tune hyperparameters for a Random Forest classifier using GridSearchCV
    and evaluate the best model on a held-out test set.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target labels.
    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV. If None, a sensible default is used.
    test_size : float, default=0.2
        Fraction of the data to reserve for the held-out test set.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default='f1_weighted'
        Scoring metric used by GridSearchCV.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.
    verbose : int, default=1
        Verbosity level for GridSearchCV.

    Returns
    -------
    results : dict
        Dictionary containing the best model, best parameters, best CV score,
        test-set metrics, predictions, and the fitted GridSearchCV object.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        }

    # Split into train/test, stratifying to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Set up grid search
    base_model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True,
    )

    # Fit on training data
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Evaluate on the held-out test set
    y_pred = best_model.predict(X_test)
    average = "binary" if len(np.unique(y)) == 2 else "weighted"

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    if verbose:
        print(f"\nBest CV {scoring}: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        print("\nTest set performance:")
        for k in ("accuracy", "precision", "recall", "f1"):
            print(f"  {k}: {test_metrics[k]:.4f}")
        print("\nClassification report:")
        print(test_metrics["classification_report"])

    return {
        "best_model": best_model,
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "test_metrics": test_metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "grid_search": grid_search,
    }


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    results = tune_and_evaluate_random_forest(data.data, data.target)