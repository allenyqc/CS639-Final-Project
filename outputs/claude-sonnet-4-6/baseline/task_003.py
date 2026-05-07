```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings("ignore")


def tune_and_evaluate_random_forest(
    X,
    y,
    param_grid: dict = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    cv: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
) -> dict:
    """
    Tune hyperparameters for a Random Forest model using GridSearchCV,
    then evaluate the best model on a held-out test set.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV. Uses a default grid if None.
    test_size : float
        Proportion of data to reserve as the final test set.
    val_size : float
        Proportion of the remaining data to use for cross-validation
        (informational only; CV is done on the train split).
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric for GridSearchCV.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs (-1 uses all available cores).
    verbose : int
        Verbosity level.

    Returns
    -------
    results : dict
        Dictionary containing the best estimator, best parameters,
        CV results, and test-set evaluation metrics.
    """
    # ------------------------------------------------------------------ #
    # 1. Train / test split                                                #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if verbose:
        print(f"Training samples : {len(X_train)}")
        print(f"Test samples     : {len(X_test)}")

    # ------------------------------------------------------------------ #
    # 2. Default hyperparameter grid                                       #
    # ------------------------------------------------------------------ #
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True, False],
        }

    # ------------------------------------------------------------------ #
    # 3. GridSearchCV                                                      #
    # ------------------------------------------------------------------ #
    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )

    if verbose:
        total_combinations = 1
        for v in param_grid.values():
            total_combinations *= len(v)
        print(f"\nFitting {cv} folds for each of {total_combinations} candidates "
              f"({cv * total_combinations} fits total)...\n")

    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    if verbose:
        print(f"\nBest CV {scoring}: {best_cv_score:.4f}")
        print(f"Best parameters  : {best_params}")

    # ------------------------------------------------------------------ #
    # 4. Evaluate on the held-out test set                                 #
    # ------------------------------------------------------------------ #
    y_pred = best_estimator.predict(X_test)
    y_proba = best_estimator.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # ROC-AUC (handles binary and multiclass)
    classes = np.unique(y)
    if len(classes) == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        roc_auc = roc_auc_score(
            y_test_bin, y_proba, multi_class="ovr", average="macro"
        )

    if verbose:
        print(f"\n{'='*50}")
        print("TEST SET EVALUATION")
        print(f"{'='*50}")
        print(f"Accuracy : {test_accuracy:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(cm)

    # ------------------------------------------------------------------ #
    # 5. Feature importances                                               #
    # ------------------------------------------------------------------ #
    feature_importances = best_estimator.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]

    if verbose:
        print("\nTop-10 Feature Importances:")
        for rank, idx in enumerate(sorted_idx[:10], 1):
            print(f"  {rank:>2}. Feature {idx:>4d} : {feature_importances[idx]:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Package and return results                                        #
    # ------------------------------------------------------------------ #
    results = {
        "best_estimator": best_estimator,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "cv_results": grid_search.cv_results_,
        "grid_search_object": grid_search,
        "test_accuracy": test_accuracy,
        "test_roc_auc": roc_auc,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importances": feature_importances,
        "feature_importance_ranking": sorted_idx,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    return results


# --------------------------------------------------------------------------- #
# Demo / smoke test                                                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Generate a synthetic classification dataset
    X, y = make_classification(
        n_samples=1_000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    # Use a smaller grid for the demo so it runs quickly
    demo_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt"],
    }

    results = tune_and_evaluate_random_forest(
        X=X,
        y=y,
        param_grid=demo_param_grid,
        test_size=0.2,