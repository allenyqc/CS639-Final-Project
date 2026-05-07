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
    param_grid=None,
    test_size=0.2,
    val_size=0.1,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=1,
):
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
    test_size : float, default=0.2
        Proportion of data reserved for the final test set.
    val_size : float, default=0.1
        Proportion of training data used for cross-validation (informational only).
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default="accuracy"
        Scoring metric for GridSearchCV.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all available cores).
    verbose : int, default=1
        Verbosity level.

    Returns
    -------
    results : dict
        Dictionary containing:
            - "best_params": best hyperparameters found
            - "best_cv_score": best cross-validation score
            - "test_accuracy": accuracy on the test set
            - "classification_report": full classification report
            - "confusion_matrix": confusion matrix
            - "roc_auc": ROC-AUC score (OvR for multiclass)
            - "best_model": fitted RandomForestClassifier with best params
            - "grid_search": fitted GridSearchCV object
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

    if verbose:
        total_combinations = 1
        for v in param_grid.values():
            total_combinations *= len(v)
        print(f"\nTotal hyperparameter combinations : {total_combinations}")
        print(f"Cross-validation folds            : {cv}")
        print(f"Total fits                        : {total_combinations * cv}\n")

    # ------------------------------------------------------------------ #
    # 3. GridSearchCV                                                      #
    # ------------------------------------------------------------------ #
    base_rf = RandomForestClassifier(random_state=random_state)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    if verbose:
        print("\n" + "=" * 60)
        print("GRID SEARCH RESULTS")
        print("=" * 60)
        print(f"Best CV {scoring}: {best_cv_score:.4f}")
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

    # ------------------------------------------------------------------ #
    # 4. Evaluate on the held-out test set                                 #
    # ------------------------------------------------------------------ #
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # ROC-AUC: binary vs. multiclass
    classes = np.unique(y)
    if len(classes) == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")

    if verbose:
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)
        print(f"Test Accuracy : {test_accuracy:.4f}")
        print(f"ROC-AUC       : {roc_auc:.4f}")
        print("\nClassification Report:")
        print(clf_report)
        print("Confusion Matrix:")
        print(conf_matrix)

    # ------------------------------------------------------------------ #
    # 5. Feature importances (top 10)                                      #
    # ------------------------------------------------------------------ #
    feature_importances = best_model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1]

    if verbose:
        print("\nTop-10 Feature Importances:")
        for rank, idx in enumerate(sorted_idx[:10], start=1):
            print(f"  {rank:2d}. Feature {idx:4d} : {feature_importances[idx]:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Return results                                                    #
    # ------------------------------------------------------------------ #
    results = {
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "test_accuracy": test_accuracy,
        "classification_report": clf_report,
        "confusion_matrix": conf_matrix,
        "roc_auc": roc_auc,
        "feature_importances": feature_importances,
        "best_model": best_model,
        "grid_search": grid_search,
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

    # Use a smaller grid for the demo to keep runtime short
    demo_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "max_features":