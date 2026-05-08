import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import warnings

warnings.filterwarnings("ignore")


def automl_model_selection(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    scoring: str = "roc_auc",
    verbose: bool = True,
) -> dict:
    """
    Perform AutoML-style model comparison and selection.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector (binary or multiclass).
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random state for reproducibility.
    scoring : str, default='roc_auc'
        Scoring metric for GridSearchCV. For multiclass, 'roc_auc_ovr' is used.
    verbose : bool, default=True
        Whether to print progress and results.

    Returns
    -------
    dict with keys:
        'best_model'        : fitted best estimator
        'comparison_table'  : pd.DataFrame with per-model results
        'final_metrics'     : dict with accuracy, f1, auc on the held-out test set
        'best_model_name'   : str name of the winning model
    """
    X = np.array(X)
    y = np.array(y)

    # ------------------------------------------------------------------ #
    # Detect binary vs multiclass
    # ------------------------------------------------------------------ #
    classes = np.unique(y)
    n_classes = len(classes)
    is_binary = n_classes == 2

    # Adjust scoring for multiclass
    cv_scoring = scoring
    if not is_binary:
        if scoring == "roc_auc":
            cv_scoring = "roc_auc_ovr"
        if verbose:
            print(f"[AutoML] Multiclass problem detected ({n_classes} classes). "
                  f"Using scoring='{cv_scoring}'.")
    else:
        if verbose:
            print(f"[AutoML] Binary classification problem detected.")

    # ------------------------------------------------------------------ #
    # Stratified train / test split
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ------------------------------------------------------------------ #
    # Candidate pool with hyperparameter grids
    # ------------------------------------------------------------------ #
    candidate_pool = {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                max_iter=1000, random_state=random_state
            ),
            "param_grid": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
                "penalty": ["l2"],
            },
        },
        "SVM": {
            "estimator": SVC(
                probability=True, random_state=random_state
            ),
            "param_grid": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            },
        },
        "GradientBoosting": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "param_grid": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5],
            },
        },
        "KNeighbors": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
    }

    # ------------------------------------------------------------------ #
    # Cross-validation strategy
    # ------------------------------------------------------------------ #
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # ------------------------------------------------------------------ #
    # Grid search for each candidate
    # ------------------------------------------------------------------ #
    results = []

    for model_name, config in candidate_pool.items():
        if verbose:
            print(f"\n[AutoML] Tuning {model_name} ...")

        grid_search = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            cv=cv_strategy,
            scoring=cv_scoring,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )

        grid_search.fit(X_train, y_train)

        best_cv_score = grid_search.best_score_
        best_params = grid_search.best_params_

        if verbose:
            print(f"  Best CV {cv_scoring}: {best_cv_score:.4f}")
            print(f"  Best params: {best_params}")

        results.append(
            {
                "model_name": model_name,
                "best_cv_score": best_cv_score,
                "best_params": best_params,
                "fitted_estimator": grid_search.best_estimator_,
            }
        )

    # ------------------------------------------------------------------ #
    # Select best model
    # ------------------------------------------------------------------ #
    results_sorted = sorted(results, key=lambda r: r["best_cv_score"], reverse=True)
    best_result = results_sorted[0]
    best_model = best_result["fitted_estimator"]
    best_model_name = best_result["model_name"]

    if verbose:
        print(f"\n[AutoML] *** Best model: {best_model_name} "
              f"(CV {cv_scoring} = {best_result['best_cv_score']:.4f}) ***")

    # ------------------------------------------------------------------ #
    # Evaluate best model on held-out test set
    # ------------------------------------------------------------------ #
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary" if is_binary else "weighted")

    if is_binary:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="weighted")

    final_metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "auc": round(auc, 4),
    }

    if verbose:
        print("\n[AutoML] Final test-set metrics for best model:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # ------------------------------------------------------------------ #
    # Build comparison table
    # ------------------------------------------------------------------ #
    comparison_rows = []
    for r in results_sorted:
        comparison_rows.append(
            {
                "Model": r["model_name"],
                f"Best CV {cv_scoring}": round(r["best_cv_score"], 4),
                "Best Hyperparameters": str(r["best_params"]),
                "Selected": "✓" if r["model_name"] == best_model_name else "",
            }
        )

    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table.set_index("Model", inplace=True)

    if verbose:
        print("\n[AutoML] Model Comparison Table:")
        print(comparison_table.to_string())

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "comparison_table": comparison_table,
        "final_metrics": final_metrics,
    }


# --------------------------------------------------------------------------- #
# Demo / smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_iris

    print("=" * 60)
    print("DEMO 1: Binary classification (Breast Cancer)")
    print("=" * 60)
    data = load_breast_cancer()
    output = automl_model_selection(data.data, data.target, verbose=True)

    print("\n" + "=" * 60)
    print("DEMO 2: Multiclass classification (Iris)")
    print("=" * 60)
    data2 = load_iris()
    output2 = automl_model_selection(data2.data, data2.target, verbose=True)