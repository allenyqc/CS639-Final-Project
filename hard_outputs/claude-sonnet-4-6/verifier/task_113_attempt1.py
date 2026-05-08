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
) -> tuple:
    """
    Perform AutoML-style model comparison and selection.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split.
    random_state : int, optional (default=42)
        Random state for reproducibility.
    scoring : str, optional (default='roc_auc')
        Scoring metric for GridSearchCV. Use 'roc_auc_ovr' for multiclass.
    verbose : bool, optional (default=True)
        Whether to print progress information.

    Returns
    -------
    best_model : estimator
        The best fitted model (refitted on the full training set).
    comparison_table : pd.DataFrame
        DataFrame showing each model's best CV score and best hyperparameters.
    final_metrics : dict
        Dictionary containing accuracy, F1, and AUC on the held-out test set.
    """
    X = np.array(X)
    y = np.array(y)

    # ------------------------------------------------------------------ #
    # Detect binary vs. multiclass                                         #
    # ------------------------------------------------------------------ #
    classes = np.unique(y)
    n_classes = len(classes)
    is_multiclass = n_classes > 2

    if is_multiclass and scoring == "roc_auc":
        scoring = "roc_auc_ovr"
        if verbose:
            print(
                f"[AutoML] Multiclass problem detected ({n_classes} classes). "
                f"Switching scoring to '{scoring}'."
            )

    # ------------------------------------------------------------------ #
    # Stratified train / test split                                        #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # ------------------------------------------------------------------ #
    # Candidate pool with hyperparameter grids                             #
    # ------------------------------------------------------------------ #
    candidate_pool = {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                max_iter=1000, random_state=random_state
            ),
            "param_grid": {
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["lbfgs", "liblinear"],
                "penalty": ["l2"],
            },
        },
        "SVM": {
            "estimator": SVC(
                probability=True, random_state=random_state
            ),
            "param_grid": {
                "C": [0.1, 1, 10],
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
    # Grid search for each candidate                                       #
    # ------------------------------------------------------------------ #
    results = []

    for model_name, config in candidate_pool.items():
        if verbose:
            print(f"[AutoML] Tuning {model_name} ...")

        grid_search = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X_train, y_train)

        results.append(
            {
                "Model": model_name,
                "Best_CV_Score": round(grid_search.best_score_, 6),
                "Best_Params": grid_search.best_params_,
                "GridSearchCV_Object": grid_search,
            }
        )

        if verbose:
            print(
                f"         Best CV {scoring}: {grid_search.best_score_:.4f} | "
                f"Params: {grid_search.best_params_}"
            )

    # ------------------------------------------------------------------ #
    # Select the best model                                                #
    # ------------------------------------------------------------------ #
    results_sorted = sorted(results, key=lambda r: r["Best_CV_Score"], reverse=True)
    best_result = results_sorted[0]
    best_model = best_result["GridSearchCV_Object"].best_estimator_

    if verbose:
        print(
            f"\n[AutoML] Best model: {best_result['Model']} "
            f"(CV {scoring} = {best_result['Best_CV_Score']:.4f})"
        )

    # ------------------------------------------------------------------ #
    # Evaluate on the held-out test set                                    #
    # ------------------------------------------------------------------ #
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    if is_multiclass:
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")
    else:
        auc = roc_auc_score(y_test, y_prob[:, 1])

    final_metrics = {
        "model_name": best_result["Model"],
        "accuracy": round(accuracy, 6),
        "f1_weighted": round(f1, 6),
        "auc": round(auc, 6),
        "best_cv_score": round(best_result["Best_CV_Score"], 6),
        "best_params": best_result["Best_Params"],
    }

    if verbose:
        print("\n[AutoML] Final Test-Set Metrics for Best Model:")
        print(f"         Accuracy  : {accuracy:.4f}")
        print(f"         F1 (wtd)  : {f1:.4f}")
        print(f"         AUC       : {auc:.4f}")

    # ------------------------------------------------------------------ #
    # Build comparison table                                               #
    # ------------------------------------------------------------------ #
    comparison_table = pd.DataFrame(
        [
            {
                "Model": r["Model"],
                "Best_CV_Score": r["Best_CV_Score"],
                "Best_Params": str(r["Best_Params"]),
            }
            for r in results_sorted
        ]
    ).reset_index(drop=True)

    comparison_table.index = comparison_table.index + 1  # rank starts at 1
    comparison_table.index.name = "Rank"

    if verbose:
        print("\n[AutoML] Model Comparison Table:")
        print(comparison_table.to_string())

    return best_model, comparison_table, final_metrics


# --------------------------------------------------------------------------- #
# Quick demo / smoke test                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_iris

    print("=" * 60)
    print("DEMO 1: Binary classification (Breast Cancer)")
    print("=" * 60)
    data = load_breast_cancer()
    best_model, table, metrics = automl_model_selection(
        data.data, data.target, verbose=True
    )
    print("\nReturned metrics dict:", metrics)

    print("\n" + "=" * 60)
    print("DEMO 2: Multiclass classification (Iris)")
    print("=" * 60)
    data2 = load_iris()
    best_model2, table2, metrics2 = automl_model_selection(
        data2.data, data2.target, verbose=True
    )
    print("\nReturned metrics dict:", metrics2)