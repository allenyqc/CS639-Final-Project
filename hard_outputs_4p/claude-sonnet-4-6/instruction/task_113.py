"""
AutoML-style model comparison and selection module.

Best practices followed:
- Train/test split BEFORE any preprocessing
- Transformers fit ONLY on training data
- Hyperparameter tuning via GridSearchCV on training set only
- Final metrics reported on held-out test set
- Stratified splits throughout
- No hardcoded credentials
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Candidate model definitions
# ---------------------------------------------------------------------------

def _build_candidate_pool() -> list[dict[str, Any]]:
    """
    Return a list of dicts, each describing one candidate model.

    Each dict contains:
        name        : human-readable label
        pipeline    : sklearn Pipeline (scaler + estimator)
        param_grid  : hyperparameter grid for GridSearchCV
    """
    candidates = [
        {
            "name": "LogisticRegression",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__solver": ["lbfgs", "liblinear"],
                "clf__penalty": ["l2"],
            },
        },
        {
            "name": "SVM",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, random_state=42)),
            ]),
            "param_grid": {
                "clf__C": [0.1, 1.0, 10.0],
                "clf__kernel": ["rbf", "linear"],
                "clf__gamma": ["scale", "auto"],
            },
        },
        {
            "name": "RandomForest",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(random_state=42)),
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_split": [2, 5],
            },
        },
        {
            "name": "GradientBoosting",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(random_state=42)),
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__max_depth": [3, 5],
            },
        },
        {
            "name": "KNearestNeighbors",
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            "param_grid": {
                "clf__n_neighbors": [3, 5, 7, 11],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"],
            },
        },
    ]
    return candidates


# ---------------------------------------------------------------------------
# Core AutoML function
# ---------------------------------------------------------------------------

def automl_select(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    test_size: float = 0.20,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> tuple[Pipeline, pd.DataFrame, dict[str, float]]:
    """
    AutoML-style model comparison and selection.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector (binary or multiclass).
    test_size : float, default=0.20
        Fraction of data reserved for final evaluation.
    cv_folds : int, default=5
        Number of stratified CV folds for hyperparameter search.
    scoring : str, default="roc_auc"
        Scoring metric used inside GridSearchCV.
    random_state : int, default=42
        Seed for reproducibility.
    n_jobs : int, default=-1
        Parallel jobs for GridSearchCV.
    verbose : bool, default=True
        Print progress information.

    Returns
    -------
    best_model : fitted Pipeline
        The best model, already fitted on the full training set.
    comparison_table : pd.DataFrame
        One row per candidate with best CV score and best hyperparameters.
    final_metrics : dict
        Accuracy, F1 (weighted), and AUC evaluated on the held-out test set.
    """
    # ------------------------------------------------------------------
    # 1. Convert inputs to numpy for consistency
    # ------------------------------------------------------------------
    X = np.array(X)
    y = np.array(y)

    classes = np.unique(y)
    n_classes = len(classes)
    is_binary = n_classes == 2

    # ------------------------------------------------------------------
    # 2. Stratified train / test split — BEFORE any preprocessing
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    if verbose:
        print(f"Dataset  : {X.shape[0]} samples, {X.shape[1]} features, "
              f"{n_classes} classes")
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set : {X_test.shape[0]} samples")
        print(f"CV folds : {cv_folds}  |  Scoring : {scoring}\n")

    # ------------------------------------------------------------------
    # 3. Inner CV strategy (stratified)
    # ------------------------------------------------------------------
    inner_cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # 4. Hyperparameter search for each candidate
    #    All fitting happens on X_train / y_train only.
    # ------------------------------------------------------------------
    candidates = _build_candidate_pool()
    results: list[dict[str, Any]] = []

    for candidate in candidates:
        name = candidate["name"]
        if verbose:
            print(f"  Tuning {name} ...", end=" ", flush=True)

        grid_search = GridSearchCV(
            estimator=candidate["pipeline"],
            param_grid=candidate["param_grid"],
            cv=inner_cv,
            scoring=scoring,
            refit=True,          # refit best params on full X_train
            n_jobs=n_jobs,
            error_score="raise",
        )
        grid_search.fit(X_train, y_train)

        best_cv_score = grid_search.best_score_
        best_params = {
            k.replace("clf__", ""): v
            for k, v in grid_search.best_params_.items()
        }

        results.append({
            "name": name,
            "best_cv_score": best_cv_score,
            "best_params": best_params,
            "fitted_pipeline": grid_search.best_estimator_,
        })

        if verbose:
            print(f"best CV {scoring} = {best_cv_score:.4f}")

    # ------------------------------------------------------------------
    # 5. Select best model by CV score (inner loop only — no test leakage)
    # ------------------------------------------------------------------
    results_sorted = sorted(results, key=lambda r: r["best_cv_score"], reverse=True)
    best_result = results_sorted[0]
    best_model: Pipeline = best_result["fitted_pipeline"]

    if verbose:
        print(f"\nBest model (by CV): {best_result['name']} "
              f"(CV {scoring} = {best_result['best_cv_score']:.4f})")

    # ------------------------------------------------------------------
    # 6. Final evaluation on held-out test set
    #    The test set is touched ONLY here, after model selection is done.
    # ------------------------------------------------------------------
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    if is_binary:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        auc = roc_auc_score(
            y_test, y_proba,
            multi_class="ovr",
            average="weighted",
        )

    final_metrics: dict[str, float] = {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "auc": auc,
    }

    if verbose:
        print("\n=== Final Test-Set Metrics ===")
        for metric, value in final_metrics.items():
            print(f"  {metric:15s}: {value:.4f}")

    # ------------------------------------------------------------------
    # 7. Build comparison table
    # ------------------------------------------------------------------
    comparison_rows = []
    for r in results_sorted:
        comparison_rows.append({
            "model": r["name"],
            f"best_cv_{scoring}": round(r["best_cv_score"], 4),
            "best_hyperparameters": str(r["best_params"]),
        })

    comparison_table = pd.DataFrame(comparison_rows).reset_index(drop=True)

    if verbose:
        print("\n=== Model Comparison Table ===")
        print(comparison_table.to_string(index=False))

    return best_model, comparison_table, final_metrics


# ---------------------------------------------------------------------------
# Convenience wrapper that also returns a rich summary dict
# ---------------------------------------------------------------------------

def run_automl(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Thin wrapper around :func:`automl_select` that bundles all outputs
    into a single dictionary for easy downstream use.

    Returns
    -------
    dict with keys:
        best_model        : fitted Pipeline
        comparison_table  : pd.DataFrame
        final_metrics     : dict[str, float]
    """
    best_model, comparison_table, final_metrics = automl_select(X, y, **kwargs)
    return {
        "best_model": best_model,
        "comparison_table": comparison_table,
        "final_metrics": final_metrics,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test (runs only when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    result = run_automl(data.data, data.target, verbose=True)

    print("\nReturned keys:", list(result.keys()))
    print("Best model type:", type(result["best_model"].named_steps["clf"]).__name__)
    print("Final metrics:", result["final_metrics"])