"""
AutoML-style model comparison and selection module.

Best practices followed:
- Train/test split BEFORE any preprocessing
- Transformers fit ONLY on training data
- Test set used ONLY for final evaluation
- Stratified splits throughout
- Appropriate metrics reported
- No hardcoded credentials
- Specific exception handling
- No mutable default arguments in function signatures
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Candidate model definitions
# ---------------------------------------------------------------------------

def _build_candidate_pool() -> dict[str, tuple[Pipeline, dict]]:
    """
    Return a dict mapping model name -> (pipeline, param_grid).

    All pipelines include a StandardScaler so that the scaler is fitted
    inside cross-validation folds (no data leakage).
    """
    candidates: dict[str, tuple[Pipeline, dict]] = {}

    # 1. Logistic Regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs", "liblinear"],
        "clf__penalty": ["l2"],
    }
    candidates["LogisticRegression"] = (lr_pipe, lr_grid)

    # 2. Support Vector Machine
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42)),
    ])
    svm_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
    }
    candidates["SVM"] = (svm_pipe, svm_grid)

    # 3. Random Forest
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42)),
    ])
    rf_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
    }
    candidates["RandomForest"] = (rf_pipe, rf_grid)

    # 4. Gradient Boosting
    gb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])
    gb_grid = {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth": [3, 5],
    }
    candidates["GradientBoosting"] = (gb_pipe, gb_grid)

    # 5. K-Nearest Neighbors
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ])
    knn_grid = {
        "clf__n_neighbors": [3, 5, 7, 11],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    }
    candidates["KNN"] = (knn_pipe, knn_grid)

    return candidates


# ---------------------------------------------------------------------------
# Core AutoML function
# ---------------------------------------------------------------------------

def automl_select(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """
    AutoML-style model comparison and selection.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        Binary or multiclass target labels.
    test_size : float
        Fraction of data reserved for final evaluation.
    val_size : float
        Fraction of *training* data reserved for threshold tuning /
        probability calibration (kept separate from the test set).
    random_state : int
        Random seed for reproducibility.
    cv_folds : int
        Number of stratified folds for inner GridSearchCV.
    scoring : str
        Scoring metric used by GridSearchCV (default: "roc_auc").
    n_jobs : int
        Parallel jobs for GridSearchCV.
    verbose : int
        Verbosity level.

    Returns
    -------
    best_model : fitted Pipeline
        The best model fitted on the full training partition.
    comparison_table : pd.DataFrame
        One row per candidate with best CV score and best hyperparameters.
    final_metrics : dict
        Accuracy, F1 (macro), and AUC on the held-out test set.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # ------------------------------------------------------------------
    # Step 1: Train / test split BEFORE any preprocessing
    # ------------------------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # Step 2: Carve out a validation split from the training partition
    #         for threshold tuning / calibration (never touches test set)
    # ------------------------------------------------------------------
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        stratify=y_trainval,
        random_state=random_state,
    )

    if verbose:
        print(f"Dataset sizes — train: {len(X_train)}, "
              f"val: {len(X_val)}, test: {len(X_test)}")

    # ------------------------------------------------------------------
    # Step 3: Hyperparameter search on training data only
    #         (GridSearchCV uses StratifiedKFold internally)
    # ------------------------------------------------------------------
    inner_cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    candidates = _build_candidate_pool()
    results: list[dict] = []

    for name, (pipeline, param_grid) in candidates.items():
        if verbose:
            print(f"\nTuning {name} ...")

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,          # refit on full X_train after search
            return_train_score=False,
            verbose=max(0, verbose - 1),
        )

        try:
            grid_search.fit(X_train, y_train)
        except ValueError as exc:
            warnings.warn(f"Skipping {name} due to error: {exc}")
            continue

        best_cv_score = grid_search.best_score_
        best_params = {
            k.replace("clf__", ""): v
            for k, v in grid_search.best_params_.items()
        }

        results.append({
            "model_name": name,
            "best_cv_score": best_cv_score,
            "best_params": best_params,
            "_fitted_estimator": grid_search.best_estimator_,
        })

        if verbose:
            print(f"  Best CV {scoring}: {best_cv_score:.4f}")
            print(f"  Best params: {best_params}")

    if not results:
        raise RuntimeError("All candidate models failed during tuning.")

    # ------------------------------------------------------------------
    # Step 4: Select the best model by CV score
    # ------------------------------------------------------------------
    comparison_table = pd.DataFrame([
        {
            "model_name": r["model_name"],
            "best_cv_score": r["best_cv_score"],
            "best_params": str(r["best_params"]),
        }
        for r in results
    ]).sort_values("best_cv_score", ascending=False).reset_index(drop=True)

    best_result = max(results, key=lambda r: r["best_cv_score"])
    best_model: Pipeline = best_result["_fitted_estimator"]

    if verbose:
        print(f"\nSelected model: {best_result['model_name']} "
              f"(CV {scoring}: {best_result['best_cv_score']:.4f})")

    # ------------------------------------------------------------------
    # Step 5: (Optional) Threshold tuning on validation set
    #         This keeps the test set completely untouched.
    # ------------------------------------------------------------------
    is_binary = len(np.unique(y)) == 2
    optimal_threshold = 0.5

    if is_binary:
        val_proba = best_model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1_val = -1.0
        for thr in thresholds:
            preds = (val_proba >= thr).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1_val:
                best_f1_val = f1
                optimal_threshold = thr

        if verbose:
            print(f"Optimal threshold (from val set): {optimal_threshold:.2f} "
                  f"(val F1={best_f1_val:.4f})")

    # ------------------------------------------------------------------
    # Step 6: Final evaluation on the held-out TEST set only
    # ------------------------------------------------------------------
    if is_binary:
        test_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred_test = (test_proba >= optimal_threshold).astype(int)
        auc = roc_auc_score(y_test, test_proba)
    else:
        y_pred_test = best_model.predict(X_test)
        try:
            test_proba_multi = best_model.predict_proba(X_test)
            auc = roc_auc_score(
                y_test, test_proba_multi,
                multi_class="ovr", average="macro"
            )
        except (ValueError, AttributeError):
            auc = float("nan")

    accuracy = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average="macro", zero_division=0)

    final_metrics: dict = {
        "model_name": best_result["model_name"],
        "test_accuracy": accuracy,
        "test_f1_macro": f1,
        "test_auc": auc,
        "optimal_threshold": optimal_threshold,
        "cv_scoring": scoring,
        "best_cv_score": best_result["best_cv_score"],
    }

    if verbose:
        print("\n=== Final Test Metrics ===")
        for k, v in final_metrics.items():
            print(f"  {k}: {v}")

    return best_model, comparison_table, final_metrics


# ---------------------------------------------------------------------------
# Convenience wrapper that also prints a formatted report
# ---------------------------------------------------------------------------

def run_automl(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """
    High-level entry point: runs automl_select and prints a formatted report.

    Returns
    -------
    best_model, comparison_table, final_metrics
    """
    best_model, comparison_table, final_metrics = automl_select(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        cv_folds=cv_folds,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(comparison_table.to_string(index=False))

    print("\n" + "=" * 60)
    print("BEST MODEL FINAL TEST METRICS")
    print("=" * 60)
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {key:<25}: {value:.4f}")
        else:
            print(f"  {key:<25}: {value}")

    return best_model, comparison_table, final_metrics


# ---------------------------------------------------------------------------
# Example usage (runs only when executed as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Synthetic binary classification dataset
    X_demo, y_demo = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=0,
        class_sep=0.8,
    )

    best, table, metrics = run_automl(
        X=X_demo,
        y=y_demo,
        test_size=0.20,
        val_size=0.10,
        cv_folds=5,
        scoring="roc_auc",
    )