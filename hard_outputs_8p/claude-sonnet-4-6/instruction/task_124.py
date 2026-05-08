"""
Cross-validated feature importance analysis with stability assessment.

This module performs rigorous feature selection using cross-validated
permutation importance with stability metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ImportanceSummary:
    """Per-feature aggregated importance statistics."""
    feature_names: list[str]
    mean_importance: np.ndarray
    std_importance: np.ndarray
    mean_rank: np.ndarray          # lower rank = more important
    std_rank: np.ndarray
    fold_importances: np.ndarray   # shape (n_folds, n_features)
    fold_rankings: np.ndarray      # shape (n_folds, n_features)


@dataclass
class StabilityMetrics:
    """Ranking stability across folds."""
    pairwise_spearman: np.ndarray  # shape (n_folds, n_folds)
    mean_spearman: float
    std_spearman: float
    top20_frequency: np.ndarray    # fraction of folds each feature is in top-20


@dataclass
class FeatureSelectionResult:
    """Complete output of the analysis pipeline."""
    importance_summary: ImportanceSummary
    stability_metrics: StabilityMetrics
    selected_features: list[str]
    selected_feature_indices: list[int]
    final_model_performance: dict[str, float]
    final_model: Pipeline
    task_type: str


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _detect_task(y: np.ndarray) -> str:
    """Detect whether the target is for classification or regression."""
    target_type = type_of_target(y)
    if target_type in ("binary", "multiclass"):
        return "classification"
    return "regression"


def _build_model(task: str, random_state: int) -> RandomForestClassifier | RandomForestRegressor:
    """Instantiate a Random Forest appropriate for the task."""
    common_kwargs = dict(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    if task == "classification":
        return RandomForestClassifier(class_weight="balanced", **common_kwargs)
    return RandomForestRegressor(**common_kwargs)


def _compute_metrics(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute task-appropriate evaluation metrics."""
    metrics: dict[str, float] = {}
    if task == "classification":
        target_type = type_of_target(y_true)
        average = "binary" if target_type == "binary" else "macro"
        metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
        if y_prob is not None:
            try:
                if target_type == "binary":
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                    metrics["average_precision"] = float(
                        average_precision_score(y_true, y_prob[:, 1])
                    )
                else:
                    metrics["roc_auc_ovr"] = float(
                        roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                    )
            except ValueError as exc:
                warnings.warn(f"Could not compute probability-based metric: {exc}")
    else:
        metrics["r2"] = float(r2_score(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return metrics


def _rank_array(arr: np.ndarray) -> np.ndarray:
    """Return ranks (1 = highest importance) for a 1-D importance array."""
    order = np.argsort(arr)[::-1]          # indices sorted by descending importance
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_cv_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list[str]] = None,
    n_splits: int = 10,
    top_k: int = 20,
    stability_threshold: float = 0.80,
    perm_n_repeats: int = 10,
    random_state: int = 42,
) -> FeatureSelectionResult:
    """
    Cross-validated feature importance analysis with stability assessment.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    feature_names : list of str, optional
        Names for each feature column.  Defaults to "f0", "f1", …
    n_splits : int
        Number of CV folds (default 10).
    top_k : int
        Number of top features to consider for stability (default 20).
    stability_threshold : float
        Minimum fraction of folds a feature must appear in the top-k
        to be selected (default 0.80).
    perm_n_repeats : int
        Number of permutation repeats per fold (default 10).
    random_state : int
        Master random seed.

    Returns
    -------
    FeatureSelectionResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length {len(feature_names)} != n_features {n_features}"
        )

    task = _detect_task(y)
    rng = np.random.default_rng(random_state)

    # -----------------------------------------------------------------------
    # Outer CV: collect per-fold importances and rankings
    # -----------------------------------------------------------------------
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_importances = np.zeros((n_splits, n_features))
    fold_rankings = np.zeros((n_splits, n_features), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        # --- strict train/test split; all preprocessing fitted on train only ---
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
        X_test_scaled = scaler.transform(X_test)          # transform test

        model = _build_model(task, random_state=int(rng.integers(0, 2**31)))
        model.fit(X_train_scaled, y_train)

        # Permutation importance evaluated on the held-out test fold
        perm_result = permutation_importance(
            model,
            X_test_scaled,
            y_test,
            n_repeats=perm_n_repeats,
            random_state=int(rng.integers(0, 2**31)),
            n_jobs=-1,
        )
        importances = perm_result.importances_mean  # shape (n_features,)
        fold_importances[fold_idx] = importances
        fold_rankings[fold_idx] = _rank_array(importances)

    # -----------------------------------------------------------------------
    # Aggregate importance statistics
    # -----------------------------------------------------------------------
    mean_importance = fold_importances.mean(axis=0)
    std_importance = fold_importances.std(axis=0)
    mean_rank = fold_rankings.mean(axis=0)
    std_rank = fold_rankings.std(axis=0)

    importance_summary = ImportanceSummary(
        feature_names=feature_names,
        mean_importance=mean_importance,
        std_importance=std_importance,
        mean_rank=mean_rank,
        std_rank=std_rank,
        fold_importances=fold_importances,
        fold_rankings=fold_rankings,
    )

    # -----------------------------------------------------------------------
    # Stability: pairwise Spearman correlation between fold rankings
    # -----------------------------------------------------------------------
    pairwise_spearman = np.ones((n_splits, n_splits))
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            corr, _ = spearmanr(fold_rankings[i], fold_rankings[j])
            pairwise_spearman[i, j] = corr
            pairwise_spearman[j, i] = corr

    upper_tri = pairwise_spearman[np.triu_indices(n_splits, k=1)]
    mean_spearman = float(upper_tri.mean())
    std_spearman = float(upper_tri.std())

    # Fraction of folds each feature appears in the top-k
    in_top_k = (fold_rankings <= top_k)          # True where rank ≤ top_k
    top_k_frequency = in_top_k.mean(axis=0)      # shape (n_features,)

    stability_metrics = StabilityMetrics(
        pairwise_spearman=pairwise_spearman,
        mean_spearman=mean_spearman,
        std_spearman=std_spearman,
        top20_frequency=top_k_frequency,
    )

    # -----------------------------------------------------------------------
    # Feature selection: consistently in top-k across ≥ stability_threshold
    # -----------------------------------------------------------------------
    stable_mask = top_k_frequency >= stability_threshold
    selected_indices = np.where(stable_mask)[0].tolist()

    # Fallback: if no feature passes the threshold, take the top-k by mean rank
    if not selected_indices:
        warnings.warn(
            f"No feature met the {stability_threshold:.0%} stability threshold. "
            f"Falling back to top-{top_k} by mean rank.",
            UserWarning,
            stacklevel=2,
        )
        selected_indices = np.argsort(mean_rank)[:top_k].tolist()

    selected_features = [feature_names[i] for i in selected_indices]

    # -----------------------------------------------------------------------
    # Final model: train on stable features, evaluate with nested CV
    # -----------------------------------------------------------------------
    X_selected = X[:, selected_indices]

    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", _build_model(task, random_state=random_state)),
        ]
    )

    # Nested CV scoring — test set never touched for model selection
    if task == "classification":
        target_type = type_of_target(y)
        scoring = (
            {"f1": "f1", "roc_auc": "roc_auc"}
            if target_type == "binary"
            else {"f1_macro": "f1_macro", "roc_auc_ovr": "roc_auc_ovr"}
        )
    else:
        scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"}

    nested_cv_results = cross_validate(
        final_pipeline,
        X_selected,
        y,
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    final_model_performance: dict[str, float] = {}
    for metric_key, scores in nested_cv_results.items():
        if metric_key.startswith("test_"):
            clean_key = metric_key[len("test_"):]
            final_model_performance[f"{clean_key}_mean"] = float(scores.mean())
            final_model_performance[f"{clean_key}_std"] = float(scores.std())

    # Fit the final pipeline on the full dataset (all stable features)
    final_pipeline.fit(X_selected, y)

    return FeatureSelectionResult(
        importance_summary=importance_summary,
        stability_metrics=stability_metrics,
        selected_features=selected_features,
        selected_feature_indices=selected_indices,
        final_model_performance=final_model_performance,
        final_model=final_pipeline,
        task_type=task,
    )


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def summarise_results(result: FeatureSelectionResult) -> pd.DataFrame:
    """
    Return a tidy DataFrame summarising per-feature importance and stability.

    Columns
    -------
    feature, mean_importance, std_importance, mean_rank, std_rank,
    top_k_frequency, selected
    """
    imp = result.importance_summary
    stab = result.stability_metrics
    selected_set = set(result.selected_features)

    df = pd.DataFrame(
        {
            "feature": imp.feature_names,
            "mean_importance": imp.mean_importance,
            "std_importance": imp.std_importance,
            "mean_rank": imp.mean_rank,
            "std_rank": imp.std_rank,
            "top_k_frequency": stab.top20_frequency,
            "selected": [f in selected_set for f in imp.feature_names],
        }
    )
    return df.sort_values("mean_rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X_demo, y_demo = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=8,
        n_redundant=4,
        random_state=0,
    )
    feat_names = [f"feature_{i:02d}" for i in range(30)]

    result = run_cv_feature_importance(
        X_demo,
        y_demo,
        feature_names=feat_names,
        n_splits=10,
        top_k=20,
        stability_threshold=0.80,
        perm_n_repeats=5,
        random_state=42,
    )

    print("=== Selected features ===")
    print(result.selected_features)

    print("\n=== Stability metrics ===")
    print(f"  Mean Spearman r across folds : {result.stability_metrics.mean_spearman:.3f}")
    print(f"  Std  Spearman r across folds : {result.stability_metrics.std_spearman:.3f}")

    print("\n=== Nested CV performance ===")
    for k, v in result.final_model_performance.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Feature summary (top 10) ===")
    summary_df = summarise_results(result)
    print(summary_df.head(10).to_string(index=False))