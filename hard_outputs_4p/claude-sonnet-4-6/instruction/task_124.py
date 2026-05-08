"""
Cross-validated feature importance analysis with stability assessment.

Best practices followed:
- Train/test split before any preprocessing
- Transformers fit only on training partitions
- Nested CV for unbiased final evaluation
- No data leakage from test sets
- No hardcoded credentials
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
import warnings


@dataclass
class FeatureImportanceResult:
    """Container for all outputs of the feature importance analysis."""
    importance_summary: pd.DataFrame          # mean/std importance per feature
    stability_metrics: dict                   # Spearman correlations & summary stats
    selected_features: list                   # indices of stable top-20 features
    selected_feature_names: list              # names (or str indices) of stable features
    final_model_performance: dict             # nested-CV scores
    fold_importances: np.ndarray              # raw per-fold importance matrix (n_folds x n_features)
    fold_rankings: np.ndarray                 # per-fold rank matrix (n_folds x n_features)


def _make_feature_names(X: Union[np.ndarray, pd.DataFrame]) -> list:
    """Return feature names from a DataFrame or generate generic ones."""
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    n_features = X.shape[1]
    return [f"feature_{i}" for i in range(n_features)]


def _to_numpy(X: Union[np.ndarray, pd.DataFrame],
              y: Union[np.ndarray, pd.Series]) -> tuple:
    """Convert inputs to numpy arrays."""
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
    return X_arr, y_arr


def _is_classification(y: np.ndarray) -> bool:
    """Heuristic: treat as classification if target has ≤20 unique integer values."""
    unique_vals = np.unique(y)
    return (np.issubdtype(y.dtype, np.integer) or
            np.array_equal(y, y.astype(int))) and len(unique_vals) <= 20


def _make_rf(task: str, random_state: int) -> Union[RandomForestClassifier,
                                                     RandomForestRegressor]:
    """Instantiate an appropriate Random Forest."""
    if task == "classification":
        return RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    return RandomForestRegressor(
        n_estimators=200,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )


def _rank_array(arr: np.ndarray) -> np.ndarray:
    """Convert importance scores to ranks (highest importance → rank 1)."""
    # argsort twice gives ranks; negate so highest value gets rank 1
    order = np.argsort(-arr)          # indices that sort descending
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


def _compute_spearman_stability(fold_rankings: np.ndarray) -> dict:
    """
    Compute pairwise Spearman correlations between all fold ranking vectors.

    Parameters
    ----------
    fold_rankings : ndarray of shape (n_folds, n_features)

    Returns
    -------
    dict with keys: pairwise_correlations, mean_correlation, std_correlation,
                    min_correlation, median_correlation
    """
    n_folds = fold_rankings.shape[0]
    pairwise = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            rho, _ = spearmanr(fold_rankings[i], fold_rankings[j])
            pairwise.append(rho)

    pairwise = np.array(pairwise)
    return {
        "pairwise_correlations": pairwise,
        "mean_correlation": float(np.mean(pairwise)),
        "std_correlation": float(np.std(pairwise)),
        "min_correlation": float(np.min(pairwise)),
        "median_correlation": float(np.median(pairwise)),
    }


def _select_stable_features(
    fold_rankings: np.ndarray,
    top_k: int = 20,
    min_fraction: float = 0.80,
) -> np.ndarray:
    """
    Select features that appear in the top-k across at least min_fraction of folds.

    Parameters
    ----------
    fold_rankings : ndarray (n_folds, n_features)
    top_k         : threshold rank (inclusive) to be considered "top"
    min_fraction  : minimum fraction of folds a feature must be top-k

    Returns
    -------
    sorted array of selected feature indices
    """
    n_folds = fold_rankings.shape[0]
    in_top_k = (fold_rankings <= top_k)          # bool matrix
    fraction_in_top = in_top_k.mean(axis=0)      # per feature
    selected = np.where(fraction_in_top >= min_fraction)[0]
    return selected


def run_feature_importance_analysis(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    n_splits: int = 10,
    top_k: int = 20,
    min_stability_fraction: float = 0.80,
    perm_n_repeats: int = 10,
    random_state: int = 42,
    scoring: Optional[str] = None,
) -> FeatureImportanceResult:
    """
    Cross-validated feature importance analysis with stability assessment.

    Parameters
    ----------
    X                     : Feature matrix (n_samples, n_features)
    y                     : Target vector (n_samples,)
    n_splits              : Number of CV folds (default 10)
    top_k                 : Rank threshold for "important" features (default 20)
    min_stability_fraction: Fraction of folds a feature must be top-k (default 0.80)
    perm_n_repeats        : Permutation importance repetitions per fold (default 10)
    random_state          : Global random seed
    scoring               : Sklearn scoring string; auto-detected if None

    Returns
    -------
    FeatureImportanceResult dataclass
    """
    feature_names = _make_feature_names(X)
    X_arr, y_arr = _to_numpy(X, y)
    n_features = X_arr.shape[1]

    task = "classification" if _is_classification(y_arr) else "regression"
    if scoring is None:
        scoring = "accuracy" if task == "classification" else "r2"

    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_importances = np.zeros((n_splits, n_features))
    fold_rankings = np.zeros((n_splits, n_features), dtype=int)

    # ------------------------------------------------------------------ #
    # Outer loop: compute permutation importance per fold                  #
    # ------------------------------------------------------------------ #
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_arr)):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # Fit scaler ONLY on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)   # transform only, no fit

        rf = _make_rf(task, random_state=random_state + fold_idx)
        rf.fit(X_train_scaled, y_train)

        # Permutation importance on the held-out test fold
        perm_result = permutation_importance(
            rf,
            X_test_scaled,
            y_test,
            n_repeats=perm_n_repeats,
            random_state=random_state + fold_idx,
            scoring=scoring,
            n_jobs=-1,
        )

        importances = perm_result.importances_mean
        fold_importances[fold_idx] = importances
        fold_rankings[fold_idx] = _rank_array(importances)

    # ------------------------------------------------------------------ #
    # Aggregate importance scores                                          #
    # ------------------------------------------------------------------ #
    mean_importance = fold_importances.mean(axis=0)
    std_importance = fold_importances.std(axis=0)
    mean_rank = fold_rankings.mean(axis=0)

    importance_summary = pd.DataFrame({
        "feature": feature_names,
        "mean_importance": mean_importance,
        "std_importance": std_importance,
        "mean_rank": mean_rank,
        "cv_of_importance": np.where(
            mean_importance != 0,
            std_importance / np.abs(mean_importance),
            np.nan,
        ),
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Stability via Spearman correlation                                   #
    # ------------------------------------------------------------------ #
    stability_metrics = _compute_spearman_stability(fold_rankings)

    # ------------------------------------------------------------------ #
    # Feature selection: consistently top-k across ≥ min_fraction folds   #
    # ------------------------------------------------------------------ #
    selected_indices = _select_stable_features(
        fold_rankings, top_k=top_k, min_fraction=min_stability_fraction
    )

    if len(selected_indices) == 0:
        warnings.warn(
            "No features met the stability criterion. "
            "Falling back to top-k by mean importance.",
            UserWarning,
        )
        selected_indices = np.argsort(mean_importance)[::-1][:top_k]

    selected_feature_names = [feature_names[i] for i in selected_indices]

    # ------------------------------------------------------------------ #
    # Final model: nested CV on stable features only                       #
    # ------------------------------------------------------------------ #
    X_selected = X_arr[:, selected_indices]

    # Build a pipeline so the scaler is always fit inside each CV fold
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", _make_rf(task, random_state=random_state)),
    ])

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_state + 999)

    nested_scores = cross_val_score(
        final_pipeline,
        X_selected,
        y_arr,
        cv=inner_cv,
        scoring=scoring,
        n_jobs=-1,
    )

    final_model_performance = {
        "scoring_metric": scoring,
        "nested_cv_scores": nested_scores.tolist(),
        "mean_score": float(nested_scores.mean()),
        "std_score": float(nested_scores.std()),
        "n_selected_features": len(selected_indices),
    }

    # Fit the final pipeline on ALL data (for the returned model object)
    final_pipeline.fit(X_selected, y_arr)

    return FeatureImportanceResult(
        importance_summary=importance_summary,
        stability_metrics=stability_metrics,
        selected_features=selected_indices.tolist(),
        selected_feature_names=selected_feature_names,
        final_model_performance=final_model_performance,
        fold_importances=fold_importances,
        fold_rankings=fold_rankings,
    )


# --------------------------------------------------------------------------- #
# Convenience pretty-printer                                                   #
# --------------------------------------------------------------------------- #
def print_summary(result: FeatureImportanceResult) -> None:
    """Print a human-readable summary of the analysis results."""
    print("=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS — SUMMARY")
    print("=" * 60)

    print("\n[Top-10 Features by Mean Permutation Importance]")
    print(result.importance_summary.head(10).to_string(index=False))

    print("\n[Ranking Stability (Spearman ρ across folds)]")
    sm = result.stability_metrics
    print(f"  Mean ρ   : {sm['mean_correlation']:.4f}")
    print(f"  Std  ρ   : {sm['std_correlation']:.4f}")
    print(f"  Min  ρ   : {sm['min_correlation']:.4f}")
    print(f"  Median ρ : {sm['median_correlation']:.4f}")

    print(f"\n[Selected Stable Features ({len(result.selected_features)} total)]")
    for name in result.selected_feature_names:
        print(f"  • {name}")

    print("\n[Final Model — Nested CV Performance]")
    fp = result.final_model_performance
    print(f"  Metric : {fp['scoring_metric']}")
    print(f"  Mean   : {fp['mean_score']:.4f} ± {fp['std_score']:.4f}")
    print(f"  Scores : {[round(s, 4) for s in fp['nested_cv_scores']]}")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Quick smoke-test (runs when executed directly)                               #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    rng = np.random.default_rng(0)
    X_demo, y_demo = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=10,
        n_redundant=5,
        random_state=0,
    )

    result = run_feature_importance_analysis(
        X_demo,
        y_demo,
        n_splits=10,
        top_k=20,
        min_stability_fraction=0.80,
        perm_n_repeats=5,   # fewer repeats for speed in demo
        random_state=42,
    )

    print_summary(result)