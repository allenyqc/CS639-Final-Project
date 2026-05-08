import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr
from typing import Optional, Union
import warnings

warnings.filterwarnings("ignore")


def cross_validated_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None,
    n_splits: int = 10,
    top_k: int = 20,
    stability_threshold: float = 0.8,
    n_estimators: int = 100,
    random_state: int = 42,
    n_repeats_permutation: int = 5,
    task: str = "classification",
) -> dict:
    """
    Perform cross-validated feature importance analysis with stability assessment.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    feature_names : list, optional
        Names of features. If None, uses 'feature_0', 'feature_1', etc.
    n_splits : int
        Number of cross-validation folds (default: 10).
    top_k : int
        Number of top features to consider for stability (default: 20).
    stability_threshold : float
        Fraction of folds a feature must appear in top-k to be selected (default: 0.8).
    n_estimators : int
        Number of trees in Random Forest (default: 100).
    random_state : int
        Random seed for reproducibility (default: 42).
    n_repeats_permutation : int
        Number of repeats for permutation importance (default: 5).
    task : str
        'classification' or 'regression' (default: 'classification').

    Returns
    -------
    dict containing:
        - importance_summary: DataFrame with mean/std importance per feature
        - stability_metrics: dict with Spearman correlations and stability scores
        - selected_features: list of stably important feature names
        - selected_feature_indices: list of indices of selected features
        - final_model_performance: dict with nested CV performance metrics
        - fold_importances: array of shape (n_splits, n_features)
        - fold_rankings: array of shape (n_splits, n_features)
    """
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    feature_names = list(feature_names)

    if len(feature_names) != n_features:
        raise ValueError(
            f"Length of feature_names ({len(feature_names)}) must match "
            f"number of features ({n_features})."
        )

    top_k = min(top_k, n_features)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_importances = np.zeros((n_splits, n_features))
    fold_rankings = np.zeros((n_splits, n_features), dtype=int)

    print(f"Running {n_splits}-fold cross-validation for feature importance...")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state + fold_idx,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        perm_imp = permutation_importance(
            rf,
            X_test,
            y_test,
            n_repeats=n_repeats_permutation,
            random_state=random_state + fold_idx,
            n_jobs=-1,
        )

        importances = perm_imp.importances_mean
        fold_importances[fold_idx] = importances

        # Rank features: rank 1 = most important
        # argsort gives indices that would sort ascending, so we reverse
        sorted_indices = np.argsort(importances)[::-1]
        ranks = np.empty(n_features, dtype=int)
        ranks[sorted_indices] = np.arange(1, n_features + 1)
        fold_rankings[fold_idx] = ranks

        print(
            f"  Fold {fold_idx + 1}/{n_splits} complete. "
            f"Top feature: {feature_names[sorted_indices[0]]} "
            f"(importance={importances[sorted_indices[0]]:.4f})"
        )

    # -------------------------------------------------------------------------
    # Aggregate importance scores across folds
    # -------------------------------------------------------------------------
    mean_importances = fold_importances.mean(axis=0)
    std_importances = fold_importances.std(axis=0)
    mean_rankings = fold_rankings.mean(axis=0)
    std_rankings = fold_rankings.std(axis=0)

    importance_summary = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_importance": mean_importances,
            "std_importance": std_importances,
            "mean_rank": mean_rankings,
            "std_rank": std_rankings,
            "cv_importance": np.where(
                mean_importances != 0, std_importances / np.abs(mean_importances), np.inf
            ),
        }
    ).sort_values("mean_importance", ascending=False).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Compute feature ranking stability using Spearman correlation
    # -------------------------------------------------------------------------
    print("\nComputing ranking stability via Spearman correlations...")

    n_pairs = n_splits * (n_splits - 1) // 2
    spearman_correlations = np.zeros(n_pairs)
    pair_labels = []
    pair_idx = 0

    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            corr, _ = spearmanr(fold_rankings[i], fold_rankings[j])
            spearman_correlations[pair_idx] = corr
            pair_labels.append(f"fold_{i+1}_vs_fold_{j+1}")
            pair_idx += 1

    mean_spearman = float(np.mean(spearman_correlations))
    std_spearman = float(np.std(spearman_correlations))
    min_spearman = float(np.min(spearman_correlations))
    max_spearman = float(np.max(spearman_correlations))

    # Per-feature stability: fraction of folds where feature is in top-k
    top_k_counts = np.zeros(n_features)
    for fold_idx in range(n_splits):
        top_k_indices = np.argsort(fold_importances[fold_idx])[::-1][:top_k]
        top_k_counts[top_k_indices] += 1

    feature_stability_scores = top_k_counts / n_splits

    stability_metrics = {
        "mean_spearman_correlation": mean_spearman,
        "std_spearman_correlation": std_spearman,
        "min_spearman_correlation": min_spearman,
        "max_spearman_correlation": max_spearman,
        "pairwise_correlations": dict(zip(pair_labels, spearman_correlations.tolist())),
        "feature_stability_scores": dict(zip(feature_names, feature_stability_scores.tolist())),
        "overall_stability_interpretation": _interpret_stability(mean_spearman),
    }

    print(f"  Mean Spearman correlation: {mean_spearman:.4f} ± {std_spearman:.4f}")

    # -------------------------------------------------------------------------
    # Select features consistently ranked in top-k across >= stability_threshold folds
    # -------------------------------------------------------------------------
    stable_mask = feature_stability_scores >= stability_threshold
    selected_feature_indices = np.where(stable_mask)[0].tolist()
    selected_features = [feature_names[i] for i in selected_feature_indices]

    if len(selected_features) == 0:
        print(
            f"\nWarning: No features met the stability threshold of {stability_threshold:.0%}. "
            f"Relaxing to top-{top_k} by mean importance."
        )
        top_by_mean = np.argsort(mean_importances)[::-1][:top_k]
        selected_feature_indices = top_by_mean.tolist()
        selected_features = [feature_names[i] for i in selected_feature_indices]

    print(
        f"\nSelected {len(selected_features)} stable features "
        f"(threshold: top-{top_k} in >= {stability_threshold:.0%} of folds):"
    )
    for feat in selected_features:
        score = feature_stability_scores[feature_names.index(feat)]
        print(f"  {feat}: stability={score:.2f}")

    # -------------------------------------------------------------------------
    # Train final model using only stable features + nested CV evaluation
    # -------------------------------------------------------------------------
    print("\nEvaluating final model with nested cross-validation...")

    X_selected = X[:, selected_feature_indices]

    final_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )

    # Nested CV: outer loop for performance estimation
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    nested_accuracy = cross_val_score(
        final_rf, X_selected, y, cv=outer_cv, scoring="accuracy", n_jobs=-1
    )

    # AUC (only for binary classification)
    unique_classes = np.unique(y)
    nested_auc = None
    if len(unique_classes) == 2:
        nested_auc = cross_val_score(
            final_rf, X_selected, y, cv=outer_cv, scoring="roc_auc", n_jobs=-1
        )

    # Also evaluate full model for comparison
    full_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    full_accuracy = cross_val_score(
        full_rf, X, y, cv=outer_cv, scoring="accuracy", n_jobs=-1
    )

    final_model_performance = {
        "selected_features_accuracy_mean": float(nested_accuracy.mean()),
        "selected_features_accuracy_std": float(nested_accuracy.std()),
        "selected_features_accuracy_per_fold": nested_accuracy.tolist(),
        "full_model_accuracy_mean": float(full_accuracy.mean()),
        "full_model_accuracy_std": float(full_accuracy.std()),
        "full_model_accuracy_per_fold": full_accuracy.tolist(),
        "n_selected_features": len(selected_features),
        "n_total_features": n_features,
        "feature_reduction_pct": (1 - len(selected_features) / n_features) * 100,
    }

    if nested_auc is not None:
        full_auc = cross_val_score(
            full_rf, X, y, cv=outer_cv, scoring="roc_auc", n_jobs=-1
        )
        final_model_performance.update(
            {
                "selected_features_auc_mean": float(nested_auc.mean()),
                "selected_features_auc_std": float(nested_auc.std()),
                "selected_features_auc_per_fold": nested_auc.tolist(),
                "full_model_auc_mean": float(full_auc.mean()),
                "full_model_auc_std": float(full_auc.std()),
                "full_model_auc_per_fold": full_auc.tolist(),
            }
        )

    # Train final model on all data with selected features
    final_rf.fit(X_selected, y)

    print("\n=== Final Model Performance (Nested CV) ===")
    print(
        f"  Selected features accuracy: "
        f"{final_model_performance['selected_features_accuracy_mean']:.4f} ± "
        f"{final_model_performance['selected_features_accuracy_std']:.4f}"
    )
    print(
        f"  Full model accuracy:        "
        f"{final_model_performance['full_model_accuracy_mean']:.4f} ± "
        f"{final_model_performance['full_model_accuracy_std']:.4f}"
    )
    if nested_auc is not None:
        print(
            f"  Selected features AUC:      "
            f"{final_model_performance['selected_features_auc_mean']:.4f} ± "
            f"{final_model_performance['selected_features_auc_std']:.4f}"
        )
    print(
        f"  Feature reduction:          "
        f"{final_model_performance['feature_reduction_pct']:.1f}% "
        f"({n_features} → {len(selected_features)} features)"
    )

    return {
        "importance_summary": importance_summary,
        "stability_metrics": stability_metrics,
        "selected_features": selected_features,
        "selected_feature_indices": selected_feature_indices,
        "final_model_performance": final_model_performance,
        "final_model": final_rf,
        "fold_importances": fold_importances,
        "fold_rankings": fold_rankings,
    }


def _interpret_stability(mean_spearman: float) -> str:
    """Interpret the mean Spearman correlation as a stability label."""
    if mean_spearman >= 0.9:
        return "Very High Stability"
    elif mean_spearman >= 0.75:
        return "High Stability"
    elif mean_spearman >= 0.5:
        return "Moderate Stability"
    elif mean_spearman >= 0.25:
        return "Low Stability"
    else:
        return "Very Low Stability"


def summarize_results(results: dict) -> None:
    """
    Print a human-readable summary of the analysis results.

    Parameters
    ----------
    results : dict
        Output from cross_validated_feature_importance().
    """
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n--- Top 10 Features by Mean Importance ---")
    top10 = results["importance_summary"].head(10)
    for _, row in top10.iterrows():
        stability = results["stability_metrics"]["feature_stability_scores"].get(
            row["feature"], 0.0
        )
        print(
            f"  {row['feature']:30s}  importance={row['mean_importance']:+.4f} ± {row['std_importance']:.4f}  "
            f"rank={row['mean_rank']:.1f} ± {row['std_rank']:.1f}  stability={stability:.2f}"
        )

    print("\n--- Ranking Stability ---")
    sm = results["stability_metrics"]
    print(f"  Mean Spearman correlation: {sm['mean_spearman_correlation']:.4f} ± {sm['std_spearman_correlation']:.4f}")
    print(f"  Range: [{sm['min_spearman_correlation']:.4f}, {sm['max_spearman_correlation']:.4f}]")
    print(f"  Interpretation: {sm['overall_stability_interpretation']}")

    print("\n--- Selected Stable Features ---")
    for feat in results["selected_features"]:
        score = sm["feature_stability_scores"].get(feat, 0.0)
        print(f"  {feat} (stability={score:.2f})")

    print("\n--- Final Model Performance ---")
    perf = results["final_model_performance"]
    print(f"  Accuracy (selected): {perf['selected_features_accuracy_mean']:.4f} ± {perf['selected_features_accuracy_std']:.4f}")
    print(f"  Accuracy (full):     {perf['full_model_accuracy_mean']:.4f} ± {perf['full_model_accuracy_std']:.4f}")
    if "selected_features_auc_mean" in perf:
        print(f"  AUC (selected):      {perf['selected_features_auc_mean']:.4f} ± {perf['selected_features_auc_std']:.4f}")
        print(f"  AUC (full):          {perf['full_model_auc_mean']:.4f} ± {perf['full_model_auc_std']:.4f}")
    print(
        f"  Feature reduction:   {perf['n_total_features']} → {perf['n_selected_features']} "
        f"({perf['feature_reduction_pct']:.1f}% reduction)"
    )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    print("Generating synthetic dataset...")
    X_demo, y_demo = make_classification(
        n_samples=500,
        n_features=30,
        n_informative=10,
        n_redundant=5,
        n_repeated=2,
        random_state=0,
    )
    feature_names_demo = [f"feat_{i:02d}" for i in range(X_demo.shape[1])]

    results = cross_validated_feature_importance(
        X=X_demo,
        y=y_demo,
        feature_names=feature_names_demo,
        n_splits=10,
        top_k=20,
        stability_threshold=0.8,
        n_estimators=100,
        random_state=42,
        n_repeats_permutation=5,
    )

    summarize_results(results)