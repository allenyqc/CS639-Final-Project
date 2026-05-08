import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import spearmanr
from typing import Optional, Union
import warnings
warnings.filterwarnings('ignore')


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
        Fraction of folds in which a feature must appear in top-k (default: 0.8).
    n_estimators : int
        Number of trees in the Random Forest (default: 100).
    random_state : int
        Random seed for reproducibility (default: 42).
    n_repeats_permutation : int
        Number of repeats for permutation importance (default: 5).

    Returns
    -------
    dict with keys:
        - importance_summary : pd.DataFrame with mean/std importance per feature
        - stability_metrics  : dict with Spearman correlations and mean stability
        - selected_features  : list of feature names selected as stable
        - final_model_performance : dict with nested CV scores
        - fold_importances   : np.ndarray of shape (n_splits, n_features)
        - fold_rankings      : np.ndarray of shape (n_splits, n_features)
    """
    X = np.asarray(X)
    y = np.asarray(y)
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

    # ------------------------------------------------------------------ #
    # Step 1 – 10-fold CV: train RF, compute permutation importance        #
    # ------------------------------------------------------------------ #
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_importances = np.zeros((n_splits, n_features))   # mean importance per fold
    fold_rankings    = np.zeros((n_splits, n_features), dtype=int)  # rank per feature

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state + fold_idx,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        perm_result = permutation_importance(
            rf, X_test, y_test,
            n_repeats=n_repeats_permutation,
            random_state=random_state + fold_idx,
            n_jobs=-1,
        )

        importances = perm_result.importances_mean
        fold_importances[fold_idx] = importances

        # Rank: rank 1 = most important (highest importance value)
        # argsort gives ascending order; we want descending → negate
        order = np.argsort(-importances)          # indices sorted best→worst
        ranks = np.empty(n_features, dtype=int)
        ranks[order] = np.arange(1, n_features + 1)
        fold_rankings[fold_idx] = ranks

    # ------------------------------------------------------------------ #
    # Step 2 – Aggregate importance scores across folds                   #
    # ------------------------------------------------------------------ #
    mean_importance = fold_importances.mean(axis=0)
    std_importance  = fold_importances.std(axis=0)
    mean_rank       = fold_rankings.mean(axis=0)
    std_rank        = fold_rankings.std(axis=0)

    importance_summary = pd.DataFrame({
        "feature":         feature_names,
        "mean_importance": mean_importance,
        "std_importance":  std_importance,
        "mean_rank":       mean_rank,
        "std_rank":        std_rank,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Step 3 – Ranking stability via Spearman correlation between folds   #
    # ------------------------------------------------------------------ #
    spearman_matrix = np.ones((n_splits, n_splits))
    pairwise_correlations = []

    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            corr, _ = spearmanr(fold_rankings[i], fold_rankings[j])
            spearman_matrix[i, j] = corr
            spearman_matrix[j, i] = corr
            pairwise_correlations.append(corr)

    pairwise_correlations = np.array(pairwise_correlations)
    mean_spearman = float(np.mean(pairwise_correlations))
    std_spearman  = float(np.std(pairwise_correlations))
    min_spearman  = float(np.min(pairwise_correlations))
    max_spearman  = float(np.max(pairwise_correlations))

    stability_metrics = {
        "spearman_matrix":       spearman_matrix,
        "pairwise_correlations": pairwise_correlations,
        "mean_spearman":         mean_spearman,
        "std_spearman":          std_spearman,
        "min_spearman":          min_spearman,
        "max_spearman":          max_spearman,
    }

    # ------------------------------------------------------------------ #
    # Step 4 – Select features consistently in top-k across ≥ threshold   #
    # ------------------------------------------------------------------ #
    # For each feature, count how many folds it appears in the top-k
    in_top_k = (fold_rankings <= top_k)          # shape (n_splits, n_features)
    fraction_in_top_k = in_top_k.mean(axis=0)    # shape (n_features,)

    stable_mask = fraction_in_top_k >= stability_threshold
    selected_feature_names = [
        feature_names[i] for i in range(n_features) if stable_mask[i]
    ]
    selected_feature_indices = np.where(stable_mask)[0]

    # Add stability info to summary
    importance_summary["fraction_in_top_k"] = [
        fraction_in_top_k[feature_names.index(f)]
        for f in importance_summary["feature"]
    ]
    importance_summary["is_stable"] = [
        f in selected_feature_names for f in importance_summary["feature"]
    ]

    # ------------------------------------------------------------------ #
    # Step 5 – Train final model on stable features + nested CV           #
    # ------------------------------------------------------------------ #
    if len(selected_feature_indices) == 0:
        warnings.warn(
            "No features met the stability threshold. "
            "Using all features for the final model.",
            UserWarning,
        )
        selected_feature_indices = np.arange(n_features)
        selected_feature_names   = feature_names

    X_stable = X[:, selected_feature_indices]

    final_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    final_rf.fit(X_stable, y)

    # Nested CV for unbiased performance estimate
    nested_cv_scores = cross_val_score(
        RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        ),
        X_stable, y,
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        scoring="accuracy",
        n_jobs=-1,
    )

    final_model_performance = {
        "nested_cv_scores":    nested_cv_scores,
        "mean_accuracy":       float(nested_cv_scores.mean()),
        "std_accuracy":        float(nested_cv_scores.std()),
        "n_selected_features": len(selected_feature_names),
        "n_total_features":    n_features,
    }

    return {
        "importance_summary":      importance_summary,
        "stability_metrics":       stability_metrics,
        "selected_features":       selected_feature_names,
        "final_model":             final_rf,
        "final_model_performance": final_model_performance,
        "fold_importances":        fold_importances,
        "fold_rankings":           fold_rankings,
    }


def print_report(results: dict) -> None:
    """Pretty-print the analysis results."""
    perf   = results["final_model_performance"]
    stab   = results["stability_metrics"]
    sel    = results["selected_features"]
    summary = results["importance_summary"]

    print("=" * 65)
    print("  CROSS-VALIDATED FEATURE IMPORTANCE ANALYSIS – REPORT")
    print("=" * 65)

    print("\n[1] TOP-10 FEATURES BY MEAN PERMUTATION IMPORTANCE")
    print("-" * 65)
    top10 = summary.head(10)[
        ["feature", "mean_importance", "std_importance",
         "mean_rank", "fraction_in_top_k", "is_stable"]
    ]
    print(top10.to_string(index=False))

    print("\n[2] RANKING STABILITY (SPEARMAN CORRELATION BETWEEN FOLDS)")
    print("-" * 65)
    print(f"  Mean  Spearman r : {stab['mean_spearman']:.4f}")
    print(f"  Std   Spearman r : {stab['std_spearman']:.4f}")
    print(f"  Min   Spearman r : {stab['min_spearman']:.4f}")
    print(f"  Max   Spearman r : {stab['max_spearman']:.4f}")

    print("\n[3] SELECTED STABLE FEATURES")
    print("-" * 65)
    print(f"  {perf['n_selected_features']} / {perf['n_total_features']} features selected:")
    for f in sel:
        print(f"    • {f}")

    print("\n[4] FINAL MODEL PERFORMANCE (NESTED 10-FOLD CV)")
    print("-" * 65)
    print(f"  Mean Accuracy : {perf['mean_accuracy']:.4f}")
    print(f"  Std  Accuracy : {perf['std_accuracy']:.4f}")
    fold_scores = "  ".join(f"{s:.4f}" for s in perf["nested_cv_scores"])
    print(f"  Fold scores   : {fold_scores}")
    print("=" * 65)


# ------------------------------------------------------------------ #
# Demo / self-test                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X_demo, y_demo = make_classification(
        n_samples=600,
        n_features=30,
        n_informative=10,
        n_redundant=5,
        n_repeated=2,
        random_state=0,
    )
    feature_names_demo = [f"feat_{i:02d}" for i in range(30)]

    results = cross_validated_feature_importance(
        X_demo,
        y_demo,
        feature_names=feature_names_demo,
        n_splits=10,
        top_k=20,
        stability_threshold=0.8,
        n_estimators=100,
        random_state=42,
        n_repeats_permutation=5,
    )

    print_report(results)