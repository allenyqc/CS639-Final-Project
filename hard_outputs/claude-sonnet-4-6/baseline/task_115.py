"""
Model Interpretability and Simplification Pipeline

This module builds a pipeline that:
1. Accepts X and y arrays
2. Splits into train/test sets
3. Trains a GradientBoostingClassifier (complex model)
4. Computes SHAP values or permutation-based feature importance
5. Builds a simplified LogisticRegression using top-10 features
6. Compares both models' performance
7. Returns models, importance rankings, and comparison table
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# SHAP / Permutation Importance helpers
# ─────────────────────────────────────────────

def _try_shap_importance(model, X_train: np.ndarray, feature_names: list) -> np.ndarray | None:
    """
    Attempt to compute SHAP-based feature importances.
    Returns an array of mean |SHAP| values per feature, or None if shap is unavailable.
    """
    try:
        import shap  # type: ignore
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # For binary classification shap may return a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]          # positive-class SHAP values

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return mean_abs_shap

    except Exception:
        return None


def _permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
    metric: str = "accuracy",
) -> np.ndarray:
    """
    Manual permutation importance.

    For each feature, randomly shuffle its values `n_repeats` times and
    measure the average drop in the chosen metric.  A larger drop means
    the feature is more important.

    Parameters
    ----------
    model       : fitted estimator with a predict (and optionally predict_proba) method
    X           : 2-D array, shape (n_samples, n_features)
    y           : 1-D array of true labels
    n_repeats   : number of shuffle repetitions per feature
    random_state: seed for reproducibility
    metric      : 'accuracy' | 'f1' | 'auc'

    Returns
    -------
    importances : 1-D array of mean importance scores, shape (n_features,)
    """
    rng = np.random.default_rng(random_state)

    def _score(X_eval: np.ndarray) -> float:
        if metric == "auc":
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_eval)[:, 1]
            else:
                proba = model.decision_function(X_eval)
            return roc_auc_score(y, proba)
        preds = model.predict(X_eval)
        if metric == "f1":
            return f1_score(y, preds, average="weighted", zero_division=0)
        return accuracy_score(y, preds)

    baseline = _score(X)
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    for feat_idx in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])
            permuted_score = _score(X_permuted)
            drops.append(baseline - permuted_score)
        importances[feat_idx] = np.mean(drops)

    return importances


def _compute_feature_importance(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute feature importances, preferring SHAP and falling back to
    permutation importance.

    Returns a DataFrame sorted by importance (descending) with columns
    ['feature', 'importance', 'method'].
    """
    method = "shap"
    importances = _try_shap_importance(model, X_train, feature_names)

    if importances is None:
        method = "permutation"
        importances = _permutation_importance(
            model, X_train, y_train,
            n_repeats=n_repeats,
            random_state=random_state,
            metric="accuracy",
        )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "method": method,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
# Performance evaluation helper
# ─────────────────────────────────────────────

def _evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Return a dict with accuracy, F1 (weighted), and AUC for a fitted model."""
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if proba.shape[1] == 2:
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
        else:
            scores = model.decision_function(X_test)
            auc = roc_auc_score(y_test, scores)
    except Exception:
        auc = float("nan")

    return {"model": model_name, "accuracy": accuracy, "f1_weighted": f1, "auc": auc}


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def build_interpretability_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list | None = None,
    top_n: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    gb_params: dict | None = None,
    lr_params: dict | None = None,
    n_permutation_repeats: int = 10,
) -> dict:
    """
    Build a model interpretability and simplification pipeline.

    Parameters
    ----------
    X                    : Feature matrix, shape (n_samples, n_features)
    y                    : Target vector, shape (n_samples,)
    feature_names        : Optional list of feature names (length == n_features).
                           Defaults to ['f0', 'f1', ...].
    top_n                : Number of top features to select for the simplified model.
    test_size            : Fraction of data reserved for testing.
    random_state         : Random seed for reproducibility.
    gb_params            : Optional dict of kwargs for GradientBoostingClassifier.
    lr_params            : Optional dict of kwargs for LogisticRegression.
    n_permutation_repeats: Repeats used in permutation importance fallback.

    Returns
    -------
    result : dict with keys
        'complex_model'       – fitted GradientBoostingClassifier
        'simplified_model'    – fitted Pipeline(scaler + LogisticRegression)
        'importance_rankings' – DataFrame sorted by importance (desc)
        'top_features'        – list of top-N feature names
        'top_feature_indices' – list of top-N feature column indices
        'comparison_table'    – DataFrame comparing accuracy / F1 / AUC
        'X_train'             – training features (full)
        'X_test'              – test features (full)
        'y_train'             – training labels
        'y_test'              – test labels
        'importance_method'   – 'shap' or 'permutation'
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")

    n_samples, n_features = X.shape
    actual_top_n = min(top_n, n_features)

    # Default feature names
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    else:
        feature_names = list(feature_names)
        if len(feature_names) != n_features:
            raise ValueError(
                f"len(feature_names)={len(feature_names)} != n_features={n_features}"
            )

    # ── 1. Train / test split ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ── 2. Train complex model ─────────────────────────────────────────────
    default_gb_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
    )
    if gb_params:
        default_gb_params.update(gb_params)

    complex_model = GradientBoostingClassifier(**default_gb_params)
    complex_model.fit(X_train, y_train)

    # ── 3. Feature importance (SHAP or permutation) ────────────────────────
    importance_df = _compute_feature_importance(
        model=complex_model,
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_names,
        n_repeats=n_permutation_repeats,
        random_state=random_state,
    )
    importance_method = importance_df["method"].iloc[0]

    # ── 4. Select top-N features ───────────────────────────────────────────
    top_features = importance_df["feature"].head(actual_top_n).tolist()
    top_feature_indices = [feature_names.index(f) for f in top_features]

    X_train_top = X_train[:, top_feature_indices]
    X_test_top = X_test[:, top_feature_indices]

    # ── 5. Train simplified model ──────────────────────────────────────────
    default_lr_params = dict(
        max_iter=1000,
        random_state=random_state,
        C=1.0,
        solver="lbfgs",
        multi_class="auto",
    )
    if lr_params:
        default_lr_params.update(lr_params)

    simplified_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**default_lr_params)),
    ])
    simplified_model.fit(X_train_top, y_train)

    # ── 6. Evaluate both models ────────────────────────────────────────────
    complex_metrics = _evaluate_model(complex_model, X_test, y_test, "GradientBoosting (complex)")
    simplified_metrics = _evaluate_model(simplified_model, X_test_top, y_test, "LogisticRegression (simplified)")

    comparison_table = pd.DataFrame([complex_metrics, simplified_metrics]).set_index("model")

    # ── 7. Print summary ───────────────────────────────────────────────────
    _print_summary(importance_df, top_features, comparison_table, importance_method, actual_top_n)

    return {
        "complex_model": complex_model,
        "simplified_model": simplified_model,
        "importance_rankings": importance_df,
        "top_features": top_features,
        "top_feature_indices": top_feature_indices,
        "comparison_table": comparison_table,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "importance_method": importance_method,
    }


# ─────────────────────────────────────────────
# Pretty-print helper
# ─────────────────────────────────────────────

def _print_summary(
    importance_df: pd.DataFrame,
    top_features: list,
    comparison_table: pd.DataFrame,
    method: str,
    top_n: int,
) -> None:
    sep = "=" * 60
    print(sep)
    print("  MODEL INTERPRETABILITY & SIMPLIFICATION PIPELINE")
    print(sep)

    print(f"\n[Feature Importance Method]: {method.upper()}")
    print(f"\nTop-{top_n} Features:")
    print(importance_df.head(top_n)[["feature", "importance"]].to_string(index=False))

    print(f"\n[Performance Comparison]")
    print(comparison_table.to_string())
    print()

    # Highlight accuracy delta
    models = comparison_table.index.tolist()
    if len(models) == 2:
        acc_complex = comparison_table.loc[models[0], "accuracy"]
        acc_simple = comparison_table.loc[models[1], "accuracy"]
        delta = acc_complex - acc_simple
        print(f"  Accuracy trade-off (complex − simplified): {delta:+.4f}")
        if abs(delta) < 0.02:
            print("  ✓ Simplified model achieves comparable accuracy with fewer features.")
        else:
            print("  ⚠ Notable accuracy gap; consider increasing top_n or tuning LR.")
    print(sep)


# ─────────────────────────────────────────────
# Convenience wrapper / demo
# ─────────────────────────────────────────────

def demo():
    """Quick demo using sklearn's breast-cancer dataset."""
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)

    result = build_interpretability_pipeline(
        X=X,
        y=y,
        feature_names=feature_names,
        top_n=10,
        test_size=0.2,
        random_state=42,
    )

    print("\nReturned keys:", list(result.keys()))
    print("\nImportance Rankings (all features):")
    print(result["importance_rankings"].to_string(index=False))

    return result


if __name__ == "__main__":
    demo()