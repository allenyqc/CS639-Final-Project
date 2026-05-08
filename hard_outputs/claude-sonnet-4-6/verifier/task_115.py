"""
Model Interpretability and Simplification Pipeline

This module builds a pipeline that:
1. Accepts X and y arrays
2. Splits into train/test sets
3. Trains a GradientBoostingClassifier
4. Computes SHAP values or permutation-based feature importance
5. Builds a simplified LogisticRegression on top-10 features
6. Compares both models' performance
7. Returns models, importance rankings, and comparison table
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper: try to import shap; fall back to permutation importance
# ---------------------------------------------------------------------------
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _compute_shap_importance(model, X_train: np.ndarray, feature_names: list) -> pd.Series:
    """
    Compute mean absolute SHAP values for each feature using the shap library.

    Parameters
    ----------
    model : fitted GradientBoostingClassifier
    X_train : np.ndarray  – training features
    feature_names : list  – column names

    Returns
    -------
    pd.Series sorted descending by importance
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # For binary classification shap may return a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=feature_names, name="shap_importance")
    return importance.sort_values(ascending=False)


def _compute_permutation_importance(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.Series:
    """
    Manually compute permutation importance.

    For each feature, shuffle its values `n_repeats` times and measure the
    average drop in accuracy compared to the baseline.

    Parameters
    ----------
    model       : fitted classifier with a .predict() method
    X_val       : np.ndarray – validation / test features
    y_val       : np.ndarray – true labels
    feature_names : list
    n_repeats   : int  – number of shuffles per feature
    random_state : int

    Returns
    -------
    pd.Series sorted descending by importance
    """
    rng = np.random.RandomState(random_state)
    baseline_acc = accuracy_score(y_val, model.predict(X_val))

    importances = {}
    for col_idx, col_name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X_val.copy()
            X_permuted[:, col_idx] = rng.permutation(X_permuted[:, col_idx])
            perm_acc = accuracy_score(y_val, model.predict(X_permuted))
            drops.append(baseline_acc - perm_acc)
        importances[col_name] = np.mean(drops)

    importance = pd.Series(importances, name="permutation_importance")
    return importance.sort_values(ascending=False)


def _evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute accuracy, F1 (weighted), and AUC for a fitted classifier.

    Parameters
    ----------
    model  : fitted classifier
    X_test : np.ndarray
    y_test : np.ndarray

    Returns
    -------
    dict with keys 'accuracy', 'f1', 'auc'
    """
    y_pred = model.predict(X_test)

    # AUC requires probability estimates
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    else:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_interpretability_pipeline(
    X,
    y,
    feature_names=None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_top_features: int = 10,
    gb_params: dict = None,
    lr_params: dict = None,
    use_shap: bool = True,
    permutation_repeats: int = 10,
):
    """
    Build a model interpretability and simplification pipeline.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector (classification labels).
    feature_names : list or None
        Names for each feature column. Auto-generated if None.
    test_size : float
        Fraction of data reserved for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.
    n_top_features : int
        Number of top features to keep for the simplified model (default 10).
    gb_params : dict or None
        Extra keyword arguments forwarded to GradientBoostingClassifier.
    lr_params : dict or None
        Extra keyword arguments forwarded to LogisticRegression.
    use_shap : bool
        If True (and shap is installed), use SHAP values; otherwise fall back
        to permutation importance.
    permutation_repeats : int
        Number of shuffles per feature when computing permutation importance.

    Returns
    -------
    results : dict with keys
        'complex_model'      – fitted GradientBoostingClassifier
        'simple_model'       – fitted Pipeline(scaler + LogisticRegression)
        'importance_ranking' – pd.Series of feature importances (all features)
        'top_features'       – list of top-N feature names
        'comparison_table'   – pd.DataFrame comparing both models
        'X_train'            – training features (np.ndarray)
        'X_test'             – test features (np.ndarray)
        'y_train'            – training labels
        'y_test'             – test labels
        'feature_names'      – list of all feature names
        'importance_method'  – str, 'shap' or 'permutation'
    """
    # ------------------------------------------------------------------
    # 1. Coerce inputs
    # ------------------------------------------------------------------
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    n_samples, n_features = X.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    else:
        feature_names = list(feature_names)

    if len(feature_names) != n_features:
        raise ValueError(
            f"len(feature_names)={len(feature_names)} does not match "
            f"n_features={n_features}."
        )

    n_top_features = min(n_top_features, n_features)

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[Pipeline] Dataset: {n_samples} samples, {n_features} features")
    print(f"[Pipeline] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Train complex model (GradientBoostingClassifier)
    # ------------------------------------------------------------------
    _gb_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
    )
    if gb_params:
        _gb_params.update(gb_params)

    print("[Pipeline] Training GradientBoostingClassifier …")
    complex_model = GradientBoostingClassifier(**_gb_params)
    complex_model.fit(X_train, y_train)
    print("[Pipeline] Complex model trained.")

    # ------------------------------------------------------------------
    # 4. Compute feature importance (SHAP or permutation)
    # ------------------------------------------------------------------
    importance_method = "permutation"

    if use_shap and SHAP_AVAILABLE:
        print("[Pipeline] Computing SHAP values …")
        try:
            importance_ranking = _compute_shap_importance(
                complex_model, X_train, feature_names
            )
            importance_method = "shap"
            print("[Pipeline] SHAP values computed.")
        except Exception as exc:
            print(f"[Pipeline] SHAP failed ({exc}); falling back to permutation importance.")
            importance_ranking = _compute_permutation_importance(
                complex_model, X_test, y_test, feature_names,
                n_repeats=permutation_repeats, random_state=random_state,
            )
    else:
        if use_shap and not SHAP_AVAILABLE:
            print("[Pipeline] shap not installed; using permutation importance.")
        else:
            print("[Pipeline] Computing permutation importance …")
        importance_ranking = _compute_permutation_importance(
            complex_model, X_test, y_test, feature_names,
            n_repeats=permutation_repeats, random_state=random_state,
        )
        print("[Pipeline] Permutation importance computed.")

    top_features = importance_ranking.index[:n_top_features].tolist()
    top_indices = [feature_names.index(f) for f in top_features]

    print(f"[Pipeline] Top-{n_top_features} features ({importance_method}): {top_features}")

    # ------------------------------------------------------------------
    # 5. Build simplified logistic regression on top-N features
    # ------------------------------------------------------------------
    _lr_params = dict(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
        multi_class="auto",
    )
    if lr_params:
        _lr_params.update(lr_params)

    X_train_top = X_train[:, top_indices]
    X_test_top = X_test[:, top_indices]

    simple_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**_lr_params)),
        ]
    )
    print("[Pipeline] Training simplified LogisticRegression …")
    simple_model.fit(X_train_top, y_train)
    print("[Pipeline] Simplified model trained.")

    # ------------------------------------------------------------------
    # 6. Compare performance
    # ------------------------------------------------------------------
    complex_metrics = _evaluate_model(complex_model, X_test, y_test)
    simple_metrics = _evaluate_model(simple_model, X_test_top, y_test)

    comparison_table = pd.DataFrame(
        {
            "GradientBoosting (all features)": complex_metrics,
            f"LogisticRegression (top-{n_top_features} features)": simple_metrics,
        }
    ).T
    comparison_table.index.name = "Model"
    comparison_table = comparison_table[["accuracy", "f1", "auc"]]

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(comparison_table.to_string())
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 7. Return everything
    # ------------------------------------------------------------------
    return {
        "complex_model": complex_model,
        "simple_model": simple_model,
        "importance_ranking": importance_ranking,
        "top_features": top_features,
        "comparison_table": comparison_table,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "importance_method": importance_method,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper: predict with the simplified model
# ---------------------------------------------------------------------------

def predict_simplified(results: dict, X_new):
    """
    Run inference with the simplified logistic regression model.

    Parameters
    ----------
    results : dict returned by build_interpretability_pipeline()
    X_new   : array-like of shape (n_samples, n_all_features)
              Must have the same number of columns as the original X.

    Returns
    -------
    np.ndarray of predicted class labels
    """
    X_new = np.asarray(X_new, dtype=float)
    feature_names = results["feature_names"]
    top_features = results["top_features"]
    top_indices = [feature_names.index(f) for f in top_features]
    return results["simple_model"].predict(X_new[:, top_indices])


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

def _demo():
    """Quick smoke-test using a synthetic dataset."""
    from sklearn.datasets import make_classification

    print("Generating synthetic dataset …")
    X, y = make_classification(
        n_samples=1_000,
        n_features=30,
        n_informative=10,
        n_redundant=5,
        random_state=0,
    )
    feature_names = [f"feat_{i:02d}" for i in range(X.shape[1])]

    results = build_interpretability_pipeline(
        X,
        y,
        feature_names=feature_names,
        test_size=0.2,
        random_state=42,
        n_top_features=10,
        use_shap=True,          # will fall back to permutation if shap absent
        permutation_repeats=5,
    )

    print("\nImportance ranking (top 10):")
    print(results["importance_ranking"].head(10))

    print("\nComparison table:")
    print(results["comparison_table"])

    # Test predict_simplified
    sample = results["X_test"][:5]
    preds = predict_simplified(results, sample)
    print(f"\nSample predictions (simplified model): {preds}")

    return results


if __name__ == "__main__":
    _demo()