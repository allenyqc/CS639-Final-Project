"""
Model Interpretability and Simplification Pipeline

This module builds a pipeline that:
1. Trains a complex GradientBoostingClassifier
2. Computes feature importance (SHAP or permutation-based)
3. Builds a simplified LogisticRegression on top-10 features
4. Compares both models' performance
"""

import warnings
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InterpretabilityResult:
    """Container for all pipeline outputs."""
    complex_model: Pipeline
    simplified_model: Pipeline
    shap_importance: pd.DataFrame          # columns: feature_index, importance
    comparison_table: pd.DataFrame         # columns: model, accuracy, f1, auc
    top_feature_indices: list[int]
    X_test: np.ndarray
    y_test: np.ndarray
    used_shap: bool = False


# ---------------------------------------------------------------------------
# SHAP / permutation importance helpers
# ---------------------------------------------------------------------------

def _compute_shap_importance(
    model: GradientBoostingClassifier,
    X_train: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """
    Try to compute SHAP values; fall back to permutation importance.

    Returns
    -------
    importances : np.ndarray of shape (n_features,)
        Mean absolute importance per feature.
    used_shap : bool
    """
    try:
        import shap  # noqa: PLC0415

        logger.info("SHAP library found — computing TreeExplainer SHAP values.")
        explainer = shap.TreeExplainer(model)
        # Use a subsample for speed when the training set is large
        sample_size = min(500, X_train.shape[0])
        rng = np.random.default_rng(42)
        idx = rng.choice(X_train.shape[0], size=sample_size, replace=False)
        shap_values = explainer.shap_values(X_train[idx])

        # For binary classification shap may return a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        importances = np.abs(shap_values).mean(axis=0)
        return importances, True

    except ImportError:
        logger.warning(
            "shap library not available — falling back to permutation importance "
            "(computed on the TRAINING set only to avoid test-set leakage)."
        )
        return _permutation_importance(model, X_train, n_features), False


def _permutation_importance(
    model: GradientBoostingClassifier,
    X: np.ndarray,
    n_features: int,
    n_repeats: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """
    Manual permutation importance computed on X (training data only).

    For each feature, shuffle its values `n_repeats` times and measure the
    average drop in the model's predict_proba output (log-loss proxy).

    Returns
    -------
    importances : np.ndarray of shape (n_features,)
    """
    rng = np.random.default_rng(random_state)

    # Baseline: mean predicted probability of the positive class
    baseline_proba = model.predict_proba(X)[:, 1]

    importances = np.zeros(n_features)
    for feat_idx in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])
            permuted_proba = model.predict_proba(X_permuted)[:, 1]
            # Importance = mean absolute change in predicted probability
            drop = np.mean(np.abs(baseline_proba - permuted_proba))
            drops.append(drop)
        importances[feat_idx] = np.mean(drops)

    return importances


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(model: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """Return accuracy, macro-F1, and AUC for a fitted pipeline."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    n_classes = len(np.unique(y))
    auc = roc_auc_score(y, y_proba) if n_classes == 2 else float("nan")

    return {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_interpretability_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_top_features: int = 10,
    gb_params: Optional[dict] = None,
    lr_params: Optional[dict] = None,
) -> InterpretabilityResult:
    """
    Build a model interpretability and simplification pipeline.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    test_size : float, default 0.2
    random_state : int, default 42
    n_top_features : int, default 10
        Number of top features to keep for the simplified model.
    gb_params : dict, optional
        Extra kwargs forwarded to GradientBoostingClassifier.
    lr_params : dict, optional
        Extra kwargs forwarded to LogisticRegression.

    Returns
    -------
    InterpretabilityResult
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_features = X.shape[1]
    n_top_features = min(n_top_features, n_features)

    # ------------------------------------------------------------------
    # 1. Train / test split — FIRST, before any preprocessing
    # ------------------------------------------------------------------
    logger.info("Splitting data into train / test sets (test_size=%.2f).", test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ------------------------------------------------------------------
    # 2. Build and fit the complex model pipeline
    #    Scaler is fit ONLY on X_train via the Pipeline.
    # ------------------------------------------------------------------
    gb_defaults = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
    )
    if gb_params:
        gb_defaults.update(gb_params)

    logger.info("Training GradientBoostingClassifier on training set.")
    complex_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(**gb_defaults)),
    ])
    complex_pipeline.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 3. Compute feature importance on TRAINING data only
    # ------------------------------------------------------------------
    logger.info("Computing feature importance (training data only).")
    # Extract the fitted scaler's transform of X_train for SHAP/permutation
    X_train_scaled = complex_pipeline.named_steps["scaler"].transform(X_train)
    gb_model = complex_pipeline.named_steps["gb"]

    importances, used_shap = _compute_shap_importance(gb_model, X_train_scaled, n_features)

    # Build importance DataFrame
    importance_df = pd.DataFrame({
        "feature_index": np.arange(n_features),
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top_feature_indices = importance_df["feature_index"].iloc[:n_top_features].tolist()
    logger.info("Top-%d feature indices: %s", n_top_features, top_feature_indices)

    # ------------------------------------------------------------------
    # 4. Build simplified LogisticRegression on top features
    #    Scaler is fit ONLY on X_train[:, top_features] via the Pipeline.
    # ------------------------------------------------------------------
    lr_defaults = dict(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
        C=1.0,
    )
    if lr_params:
        lr_defaults.update(lr_params)

    logger.info("Training simplified LogisticRegression on top-%d features.", n_top_features)

    X_train_top = X_train[:, top_feature_indices]

    simplified_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**lr_defaults)),
    ])
    simplified_pipeline.fit(X_train_top, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate BOTH models on the held-out test set ONLY
    # ------------------------------------------------------------------
    logger.info("Evaluating models on the held-out test set.")

    complex_metrics = _evaluate(complex_pipeline, X_test, y_test)

    X_test_top = X_test[:, top_feature_indices]
    simplified_metrics = _evaluate(simplified_pipeline, X_test_top, y_test)

    comparison_table = pd.DataFrame([
        {"model": "GradientBoosting (complex)", **complex_metrics},
        {"model": f"LogisticRegression (top-{n_top_features} features)", **simplified_metrics},
    ])

    logger.info("\n%s", comparison_table.to_string(index=False))

    return InterpretabilityResult(
        complex_model=complex_pipeline,
        simplified_model=simplified_pipeline,
        shap_importance=importance_df,
        comparison_table=comparison_table,
        top_feature_indices=top_feature_indices,
        X_test=X_test,
        y_test=y_test,
        used_shap=used_shap,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper that also prints a summary
# ---------------------------------------------------------------------------

def run_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    **kwargs,
) -> InterpretabilityResult:
    """
    Run the full interpretability pipeline and print a human-readable summary.

    Parameters
    ----------
    X, y : arrays
    **kwargs : forwarded to build_interpretability_pipeline

    Returns
    -------
    InterpretabilityResult
    """
    result = build_interpretability_pipeline(X, y, **kwargs)

    print("\n" + "=" * 60)
    print("INTERPRETABILITY PIPELINE SUMMARY")
    print("=" * 60)
    print(f"\nFeature importance method : {'SHAP (TreeExplainer)' if result.used_shap else 'Permutation importance (manual)'}")
    print(f"\nTop-{len(result.top_feature_indices)} feature indices:")
    print(result.shap_importance.head(len(result.top_feature_indices)).to_string(index=False))
    print("\nModel comparison (test set):")
    print(result.comparison_table.to_string(index=False))
    print("=" * 60 + "\n")

    return result


# ---------------------------------------------------------------------------
# Example usage / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    logger.info("Generating synthetic classification dataset.")
    X_demo, y_demo = make_classification(
        n_samples=2000,
        n_features=30,
        n_informative=10,
        n_redundant=5,
        random_state=0,
    )

    result = run_pipeline(X_demo, y_demo, n_top_features=10)

    # Demonstrate how to use the returned objects
    sample = X_demo[:5]
    sample_top = sample[:, result.top_feature_indices]

    print("Complex model predictions on 5 samples:", result.complex_model.predict(sample))
    print("Simplified model predictions on 5 samples:", result.simplified_model.predict(sample_top))