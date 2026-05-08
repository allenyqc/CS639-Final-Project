"""
Model Interpretability and Simplification Pipeline

This module builds a pipeline that:
1. Trains a complex GradientBoostingClassifier
2. Computes feature importance (SHAP or permutation-based)
3. Builds a simplified LogisticRegression using top-10 features
4. Compares both models on test performance
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """Holds evaluation metrics for a single model."""
    model_name: str
    accuracy: float
    f1: float
    auc: float

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
            "auc": round(self.auc, 4),
        }


@dataclass
class InterpretabilityResult:
    """Container for the full pipeline output."""
    complex_model: GradientBoostingClassifier
    simplified_model: Pipeline
    feature_importance_ranking: pd.DataFrame          # columns: feature, importance
    top_features: list[str]
    comparison_table: pd.DataFrame                    # accuracy / F1 / AUC side-by-side
    complex_metrics: ModelMetrics
    simplified_metrics: ModelMetrics
    feature_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Permutation importance (fallback when shap is unavailable)
# ---------------------------------------------------------------------------

def _permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute permutation-based feature importance on a *validation* set.

    Parameters
    ----------
    model   : fitted estimator with predict_proba
    X       : 2-D array, shape (n_samples, n_features)
    y       : 1-D array of true labels
    n_repeats : number of permutation rounds per feature
    random_state : seed for reproducibility

    Returns
    -------
    importances : 1-D array of mean importance per feature
    """
    rng = np.random.default_rng(random_state)
    baseline_proba = model.predict_proba(X)[:, 1]
    try:
        baseline_score = roc_auc_score(y, baseline_proba)
    except ValueError:
        baseline_score = accuracy_score(y, model.predict(X))

    n_features = X.shape[1]
    importances = np.zeros(n_features)

    for feat_idx in range(n_features):
        scores = np.empty(n_repeats)
        for rep in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])
            perm_proba = model.predict_proba(X_permuted)[:, 1]
            try:
                scores[rep] = roc_auc_score(y, perm_proba)
            except ValueError:
                scores[rep] = accuracy_score(y, model.predict(X_permuted))
        importances[feat_idx] = baseline_score - scores.mean()

    return importances


# ---------------------------------------------------------------------------
# SHAP importance (preferred)
# ---------------------------------------------------------------------------

def _shap_importance(
    model: GradientBoostingClassifier,
    X_train: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute mean |SHAP| values using the shap library.

    Parameters
    ----------
    model        : fitted GradientBoostingClassifier
    X_train      : training features (numpy array)
    feature_names: list of feature name strings

    Returns
    -------
    DataFrame with columns ['feature', 'importance'] sorted descending
    """
    import shap  # type: ignore

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # For binary classification shap may return a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_interpretability_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list[str]] = None,
    test_size: float = 0.20,
    val_size: float = 0.15,
    n_top_features: int = 10,
    random_state: int = 42,
    gb_params: Optional[dict] = None,
    lr_params: Optional[dict] = None,
    permutation_repeats: int = 10,
) -> InterpretabilityResult:
    """
    Build a model interpretability and simplification pipeline.

    Parameters
    ----------
    X               : Feature matrix, shape (n_samples, n_features)
    y               : Target vector, shape (n_samples,)
    feature_names   : Optional list of feature name strings
    test_size       : Fraction of data held out as the final test set
    val_size        : Fraction of *training* data used as validation
                      (for permutation importance; never touches test set)
    n_top_features  : Number of top features to keep for simplified model
    random_state    : Reproducibility seed
    gb_params       : Optional hyperparameters for GradientBoostingClassifier
    lr_params       : Optional hyperparameters for LogisticRegression
    permutation_repeats : Rounds per feature for permutation importance fallback

    Returns
    -------
    InterpretabilityResult dataclass
    """
    # ------------------------------------------------------------------
    # 0. Input validation
    # ------------------------------------------------------------------
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=float)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    n_samples, n_features = X.shape
    n_top_features = min(n_top_features, n_features)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError(
            f"len(feature_names)={len(feature_names)} != n_features={n_features}"
        )

    # ------------------------------------------------------------------
    # 1. Train / test split  (BEFORE any preprocessing)
    # ------------------------------------------------------------------
    logger.info("Splitting data into train and test sets (test_size=%.2f).", test_size)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ------------------------------------------------------------------
    # 2. Validation split from training data
    #    Used ONLY for permutation importance; test set is never touched.
    # ------------------------------------------------------------------
    logger.info(
        "Splitting training data into train/val (val_size=%.2f of train).", val_size
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    # ------------------------------------------------------------------
    # 3. Fit scaler on training partition ONLY
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Train complex model (GradientBoostingClassifier)
    # ------------------------------------------------------------------
    default_gb_params: dict = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": random_state,
    }
    if gb_params is not None:
        default_gb_params.update(gb_params)

    logger.info("Training GradientBoostingClassifier with params: %s", default_gb_params)
    gb_model = GradientBoostingClassifier(**default_gb_params)
    gb_model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 5. Compute feature importance (SHAP preferred, permutation fallback)
    #    Importance is computed on the VALIDATION set — never on test data.
    # ------------------------------------------------------------------
    shap_available = False
    try:
        import shap  # noqa: F401
        shap_available = True
    except ImportError:
        logger.warning("shap not installed; falling back to permutation importance.")

    if shap_available:
        logger.info("Computing SHAP values on training data.")
        importance_df = _shap_importance(gb_model, X_train_scaled, feature_names)
    else:
        logger.info(
            "Computing permutation importance on validation set (%d repeats).",
            permutation_repeats,
        )
        perm_imp = _permutation_importance(
            gb_model,
            X_val_scaled,
            y_val,
            n_repeats=permutation_repeats,
            random_state=random_state,
        )
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": perm_imp}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

    top_features: list[str] = importance_df["feature"].head(n_top_features).tolist()
    top_indices: list[int] = [feature_names.index(f) for f in top_features]

    logger.info("Top-%d features: %s", n_top_features, top_features)

    # ------------------------------------------------------------------
    # 6. Build simplified LogisticRegression on top features
    #    Scaler is re-fit on the top-feature training subset only.
    # ------------------------------------------------------------------
    default_lr_params: dict = {
        "max_iter": 1000,
        "random_state": random_state,
        "solver": "lbfgs",
        "C": 1.0,
    }
    if lr_params is not None:
        default_lr_params.update(lr_params)

    logger.info(
        "Training simplified LogisticRegression on top-%d features.", n_top_features
    )

    # Use the full training partition (train + val) for the simplified model
    X_train_full_top = X_train_full[:, top_indices]
    X_test_top = X_test[:, top_indices]

    simplified_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**default_lr_params)),
        ]
    )
    # Fit scaler inside pipeline on training data only
    simplified_pipeline.fit(X_train_full_top, y_train_full)

    # ------------------------------------------------------------------
    # 7. Evaluate both models on the held-out TEST set (final evaluation)
    # ------------------------------------------------------------------
    logger.info("Evaluating models on the held-out test set.")

    # --- Complex model ---
    gb_pred = gb_model.predict(X_test_scaled)
    gb_proba = gb_model.predict_proba(X_test_scaled)[:, 1]
    gb_metrics = ModelMetrics(
        model_name="GradientBoostingClassifier",
        accuracy=accuracy_score(y_test, gb_pred),
        f1=f1_score(y_test, gb_pred, average="weighted", zero_division=0),
        auc=roc_auc_score(y_test, gb_proba),
    )

    # --- Simplified model ---
    lr_pred = simplified_pipeline.predict(X_test_top)
    lr_proba = simplified_pipeline.predict_proba(X_test_top)[:, 1]
    lr_metrics = ModelMetrics(
        model_name="LogisticRegression (top features)",
        accuracy=accuracy_score(y_test, lr_pred),
        f1=f1_score(y_test, lr_pred, average="weighted", zero_division=0),
        auc=roc_auc_score(y_test, lr_proba),
    )

    # ------------------------------------------------------------------
    # 8. Build comparison table
    # ------------------------------------------------------------------
    comparison_table = pd.DataFrame(
        [gb_metrics.to_dict(), lr_metrics.to_dict()]
    ).set_index("model")

    logger.info("\n%s", comparison_table.to_string())

    return InterpretabilityResult(
        complex_model=gb_model,
        simplified_model=simplified_pipeline,
        feature_importance_ranking=importance_df,
        top_features=top_features,
        comparison_table=comparison_table,
        complex_metrics=gb_metrics,
        simplified_metrics=lr_metrics,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Convenience pretty-printer
# ---------------------------------------------------------------------------

def print_results(result: InterpretabilityResult) -> None:
    """Pretty-print the pipeline results."""
    sep = "=" * 60
    print(sep)
    print("FEATURE IMPORTANCE RANKING (top 10)")
    print(sep)
    print(result.feature_importance_ranking.head(10).to_string(index=False))

    print()
    print(sep)
    print("MODEL COMPARISON (test set)")
    print(sep)
    print(result.comparison_table.to_string())
    print()


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
    feat_names = [f"feat_{i:02d}" for i in range(30)]

    result = build_interpretability_pipeline(
        X=X_demo,
        y=y_demo,
        feature_names=feat_names,
        test_size=0.20,
        val_size=0.15,
        n_top_features=10,
        random_state=42,
    )

    print_results(result)