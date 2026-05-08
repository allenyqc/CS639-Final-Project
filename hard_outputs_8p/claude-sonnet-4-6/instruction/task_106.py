"""
Two-Level Stacking Ensemble Classifier Module

Implements a stacking ensemble with:
- Level 1: Logistic Regression, Random Forest, Gradient Boosting
- Level 2: Logistic Regression meta-learner
- Proper data leakage prevention via cross-validation for meta-feature generation
- Probability calibration on a held-out validation set
- Feature scaling fitted only on training data
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label: str,
) -> dict[str, Any]:
    """Compute accuracy, AUC, and macro-F1 for a single model."""
    n_classes = len(np.unique(y_true))
    multi_class = "ovr" if n_classes > 2 else "raise"

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(
                y_true, y_prob, multi_class=multi_class, average="macro"
            )
    except ValueError as exc:
        logger.warning("AUC computation failed for %s: %s", label, exc)
        auc = float("nan")

    return {
        "model": label,
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": auc,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _build_base_pipelines() -> dict[str, Pipeline]:
    """
    Return a dict of named sklearn Pipelines, each with a StandardScaler
    followed by a calibrated classifier.

    Calibration is done with cv='prefit' later (after fitting on fold data),
    so here we wrap with CalibratedClassifierCV(cv=5) to allow in-pipeline
    calibration during cross-validation meta-feature generation.
    """
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        multi_class="auto",
                        random_state=42,
                    ),
                    cv=3,
                    method="sigmoid",
                ),
            ),
        ]
    )

    rf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    cv=3,
                    method="isotonic",
                ),
            ),
        ]
    )

    gb = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        random_state=42,
                    ),
                    cv=3,
                    method="isotonic",
                ),
            ),
        ]
    )

    return {"LogisticRegression": lr, "RandomForest": rf, "GradientBoosting": gb}


# ---------------------------------------------------------------------------
# Core stacking logic
# ---------------------------------------------------------------------------

def _generate_oof_meta_features(
    base_pipelines: dict[str, Pipeline],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, dict[str, Pipeline]]:
    """
    Generate out-of-fold (OOF) predicted probabilities for the training set.

    Each base model is trained on (n_splits - 1) folds and predicts on the
    held-out fold.  This prevents data leakage into the meta-features.

    Returns
    -------
    meta_features : ndarray of shape (n_train_samples, n_models * n_classes)
    fitted_pipelines : dict of pipelines fitted on the FULL training set
                       (used later to generate test meta-features)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_classes = len(np.unique(y_train))
    n_samples = X_train.shape[0]
    n_models = len(base_pipelines)

    # Placeholder: (n_samples, n_models * n_classes)
    meta_features = np.zeros((n_samples, n_models * n_classes))

    model_names = list(base_pipelines.keys())

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info("OOF fold %d / %d", fold_idx + 1, n_splits)
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train = y_train[train_idx]

        for m_idx, name in enumerate(model_names):
            # Deep-copy the pipeline so each fold starts fresh
            import copy

            pipeline = copy.deepcopy(base_pipelines[name])
            pipeline.fit(X_fold_train, y_fold_train)
            proba = pipeline.predict_proba(X_fold_val)  # (n_val, n_classes)

            col_start = m_idx * n_classes
            col_end = col_start + n_classes
            meta_features[val_idx, col_start:col_end] = proba

    # Refit each pipeline on the FULL training set for test-time predictions
    fitted_pipelines: dict[str, Pipeline] = {}
    for name in model_names:
        import copy

        pipeline = copy.deepcopy(base_pipelines[name])
        logger.info("Fitting %s on full training set …", name)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

    return meta_features, fitted_pipelines


def _generate_test_meta_features(
    fitted_pipelines: dict[str, Pipeline],
    X_test: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Generate meta-features for the test set using fully-fitted base models.
    """
    n_models = len(fitted_pipelines)
    test_meta = np.zeros((X_test.shape[0], n_models * n_classes))

    for m_idx, (name, pipeline) in enumerate(fitted_pipelines.items()):
        proba = pipeline.predict_proba(X_test)
        col_start = m_idx * n_classes
        col_end = col_start + n_classes
        test_meta[:, col_start:col_end] = proba
        logger.info("Test meta-features generated for %s", name)

    return test_meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_stacking_ensemble(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_oof_splits: int = 5,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Build and evaluate a two-level stacking ensemble classifier.

    Parameters
    ----------
    X_train : array-like of shape (n_train, n_features)
        Training features (already split from test BEFORE calling this function).
    X_test : array-like of shape (n_test, n_features)
        Test features — used ONLY for final evaluation.
    y_train : array-like of shape (n_train,)
        Training labels.
    y_test : array-like of shape (n_test,)
        Test labels — used ONLY for final evaluation.
    n_oof_splits : int
        Number of cross-validation folds for OOF meta-feature generation.
    val_size : float
        Fraction of training data to hold out as a validation set for
        meta-learner threshold tuning / calibration.
    random_state : int
        Global random seed.

    Returns
    -------
    results : dict with keys
        "ensemble"        : dict of ensemble metrics (accuracy, auc, f1_macro)
        "base_models"     : list of dicts, one per base model
        "fitted_pipelines": dict of fitted base-model pipelines
        "meta_learner"    : fitted meta-learner pipeline
    """
    # ------------------------------------------------------------------
    # 0. Input validation & type coercion
    # ------------------------------------------------------------------
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    n_classes = len(np.unique(y_train))
    logger.info(
        "Dataset — train: %s, test: %s, classes: %d",
        X_train.shape,
        X_test.shape,
        n_classes,
    )

    # ------------------------------------------------------------------
    # 1. Carve out a validation split from training data
    #    (used for meta-learner calibration; test set stays untouched)
    # ------------------------------------------------------------------
    (
        X_tr,
        X_val,
        y_tr,
        y_val,
    ) = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=random_state,
    )
    logger.info(
        "Train/val split — train: %d, val: %d", X_tr.shape[0], X_val.shape[0]
    )

    # ------------------------------------------------------------------
    # 2. Build base model pipelines (scaler + calibrated classifier)
    # ------------------------------------------------------------------
    base_pipelines = _build_base_pipelines()

    # ------------------------------------------------------------------
    # 3. Generate OOF meta-features on X_tr (not X_train, to keep X_val clean)
    # ------------------------------------------------------------------
    logger.info("Generating OOF meta-features …")
    oof_meta, fitted_pipelines = _generate_oof_meta_features(
        base_pipelines, X_tr, y_tr, n_splits=n_oof_splits
    )

    # ------------------------------------------------------------------
    # 4. Generate validation meta-features (for meta-learner fitting)
    # ------------------------------------------------------------------
    val_meta = _generate_test_meta_features(fitted_pipelines, X_val, n_classes)

    # Combine OOF + validation meta-features to train the meta-learner
    # (OOF covers X_tr; val_meta covers X_val — together they span X_train)
    meta_X_train = np.vstack([oof_meta, val_meta])
    meta_y_train = np.concatenate([y_tr, y_val])

    # ------------------------------------------------------------------
    # 5. Train meta-learner (Level-2 Logistic Regression) with its own scaler
    # ------------------------------------------------------------------
    logger.info("Training meta-learner …")
    meta_learner = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    multi_class="auto",
                    C=1.0,
                    random_state=random_state,
                ),
            ),
        ]
    )
    meta_learner.fit(meta_X_train, meta_y_train)

    # ------------------------------------------------------------------
    # 6. Generate test meta-features & predict with meta-learner
    # ------------------------------------------------------------------
    logger.info("Generating test meta-features …")
    test_meta = _generate_test_meta_features(fitted_pipelines, X_test, n_classes)

    ensemble_proba = meta_learner.predict_proba(test_meta)
    ensemble_pred = meta_learner.predict(test_meta)

    # ------------------------------------------------------------------
    # 7. Compute ensemble metrics on test set (first and only look)
    # ------------------------------------------------------------------
    ensemble_metrics = _compute_metrics(
        y_test, ensemble_pred, ensemble_proba, label="StackingEnsemble"
    )
    logger.info("Ensemble — %s", ensemble_metrics)

    # ------------------------------------------------------------------
    # 8. Compute individual base-model metrics on test set for comparison
    # ------------------------------------------------------------------
    base_model_metrics: list[dict[str, Any]] = []
    for name, pipeline in fitted_pipelines.items():
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)
        metrics = _compute_metrics(y_test, pred, proba, label=name)
        base_model_metrics.append(metrics)
        logger.info("%s — %s", name, metrics)

    # ------------------------------------------------------------------
    # 9. Return results
    # ------------------------------------------------------------------
    return {
        "ensemble": ensemble_metrics,
        "base_models": base_model_metrics,
        "fitted_pipelines": fitted_pipelines,
        "meta_learner": meta_learner,
    }


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split as tts

    # Generate synthetic data
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=0,
    )

    # Split BEFORE any preprocessing (best practice #1)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = tts(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    results = build_stacking_ensemble(
        X_train=X_train_raw,
        X_test=X_test_raw,
        y_train=y_train_raw,
        y_test=y_test_raw,
        n_oof_splits=5,
        val_size=0.15,
        random_state=42,
    )

    print("\n=== Stacking Ensemble Results ===")
    print(f"Ensemble  : {results['ensemble']}")
    print("\nBase Models:")
    for bm in results["base_models"]:
        print(f"  {bm}")