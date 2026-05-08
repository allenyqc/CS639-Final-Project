"""
Two-Level Stacking Ensemble Classifier Module

This module implements a two-level stacking ensemble with:
- Level 1: Logistic Regression, Random Forest, Gradient Boosting
- Level 2: Logistic Regression meta-learner
- Proper data leakage prevention via cross-validation for meta-feature generation
- Probability calibration and feature scaling
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper: compute a standard metrics dictionary
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Return accuracy, AUC, and macro F1 for a set of predictions."""
    n_classes = len(np.unique(y_true))
    multi_class = "ovr" if n_classes > 2 else "raise"

    if n_classes == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_true, y_prob, multi_class=multi_class, average="macro")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": auc,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Core stacking function
# ---------------------------------------------------------------------------

def build_stacking_ensemble(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Build and evaluate a two-level stacking ensemble classifier.

    Parameters
    ----------
    X_train : array-like of shape (n_train_samples, n_features)
        Training features (already split from the full dataset).
    X_test : array-like of shape (n_test_samples, n_features)
        Test features — used ONLY for final evaluation.
    y_train : array-like of shape (n_train_samples,)
        Training labels.
    y_test : array-like of shape (n_test_samples,)
        Test labels — used ONLY for final evaluation.
    n_folds : int, default=5
        Number of stratified folds for out-of-fold meta-feature generation.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    results : dict
        {
            "ensemble": {"accuracy": ..., "auc": ..., "f1_macro": ...},
            "base_models": {
                "logistic_regression": {"accuracy": ..., "auc": ..., "f1_macro": ...},
                "random_forest":       {"accuracy": ..., "auc": ..., "f1_macro": ...},
                "gradient_boosting":   {"accuracy": ..., "auc": ..., "f1_macro": ...},
            },
            "fitted_objects": {
                "scaler": ...,
                "base_pipelines": [...],
                "meta_scaler": ...,
                "meta_learner": ...,
            },
        }
    """
    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    n_classes = len(np.unique(y_train))

    # ------------------------------------------------------------------
    # 1. Feature scaling — fit ONLY on training data
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
    X_test_scaled  = scaler.transform(X_test)         # transform only on test

    # ------------------------------------------------------------------
    # 2. Define level-1 base models (wrapped in calibrated pipelines)
    #    Calibration uses cv="prefit" after fitting, so we calibrate on
    #    held-out fold data — never on the test set.
    # ------------------------------------------------------------------
    base_model_definitions = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
            multi_class="auto",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=random_state,
        ),
    }

    model_names = list(base_model_definitions.keys())
    n_models    = len(model_names)

    # ------------------------------------------------------------------
    # 3. Generate out-of-fold (OOF) meta-features for the training set
    #    This prevents data leakage into the meta-learner.
    # ------------------------------------------------------------------
    oof_meta_features = np.zeros((len(y_train), n_models * n_classes))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    print(f"Generating out-of-fold meta-features with {n_folds}-fold CV …")

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_fold_tr, X_fold_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_fold_tr             = y_train[tr_idx]

        for m_idx, (name, base_clf) in enumerate(base_model_definitions.items()):
            # Clone-like: re-instantiate from the same class/params each fold
            clf = _clone_model(name, random_state)
            clf.fit(X_fold_tr, y_fold_tr)

            # Calibrate on the validation fold (cv="prefit" = calibrate a
            # pre-fitted estimator; the calibration data is the val fold,
            # which was NOT used to fit the base model in this fold).
            calibrated = CalibratedClassifierCV(clf, cv="prefit", method="isotonic")
            calibrated.fit(X_fold_val, y_train[val_idx])

            proba = calibrated.predict_proba(X_fold_val)  # shape (n_val, n_classes)
            col_start = m_idx * n_classes
            col_end   = col_start + n_classes
            oof_meta_features[val_idx, col_start:col_end] = proba

        print(f"  Fold {fold_idx + 1}/{n_folds} done.")

    # ------------------------------------------------------------------
    # 4. Train final base models on the FULL training set and calibrate
    #    using cross-validation (cv=5) so calibration never touches test.
    # ------------------------------------------------------------------
    print("\nTraining final base models on full training set …")
    final_base_models: Dict[str, CalibratedClassifierCV] = {}

    for name in model_names:
        clf = _clone_model(name, random_state)
        # CalibratedClassifierCV with cv=5 internally does cross-val on
        # X_train_scaled — test data is never involved.
        calibrated_final = CalibratedClassifierCV(clf, cv=5, method="isotonic")
        calibrated_final.fit(X_train_scaled, y_train)
        final_base_models[name] = calibrated_final
        print(f"  {name} trained and calibrated.")

    # ------------------------------------------------------------------
    # 5. Generate test meta-features using the final base models
    # ------------------------------------------------------------------
    test_meta_features = np.zeros((len(y_test), n_models * n_classes))

    for m_idx, name in enumerate(model_names):
        proba = final_base_models[name].predict_proba(X_test_scaled)
        col_start = m_idx * n_classes
        col_end   = col_start + n_classes
        test_meta_features[:, col_start:col_end] = proba

    # ------------------------------------------------------------------
    # 6. Scale meta-features — fit ONLY on OOF meta-features (train)
    # ------------------------------------------------------------------
    meta_scaler = StandardScaler()
    oof_meta_scaled  = meta_scaler.fit_transform(oof_meta_features)
    test_meta_scaled = meta_scaler.transform(test_meta_features)

    # ------------------------------------------------------------------
    # 7. Train level-2 meta-learner on OOF meta-features
    # ------------------------------------------------------------------
    print("\nTraining meta-learner …")
    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
        multi_class="auto",
        C=1.0,
    )
    meta_learner.fit(oof_meta_scaled, y_train)

    # ------------------------------------------------------------------
    # 8. Final evaluation on the test set (first and only time test is used)
    # ------------------------------------------------------------------
    print("\nEvaluating on test set …")

    # Ensemble predictions
    ensemble_pred  = meta_learner.predict(test_meta_scaled)
    ensemble_proba = meta_learner.predict_proba(test_meta_scaled)
    ensemble_metrics = _compute_metrics(y_test, ensemble_pred, ensemble_proba)

    # Individual base model performance on test set
    base_model_metrics: Dict[str, Dict[str, float]] = {}
    for m_idx, name in enumerate(model_names):
        col_start = m_idx * n_classes
        col_end   = col_start + n_classes
        bm_proba  = test_meta_features[:, col_start:col_end]   # raw (unscaled) probas
        bm_pred   = np.argmax(bm_proba, axis=1)
        # Map back to original class labels
        classes   = final_base_models[name].classes_
        bm_pred_labels = classes[bm_pred]
        base_model_metrics[name] = _compute_metrics(y_test, bm_pred_labels, bm_proba)

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    _print_summary(ensemble_metrics, base_model_metrics, y_test, ensemble_pred)

    return {
        "ensemble": ensemble_metrics,
        "base_models": base_model_metrics,
        "fitted_objects": {
            "scaler": scaler,
            "final_base_models": final_base_models,
            "meta_scaler": meta_scaler,
            "meta_learner": meta_learner,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone_model(name: str, random_state: int) -> Any:
    """Return a fresh (unfitted) instance of the named base model."""
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
            multi_class="auto",
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        )
    elif name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def _print_summary(
    ensemble_metrics: Dict[str, float],
    base_model_metrics: Dict[str, Dict[str, float]],
    y_test: np.ndarray,
    ensemble_pred: np.ndarray,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("STACKING ENSEMBLE — TEST SET RESULTS")
    print(sep)
    print(f"  Accuracy : {ensemble_metrics['accuracy']:.4f}")
    print(f"  AUC      : {ensemble_metrics['auc']:.4f}")
    print(f"  F1 Macro : {ensemble_metrics['f1_macro']:.4f}")

    print(f"\n{'─' * 60}")
    print("BASE MODEL COMPARISON (test set)")
    print(f"{'─' * 60}")
    header = f"{'Model':<25} {'Accuracy':>10} {'AUC':>10} {'F1 Macro':>10}"
    print(header)
    print("─" * 60)
    for name, m in base_model_metrics.items():
        print(
            f"{name:<25} {m['accuracy']:>10.4f} {m['auc']:>10.4f} {m['f1_macro']:>10.4f}"
        )
    print(f"{'Stacking Ensemble':<25} "
          f"{ensemble_metrics['accuracy']:>10.4f} "
          f"{ensemble_metrics['auc']:>10.4f} "
          f"{ensemble_metrics['f1_macro']:>10.4f}")
    print(sep)

    print("\nDetailed Classification Report (Ensemble):")
    print(classification_report(y_test, ensemble_pred, zero_division=0))


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target

    # ── BEST PRACTICE 1 ──────────────────────────────────────────────────
    # Split BEFORE any preprocessing.  The build_stacking_ensemble function
    # then fits the scaler only on X_train.
    # ─────────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = build_stacking_ensemble(
        X_train, X_test, y_train, y_test,
        n_folds=5,
        random_state=42,
    )

    print("\nReturned metrics dictionary:")
    print("  Ensemble :", results["ensemble"])
    for name, m in results["base_models"].items():
        print(f"  {name}: {m}")