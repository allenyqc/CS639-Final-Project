```python
"""
Fairness-Aware Classification System

This module builds a fairness-aware binary classifier that:
1. Splits data (train/val/test) before any preprocessing
2. Trains logistic regression on training data only
3. Computes group-level and fairness metrics
4. Applies post-hoc threshold adjustment on validation set if fairness thresholds are violated
5. Reports pre- and post-adjustment metrics on the test set
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GroupMetrics:
    """Per-group classification metrics."""
    group_value: Any
    accuracy: float
    f1: float
    fpr: float  # False Positive Rate = FP / (FP + TN)
    fnr: float  # False Negative Rate = FN / (FN + TP)
    n_samples: int
    n_positive: int


@dataclass
class FairnessMetrics:
    """Dataset-level fairness metrics."""
    demographic_parity_difference: float   # |P(ŷ=1|A=0) - P(ŷ=1|A=1)|
    equalized_odds_difference: float       # max(|ΔFPR|, |ΔFNR|)
    disparate_impact_ratio: float          # min(P(ŷ=1|A=a)) / max(P(ŷ=1|A=a))
    fpr_difference: float                  # |FPR_group0 - FPR_group1|
    fnr_difference: float                  # |FNR_group0 - FNR_group1|


@dataclass
class FairnessAwareResult:
    """Full result object returned by the fairness-aware classifier."""
    model: Pipeline
    pre_adjustment: dict[Any, GroupMetrics]
    post_adjustment: dict[Any, GroupMetrics] | None
    pre_fairness: FairnessMetrics
    post_fairness: FairnessMetrics | None
    adjusted_thresholds: dict[Any, float] | None
    fairness_violated: bool
    feature_names: list[str]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _stratified_split_with_protected(
    X: pd.DataFrame,
    y: pd.Series,
    protected: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
    pd.Series, pd.Series, pd.Series,
]:
    """
    Stratified split by (target × protected attribute) into train / val / test.

    Returns
    -------
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    prot_train, prot_val, prot_test
    """
    # Create a combined stratification key
    strat_key = y.astype(str) + "_" + protected.astype(str)

    # First split: separate test set
    X_trainval, X_test, y_trainval, y_test, prot_trainval, prot_test = train_test_split(
        X, y, protected,
        test_size=test_size,
        stratify=strat_key,
        random_state=random_state,
    )

    # Second split: separate validation set from train
    strat_key_trainval = y_trainval.astype(str) + "_" + prot_trainval.astype(str)
    relative_val_size = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val, prot_train, prot_val = train_test_split(
        X_trainval, y_trainval, prot_trainval,
        test_size=relative_val_size,
        stratify=strat_key_trainval,
        random_state=random_state,
    )

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        prot_train, prot_val, prot_test,
    )


def _compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
) -> dict[Any, GroupMetrics]:
    """Compute per-group accuracy, F1, FPR, FNR."""
    groups = np.unique(protected)
    metrics: dict[Any, GroupMetrics] = {}

    for g in groups:
        mask = protected == g
        yt = y_true[mask]
        yp = y_pred[mask]

        acc = accuracy_score(yt, yp)
        # Use zero_division=0 to handle edge cases gracefully
        f1 = f1_score(yt, yp, zero_division=0)

        # Confusion matrix: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        metrics[g] = GroupMetrics(
            group_value=g,
            accuracy=float(acc),
            f1=float(f1),
            fpr=float(fpr),
            fnr=float(fnr),
            n_samples=int(mask.sum()),
            n_positive=int(yt.sum()),
        )

    return metrics


def _compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
) -> FairnessMetrics:
    """
    Compute demographic parity difference, equalized odds difference,
    and disparate impact ratio for a binary protected attribute.
    """
    groups = np.unique(protected)
    if len(groups) != 2:
        raise ValueError(
            f"Fairness metrics currently support exactly 2 protected groups; "
            f"got {len(groups)}: {groups}"
        )

    g0, g1 = groups[0], groups[1]
    mask0 = protected == g0
    mask1 = protected == g1

    # Positive prediction rates
    ppr0 = y_pred[mask0].mean()
    ppr1 = y_pred[mask1].mean()

    # Demographic parity difference
    dp_diff = abs(ppr0 - ppr1)

    # FPR / FNR per group
    gm = _compute_group_metrics(y_true, y_pred, protected)
    fpr0, fnr0 = gm[g0].fpr, gm[g0].fnr
    fpr1, fnr1 = gm[g1].fpr, gm[g1].fnr

    fpr_diff = abs(fpr0 - fpr1)
    fnr_diff = abs(fnr0 - fnr1)
    eo_diff = max(fpr_diff, fnr_diff)

    # Disparate impact ratio (avoid division by zero)
    max_ppr = max(ppr0, ppr1)
    min_ppr = min(ppr0, ppr1)
    di_ratio = (min_ppr / max_ppr) if max_ppr > 0 else 1.0

    return FairnessMetrics(
        demographic_parity_difference=float(dp_diff),
        equalized_odds_difference=float(eo_diff),
        disparate_impact_ratio=float(di_ratio),
        fpr_difference=float(fpr_diff),
        fnr_difference=float(fnr_diff),
    )


def _apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to binary predictions using a threshold."""
    return (proba >= threshold).astype(int)


def _equalize_fpr_thresholds(
    y_val: np.ndarray,
    probas_val: np.ndarray,
    protected_val: np.ndarray,
    target_fpr: float | None = None,
    n_thresholds: int = 200,
) -> dict[Any, float]:
    """
    Find per-group thresholds on the VALIDATION set that equalize FPR across groups.

    Strategy
    --------
    For each group, sweep thresholds and pick the one whose FPR is closest
    to the target FPR (mean FPR across groups at the default 0.5 threshold).

    Parameters
    ----------
    y_val : ground-truth labels (validation)
    probas_val : predicted probabilities (validation)
    protected_val : protected attribute values (validation)
    target_fpr : desired FPR; if None, uses mean FPR at default threshold
    n_thresholds : number of candidate thresholds to sweep

    Returns
    -------
    dict mapping group value → optimal threshold
    """
    groups = np.unique(protected_val)
    thresholds_grid = np.linspace(0.01, 0.99, n_thresholds)

    # Determine target FPR from default threshold if not provided
    if target_fpr is None:
        default_preds = _apply_threshold(probas_val, 0.5)
        gm_default = _compute_group_metrics(y_val, default_preds, protected_val)
        target_fpr = float(np.mean([gm_default[g].fpr for g in groups]))
        logger.info("Target FPR for equalization: %.4f", target_fpr)

    adjusted_thresholds: dict[Any, float] = {}

    for g in groups:
        mask = protected_val == g
        yt_g = y_val[mask]
        prob_g = probas_val[mask]

        best_thresh = 0.5
        best_fpr_diff = float("inf")

        for t in thresholds_grid:
            preds_g = _apply_threshold(prob_g, t)
            cm = confusion_matrix(yt_g, preds_g, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
            fpr_g = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            diff = abs(fpr_g - target_fpr)
            if diff < best_fpr_diff:
                best_fpr_diff = diff
                best_thresh = t

        adjusted_thresholds[g] = float(best_thresh)
        logger.info("Group %s → adjusted threshold: %.4f", g, best_thresh)

    return adjusted_thresholds


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def build_fairness_aware_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    protected: pd.Series,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    fairness_threshold_dp: float = 0.10,   # demographic parity difference
    fairness_threshold_eo: float = 0.10,   # equalized odds difference
    fairness_threshold_di: float = 0.80,   # disparate impact ratio (lower bound)
    logistic_regression_kwargs: dict[str, Any] | None = None,
) -> FairnessAwareResult:
    """
    Build a fairness-aware binary classifier.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric or will be coerced to numeric).
    y : pd.Series
        Binary target (0/1).
    protected : pd.Series
        Protected attribute (binary categorical, e.g., 0/1 or 'M'/'F').
    test_size : float
        Fraction of data held out as the final test set.
    val_size : float
        Fraction of data used for validation (threshold tuning).
    random_state : int
        Random seed for reproducibility.
    fairness_threshold_dp : float
        Maximum allowed demographic parity difference before adjustment.
    fairness_threshold_eo : float
        Maximum allowed equalized odds difference before adjustment.
    fairness_threshold_di : float
        Minimum allowed disparate impact ratio before adjustment.
    logistic_regression_kwargs : dict | None
        Extra keyword arguments forwarded to LogisticRegression.

    Returns
    -------
    FairnessAwareResult
    """
    if logistic_regression_kwargs is None:
        logistic_regression_kwargs = {}

    # -----------------------------------------------------------------------
    # 1. Validate inputs
    # -----------------------------------------------------------------------
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")
    if not isinstance(protected, pd.Series):
        raise TypeError("protected must be a pandas Series.")
    if set(y.unique()) - {0, 1}:
        raise ValueError("y must be binary (0/1).")
    if len(np.unique(protected)) != 2:
        raise ValueError("protected attribute must have exactly 2 unique values.")

    feature_names = list(X.columns)

    # -----------------------------------------------------------------------
    # 2. Train / Val / Test split — BEFORE any preprocessing
    # -----------------------------------------------------------------------
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        prot_train, prot_val, prot_test,
    ) = _stratified_split_with_protected(
        X, y, protected,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    # -----------------------------------------------------------------------
    # 3. Build pipeline: scaler + logistic regression
    #    Fit ONLY on training data
    # -----------------------------------------------------------------------
    lr_defaults: dict[str, Any] = {
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": random_state,
        "class_weight": "balanced",
    }
    lr_defaults.update(logistic_regression_kwargs)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**lr_defaults)),
    ])

    logger.info("Fitting logistic regression pipeline on training data (%d samples)...", len(X_train))
    pipeline.fit(X_train, y_train)

    # -----------------------------------------------------------------------
    # 4. Predict on validation and test sets
    # -----------------------------------------------------------------------
    probas_val = pipeline.predict_proba(X_val)[:, 1]
    probas_test = pipeline.predict_proba(X_test)[:, 1]

    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()
    prot_val_np = prot_val.to_numpy()
    prot_test_np = prot_test.to_numpy()

    # Default threshold = 0.5 predictions on test set
    preds_test_default = _apply_threshold(probas_test, 0.5)

    # -----------------------------------------------------------------------
    # 5. Pre-adjustment metrics (test set, threshold = 0.5)
    # -----------------------------------------------------------------------
    pre_group_metrics = _compute_group_metrics(y_test_np, preds_test_default, prot_test_np)
    pre_fairness = _compute_fairness_metrics(y_test_np, preds_test_default, prot_test_np)

    logger.info("Pre-adjustment fairness metrics:")
    logger.info("  Demographic Parity Difference : %.4f", pre_fairness.demographic_parity_difference)
    logger.info("  Equalized Odds Difference     : %.4f", pre_fairness.equalized_odds_difference)
    logger.info("  Disparate Impact Ratio        : %.4f", pre_fairness.disparate_impact_ratio)

    # -----------------------------------------------------------------------
    # 6. Check fairness thresholds
    # -----------------------------------------------------------------------
    fairness_violated = (
        pre_fairness.demographic_parity_difference > fairness_threshold_dp
        or pre_fairness.equalized_odds_difference > fairness_threshold_eo
        or pre_fairness.disparate_impact_ratio < fairness_threshold_di
    )

    post_group_metrics: dict[Any, GroupMetrics] | None = None
    post_fairness: FairnessMetrics | None = None
    adjusted_thresholds: dict[Any, float] | None = None

    if fairness_violated:
        logger.info(
            "Fairness threshold violated. Applying post-hoc FPR equalization "
            "using VALIDATION set..."
        )

        # -----------------------------------------------------------------------
        # 7. Threshold adjustment — tuned on VALIDATION set only
        # -----------------------------------------------------------------------
        adjusted_thresholds = _equalize_fpr_thresholds(
            y_val_np, probas_val, prot_val_np
        )

        # -----------------------------------------------------------------------
        # 8. Apply adjusted thresholds to TEST set for final evaluation
        # -----------------------------------------------------------------------
        preds_test_adjusted = np.zeros(len(y_test_np), dtype=int)
        for g, thresh in adjusted_thresholds.items():
            mask = prot_test_np == g
            preds_test_adjusted[mask] = _apply_threshold(probas_test[mask], thresh)

        post_group_metrics = _compute_group_metrics(
            y_test_np, preds_test_adjusted, prot_test_np
        )
        post_fairness = _compute_fairness_metrics(
            y_test_np, preds_test_adjusted, prot_test_np
        )

        logger.info("Post-adjustment fairness metrics:")
        logger.info("  Demographic Parity Difference : %.4f", post_fairness.demographic_parity_difference)
        logger.info("  Equalized Odds Difference     : %.4f", post_fairness.equalized_odds_difference)
        logger.info("  Disparate Impact Ratio        : %.4f", post_fairness.disparate_impact_ratio)
    else:
        logger.info("Model satisfies all fairness thresholds. No adjustment needed.")

    return FairnessAwareResult(
        model=pipeline,
        pre_adjustment=pre_group_metrics,
        post_adjustment=post_group_metrics,
        pre_fairness=pre_fairness,
        post_fairness=post_fairness,
        adjusted_thresholds=adjusted_thresholds,
        fairness_violated=fairness_violated,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Reporting utility
# ---------------------------------------------------------------------------

def print_fairness_report(result: FairnessAwareResult) -> None:
    """Pretty-print a summary of the fairness-aware classification result."""
    sep = "=" * 60

    print(sep)
    print("FAIRNESS-AWARE CLASSIFICATION REPORT")
    print(sep)

    print("\n[PRE-ADJUSTMENT] Per-Group Metrics (threshold = 0.50)")
    print(f"{'Group':<12} {'Acc':>8} {'F1':>8} {'FPR':>8} {'FNR':>8} {'N':>6} {'N+':>6}")
    print("-" * 60