```python
"""
Fairness-Aware Classification System

This module builds a fairness-aware binary classifier that:
- Trains on features with a protected attribute
- Evaluates group-level and fairness metrics
- Applies post-hoc threshold adjustment if fairness thresholds are violated
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GroupMetrics:
    """Per-group classification metrics."""
    group_value: object
    accuracy: float
    f1: float
    fpr: float  # False Positive Rate = FP / (FP + TN)
    fnr: float  # False Negative Rate = FN / (FN + TP)
    n_samples: int
    n_positive: int


@dataclass
class FairnessMetrics:
    """Dataset-level fairness metrics between two groups."""
    demographic_parity_difference: float   # |P(ŷ=1|A=0) - P(ŷ=1|A=1)|
    equalized_odds_difference: float       # max(|ΔFPR|, |ΔFNR|)
    disparate_impact_ratio: float          # min(P(ŷ=1|A=a)) / max(P(ŷ=1|A=a))
    fpr_difference: float
    fnr_difference: float


@dataclass
class FairnessAwareResult:
    """Complete result object returned by the system."""
    model: Pipeline
    scaler: StandardScaler  # fitted on train only
    pre_adjustment_group_metrics: Dict[object, GroupMetrics]
    post_adjustment_group_metrics: Optional[Dict[object, GroupMetrics]]
    pre_adjustment_fairness: FairnessMetrics
    post_adjustment_fairness: Optional[FairnessMetrics]
    adjusted_thresholds: Optional[Dict[object, float]]
    fairness_violated: bool
    protected_attribute: str
    feature_columns: list


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _stratify_key(y: pd.Series, protected: pd.Series) -> pd.Series:
    """Create a combined stratification key from target and protected attribute."""
    return y.astype(str) + "_" + protected.astype(str)


def _compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_value: object,
) -> GroupMetrics:
    """Compute classification metrics for a single group."""
    n = len(y_true)
    if n == 0:
        raise ValueError(f"Group '{group_value}' has no samples.")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return GroupMetrics(
        group_value=group_value,
        accuracy=float(acc),
        f1=float(f1),
        fpr=float(fpr),
        fnr=float(fnr),
        n_samples=int(n),
        n_positive=int(np.sum(y_true)),
    )


def _compute_fairness_metrics(
    group_metrics: Dict[object, GroupMetrics],
    y_pred_by_group: Dict[object, np.ndarray],
) -> FairnessMetrics:
    """
    Compute fairness metrics across all groups.
    For binary protected attribute, computes pairwise differences.
    For multi-group, computes max pairwise difference.
    """
    groups = list(group_metrics.keys())

    # Positive prediction rates per group
    pos_rates = {
        g: np.mean(y_pred_by_group[g]) for g in groups
    }
    fprs = {g: group_metrics[g].fpr for g in groups}
    fnrs = {g: group_metrics[g].fnr for g in groups}

    # Demographic parity difference: max - min positive prediction rate
    dp_diff = max(pos_rates.values()) - min(pos_rates.values())

    # Equalized odds difference: max of FPR diff and FNR diff
    fpr_diff = max(fprs.values()) - min(fprs.values())
    fnr_diff = max(fnrs.values()) - min(fnrs.values())
    eo_diff = max(fpr_diff, fnr_diff)

    # Disparate impact ratio: min / max positive prediction rate
    max_rate = max(pos_rates.values())
    min_rate = min(pos_rates.values())
    di_ratio = (min_rate / max_rate) if max_rate > 0 else 0.0

    return FairnessMetrics(
        demographic_parity_difference=float(dp_diff),
        equalized_odds_difference=float(eo_diff),
        disparate_impact_ratio=float(di_ratio),
        fpr_difference=float(fpr_diff),
        fnr_difference=float(fnr_diff),
    )


def _find_threshold_for_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float,
) -> float:
    """
    Binary search for a decision threshold that achieves a target FPR.
    Searches over candidate thresholds from the predicted probabilities.
    """
    negatives = y_true == 0
    if negatives.sum() == 0:
        return 0.5  # fallback

    # Candidate thresholds
    thresholds = np.sort(np.unique(y_prob))[::-1]

    best_threshold = 0.5
    best_fpr_diff = float("inf")

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        fp = np.sum((y_pred_t == 1) & (y_true == 0))
        tn = np.sum((y_pred_t == 0) & (y_true == 0))
        fpr_t = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        diff = abs(fpr_t - target_fpr)
        if diff < best_fpr_diff:
            best_fpr_diff = diff
            best_threshold = t

    return float(best_threshold)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_fairness_aware_classifier(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str,
    protected_attribute: str,
    test_size: float = 0.2,
    random_state: int = 42,
    fairness_threshold_fpr_diff: float = 0.1,
    fairness_threshold_dp_diff: float = 0.1,
    logistic_regression_kwargs: Optional[dict] = None,
) -> FairnessAwareResult:
    """
    Build a fairness-aware classification system.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features, target, and protected attribute.
    feature_columns : list
        List of feature column names to use for training.
    target_column : str
        Name of the binary target column (0/1).
    protected_attribute : str
        Name of the protected attribute column (e.g., 'gender', 'race').
    test_size : float
        Fraction of data to hold out for testing.
    random_state : int
        Random seed for reproducibility.
    fairness_threshold_fpr_diff : float
        Maximum allowed FPR difference across groups before adjustment.
    fairness_threshold_dp_diff : float
        Maximum allowed demographic parity difference before adjustment.
    logistic_regression_kwargs : dict, optional
        Additional keyword arguments for LogisticRegression.

    Returns
    -------
    FairnessAwareResult
        Complete result object with model, metrics, and adjusted thresholds.
    """
    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------
    required_cols = set(feature_columns + [target_column, protected_attribute])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    y = df[target_column].astype(int)
    unique_targets = set(y.unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target column must be binary (0/1). Found: {unique_targets}")

    protected = df[protected_attribute]
    groups = sorted(protected.unique())
    if len(groups) < 2:
        raise ValueError(
            f"Protected attribute '{protected_attribute}' must have at least 2 unique values."
        )

    # ------------------------------------------------------------------
    # 1. Train/test split BEFORE any preprocessing
    #    Stratify by combined (target, protected_attribute) key
    # ------------------------------------------------------------------
    strat_key = _stratify_key(y, protected)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(df, strat_key))

    X_train_raw = df.iloc[train_idx][feature_columns].copy()
    X_test_raw = df.iloc[test_idx][feature_columns].copy()
    y_train = y.iloc[train_idx].values
    y_test = y.iloc[test_idx].values
    prot_train = protected.iloc[train_idx].values
    prot_test = protected.iloc[test_idx].values

    # ------------------------------------------------------------------
    # 2. Fit scaler ONLY on training data
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)  # transform only, never fit

    # ------------------------------------------------------------------
    # 3. Train logistic regression classifier
    # ------------------------------------------------------------------
    lr_kwargs = {
        "max_iter": 1000,
        "random_state": random_state,
        "solver": "lbfgs",
    }
    if logistic_regression_kwargs:
        lr_kwargs.update(logistic_regression_kwargs)

    clf = LogisticRegression(**lr_kwargs)
    clf.fit(X_train, y_train)

    # Wrap in a pipeline for convenience (scaler already fitted separately)
    # We store the fitted scaler and model independently to maintain clarity
    pipeline = Pipeline([
        ("scaler", scaler),
        ("classifier", clf),
    ])
    # Note: pipeline.fit would refit scaler; we use it only for prediction
    # We manually set the scaler as already fitted
    pipeline.named_steps["scaler"] = scaler

    # ------------------------------------------------------------------
    # 4. Pre-adjustment evaluation on TEST set
    # ------------------------------------------------------------------
    y_prob_test = clf.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    pre_group_metrics: Dict[object, GroupMetrics] = {}
    y_pred_by_group: Dict[object, np.ndarray] = {}

    for g in groups:
        mask = prot_test == g
        if mask.sum() == 0:
            warnings.warn(f"Group '{g}' has no test samples. Skipping.")
            continue
        gm = _compute_group_metrics(y_test[mask], y_pred_test[mask], g)
        pre_group_metrics[g] = gm
        y_pred_by_group[g] = y_pred_test[mask]

    pre_fairness = _compute_fairness_metrics(pre_group_metrics, y_pred_by_group)

    # ------------------------------------------------------------------
    # 5. Check fairness violation
    # ------------------------------------------------------------------
    fairness_violated = (
        pre_fairness.fpr_difference > fairness_threshold_fpr_diff
        or pre_fairness.demographic_parity_difference > fairness_threshold_dp_diff
    )

    print(f"\n{'='*60}")
    print("PRE-ADJUSTMENT METRICS")
    print(f"{'='*60}")
    _print_group_metrics(pre_group_metrics)
    _print_fairness_metrics(pre_fairness)
    print(f"\nFairness violated: {fairness_violated}")

    # ------------------------------------------------------------------
    # 6. Post-hoc threshold adjustment (if fairness violated)
    #    Strategy: equalize FPR across groups by adjusting per-group thresholds
    #    Thresholds are selected using TRAINING data probabilities only
    #    (never using test labels for threshold selection)
    # ------------------------------------------------------------------
    adjusted_thresholds: Optional[Dict[object, float]] = None
    post_group_metrics: Optional[Dict[object, GroupMetrics]] = None
    post_fairness: Optional[FairnessMetrics] = None

    if fairness_violated:
        print(f"\n{'='*60}")
        print("APPLYING POST-HOC THRESHOLD ADJUSTMENT")
        print(f"{'='*60}")

        # Compute per-group FPR on TRAINING data to find target FPR
        y_prob_train = clf.predict_proba(X_train)[:, 1]
        y_pred_train_default = (y_prob_train >= 0.5).astype(int)

        train_group_fprs: Dict[object, float] = {}
        for g in groups:
            mask = prot_train == g
            if mask.sum() == 0:
                continue
            gm_train = _compute_group_metrics(
                y_train[mask], y_pred_train_default[mask], g
            )
            train_group_fprs[g] = gm_train.fpr

        # Target FPR: mean of per-group FPRs on training data
        target_fpr = float(np.mean(list(train_group_fprs.values())))
        print(f"Target FPR (mean across groups, from train): {target_fpr:.4f}")

        # Find per-group threshold on TRAINING data that achieves target FPR
        adjusted_thresholds = {}
        for g in groups:
            mask = prot_train == g
            if mask.sum() == 0:
                adjusted_thresholds[g] = 0.5
                continue
            t = _find_threshold_for_fpr(
                y_train[mask], y_prob_train[mask], target_fpr
            )
            adjusted_thresholds[g] = t
            print(f"  Group '{g}': threshold = {t:.4f} (train FPR was {train_group_fprs.get(g, 'N/A'):.4f})")

        # Apply adjusted thresholds to TEST data (evaluation only)
        y_pred_adjusted = np.zeros(len(y_test), dtype=int)
        for g in groups:
            mask = prot_test == g
            if mask.sum() == 0:
                continue
            t = adjusted_thresholds[g]
            y_pred_adjusted[mask] = (y_prob_test[mask] >= t).astype(int)

        # Compute post-adjustment metrics on test set
        post_group_metrics = {}
        y_pred_adj_by_group: Dict[object, np.ndarray] = {}

        for g in groups:
            mask = prot_test == g
            if mask.sum() == 0:
                continue
            gm = _compute_group_metrics(y_test[mask], y_pred_adjusted[mask], g)
            post_group_metrics[g] = gm
            y_pred_adj_by_group[g] = y_pred_adjusted[mask]

        post_fairness = _compute_fairness_metrics(post_group_metrics, y_pred_adj_by_group)

        print(f"\n{'='*60}")
        print("POST-ADJUSTMENT METRICS")
        print(f"{'='*60}")
        _print_group_metrics(post_group_metrics)
        _print_fairness_metrics(post_fairness)

        # Comparison summary
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Metric':<35} {'Pre':>10} {'Post':>10} {'Δ':>10}")
        print("-" * 65)
        metrics_to_compare = [
            ("FPR Difference", pre_fairness.fpr_difference, post_fairness.fpr_difference),
            ("FNR Difference", pre_fairness.fnr_difference, post_fairness.fnr_difference),
            ("Demographic Parity Diff", pre_fairness.demographic_parity_difference,
             post_fairness.demographic_parity_difference),
            ("Equalized Odds Diff", pre_fairness.equalized_odds_difference,
             post_fairness.equalized_odds_difference),
            ("Disparate Impact Ratio", pre_fairness.disparate_impact_ratio,
             post_fairness.disparate_impact_ratio),
        ]
        for name, pre_val, post_val in metrics_to_compare:
            delta = post_val - pre_val
            print(f"{name:<35} {pre_val:>10.4f} {post_val:>10.4f} {delta:>+10.4f}")

    return FairnessAwareResult(
        model=pipeline,
        scaler=scaler,
        pre_adjustment_group_metrics=pre_group_metrics,
        post_adjustment_group_metrics=post_group_metrics,
        pre_adjustment_fairness=pre_fairness,
        post_adjustment_fairness=post_fairness,
        adjusted_thresholds=adjusted_thresholds,
        fairness_violated=fairness_violated,
        protected_attribute=protected_attribute,
        feature_columns=feature_columns,
    )


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_group_metrics(group_metrics: Dict[object, GroupMetrics]) -> None:
    print(f"\n{'Group':<15} {'N':>6} {'N_pos':>6} {'Acc':>8} {'F1':>8} {'FPR':>8} {'FNR':>8}")
    print("-" * 65)
    for g, gm in group_metrics.items():
        print(
            f"{str(g):<15} {gm.n_samples:>6} {gm.n_positive:>6} "
            f"{gm.accuracy:>8.4f} {gm.f1:>8.4f} {gm.fpr:>8.4f} {gm.fnr:>8.4f}"
        )


def _print_fairness_metrics(fm: FairnessMetrics) -> None:
    print(f"\nFairness Metrics:")
    print(f"  Demographic Parity Difference : {fm.demographic_parity_difference:.4f}")
    print(f"  Equalized Odds Difference     : {fm.equalized_odds_difference:.4f}")
    print(f"  Disparate Impact Ratio        : {fm.disparate_impact_ratio:.4f}")
    print(f"  FPR Difference                : {fm.fpr_difference:.4f}")
    print(f"  FNR Difference                : {fm.fnr_difference:.4f}")


# ---------------------------------------------------------------------------
# Prediction helper (uses adjusted thresholds if available)
# ---------------------------------------------------------------------------

def predict_with_fairness(
    result: FairnessAwareResult,
    X_new: pd.DataFrame,
    protected_new: pd.Series,
) -> np.ndarray:
    """
    Make predictions using the fairness-aware model.
    If adjusted thresholds exist, applies per-group thresholds.

    Parameters
    ----------
    result : Fairness