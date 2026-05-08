```python
"""
Fairness-Aware Classification System

This module builds a fairness-aware binary classification pipeline that:
- Trains a logistic regression model
- Computes group-level and fairness metrics
- Applies post-hoc threshold adjustment if fairness thresholds are violated
  (thresholds are tuned on a validation split, NOT the test set)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GroupMetrics:
    """Per-group classification metrics."""
    group_value: Any
    n_samples: int
    accuracy: float
    f1: float
    fpr: float
    fnr: float
    positive_rate: float

    def to_dict(self) -> Dict:
        return {
            "group_value": self.group_value,
            "n_samples": self.n_samples,
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
            "fnr": round(self.fnr, 4),
            "positive_rate": round(self.positive_rate, 4),
        }


@dataclass
class FairnessMetrics:
    """Fairness metrics computed across groups."""
    demographic_parity_difference: float
    equalized_odds_difference: float
    disparate_impact_ratio: float
    max_fpr_difference: float
    max_fnr_difference: float
    fairness_violated: bool
    threshold_used: float

    def to_dict(self) -> Dict:
        return {
            "demographic_parity_difference": round(self.demographic_parity_difference, 4),
            "equalized_odds_difference": round(self.equalized_odds_difference, 4),
            "disparate_impact_ratio": round(self.disparate_impact_ratio, 4),
            "max_fpr_difference": round(self.max_fpr_difference, 4),
            "max_fnr_difference": round(self.max_fnr_difference, 4),
            "fairness_violated": self.fairness_violated,
            "threshold_used": self.threshold_used,
        }


@dataclass
class FairnessAwareResult:
    """Full result object returned by the pipeline."""
    model: Pipeline
    scaler_fitted: bool
    train_size: int
    val_size: int
    test_size: int

    # Pre-adjustment (evaluated on test set)
    pre_group_metrics: Dict[Any, GroupMetrics]
    pre_fairness_metrics: FairnessMetrics

    # Post-adjustment (None if no adjustment was needed)
    post_group_metrics: Optional[Dict[Any, GroupMetrics]]
    post_fairness_metrics: Optional[FairnessMetrics]
    # Thresholds tuned on validation set
    adjusted_thresholds: Optional[Dict[Any, float]]

    # Raw test data for reproducibility
    X_test: pd.DataFrame
    y_test: pd.Series
    protected_test: pd.Series

    def summary(self) -> str:
        lines = ["=" * 60, "FAIRNESS-AWARE CLASSIFICATION SUMMARY", "=" * 60]

        lines.append(f"\nTrain samples      : {self.train_size}")
        lines.append(f"Validation samples : {self.val_size}")
        lines.append(f"Test  samples      : {self.test_size}")

        lines.append("\n--- PRE-ADJUSTMENT GROUP METRICS (test set) ---")
        for g, m in self.pre_group_metrics.items():
            lines.append(
                f"  Group={g}: acc={m.accuracy:.3f}, f1={m.f1:.3f}, "
                f"fpr={m.fpr:.3f}, fnr={m.fnr:.3f}, "
                f"pos_rate={m.positive_rate:.3f} (n={m.n_samples})"
            )

        lines.append("\n--- PRE-ADJUSTMENT FAIRNESS METRICS ---")
        for k, v in self.pre_fairness_metrics.to_dict().items():
            lines.append(f"  {k}: {v}")

        if self.adjusted_thresholds is not None:
            lines.append("\n--- ADJUSTED THRESHOLDS (tuned on validation set) ---")
            for g, t in self.adjusted_thresholds.items():
                lines.append(f"  Group={g}: threshold={t:.4f}")

            lines.append("\n--- POST-ADJUSTMENT GROUP METRICS (test set) ---")
            for g, m in self.post_group_metrics.items():
                lines.append(
                    f"  Group={g}: acc={m.accuracy:.3f}, f1={m.f1:.3f}, "
                    f"fpr={m.fpr:.3f}, fnr={m.fnr:.3f}, "
                    f"pos_rate={m.positive_rate:.3f} (n={m.n_samples})"
                )

            lines.append("\n--- POST-ADJUSTMENT FAIRNESS METRICS ---")
            for k, v in self.post_fairness_metrics.to_dict().items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(
                "\nNo threshold adjustment required (fairness constraints satisfied)."
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def _stratified_split_with_protected(
    X: pd.DataFrame,
    y: pd.Series,
    protected: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Stratified split by a combined stratum of (target, protected attribute).
    Falls back to target-only stratification if any stratum is too small.
    """
    strata = y.astype(str) + "_" + protected.astype(str)
    min_count = strata.value_counts().min()

    if min_count < 2:
        warnings.warn(
            "Some (target, protected) strata have fewer than 2 samples. "
            "Falling back to target-only stratification.",
            UserWarning,
        )
        stratify_col = y
    else:
        stratify_col = strata

    X_a, X_b, y_a, y_b, p_a, p_b = train_test_split(
        X, y, protected,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )
    return X_a, X_b, y_a, y_b, p_a, p_b


def _compute_group_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    protected: pd.Series,
) -> Dict[Any, GroupMetrics]:
    """Compute per-group accuracy, F1, FPR, FNR, and positive rate."""
    metrics = {}
    groups = protected.unique()

    for g in sorted(groups):
        mask = protected == g
        yt = y_true[mask]
        yp = y_pred[mask]

        if len(yt) == 0:
            continue

        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, zero_division=0)
        pos_rate = yp.mean()

        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        metrics[g] = GroupMetrics(
            group_value=g,
            n_samples=int(mask.sum()),
            accuracy=acc,
            f1=f1,
            fpr=fpr,
            fnr=fnr,
            positive_rate=pos_rate,
        )

    return metrics


def _compute_fairness_metrics(
    group_metrics: Dict[Any, GroupMetrics],
    fairness_threshold: float = 0.1,
) -> FairnessMetrics:
    """Compute demographic parity difference, equalized odds difference, disparate impact."""
    pos_rates = [m.positive_rate for m in group_metrics.values()]
    fprs = [m.fpr for m in group_metrics.values()]
    fnrs = [m.fnr for m in group_metrics.values()]

    dp_diff = max(pos_rates) - min(pos_rates)
    max_fpr_diff = max(fprs) - min(fprs)
    max_fnr_diff = max(fnrs) - min(fnrs)
    eo_diff = max(max_fpr_diff, max_fnr_diff)
    di_ratio = (min(pos_rates) / max(pos_rates)) if max(pos_rates) > 0 else 1.0

    violated = (dp_diff > fairness_threshold) or (eo_diff > fairness_threshold)

    return FairnessMetrics(
        demographic_parity_difference=dp_diff,
        equalized_odds_difference=eo_diff,
        disparate_impact_ratio=di_ratio,
        max_fpr_difference=max_fpr_diff,
        max_fnr_difference=max_fnr_diff,
        fairness_violated=violated,
        threshold_used=fairness_threshold,
    )


def _find_threshold_for_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float,
    n_thresholds: int = 1000,
) -> float:
    """
    Search over thresholds to find the one that achieves the target FPR
    (or as close as possible) for a single group.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_thresh = 0.5
    best_diff = float("inf")

    negatives = y_true == 0
    if negatives.sum() == 0:
        return 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        diff = abs(fpr - target_fpr)
        if diff < best_diff:
            best_diff = diff
            best_thresh = t

    return best_thresh


def _tune_thresholds_on_validation(
    y_val: pd.Series,
    y_prob_val: np.ndarray,
    protected_val: pd.Series,
) -> Dict[Any, float]:
    """
    Tune per-group thresholds to equalise FPR using the VALIDATION set only.

    Strategy: compute the mean FPR across groups on the validation set as the
    target, then find per-group thresholds that achieve that target FPR.

    Parameters
    ----------
    y_val : pd.Series
        True labels for the validation split.
    y_prob_val : np.ndarray
        Predicted probabilities for the validation split.
    protected_val : pd.Series
        Protected attribute for the validation split.

    Returns
    -------
    Dict[Any, float]
        Per-group thresholds tuned on the validation set.
    """
    groups = sorted(protected_val.unique())

    # Compute per-group FPR at the default 0.5 threshold on validation data
    val_group_fprs: Dict[Any, float] = {}
    for g in groups:
        mask = (protected_val == g).values
        yt_g = y_val.values[mask]
        yp_g = (y_prob_val[mask] >= 0.5).astype(int)
        fp = ((yp_g == 1) & (yt_g == 0)).sum()
        tn = ((yp_g == 0) & (yt_g == 0)).sum()
        val_group_fprs[g] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    target_fpr = float(np.mean(list(val_group_fprs.values())))

    adjusted_thresholds: Dict[Any, float] = {}
    for g in groups:
        mask = (protected_val == g).values
        yt_g = y_val.values[mask]
        yp_g = y_prob_val[mask]
        thresh = _find_threshold_for_fpr(yt_g, yp_g, target_fpr=target_fpr)
        adjusted_thresholds[g] = thresh

    return adjusted_thresholds


def _apply_adjusted_thresholds(
    y_prob: np.ndarray,
    protected: pd.Series,
    adjusted_thresholds: Dict[Any, float],
) -> np.ndarray:
    """
    Apply pre-computed per-group thresholds to produce binary predictions.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities.
    protected : pd.Series
        Protected attribute aligned with y_prob.
    adjusted_thresholds : Dict[Any, float]
        Per-group thresholds (tuned on validation set).

    Returns
    -------
    np.ndarray
        Binary predictions.
    """
    y_pred = np.zeros(len(y_prob), dtype=int)
    for g, thresh in adjusted_thresholds.items():
        mask = (protected == g).values
        y_pred[mask] = (y_prob[mask] >= thresh).astype(int)
    return y_pred


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def build_fairness_aware_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    protected: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    fairness_threshold: float = 0.1,
    C: float = 1.0,
    max_iter: int = 1000,
    scale_features: bool = True,
) -> FairnessAwareResult:
    """
    Build a fairness-aware logistic regression classifier.

    The data is split into three non-overlapping parts:
      - **Train**      : used to fit the model.
      - **Validation** : used exclusively to tune per-group decision thresholds.
      - **Test**       : used only for final evaluation (never seen during tuning).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric).
    y : pd.Series
        Binary target (0/1).
    protected : pd.Series
        Protected attribute column (categorical or binary).
    test_size : float
        Fraction of the full dataset to hold out as the test set.
    val_size : float
        Fraction of the *remaining* (train+val) data to use as the validation
        set for threshold tuning.
    random_state : int
        Random seed for reproducibility.
    fairness_threshold : float
        Maximum allowed demographic parity / equalized odds difference.
        If exceeded, post-hoc threshold adjustment is applied (tuned on val).
    C : float
        Inverse regularisation strength for logistic regression.
    max_iter : int
        Maximum iterations for logistic regression solver.
    scale_features : bool
        Whether to apply StandardScaler before logistic regression.

    Returns
    -------
    FairnessAwareResult
        Dataclass containing model, metrics, and adjusted thresholds.
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name="target")
    if not isinstance(protected, pd.Series):
        protected = pd.Series(protected, name="protected")

    assert len(X) == len(y) == len(protected), (
        "X, y, and protected must have the same length."
    )
    assert set(y.unique()).issubset({0, 1}), "Target y must be binary (0/1)."

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    protected = protected.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Three-way split: train / val / test
    #    Step A: split off the test set (stratified).
    #    Step B: split the remainder into train and val (stratified).
    # ------------------------------------------------------------------
    # Step A – hold out test set
    X_trainval, X_test, y_trainval, y_test, p_trainval, p_test = (
        _stratified_split_with_protected(
            X, y, protected,
            test_size=test_size,
            random_state=random_state,
        )
    )

    # Reset indices after first split
    X_trainval = X_trainval.reset_index(drop=True)
    y_trainval = y_trainval.reset_index(drop=True)
    p_trainval = p_trainval.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    p_test = p_test.reset_index(drop=True)

    # Step B – split train+val into train and val
    X_train, X_val, y_train, y_val, p_train, p_val = (
        _stratified_split_with_protected(
            X_trainval, y_trainval, p_trainval,
            test_size=val_size,
            random_state=random_state,
        )
    )

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    p_train = p_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    p_val = p_val.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Build and train logistic regression pipeline (on train only)
    # ------------------------------------------------------------------
    steps = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=random_state))
    )

    model = Pipeline(steps)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Predict on test set (pre-adjustment, default threshold = 0.5)
    # ------------------------------------------------------------------
    y_pred_pre = model.predict(X_test)          # uses default 0.5 threshold
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # 5. Pre-adjustment group & fairness metrics (on test set)
    # ------------------------------------------------------------------
    pre_group_metrics = _compute_group_metrics(y_test, y_pred_pre, p_test)
    pre_fairness_metrics = _compute_fairness_metrics(
        pre_group_metrics, fairness_threshold
    )

    # ------------------------------------------------------------------
    # 6. Post-hoc threshold adjustment (if fairness violated)
    #    Thresholds are tuned on the VALIDATION set, then applied to test.
    # ------------------------------------------------------------------
    post_group_metrics = None
    post_fairness_metrics = None
    adjusted_thresholds = None

    if pre_fairness_metrics.fairness_violated:
        # Obtain probabilities on