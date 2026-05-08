```python
"""
Fairness-Aware Classification System

This module builds a fairness-aware binary classification pipeline that:
- Trains a logistic regression model
- Computes group-level and fairness metrics
- Applies post-hoc threshold adjustment if fairness thresholds are violated
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
    fpr: float          # False Positive Rate  = FP / (FP + TN)
    fnr: float          # False Negative Rate  = FN / (FN + TP)
    positive_rate: float  # P(ŷ=1 | group)

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
    demographic_parity_difference: float   # max(P(ŷ=1|g)) - min(P(ŷ=1|g))
    equalized_odds_difference: float       # max over FPR & FNR differences
    disparate_impact_ratio: float          # min(P(ŷ=1|g)) / max(P(ŷ=1|g))
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
    test_size: int

    # Pre-adjustment
    pre_group_metrics: Dict[Any, GroupMetrics]
    pre_fairness_metrics: FairnessMetrics

    # Post-adjustment (None if no adjustment was needed)
    post_group_metrics: Optional[Dict[Any, GroupMetrics]]
    post_fairness_metrics: Optional[FairnessMetrics]
    adjusted_thresholds: Optional[Dict[Any, float]]

    # Raw test data for reproducibility
    X_test: pd.DataFrame
    y_test: pd.Series
    protected_test: pd.Series

    def summary(self) -> str:
        lines = ["=" * 60, "FAIRNESS-AWARE CLASSIFICATION SUMMARY", "=" * 60]

        lines.append(f"\nTrain samples : {self.train_size}")
        lines.append(f"Test  samples : {self.test_size}")

        lines.append("\n--- PRE-ADJUSTMENT GROUP METRICS ---")
        for g, m in self.pre_group_metrics.items():
            lines.append(f"  Group={g}: acc={m.accuracy:.3f}, f1={m.f1:.3f}, "
                         f"fpr={m.fpr:.3f}, fnr={m.fnr:.3f}, "
                         f"pos_rate={m.positive_rate:.3f} (n={m.n_samples})")

        lines.append("\n--- PRE-ADJUSTMENT FAIRNESS METRICS ---")
        for k, v in self.pre_fairness_metrics.to_dict().items():
            lines.append(f"  {k}: {v}")

        if self.adjusted_thresholds is not None:
            lines.append("\n--- ADJUSTED THRESHOLDS ---")
            for g, t in self.adjusted_thresholds.items():
                lines.append(f"  Group={g}: threshold={t:.4f}")

            lines.append("\n--- POST-ADJUSTMENT GROUP METRICS ---")
            for g, m in self.post_group_metrics.items():
                lines.append(f"  Group={g}: acc={m.accuracy:.3f}, f1={m.f1:.3f}, "
                             f"fpr={m.fpr:.3f}, fnr={m.fnr:.3f}, "
                             f"pos_rate={m.positive_rate:.3f} (n={m.n_samples})")

            lines.append("\n--- POST-ADJUSTMENT FAIRNESS METRICS ---")
            for k, v in self.post_fairness_metrics.to_dict().items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append("\nNo threshold adjustment required (fairness constraints satisfied).")

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

    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y, protected,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )
    return X_train, X_test, y_train, y_test, p_train, p_test


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

        # Confusion matrix: [[TN, FP], [FN, TP]]
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

    # Violation: DP difference > threshold OR equalized odds difference > threshold
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
    Binary search over thresholds to find the one that achieves the target FPR
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


def _apply_threshold_adjustment(
    y_true: pd.Series,
    y_prob: np.ndarray,
    protected: pd.Series,
    pre_group_metrics: Dict[Any, GroupMetrics],
) -> Tuple[np.ndarray, Dict[Any, float]]:
    """
    Post-hoc threshold adjustment to equalize FPR across groups.

    Strategy: compute the mean FPR across groups as the target, then find
    per-group thresholds that achieve that target FPR.
    """
    groups = list(pre_group_metrics.keys())
    mean_fpr = np.mean([pre_group_metrics[g].fpr for g in groups])

    adjusted_thresholds: Dict[Any, float] = {}
    y_pred_adjusted = np.zeros(len(y_true), dtype=int)

    for g in groups:
        mask = (protected == g).values
        yt_g = y_true.values[mask]
        yp_g = y_prob[mask]

        thresh = _find_threshold_for_fpr(yt_g, yp_g, target_fpr=mean_fpr)
        adjusted_thresholds[g] = thresh
        y_pred_adjusted[mask] = (yp_g >= thresh).astype(int)

    return y_pred_adjusted, adjusted_thresholds


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def build_fairness_aware_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    protected: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    fairness_threshold: float = 0.1,
    C: float = 1.0,
    max_iter: int = 1000,
    scale_features: bool = True,
) -> FairnessAwareResult:
    """
    Build a fairness-aware logistic regression classifier.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric).
    y : pd.Series
        Binary target (0/1).
    protected : pd.Series
        Protected attribute column (categorical or binary).
    test_size : float
        Fraction of data to use as test set.
    random_state : int
        Random seed for reproducibility.
    fairness_threshold : float
        Maximum allowed demographic parity / equalized odds difference.
        If exceeded, post-hoc threshold adjustment is applied.
    C : float
        Inverse regularization strength for logistic regression.
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

    assert len(X) == len(y) == len(protected), "X, y, and protected must have the same length."
    assert set(y.unique()).issubset({0, 1}), "Target y must be binary (0/1)."

    # Reset indices for safe masking
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    protected = protected.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Stratified train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, p_train, p_test = _stratified_split_with_protected(
        X, y, protected, test_size=test_size, random_state=random_state
    )

    # ------------------------------------------------------------------
    # 3. Build and train logistic regression pipeline
    # ------------------------------------------------------------------
    steps = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)))

    model = Pipeline(steps)
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Predict on test set
    # ------------------------------------------------------------------
    y_pred_pre = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # 5. Pre-adjustment group & fairness metrics
    # ------------------------------------------------------------------
    pre_group_metrics = _compute_group_metrics(y_test, y_pred_pre, p_test)
    pre_fairness_metrics = _compute_fairness_metrics(pre_group_metrics, fairness_threshold)

    # ------------------------------------------------------------------
    # 6. Post-hoc threshold adjustment (if fairness violated)
    # ------------------------------------------------------------------
    post_group_metrics = None
    post_fairness_metrics = None
    adjusted_thresholds = None

    if pre_fairness_metrics.fairness_violated:
        y_pred_post, adjusted_thresholds = _apply_threshold_adjustment(
            y_test, y_prob, p_test, pre_group_metrics
        )
        post_group_metrics = _compute_group_metrics(y_test, y_pred_post, p_test)
        post_fairness_metrics = _compute_fairness_metrics(post_group_metrics, fairness_threshold)

    # ------------------------------------------------------------------
    # 7. Package and return results
    # ------------------------------------------------------------------
    return FairnessAwareResult(
        model=model,
        scaler_fitted=scale_features,
        train_size=len(X_train),
        test_size=len(X_test),
        pre_group_metrics=pre_group_metrics,
        pre_fairness_metrics=pre_fairness_metrics,
        post_group_metrics=post_group_metrics,
        post_fairness_metrics=post_fairness_metrics,
        adjusted_thresholds=adjusted_thresholds,
        X_test=X_test,
        y_test=y_test,
        protected_test=p_test,
    )


# ---------------------------------------------------------------------------
# Convenience: predict with adjusted thresholds
# ---------------------------------------------------------------------------

def predict_with_adjusted_thresholds(
    result: FairnessAwareResult,
    X_new: pd.DataFrame,
    protected_new: pd.Series,
) -> np.ndarray:
    """
    Apply the trained model with per-group adjusted thresholds to new data.

    Parameters
    ----------
    result : FairnessAwareResult
        Output of build_fairness_aware_classifier.
    X_new : pd.DataFrame
        New feature matrix.
    protected_new : pd.Series
        Protected attribute for new samples.

    Returns
    -------
    np.ndarray
        Binary predictions using adjusted thresholds (or 0.5 if no adjustment).
    """
    if not isinstance(X_new, pd.DataFrame):
        X_new = pd.DataFrame(X_new)
    if not isinstance(protected_new, pd.Series):
        protected_new = pd.Series(protected_new)

    X_new = X_new.reset_index(drop=True)
    protected_new = protected_new.reset_index(drop=True)

    y_prob = result.model.predict_proba(X_new)[:, 1]
    y_pred = np.zeros(len(X_new), dtype=int)

    thresholds = result.adjusted_thresholds or {}

    for g in protected_new.unique():
        mask = (protected_new == g).values
        thresh = thresholds.get(g, 0.5)
        y_pred[mask] = (y_prob[mask] >= thresh).astype(int)

    return y_pred


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _generate_synthetic_dataset(
    n_samples: int = 1000,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Generate a synthetic biased dataset for demonstration.

    Group A (protected=0) has a higher base positive rate than Group B (protected=1),
    introducing demographic disparity.
    """
    rng = np.random.RandomState(random_state)

    n_a = n_samples // 2
    n_b = n_samples - n_a

    # Group A: higher positive rate
    X_a = rng.randn(n_a, 4) + np.array([0.5, -0.3, 0.2, 0.1])
    y_a = (rng.rand(n_a) < 0.65).astype(int)

    # Group B: lower positive rate
    X_b = rng.randn(n_b, 4) + np.array([-0.5, 0.3, -0.2, -0.1])
    y_b =