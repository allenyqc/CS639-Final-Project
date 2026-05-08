```python
"""
Fairness-Aware Classification System

This module builds a fairness-aware binary classification pipeline that:
- Trains a logistic regression classifier
- Computes group-level and fairness metrics
- Applies post-hoc threshold adjustment if fairness thresholds are violated
- Returns comprehensive metrics and adjusted thresholds
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
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
    threshold: float = 0.5

    def to_dict(self) -> Dict:
        return {
            "group_value": self.group_value,
            "n_samples": self.n_samples,
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.fpr, 4),
            "fnr": round(self.fnr, 4),
            "positive_rate": round(self.positive_rate, 4),
            "threshold": round(self.threshold, 4),
        }


@dataclass
class FairnessMetrics:
    """Fairness metrics computed across groups."""
    demographic_parity_difference: float   # |P(ŷ=1|A=0) - P(ŷ=1|A=1)|
    equalized_odds_difference: float       # max(|ΔFPR|, |ΔFNR|)
    disparate_impact_ratio: float          # min(P(ŷ=1|A=a)) / max(P(ŷ=1|A=a))
    fpr_difference: float
    fnr_difference: float
    violates_threshold: bool
    threshold_used: float

    def to_dict(self) -> Dict:
        return {
            "demographic_parity_difference": round(self.demographic_parity_difference, 4),
            "equalized_odds_difference": round(self.equalized_odds_difference, 4),
            "disparate_impact_ratio": round(self.disparate_impact_ratio, 4),
            "fpr_difference": round(self.fpr_difference, 4),
            "fnr_difference": round(self.fnr_difference, 4),
            "violates_threshold": self.violates_threshold,
            "threshold_used": self.threshold_used,
        }


@dataclass
class FairnessResult:
    """Complete result object returned by the fairness pipeline."""
    model: Pipeline
    scaler: StandardScaler
    feature_columns: list
    target_column: str
    protected_attribute: str
    group_values: list

    # Pre-adjustment
    pre_group_metrics: Dict[Any, GroupMetrics]
    pre_fairness_metrics: FairnessMetrics

    # Post-adjustment (None if no adjustment was needed)
    post_group_metrics: Optional[Dict[Any, GroupMetrics]] = None
    post_fairness_metrics: Optional[FairnessMetrics] = None
    adjusted_thresholds: Optional[Dict[Any, float]] = None

    # Test data (stored for reproducibility)
    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None
    protected_test: Optional[pd.Series] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "FAIRNESS-AWARE CLASSIFICATION SUMMARY",
            "=" * 60,
            f"Protected attribute : {self.protected_attribute}",
            f"Groups              : {self.group_values}",
            "",
            "--- PRE-ADJUSTMENT GROUP METRICS ---",
        ]
        for gv, gm in self.pre_group_metrics.items():
            lines.append(f"  Group={gv}: acc={gm.accuracy:.4f}, f1={gm.f1:.4f}, "
                         f"fpr={gm.fpr:.4f}, fnr={gm.fnr:.4f}, "
                         f"pos_rate={gm.positive_rate:.4f}")
        lines += [
            "",
            "--- PRE-ADJUSTMENT FAIRNESS METRICS ---",
        ]
        for k, v in self.pre_fairness_metrics.to_dict().items():
            lines.append(f"  {k}: {v}")

        if self.post_group_metrics is not None:
            lines += [
                "",
                "--- POST-ADJUSTMENT GROUP METRICS ---",
            ]
            for gv, gm in self.post_group_metrics.items():
                lines.append(f"  Group={gv} (thresh={gm.threshold:.4f}): "
                             f"acc={gm.accuracy:.4f}, f1={gm.f1:.4f}, "
                             f"fpr={gm.fpr:.4f}, fnr={gm.fnr:.4f}, "
                             f"pos_rate={gm.positive_rate:.4f}")
            lines += [
                "",
                "--- POST-ADJUSTMENT FAIRNESS METRICS ---",
            ]
            for k, v in self.post_fairness_metrics.to_dict().items():
                lines.append(f"  {k}: {v}")
            lines += [
                "",
                "--- ADJUSTED THRESHOLDS ---",
            ]
            for gv, thr in self.adjusted_thresholds.items():
                lines.append(f"  Group={gv}: threshold={thr:.4f}")
        else:
            lines.append("\nNo threshold adjustment applied (fairness threshold satisfied).")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _compute_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return (FPR, FNR) from true and predicted labels."""
    if len(y_true) == 0:
        return 0.0, 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return fpr, fnr


def _group_metrics_from_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_value: Any,
    threshold: float = 0.5,
) -> GroupMetrics:
    """Compute GroupMetrics for a single group given predicted probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    # f1 with zero_division guard
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr, fnr = _compute_rates(y_true, y_pred)
    pos_rate = y_pred.mean()
    return GroupMetrics(
        group_value=group_value,
        n_samples=len(y_true),
        accuracy=acc,
        f1=f1,
        fpr=fpr,
        fnr=fnr,
        positive_rate=pos_rate,
        threshold=threshold,
    )


def _compute_fairness_metrics(
    group_metrics: Dict[Any, GroupMetrics],
    fairness_threshold: float,
) -> FairnessMetrics:
    """Compute fairness metrics from a dict of GroupMetrics."""
    groups = list(group_metrics.keys())
    pos_rates = [group_metrics[g].positive_rate for g in groups]
    fprs = [group_metrics[g].fpr for g in groups]
    fnrs = [group_metrics[g].fnr for g in groups]

    dp_diff = max(pos_rates) - min(pos_rates)
    fpr_diff = max(fprs) - min(fprs)
    fnr_diff = max(fnrs) - min(fnrs)
    eq_odds_diff = max(fpr_diff, fnr_diff)

    # Disparate impact: ratio of min to max positive rate
    if max(pos_rates) > 0:
        di_ratio = min(pos_rates) / max(pos_rates)
    else:
        di_ratio = 1.0

    violates = (dp_diff > fairness_threshold) or (eq_odds_diff > fairness_threshold)

    return FairnessMetrics(
        demographic_parity_difference=dp_diff,
        equalized_odds_difference=eq_odds_diff,
        disparate_impact_ratio=di_ratio,
        fpr_difference=fpr_diff,
        fnr_difference=fnr_diff,
        violates_threshold=violates,
        threshold_used=fairness_threshold,
    )


def _find_threshold_for_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float,
    n_thresholds: int = 1000,
) -> float:
    """
    Binary-search / grid-search for the threshold that achieves a target FPR.
    Returns the threshold whose resulting FPR is closest to target_fpr.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_thresh = 0.5
    best_diff = float("inf")
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fpr, _ = _compute_rates(y_true, y_pred)
        diff = abs(fpr - target_fpr)
        if diff < best_diff:
            best_diff = diff
            best_thresh = t
    return best_thresh


# ---------------------------------------------------------------------------
# Stratified Split Helper
# ---------------------------------------------------------------------------

def _stratified_split_by_target_and_protected(
    X: pd.DataFrame,
    y: pd.Series,
    protected: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Stratify by a combined stratum of (target, protected_attribute) so that
    both class balance and group balance are preserved in train/test splits.
    """
    strata = y.astype(str) + "_" + protected.astype(str)

    # Some strata may have only 1 sample; fall back to simple split for those
    strata_counts = strata.value_counts()
    rare_strata = strata_counts[strata_counts < 2].index.tolist()

    if rare_strata:
        warnings.warn(
            f"Some strata have < 2 samples and will not be stratified: {rare_strata}",
            UserWarning,
        )
        # Use only the strata with >= 2 samples for stratification
        mask_rare = strata.isin(rare_strata)
        X_rare, y_rare, p_rare = X[mask_rare], y[mask_rare], protected[mask_rare]
        X_common, y_common, p_common = X[~mask_rare], y[~mask_rare], protected[~mask_rare]
        strata_common = strata[~mask_rare]

        X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(
            X_common, y_common, p_common,
            test_size=test_size,
            stratify=strata_common,
            random_state=random_state,
        )
        # Append rare samples to training set
        X_tr = pd.concat([X_tr, X_rare])
        y_tr = pd.concat([y_tr, y_rare])
        p_tr = pd.concat([p_tr, p_rare])
    else:
        X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(
            X, y, protected,
            test_size=test_size,
            stratify=strata,
            random_state=random_state,
        )

    return X_tr, X_te, y_tr, y_te, p_tr, p_te


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def build_fairness_aware_classifier(
    data: pd.DataFrame,
    feature_columns: list,
    target_column: str,
    protected_attribute: str,
    test_size: float = 0.2,
    random_state: int = 42,
    fairness_threshold: float = 0.1,
    logistic_regression_kwargs: Optional[Dict] = None,
) -> FairnessResult:
    """
    Build a fairness-aware binary classification system.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing features, target, and protected attribute.
    feature_columns : list
        Column names to use as features.
    target_column : str
        Name of the binary target column (values should be 0/1 or bool).
    protected_attribute : str
        Name of the protected attribute column (e.g., 'gender', 'race').
    test_size : float
        Fraction of data to use for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.
    fairness_threshold : float
        Maximum allowed demographic parity difference or equalized odds
        difference before post-hoc adjustment is triggered (default 0.1).
    logistic_regression_kwargs : dict, optional
        Additional keyword arguments passed to LogisticRegression.

    Returns
    -------
    FairnessResult
        Dataclass containing model, metrics, and adjusted thresholds.
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    required_cols = feature_columns + [target_column, protected_attribute]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    X = data[feature_columns].copy()
    y = data[target_column].astype(int).copy()
    protected = data[protected_attribute].copy()

    group_values = sorted(protected.unique().tolist())
    if len(group_values) < 2:
        raise ValueError("Protected attribute must have at least 2 distinct values.")

    # ------------------------------------------------------------------
    # 2. Stratified train/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, p_train, p_test = (
        _stratified_split_by_target_and_protected(
            X, y, protected,
            test_size=test_size,
            random_state=random_state,
        )
    )

    # ------------------------------------------------------------------
    # 3. Train logistic regression (with StandardScaler in a Pipeline)
    # ------------------------------------------------------------------
    lr_kwargs = {
        "max_iter": 1000,
        "random_state": random_state,
        "solver": "lbfgs",
    }
    if logistic_regression_kwargs:
        lr_kwargs.update(logistic_regression_kwargs)

    scaler = StandardScaler()
    clf = LogisticRegression(**lr_kwargs)
    model = Pipeline([("scaler", scaler), ("clf", clf)])
    model.fit(X_train, y_train)

    # Predicted probabilities for the positive class on the test set
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # 4. Pre-adjustment group-level metrics
    # ------------------------------------------------------------------
    pre_group_metrics: Dict[Any, GroupMetrics] = {}
    for gv in group_values:
        mask = (p_test == gv).values
        if mask.sum() == 0:
            warnings.warn(f"Group {gv} has no test samples; skipping.")
            continue
        gm = _group_metrics_from_probs(
            y_true=y_test.values[mask],
            y_prob=y_prob_test[mask],
            group_value=gv,
            threshold=0.5,
        )
        pre_group_metrics[gv] = gm

    # ------------------------------------------------------------------
    # 5. Pre-adjustment fairness metrics
    # ------------------------------------------------------------------
    pre_fairness_metrics = _compute_fairness_metrics(pre_group_metrics, fairness_threshold)

    # ------------------------------------------------------------------
    # 6. Post-hoc threshold adjustment (if fairness is violated)
    # ------------------------------------------------------------------
    post_group_metrics = None
    post_fairness_metrics = None
    adjusted_thresholds = None

    if pre_fairness_metrics.violates_threshold:
        # Strategy: equalize FPR across groups by adjusting per-group thresholds.
        # Target FPR = mean FPR across groups (a neutral reference point).
        target_fpr = np.mean([pre_group_metrics[gv].fpr for gv in pre_group_metrics])

        adjusted_thresholds = {}
        for gv in pre_group_metrics:
            mask = (p_test == gv).values
            thr = _find_threshold_for_fpr(
                y_true=y_test.values[mask],
                y_prob=y_prob_test[mask],
                target_fpr=target_fpr,
            )
            adjusted_thresholds[gv] = thr

        # Recompute group metrics with adjusted thresholds
        post_group_metrics = {}
        for gv in pre_group_metrics:
            mask = (p_test == gv).values
            gm = _group_metrics_from_probs(
                y_true=y_test.values[mask],
                y_prob=y_prob_test[mask],
                group_value=gv,
                threshold=adjusted_thresholds[gv],
            )
            post_group_metrics[gv] = gm

        post_fairness_metrics = _compute_fairness_metrics(
            post_group_metrics, fairness_threshold
        )

    # ------------------------------------------------------------------
    # 7. Assemble and return result
    # ------------------------------------------------------------------
    result = FairnessResult(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        target_column=target_column,
        protected_attribute=protected_attribute,
        group_values=group_values,
        pre_group_metrics=pre_group_metrics,
        pre_fairness_metrics=pre_fairness_metrics,
        post_group_metrics=post_group_metrics,
        post_fairness_metrics=post_fairness_metrics,
        adjusted_thresholds=adjusted_thresholds,
        X_test=X_test,
        y_test=y_test,
        protected_test=p_test,
    )
    return result


# ---------------------------------------------------------------------------
# Prediction Helper (uses adjusted thresholds if available)
# ---------------------------------------------------------------------------

def predict_with_fairness(
    result: FairnessResult,
    X_new: pd.DataFrame,
    protected_new: pd.Series,
    use_adjusted_thresholds: bool = True,
) -> np.ndarray:
    """
    Generate predictions for new data using the trained model.

    If `use_adjusted_thresholds=True` and adjusted thresholds exist,
    each sample is classified using the threshold for its group.

    Parameters
    ----------
    result : FairnessResult
        Output from `build_fairness_aware_classifier`.
    X_new : pd.DataFrame
        New feature data.
    protected_new : pd.Series
        Protected attribute values for new data.
    use_adjusted_thresholds : bool
        Whether to apply per-group thresholds (default True).

    Returns
    -------
    np.ndarray
        Binary predictions (0 or 1).
    """
    y_prob = result.model.predict_proba(X