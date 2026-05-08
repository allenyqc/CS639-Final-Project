```python
"""
Survival Analysis Pipeline for Patient Outcome Prediction

This module builds a comprehensive survival analysis pipeline using Cox Proportional
Hazards modeling with proper handling of right-censored observations.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# Optional imports with graceful fallbacks
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    warnings.warn(
        "lifelines not available. Install with: pip install lifelines. "
        "Falling back to manual implementation.",
        ImportWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SurvivalPipelineResult:
    """Container for all outputs of the survival analysis pipeline."""
    fitted_model: object
    hazard_ratios: pd.DataFrame
    survival_curves: Dict[str, pd.DataFrame]
    evaluation_metrics: Dict[str, float]
    risk_group_assignments: pd.Series
    log_rank_results: Dict[str, object]
    scaler: StandardScaler
    feature_names: List[str]
    train_indices: np.ndarray
    test_indices: np.ndarray
    val_indices: np.ndarray


# ---------------------------------------------------------------------------
# Manual Cox PH Implementation (fallback)
# ---------------------------------------------------------------------------

class ManualCoxPH:
    """
    Minimal Cox Proportional Hazards model using partial likelihood maximisation
    via gradient ascent (Newton-Raphson).

    Only used when lifelines is not installed.
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-6, step_size: float = 0.1):
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.coef_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: np.ndarray, durations: np.ndarray, events: np.ndarray) -> "ManualCoxPH":
        """Fit Cox model via partial likelihood."""
        n, p = X.shape
        beta = np.zeros(p)

        # Sort by duration descending for risk-set computation
        order = np.argsort(-durations)
        X_s = X[order]
        T_s = durations[order]
        E_s = events[order]  # 1 = event observed, 0 = censored

        for iteration in range(self.max_iter):
            grad = np.zeros(p)
            hess = np.zeros((p, p))

            for i in range(n):
                if E_s[i] == 0:
                    continue  # skip censored
                # Risk set: all j with T_j >= T_i
                risk_mask = T_s >= T_s[i]
                X_risk = X_s[risk_mask]
                exp_xb = np.exp(X_risk @ beta)
                sum_exp = exp_xb.sum()
                weighted_x = (exp_xb[:, None] * X_risk).sum(axis=0) / sum_exp
                weighted_xx = (
                    exp_xb[:, None, None] * X_risk[:, :, None] * X_risk[:, None, :]
                ).sum(axis=0) / sum_exp

                grad += X_s[i] - weighted_x
                hess -= weighted_xx - np.outer(weighted_x, weighted_x)

            # Newton-Raphson update with regularisation
            try:
                delta = np.linalg.solve(hess + 1e-8 * np.eye(p), grad)
            except np.linalg.LinAlgError:
                delta = grad * self.step_size

            beta_new = beta + self.step_size * delta
            if np.linalg.norm(beta_new - beta) < self.tol:
                break
            beta = beta_new

        self.coef_ = beta
        return self

    def predict_log_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not fitted yet.")
        return X @ self.coef_

    def predict_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_partial_hazard(X))


# ---------------------------------------------------------------------------
# Kaplan-Meier (manual fallback)
# ---------------------------------------------------------------------------

def _kaplan_meier(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Compute Kaplan-Meier survival estimate."""
    df = pd.DataFrame({"duration": durations, "event": events})
    df = df.sort_values("duration").reset_index(drop=True)

    times = [0]
    survival = [1.0]
    n_at_risk = len(df)
    S = 1.0

    for t, group in df.groupby("duration"):
        n_events = group["event"].sum()
        if n_events > 0:
            S *= 1 - n_events / n_at_risk
        n_at_risk -= len(group)
        times.append(t)
        survival.append(S)

    return pd.DataFrame({"timeline": times, "KM_estimate": survival})


# ---------------------------------------------------------------------------
# Log-rank test (manual fallback)
# ---------------------------------------------------------------------------

def _logrank_test_manual(
    durations_a: np.ndarray,
    events_a: np.ndarray,
    durations_b: np.ndarray,
    events_b: np.ndarray,
) -> Dict[str, float]:
    """Two-sample log-rank test."""
    all_times = np.unique(
        np.concatenate([durations_a[events_a == 1], durations_b[events_b == 1]])
    )

    O1, E1 = 0.0, 0.0
    V = 0.0

    for t in all_times:
        n1 = (durations_a >= t).sum()
        n2 = (durations_b >= t).sum()
        d1 = ((durations_a == t) & (events_a == 1)).sum()
        d2 = ((durations_b == t) & (events_b == 1)).sum()
        n = n1 + n2
        d = d1 + d2

        if n < 2:
            continue

        e1 = n1 * d / n
        O1 += d1
        E1 += e1
        V += n1 * n2 * d * (n - d) / (n**2 * (n - 1)) if n > 1 and d < n else 0.0

    if V == 0:
        return {"test_statistic": 0.0, "p_value": 1.0}

    chi2 = (O1 - E1) ** 2 / V
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return {"test_statistic": float(chi2), "p_value": float(p_value)}


# ---------------------------------------------------------------------------
# Concordance Index (manual fallback)
# ---------------------------------------------------------------------------

def _concordance_index_manual(
    durations: np.ndarray, events: np.ndarray, risk_scores: np.ndarray
) -> float:
    """Harrell's concordance index."""
    concordant = 0
    discordant = 0
    tied = 0

    n = len(durations)
    for i in range(n):
        for j in range(i + 1, n):
            if events[i] == 0 and events[j] == 0:
                continue
            if durations[i] == durations[j]:
                continue

            # Determine which has shorter duration and observed event
            if events[i] == 1 and durations[i] < durations[j]:
                comparable = True
                i_first = True
            elif events[j] == 1 and durations[j] < durations[i]:
                comparable = True
                i_first = False
            else:
                comparable = False
                i_first = False

            if not comparable:
                continue

            if i_first:
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
                else:
                    tied += 1
            else:
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                elif risk_scores[j] < risk_scores[i]:
                    discordant += 1
                else:
                    tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"event_time", "censored"}
FEATURE_COLUMNS = ["age", "treatment_type", "biomarker_levels", "comorbidity_count"]


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the DataFrame has required columns and sensible values."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if (df["event_time"] <= 0).any():
        raise ValueError("'event_time' must be strictly positive.")

    if not df["censored"].isin([0, 1]).all():
        raise ValueError("'censored' must be binary (0 or 1).")

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not available_features:
        raise ValueError(
            f"No feature columns found. Expected at least one of: {FEATURE_COLUMNS}"
        )


def _encode_treatment_type(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One-hot encode treatment_type, fitting only on training data."""
    if "treatment_type" not in train_df.columns:
        return train_df, val_df, test_df

    categories = sorted(train_df["treatment_type"].dropna().unique())
    for split_df in [train_df, val_df, test_df]:
        unseen = set(split_df["treatment_type"].dropna().unique()) - set(categories)
        if unseen:
            warnings.warn(
                f"Unseen treatment categories in split: {unseen}. "
                "They will be encoded as all-zeros.",
                UserWarning,
                stacklevel=3,
            )

    def _ohe(df: pd.DataFrame) -> pd.DataFrame:
        dummies = pd.get_dummies(
            df["treatment_type"].astype(
                pd.CategoricalDtype(categories=categories, ordered=False)
            ),
            prefix="treatment",
            drop_first=True,
        )
        return pd.concat([df.drop(columns=["treatment_type"]), dummies], axis=1)

    return _ohe(train_df), _ohe(val_df), _ohe(test_df)


def _preprocess(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Scale numeric features using statistics from training data only.

    Returns scaled arrays for train/val/test, the fitted scaler, and
    the final list of feature column names.
    """
    train_df, val_df, test_df = _encode_treatment_type(train_df, val_df, test_df)

    # Recompute feature columns after encoding
    non_feature = {"event_time", "censored"}
    all_cols = [c for c in train_df.columns if c not in non_feature]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[all_cols].values.astype(float))
    X_val = scaler.transform(val_df[all_cols].values.astype(float))
    X_test = scaler.transform(test_df[all_cols].values.astype(float))

    return X_train, X_val, X_test, scaler, all_cols


# ---------------------------------------------------------------------------
# Risk Group Assignment
# ---------------------------------------------------------------------------

def _assign_risk_groups(
    risk_scores: np.ndarray,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
    thresholds: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Assign low/medium/high risk groups.

    If thresholds are provided (fitted on validation set), they are used directly.
    Otherwise, quantile-based thresholds are computed from the provided scores.
    """
    if thresholds is None:
        q_low = float(np.quantile(risk_scores, low_quantile))
        q_high = float(np.quantile(risk_scores, high_quantile))
        thresholds = (q_low, q_high)

    q_low, q_high = thresholds
    groups = np.where(
        risk_scores <= q_low,
        "low",
        np.where(risk_scores <= q_high, "medium", "high"),
    )
    return groups, thresholds


# ---------------------------------------------------------------------------
# Survival Curves
# ---------------------------------------------------------------------------

def _compute_survival_curves(
    df: pd.DataFrame,
    risk_groups: np.ndarray,
    group_labels: List[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute Kaplan-Meier survival curves per risk group."""
    if group_labels is None:
        group_labels = ["low", "medium", "high"]

    curves: Dict[str, pd.DataFrame] = {}

    for label in group_labels:
        mask = risk_groups == label
        if mask.sum() == 0:
            continue

        sub = df[mask]
        durations = sub["event_time"].values
        # events: 1 if event observed (censored == 0)
        events = (1 - sub["censored"]).values

        if LIFELINES_AVAILABLE:
            kmf = KaplanMeierFitter()
            kmf.fit(durations, event_observed=events, label=label)
            curve_df = pd.DataFrame(
                {
                    "timeline": kmf.timeline,
                    "KM_estimate": kmf.survival_function_[label].values,
                }
            )
        else:
            curve_df = _kaplan_meier(durations, events)

        curves[label] = curve_df

    return curves


# ---------------------------------------------------------------------------
# Log-rank Tests
# ---------------------------------------------------------------------------

def _run_logrank_tests(
    df: pd.DataFrame,
    risk_groups: np.ndarray,
) -> Dict[str, object]:
    """
    Run pairwise log-rank tests between risk groups.
    Apply Bonferroni correction for multiple comparisons.
    """
    group_labels = ["low", "medium", "high"]
    pairs = [
        ("low", "medium"),
        ("low", "high"),
        ("medium", "high"),
    ]

    raw_results: Dict[str, Dict] = {}

    for g1, g2 in pairs:
        mask1 = risk_groups == g1
        mask2 = risk_groups == g2

        if mask1.sum() == 0 or mask2.sum() == 0:
            continue

        d1 = df[mask1]["event_time"].values
        e1 = (1 - df[mask1]["censored"]).values
        d2 = df[mask2]["event_time"].values
        e2 = (1 - df[mask2]["censored"]).values

        key = f"{g1}_vs_{g2}"

        if LIFELINES_AVAILABLE:
            result = logrank_test(d1, d2, event_observed_A=e1, event_observed_B=e2)
            raw_results[key] = {
                "test_statistic": float(result.test_statistic),
                "p_value": float(result.p_value),
                "result_object": result,
            }
        else:
            manual = _logrank_test_manual(d1, e1, d2, e2)
            raw_results[key] = {
                "test_statistic": manual["test_statistic"],
                "p_value": manual["p_value"],
                "result_object": None,
            }

    # Bonferroni correction
    n_tests = len(raw_results)
    corrected_results: Dict[str, object] = {}
    for key, res in raw_results.items():
        corrected_p = min(res["p_value"] * n_tests, 1.0)
        corrected_results[key] = {
            "test_statistic": res["test_statistic"],
            "p_value_raw": res["p_value"],
            "p_value_bonferroni": corrected_p,
            "significant_bonferroni": corrected_p < 0.05,
            "result_object": res.get("result_object"),
        }

    # Multivariate log-rank (overall) if lifelines available
    if LIFELINES_AVAILABLE and len(np.unique(risk_groups)) > 1:
        try:
            overall = multivariate_logrank_test(
                df["event_time"].values,
                risk_groups,
                event_observed=(1 - df["censored"]).values,
            )
            corrected_results["overall"] = {
                "test_statistic": float(overall.test_statistic),
                "p_value_raw": float(overall.p_value),
                "p_value_bonferroni": float(overall.p_value),  # single test
                "significant_bonferroni": float(overall.p_value) < 0.05,
                "result_object": overall,
            }
        except Exception as exc:
            warnings.warn(f"Multivariate log-rank test failed: {exc}", RuntimeWarning, stacklevel=2)

    return corrected_results


# ---------------------------------------------------------------------------
# Model Fitting
# ---------------------------------------------------------------------------

def _fit_cox_model(
    train_df: pd.DataFrame,
    X_train: np.ndarray,
    feature_names: List[str],
) -> object:
    """Fit Cox PH model on training data."""
    if LIFELINES_AVAILABLE:
        cox_df = pd.DataFrame(X_train, columns=feature_names)
        cox_df["event_time"] = train_df["event_time"].values
        cox_df["event_observed"] = (1 - train_df["censored"]).values

        model = CoxPHFitter(penalizer=0.1)
        model.fit(
            cox_df,
            duration_col="event_time",
            event_col="event_observed",
        )
    else:
        durations = train_df["event_time"].values
        events = (1 - train_df["censored"]).values
        model = ManualCoxPH(max_iter=200, tol=1e-6)
        model.fit(X_train, durations, events)
        model.feature_names_ = feature_names

    return model


def _extract_hazard_ratios(model: object, feature_names: List[str]) -> pd.DataFrame:
    """Extract hazard ratios and confidence intervals from the fitted model."""
    if LIFELINES_AVAILABLE and isinstance(model, CoxPHFitter):
        summary = model.summary.copy()
        hr_df = pd.DataFrame(
            {
                "feature": summary.index,
                "coef": summary["coef"].values,
                "hazard_ratio": summary["exp(coef)"].values,