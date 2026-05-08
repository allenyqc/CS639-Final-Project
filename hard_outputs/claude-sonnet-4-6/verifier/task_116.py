```python
"""
Survival Analysis Pipeline for Predicting Patient Outcomes
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import lifelines; if not available, use manual implementation
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("lifelines not available. Using manual implementation.")

from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SurvivalResults:
    """Container for all survival analysis outputs."""
    fitted_model: Any
    hazard_ratios: pd.DataFrame
    survival_curves: Dict[str, pd.DataFrame]
    evaluation_metrics: Dict[str, float]
    log_rank_results: Dict[str, Any]
    risk_scores: np.ndarray
    risk_groups: np.ndarray
    feature_names: list
    scaler: Any


# ─────────────────────────────────────────────────────────────────────────────
# Manual Cox PH Implementation (fallback)
# ─────────────────────────────────────────────────────────────────────────────

class ManualCoxPH:
    """
    Manual Cox Proportional Hazards model using partial likelihood maximisation.
    Handles right-censored data via the Breslow approximation for ties.
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-6, l2_reg: float = 0.01):
        self.max_iter = max_iter
        self.tol = tol
        self.l2_reg = l2_reg
        self.coef_: Optional[np.ndarray] = None
        self.baseline_hazard_: Optional[pd.DataFrame] = None
        self.feature_names_: Optional[list] = None
        self.se_: Optional[np.ndarray] = None

    # ── Partial log-likelihood (Breslow) ──────────────────────────────────────
    def _neg_partial_log_likelihood(self, beta: np.ndarray,
                                    X: np.ndarray,
                                    time: np.ndarray,
                                    event: np.ndarray) -> float:
        """Negative partial log-likelihood with L2 regularisation."""
        xb = X @ beta
        log_lik = 0.0
        unique_times = np.unique(time[event == 1])

        for t in unique_times:
            at_risk = time >= t
            events_at_t = (time == t) & (event == 1)
            d = events_at_t.sum()
            if d == 0:
                continue
            log_lik += xb[events_at_t].sum() - d * np.log(np.exp(xb[at_risk]).sum())

        penalty = 0.5 * self.l2_reg * np.dot(beta, beta)
        return -(log_lik - penalty)

    def _gradient(self, beta: np.ndarray,
                  X: np.ndarray,
                  time: np.ndarray,
                  event: np.ndarray) -> np.ndarray:
        """Gradient of the negative partial log-likelihood."""
        xb = X @ beta
        exp_xb = np.exp(xb)
        grad = np.zeros_like(beta)
        unique_times = np.unique(time[event == 1])

        for t in unique_times:
            at_risk = time >= t
            events_at_t = (time == t) & (event == 1)
            d = events_at_t.sum()
            if d == 0:
                continue
            denom = exp_xb[at_risk].sum()
            weighted_X = (exp_xb[at_risk, np.newaxis] * X[at_risk]).sum(axis=0)
            grad += X[events_at_t].sum(axis=0) - d * weighted_X / denom

        grad -= self.l2_reg * beta
        return -grad

    # ── Hessian for standard errors ───────────────────────────────────────────
    def _hessian(self, beta: np.ndarray,
                 X: np.ndarray,
                 time: np.ndarray,
                 event: np.ndarray) -> np.ndarray:
        xb = X @ beta
        exp_xb = np.exp(xb)
        p = X.shape[1]
        H = np.zeros((p, p))
        unique_times = np.unique(time[event == 1])

        for t in unique_times:
            at_risk = time >= t
            events_at_t = (time == t) & (event == 1)
            d = events_at_t.sum()
            if d == 0:
                continue
            denom = exp_xb[at_risk].sum()
            w = exp_xb[at_risk] / denom
            Xr = X[at_risk]
            mean_X = (w[:, np.newaxis] * Xr).sum(axis=0)
            cov_X = (w[:, np.newaxis] * (Xr - mean_X)).T @ (Xr - mean_X)
            H += d * cov_X

        H += self.l2_reg * np.eye(p)
        return H

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray,
            feature_names: Optional[list] = None):
        self.feature_names_ = feature_names or [f"x{i}" for i in range(X.shape[1])]
        beta0 = np.zeros(X.shape[1])

        result = minimize(
            fun=self._neg_partial_log_likelihood,
            x0=beta0,
            jac=self._gradient,
            args=(X, time, event),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        self.coef_ = result.x

        # Standard errors via observed Fisher information
        H = self._hessian(self.coef_, X, time, event)
        try:
            cov = np.linalg.inv(H)
            self.se_ = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            self.se_ = np.full(len(self.coef_), np.nan)

        self._compute_baseline_hazard(X, time, event)
        return self

    # ── Baseline hazard (Nelson-Aalen) ────────────────────────────────────────
    def _compute_baseline_hazard(self, X: np.ndarray,
                                  time: np.ndarray,
                                  event: np.ndarray):
        xb = X @ self.coef_
        exp_xb = np.exp(xb)
        unique_times = np.sort(np.unique(time[event == 1]))
        h0 = []
        for t in unique_times:
            at_risk = time >= t
            d = ((time == t) & (event == 1)).sum()
            denom = exp_xb[at_risk].sum()
            h0.append(d / denom if denom > 0 else 0.0)

        self.baseline_hazard_ = pd.DataFrame({
            'time': unique_times,
            'baseline_hazard': h0,
            'baseline_cumhazard': np.cumsum(h0)
        })

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict_log_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def predict_survival_function(self, X: np.ndarray,
                                   times: Optional[np.ndarray] = None) -> pd.DataFrame:
        if self.baseline_hazard_ is None:
            raise RuntimeError("Model not fitted yet.")
        if times is None:
            times = self.baseline_hazard_['time'].values

        log_ph = self.predict_log_partial_hazard(X)
        results = {}
        for i, lph in enumerate(log_ph):
            cumhaz = np.interp(
                times,
                self.baseline_hazard_['time'].values,
                self.baseline_hazard_['baseline_cumhazard'].values,
                left=0.0
            )
            results[i] = np.exp(-cumhaz * np.exp(lph))

        return pd.DataFrame(results, index=times)

    # ── Summary ───────────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        z = self.coef_ / self.se_
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))
        hr = np.exp(self.coef_)
        hr_lo = np.exp(self.coef_ - 1.96 * self.se_)
        hr_hi = np.exp(self.coef_ + 1.96 * self.se_)
        return pd.DataFrame({
            'coef': self.coef_,
            'exp(coef)': hr,
            'se(coef)': self.se_,
            'z': z,
            'p': p,
            'exp(coef) lower 95%': hr_lo,
            'exp(coef) upper 95%': hr_hi
        }, index=self.feature_names_)


# ─────────────────────────────────────────────────────────────────────────────
# Kaplan-Meier (manual fallback)
# ─────────────────────────────────────────────────────────────────────────────

class ManualKaplanMeier:
    """Simple Kaplan-Meier estimator."""

    def __init__(self):
        self.timeline_: Optional[np.ndarray] = None
        self.survival_: Optional[np.ndarray] = None
        self.n_at_risk_: Optional[np.ndarray] = None

    def fit(self, time: np.ndarray, event: np.ndarray):
        order = np.argsort(time)
        time = time[order]
        event = event[order]
        n = len(time)
        unique_times = np.unique(time)
        S = 1.0
        timeline = [0]
        survival = [1.0]
        n_at_risk = []

        for t in unique_times:
            mask = time >= t
            n_risk = mask.sum()
            d = ((time == t) & (event == 1)).sum()
            if d > 0:
                S *= (1 - d / n_risk)
            timeline.append(t)
            survival.append(S)
            n_at_risk.append(n_risk)

        self.timeline_ = np.array(timeline)
        self.survival_ = np.array(survival)
        return self

    def predict(self, times: np.ndarray) -> np.ndarray:
        return np.interp(times, self.timeline_, self.survival_,
                         left=1.0, right=self.survival_[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Log-rank test (manual)
# ─────────────────────────────────────────────────────────────────────────────

def manual_logrank_test(time1: np.ndarray, event1: np.ndarray,
                         time2: np.ndarray, event2: np.ndarray) -> Dict[str, float]:
    """Two-sample log-rank test."""
    all_times = np.sort(np.unique(np.concatenate([
        time1[event1 == 1], time2[event2 == 1]
    ])))

    O1_total = E1_total = O2_total = E2_total = 0.0
    V = 0.0

    for t in all_times:
        n1 = (time1 >= t).sum()
        n2 = (time2 >= t).sum()
        d1 = ((time1 == t) & (event1 == 1)).sum()
        d2 = ((time2 == t) & (event2 == 1)).sum()
        n = n1 + n2
        d = d1 + d2
        if n < 2 or d == 0:
            continue
        e1 = n1 * d / n
        e2 = n2 * d / n
        O1_total += d1
        E1_total += e1
        O2_total += d2
        E2_total += e2
        V += (n1 * n2 * d * (n - d)) / (n ** 2 * (n - 1)) if n > 1 else 0.0

    if V == 0:
        return {'test_statistic': 0.0, 'p_value': 1.0}

    chi2 = (O1_total - E1_total) ** 2 / V
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return {'test_statistic': chi2, 'p_value': p_value}


def manual_multivariate_logrank(times: list, events: list) -> Dict[str, float]:
    """Multi-group log-rank test (generalised)."""
    k = len(times)
    all_event_times = np.sort(np.unique(np.concatenate([
        t[e == 1] for t, e in zip(times, events)
    ])))

    O = np.zeros(k)
    E = np.zeros(k)

    for t in all_event_times:
        n = np.array([(ti >= t).sum() for ti in times], dtype=float)
        d = np.array([((ti == t) & (ei == 1)).sum()
                      for ti, ei in zip(times, events)], dtype=float)
        N = n.sum()
        D = d.sum()
        if N == 0 or D == 0:
            continue
        O += d
        E += n * D / N

    # Chi-squared statistic (k-1 df)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.sum((O - E) ** 2 / np.where(E > 0, E, 1))
    p_value = 1 - stats.chi2.cdf(chi2, df=k - 1)
    return {'test_statistic': chi2, 'p_value': p_value, 'df': k - 1}


# ─────────────────────────────────────────────────────────────────────────────
# Concordance index (manual)
# ─────────────────────────────────────────────────────────────────────────────

def manual_concordance_index(time: np.ndarray, event: np.ndarray,
                              risk_score: np.ndarray) -> float:
    """Harrell's C-index."""
    concordant = discordant = tied = 0
    n = len(time)
    for i in range(n):
        if event[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[j] > time[i]:
                if risk_score[i] > risk_score[j]:
                    concordant += 1
                elif risk_score[i] < risk_score[j]:
                    discordant += 1
                else:
                    tied += 1
    total = concordant + discordant + tied
    return (concordant + 0.5 * tied) / total if total > 0 else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Data Validation & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = {'event_time', 'censored'}
FEATURE_COLUMNS = ['age', 'treatment_type', 'biomarker_levels', 'comorbidity_count']


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the input DataFrame."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Ensure numeric types
    df['event_time'] = pd.to_numeric(df['event_time'], errors='coerce')
    df['censored'] = pd.to_numeric(df['censored'], errors='coerce')

    # Drop rows with NaN in critical columns
    before = len(df)
    df.dropna(subset=['event_time', 'censored'], inplace=True)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with NaN in event_time/censored.")

    # Validate event_time > 0
    invalid_time = df['event_time'] <= 0
    if invalid_time.any():
        print(f"Dropping {invalid_time.sum()} rows with event_time <= 0.")
        df = df[~invalid_time]

    # Validate censored is binary
    if not set(df['censored'].unique()).issubset({0, 1}):
        raise ValueError("'censored' column must contain only 0 or 1.")

    # Derive 'event' column: 1 if event observed (not censored), 0 if censored
    df['event'] = 1 - df['censored']

    # Handle missing feature columns
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not available_features:
        raise ValueError(f"No feature columns found. Expected: {FEATURE_COLUMNS}")

    # Fill missing feature values with median
    for col in available_features:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled NaN in '{col}' with median ({median_val:.4f}).")

    return df, available_features


# ─────────────────────────────────────────────────────────────────────────────
# Risk Stratification
# ─────────────────────────────────────────────────────────────────────────────

def assign_risk_groups(risk_scores: np.ndarray,
                        low_q: float = 0.33,
                        high_q: float = 0.67) -> np.ndarray:
    """Assign patients to low / medium / high risk groups."""
    lo = np.quantile(risk_scores, low_q)
    hi = np.quantile(risk_scores, high_q)
    groups = np.where(risk_scores <= lo, 'low',
              np.where(risk_scores <= hi, 'medium', 'high'))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Survival Curve Estimation per Risk Group
# ─────────────────────────────────────────────────────────────────────────────

def estimate_group_survival_curves(df: pd.DataFrame,
                                    risk_groups: np.ndarray,
                                    time_col: str = 'event_time',
                                    event_col: str = 'event') -> Dict[str, pd.DataFrame]:
    """Estimate KM survival curves for each risk group."""
    curves = {}
    group_labels = ['low', 'medium', 'high']

    for grp in group_labels:
        mask = risk_groups == grp
        if mask.sum() < 2: