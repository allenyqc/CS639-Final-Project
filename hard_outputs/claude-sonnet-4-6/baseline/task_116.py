```python
"""
Survival Analysis Pipeline for Predicting Patient Outcomes
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import lifelines, provide fallback
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SurvivalResults:
    """Container for all survival analysis results."""
    fitted_model: Any
    hazard_ratios: pd.DataFrame
    survival_curves: Dict[str, pd.DataFrame]
    evaluation_metrics: Dict[str, float]
    risk_scores: np.ndarray
    risk_groups: np.ndarray
    logrank_results: Dict[str, Any]
    feature_names: list
    scaler: Any


# ─────────────────────────────────────────────────────────────────────────────
# Manual Cox PH Implementation (fallback)
# ─────────────────────────────────────────────────────────────────────────────

class ManualCoxPH:
    """
    Manual implementation of Cox Proportional Hazards model
    using partial likelihood maximization via Newton-Raphson.
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-6, l2_reg: float = 0.01):
        self.max_iter = max_iter
        self.tol = tol
        self.l2_reg = l2_reg
        self.coef_ = None
        self.baseline_hazard_ = None
        self.feature_names_ = None
        self.converged_ = False

    def _partial_log_likelihood(self, beta: np.ndarray,
                                 X: np.ndarray,
                                 time: np.ndarray,
                                 event: np.ndarray) -> float:
        """Compute negative partial log-likelihood with L2 regularization."""
        linear_pred = X @ beta
        n = len(time)
        log_lik = 0.0

        # Sort by time descending for risk set computation
        order = np.argsort(-time)
        time_sorted = time[order]
        event_sorted = event[order]
        lp_sorted = linear_pred[order]

        for i in range(n):
            if event_sorted[i] == 1:  # observed event
                # Risk set: all j with time[j] >= time[i]
                risk_set = lp_sorted[i:]
                log_lik += lp_sorted[i] - np.log(np.sum(np.exp(risk_set - np.max(risk_set))) + 1e-10) - np.max(risk_set)

        # L2 regularization
        log_lik -= 0.5 * self.l2_reg * np.sum(beta ** 2)
        return -log_lik  # negative for minimization

    def _gradient(self, beta: np.ndarray,
                  X: np.ndarray,
                  time: np.ndarray,
                  event: np.ndarray) -> np.ndarray:
        """Compute gradient of negative partial log-likelihood."""
        linear_pred = X @ beta
        n = len(time)
        grad = np.zeros_like(beta)

        order = np.argsort(-time)
        time_sorted = time[order]
        event_sorted = event[order]
        lp_sorted = linear_pred[order]
        X_sorted = X[order]

        for i in range(n):
            if event_sorted[i] == 1:
                risk_lp = lp_sorted[i:]
                risk_X = X_sorted[i:]
                exp_lp = np.exp(risk_lp - np.max(risk_lp))
                weights = exp_lp / (np.sum(exp_lp) + 1e-10)
                expected_X = weights @ risk_X
                grad += X_sorted[i] - expected_X

        grad -= self.l2_reg * beta
        return -grad  # negative for minimization

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray,
            feature_names: Optional[list] = None):
        """Fit Cox PH model."""
        self.feature_names_ = feature_names or [f"x{i}" for i in range(X.shape[1])]
        beta_init = np.zeros(X.shape[1])

        result = minimize(
            fun=self._partial_log_likelihood,
            x0=beta_init,
            jac=self._gradient,
            args=(X, time, event),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )

        self.coef_ = result.x
        self.converged_ = result.success
        self._compute_baseline_hazard(X, time, event)
        self._compute_variance(X, time, event)
        return self

    def _compute_baseline_hazard(self, X: np.ndarray,
                                  time: np.ndarray,
                                  event: np.ndarray):
        """Estimate baseline cumulative hazard using Breslow estimator."""
        linear_pred = X @ self.coef_
        order = np.argsort(time)
        time_sorted = time[order]
        event_sorted = event[order]
        lp_sorted = linear_pred[order]

        unique_times = np.unique(time_sorted[event_sorted == 1])
        baseline_hazard = {}

        for t in unique_times:
            at_risk = lp_sorted[time_sorted >= t]
            denom = np.sum(np.exp(at_risk))
            num_events = np.sum((time_sorted == t) & (event_sorted == 1))
            baseline_hazard[t] = num_events / (denom + 1e-10)

        self.baseline_hazard_ = baseline_hazard
        # Cumulative baseline hazard
        times = sorted(baseline_hazard.keys())
        cum_hazard = np.cumsum([baseline_hazard[t] for t in times])
        self.baseline_cumhazard_ = dict(zip(times, cum_hazard))

    def _compute_variance(self, X: np.ndarray,
                           time: np.ndarray,
                           event: np.ndarray):
        """Compute variance of coefficients via observed information matrix."""
        linear_pred = X @ self.coef_
        n, p = X.shape
        info_matrix = np.zeros((p, p))

        order = np.argsort(-time)
        time_sorted = time[order]
        event_sorted = event[order]
        lp_sorted = linear_pred[order]
        X_sorted = X[order]

        for i in range(n):
            if event_sorted[i] == 1:
                risk_lp = lp_sorted[i:]
                risk_X = X_sorted[i:]
                exp_lp = np.exp(risk_lp - np.max(risk_lp))
                weights = exp_lp / (np.sum(exp_lp) + 1e-10)
                expected_X = weights @ risk_X
                expected_XX = (risk_X.T * weights) @ risk_X
                info_matrix += expected_XX - np.outer(expected_X, expected_X)

        info_matrix += self.l2_reg * np.eye(p)
        try:
            self.coef_var_ = np.diag(np.linalg.inv(info_matrix))
        except np.linalg.LinAlgError:
            self.coef_var_ = np.ones(p) * 0.01

    def predict_log_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        """Predict log partial hazard (risk score)."""
        return X @ self.coef_

    def predict_survival_function(self, X: np.ndarray,
                                   times: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Predict survival function for given covariates."""
        if times is None:
            times = sorted(self.baseline_cumhazard_.keys())

        lp = X @ self.coef_
        survival_df = pd.DataFrame(index=times)

        for i, lp_i in enumerate(lp):
            surv = []
            for t in times:
                # Find cumulative baseline hazard at time t
                cum_h = 0.0
                for bt, bh in self.baseline_cumhazard_.items():
                    if bt <= t:
                        cum_h = bh
                surv.append(np.exp(-cum_h * np.exp(lp_i)))
            survival_df[f"patient_{i}"] = surv

        return survival_df

    def get_hazard_ratios(self) -> pd.DataFrame:
        """Return hazard ratios with confidence intervals."""
        se = np.sqrt(self.coef_var_)
        z = 1.96  # 95% CI
        hr = np.exp(self.coef_)
        hr_lower = np.exp(self.coef_ - z * se)
        hr_upper = np.exp(self.coef_ + z * se)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(self.coef_ / (se + 1e-10))))

        return pd.DataFrame({
            'feature': self.feature_names_,
            'coef': self.coef_,
            'hazard_ratio': hr,
            'hr_lower_95': hr_lower,
            'hr_upper_95': hr_upper,
            'se': se,
            'z_score': self.coef_ / (se + 1e-10),
            'p_value': p_values
        })


# ─────────────────────────────────────────────────────────────────────────────
# Log-Rank Test (manual implementation)
# ─────────────────────────────────────────────────────────────────────────────

def manual_logrank_test(time1: np.ndarray, event1: np.ndarray,
                         time2: np.ndarray, event2: np.ndarray) -> Dict[str, float]:
    """
    Manual log-rank test between two groups.
    Returns test statistic and p-value.
    """
    all_times = np.unique(np.concatenate([time1[event1 == 1], time2[event2 == 1]]))

    O1_total, E1_total = 0.0, 0.0
    V_total = 0.0

    for t in all_times:
        n1 = np.sum(time1 >= t)
        n2 = np.sum(time2 >= t)
        d1 = np.sum((time1 == t) & (event1 == 1))
        d2 = np.sum((time2 == t) & (event2 == 1))
        n = n1 + n2
        d = d1 + d2

        if n == 0:
            continue

        e1 = n1 * d / n
        E1_total += e1
        O1_total += d1

        if n > 1:
            V_total += (n1 * n2 * d * (n - d)) / (n ** 2 * (n - 1))

    if V_total == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'test': 'log-rank'}

    chi2 = (O1_total - E1_total) ** 2 / V_total
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        'statistic': chi2,
        'p_value': p_value,
        'test': 'log-rank',
        'observed_group1': O1_total,
        'expected_group1': E1_total
    }


def pairwise_logrank_tests(time: np.ndarray, event: np.ndarray,
                            groups: np.ndarray) -> Dict[str, Dict]:
    """Perform pairwise log-rank tests between all groups."""
    unique_groups = np.unique(groups)
    results = {}

    for i in range(len(unique_groups)):
        for j in range(i + 1, len(unique_groups)):
            g1, g2 = unique_groups[i], unique_groups[j]
            mask1 = groups == g1
            mask2 = groups == g2

            if LIFELINES_AVAILABLE:
                lr = logrank_test(
                    time[mask1], time[mask2],
                    event_observed_A=event[mask1],
                    event_observed_B=event[mask2]
                )
                result = {
                    'statistic': lr.test_statistic,
                    'p_value': lr.p_value,
                    'test': 'log-rank'
                }
            else:
                result = manual_logrank_test(
                    time[mask1], event[mask1],
                    time[mask2], event[mask2]
                )

            results[f"{g1}_vs_{g2}"] = result

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Kaplan-Meier Estimator (manual)
# ─────────────────────────────────────────────────────────────────────────────

def kaplan_meier_estimate(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """
    Manual Kaplan-Meier survival estimator.
    Returns DataFrame with time, survival probability, and confidence intervals.
    """
    order = np.argsort(time)
    time_sorted = time[order]
    event_sorted = event[order]

    unique_times = [0]
    survival = [1.0]
    n_at_risk_list = [len(time)]
    n_events_list = [0]
    variance = [0.0]  # Greenwood's formula variance

    n = len(time)
    S = 1.0
    cum_var = 0.0

    for t in np.unique(time_sorted):
        n_at_risk = np.sum(time_sorted >= t)
        n_events = np.sum((time_sorted == t) & (event_sorted == 1))

        if n_events > 0:
            S *= (1 - n_events / n_at_risk)
            cum_var += n_events / (n_at_risk * (n_at_risk - n_events + 1e-10))

        unique_times.append(t)
        survival.append(S)
        n_at_risk_list.append(n_at_risk)
        n_events_list.append(n_events)
        variance.append(cum_var)

    survival_arr = np.array(survival)
    variance_arr = np.array(variance)

    # Greenwood's formula for CI
    se = survival_arr * np.sqrt(variance_arr)
    ci_lower = np.maximum(0, survival_arr - 1.96 * se)
    ci_upper = np.minimum(1, survival_arr + 1.96 * se)

    return pd.DataFrame({
        'time': unique_times,
        'survival': survival,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_at_risk': n_at_risk_list,
        'n_events': n_events_list
    })


# ─────────────────────────────────────────────────────────────────────────────
# Concordance Index (manual)
# ─────────────────────────────────────────────────────────────────────────────

def compute_concordance_index(time: np.ndarray, event: np.ndarray,
                               risk_scores: np.ndarray) -> float:
    """
    Compute Harrell's C-index (concordance index).
    Higher risk score should correspond to shorter survival time.
    """
    concordant = 0
    discordant = 0
    tied = 0
    n = len(time)

    for i in range(n):
        for j in range(i + 1, n):
            if event[i] == 0 and event[j] == 0:
                continue  # Both censored, skip

            if event[i] == 1 and event[j] == 1:
                # Both events observed
                if time[i] != time[j]:
                    earlier = i if time[i] < time[j] else j
                    later = j if time[i] < time[j] else i
                    if risk_scores[earlier] > risk_scores[later]:
                        concordant += 1
                    elif risk_scores[earlier] < risk_scores[later]:
                        discordant += 1
                    else:
                        tied += 0.5
            elif event[i] == 1 and event[j] == 0:
                # i had event, j censored
                if time[i] < time[j]:
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] < risk_scores[j]:
                        discordant += 1
                    else:
                        tied += 0.5
            elif event[j] == 1 and event[i] == 0:
                # j had event, i censored
                if time[j] < time[i]:
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] < risk_scores[i]:
                        discordant += 1
                    else:
                        tied += 0.5

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validated C-Index
# ─────────────────────────────────────────────────────────────────────────────

def cross_validated_cindex(X: np.ndarray, time: np.ndarray,
                            event: np.ndarray, n_splits: int = 5) -> Dict[str, float]:
    """Compute cross-validated concordance index."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    c_indices = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        t_train, t_test = time[train_idx], time[test_idx]
        e_train, e_test = event[train_idx], event[test_idx]

        model = ManualCoxPH(l2_reg=0.1)
        try:
            model.fit(X_train, t_train, e_train)
            risk_scores = model.predict_log_partial_hazard(X_test)
            c_idx = compute_concordance_index(t_test, e_test, risk_scores)
            c_indices.append(c_idx)
        except Exception:
            pass

    if not c_indices:
        return {'mean': 0.5, 'std': 0.0, 'values': []}

    return {
        'mean': np.mean(c_indices),
        'std': np.std(c_indices),
        'values': c_indices
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataframe(df: