```python
"""
Survival Analysis Pipeline for Predicting Patient Outcomes
==========================================================
Builds a Cox Proportional Hazards model with proper train/test splitting,
hazard ratio computation, survival curve estimation, and model evaluation.
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Optional lifelines import ──────────────────────────────────────────────────
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test

    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print(
        "lifelines not installed – falling back to manual Cox PH implementation. "
        "Install with: pip install lifelines"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class SurvivalPipelineResult:
    """Container for all outputs of the survival analysis pipeline."""

    fitted_model: object
    hazard_ratios: pd.DataFrame
    survival_curves: Dict[str, pd.DataFrame]  # key: risk group label
    evaluation_metrics: Dict[str, float]
    risk_group_assignments: pd.Series  # test-set risk groups
    log_rank_results: Dict[str, object]
    scaler: StandardScaler
    feature_names: List[str]
    train_concordance: float
    test_concordance: float


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════


def _validate_dataframe(df: pd.DataFrame) -> None:
    required = {"event_time", "censored"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if (df["event_time"] < 0).any():
        raise ValueError("'event_time' must be non-negative.")
    if not df["censored"].isin([0, 1]).all():
        raise ValueError("'censored' must contain only 0 or 1.")


def _assign_risk_groups(
    risk_scores: np.ndarray,
    low_q: float = 0.33,
    high_q: float = 0.67,
    thresholds: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Assign low / medium / high risk groups.

    Parameters
    ----------
    risk_scores : array of predicted log-partial-hazard scores
    low_q, high_q : quantile cut-points (computed on training data only)
    thresholds : pre-computed (low_thresh, high_thresh) from training set

    Returns
    -------
    labels : array of strings
    thresholds : (low_thresh, high_thresh) – pass back for test-set use
    """
    if thresholds is None:
        low_thresh = np.quantile(risk_scores, low_q)
        high_thresh = np.quantile(risk_scores, high_q)
    else:
        low_thresh, high_thresh = thresholds

    labels = np.where(
        risk_scores <= low_thresh,
        "low",
        np.where(risk_scores <= high_thresh, "medium", "high"),
    )
    return labels, (low_thresh, high_thresh)


def _concordance_index(
    event_times: np.ndarray,
    predicted_scores: np.ndarray,
    event_observed: np.ndarray,
) -> float:
    """
    Harrell's concordance index (C-index).
    Higher predicted score → higher risk → shorter expected survival.
    """
    n = len(event_times)
    concordant = 0
    discordant = 0
    tied_risk = 0
    permissible = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Only permissible pairs: at least one event observed
            if event_observed[i] == 0 and event_observed[j] == 0:
                continue
            # If i is censored before j's event time, pair is not permissible
            if event_observed[i] == 0 and event_times[i] <= event_times[j]:
                continue
            if event_observed[j] == 0 and event_times[j] <= event_times[i]:
                continue

            permissible += 1
            if event_times[i] < event_times[j]:
                # i had event first → higher risk score for i is concordant
                if predicted_scores[i] > predicted_scores[j]:
                    concordant += 1
                elif predicted_scores[i] < predicted_scores[j]:
                    discordant += 1
                else:
                    tied_risk += 1
            elif event_times[j] < event_times[i]:
                if predicted_scores[j] > predicted_scores[i]:
                    concordant += 1
                elif predicted_scores[j] < predicted_scores[i]:
                    discordant += 1
                else:
                    tied_risk += 1
            else:
                # Tied event times
                tied_risk += 1

    if permissible == 0:
        return float("nan")
    return (concordant + 0.5 * tied_risk) / permissible


# ══════════════════════════════════════════════════════════════════════════════
# Manual Cox PH (Newton-Raphson) – used when lifelines is unavailable
# ══════════════════════════════════════════════════════════════════════════════


class ManualCoxPH:
    """
    Minimal Cox Proportional Hazards model fitted via Newton-Raphson.
    Handles right-censored data through the partial likelihood.
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-7, step_size: float = 0.5):
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        self.coef_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.baseline_hazard_: Optional[pd.DataFrame] = None

    def fit(
        self,
        X: np.ndarray,
        event_times: np.ndarray,
        event_observed: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "ManualCoxPH":
        n, p = X.shape
        self.feature_names_ = feature_names or [f"x{i}" for i in range(p)]
        beta = np.zeros(p)

        # Sort by event time (ascending) for efficient risk-set computation
        order = np.argsort(event_times)
        X_s = X[order]
        t_s = event_times[order]
        e_s = event_observed[order]

        for iteration in range(self.max_iter):
            theta = np.exp(X_s @ beta)  # (n,)
            grad = np.zeros(p)
            hess = np.zeros((p, p))

            for i in range(n):
                if e_s[i] == 0:
                    continue  # censored – skip
                # Risk set: all j with t_j >= t_i
                risk_mask = t_s >= t_s[i]
                theta_risk = theta[risk_mask]
                X_risk = X_s[risk_mask]

                denom = theta_risk.sum()
                w = theta_risk / denom  # weights
                mu = X_risk.T @ w  # weighted mean of covariates
                sigma = (X_risk * w[:, None]).T @ X_risk - np.outer(mu, mu)

                grad += X_s[i] - mu
                hess -= sigma

            # Newton-Raphson update with step-size damping
            try:
                delta = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(hess, grad, rcond=None)[0]

            beta_new = beta - self.step_size * delta
            if np.max(np.abs(beta_new - beta)) < self.tol:
                beta = beta_new
                break
            beta = beta_new

        self.coef_ = beta

        # Breslow baseline hazard estimator
        self.baseline_hazard_ = self._breslow(X_s, t_s, e_s, beta)
        return self

    def _breslow(
        self,
        X_s: np.ndarray,
        t_s: np.ndarray,
        e_s: np.ndarray,
        beta: np.ndarray,
    ) -> pd.DataFrame:
        theta = np.exp(X_s @ beta)
        event_times_unique = np.unique(t_s[e_s == 1])
        h0 = []
        for t in event_times_unique:
            d = (e_s == 1) & (t_s == t)
            risk = t_s >= t
            h0.append(d.sum() / theta[risk].sum())
        return pd.DataFrame({"time": event_times_unique, "baseline_hazard": h0})

    def predict_log_partial_hazard(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_

    def predict_survival_function(
        self, X: np.ndarray, times: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Return survival probability S(t|x) for each row of X."""
        if self.baseline_hazard_ is None:
            raise RuntimeError("Model not fitted yet.")
        bh = self.baseline_hazard_
        if times is None:
            times = bh["time"].values

        log_ph = self.predict_log_partial_hazard(X)  # (n,)
        H0 = np.array(
            [bh.loc[bh["time"] <= t, "baseline_hazard"].sum() for t in times]
        )  # cumulative baseline hazard

        # S(t|x) = exp(-H0(t) * exp(beta'x))
        survival = np.exp(-np.outer(np.exp(log_ph), H0))  # (n, len(times))
        df = pd.DataFrame(survival.T, index=times)
        return df

    def get_hazard_ratios(self) -> pd.DataFrame:
        hr = np.exp(self.coef_)
        return pd.DataFrame(
            {"feature": self.feature_names_, "coef": self.coef_, "hazard_ratio": hr}
        ).set_index("feature")


# ══════════════════════════════════════════════════════════════════════════════
# Kaplan-Meier (manual) – used when lifelines is unavailable
# ══════════════════════════════════════════════════════════════════════════════


def _kaplan_meier(
    event_times: np.ndarray, event_observed: np.ndarray
) -> pd.DataFrame:
    """Compute Kaplan-Meier survival estimate."""
    df = pd.DataFrame({"time": event_times, "event": event_observed})
    df = df.sort_values("time").reset_index(drop=True)

    unique_times = np.unique(df["time"])
    n = len(df)
    S = 1.0
    records = []
    at_risk = n

    for t in unique_times:
        mask_t = df["time"] == t
        d = df.loc[mask_t, "event"].sum()  # events at time t
        c = (mask_t & (df["event"] == 0)).sum()  # censored at time t
        if at_risk > 0:
            S *= 1 - d / at_risk
        records.append({"time": t, "survival": S, "at_risk": at_risk, "events": d})
        at_risk -= d + c

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Log-rank test (manual) – used when lifelines is unavailable
# ══════════════════════════════════════════════════════════════════════════════


def _manual_logrank_test(
    group_a_times: np.ndarray,
    group_a_events: np.ndarray,
    group_b_times: np.ndarray,
    group_b_events: np.ndarray,
) -> Dict[str, float]:
    """Two-sample log-rank test."""
    all_times = np.unique(
        np.concatenate(
            [
                group_a_times[group_a_events == 1],
                group_b_times[group_b_events == 1],
            ]
        )
    )

    O_a, E_a = 0.0, 0.0
    V = 0.0

    for t in all_times:
        n_a = (group_a_times >= t).sum()
        n_b = (group_b_times >= t).sum()
        d_a = ((group_a_times == t) & (group_a_events == 1)).sum()
        d_b = ((group_b_times == t) & (group_b_events == 1)).sum()
        n = n_a + n_b
        d = d_a + d_b

        if n < 2:
            continue

        e_a = n_a * d / n
        O_a += d_a
        E_a += e_a
        V += n_a * n_b * d * (n - d) / (n**2 * (n - 1)) if n > 1 else 0.0

    if V == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    chi2 = (O_a - E_a) ** 2 / V
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return {"statistic": float(chi2), "p_value": float(p_value)}


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════


def build_survival_pipeline(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    penalizer: float = 0.1,  # L2 regularisation for Cox model
    low_risk_quantile: float = 0.33,
    high_risk_quantile: float = 0.67,
) -> SurvivalPipelineResult:
    """
    Build a survival analysis pipeline.

    Parameters
    ----------
    df : DataFrame with columns including 'event_time', 'censored',
         and patient feature columns.
    feature_cols : list of feature column names; if None, all columns
                   except 'event_time' and 'censored' are used.
    test_size : fraction of data held out for evaluation.
    random_state : reproducibility seed.
    penalizer : L2 regularisation strength for Cox PH (lifelines only).
    low_risk_quantile : quantile threshold separating low from medium risk.
    high_risk_quantile : quantile threshold separating medium from high risk.

    Returns
    -------
    SurvivalPipelineResult
    """
    # ── 0. Validate ────────────────────────────────────────────────────────────
    _validate_dataframe(df)
    df = df.copy().reset_index(drop=True)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ("event_time", "censored")]

    # ── 1. Train / test split BEFORE any preprocessing ────────────────────────
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train_raw = train_df[feature_cols].values.astype(float)
    X_test_raw = test_df[feature_cols].values.astype(float)

    t_train = train_df["event_time"].values
    e_train = (1 - train_df["censored"].values).astype(int)  # 1 = event observed

    t_test = test_df["event_time"].values
    e_test = (1 - test_df["censored"].values).astype(int)

    # ── 2. Fit scaler on TRAINING data only ───────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)  # transform only

    # ── 3. Fit Cox PH model ───────────────────────────────────────────────────
    if LIFELINES_AVAILABLE:
        fitted_model, hazard_ratios_df, train_scores, test_scores = _fit_lifelines_cox(
            X_train,
            t_train,
            e_train,
            X_test,
            feature_cols,
            penalizer,
        )
    else:
        fitted_model, hazard_ratios_df, train_scores, test_scores = _fit_manual_cox(
            X_train,
            t_train,
            e_train,
            X_test,
            feature_cols,
        )

    # ── 4. Assign risk groups (thresholds from TRAINING scores only) ──────────
    train_groups, thresholds = _assign_risk_groups(
        train_scores,
        low_q=low_risk_quantile,
        high_q=high_risk_quantile,
    )
    test_groups, _ = _assign_risk_groups(
        test_scores,
        thresholds=thresholds,
    )
    test_risk_series = pd.Series(test_groups, index=test_df.index, name="risk_group")

    # ── 5. Survival curves per risk group (Kaplan-Meier on TEST set) ──────────
    survival_curves = _compute_survival_curves(t_test, e_test, test_groups)

    # ── 6. Log-rank tests between risk groups (Bonferroni corrected) ──────────
    log_rank_results = _run_log_rank_tests(t_test, e_test, test_groups)

    # ── 7. Evaluation metrics ─────────────────────────────────────────────────
    train_ci = _concordance_index(t_train, train_scores, e_train)
    test_ci = _concordance_index(t_test, test_scores, e_test)

    evaluation_metrics = {
        "train_concordance_index": train_ci,
        "test_concordance_index": test_ci,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_events_train": int(e_train.sum()),
        "n_events_test": int(e_test.sum()),
        "n_censored_train": int((1 - e_train).sum()),
        "n_censored_test": int((1 - e_test).sum()),
    }

    # Add log-rank p-values to metrics
    for pair, result in log_rank_results.items():
        p_val = result.get("p_value_corrected", result.get("p_value", float("nan")))
        evaluation_metrics[f"logrank_p_{pair}"] = p_val

    return SurvivalPipelineResult(
        fitted_model=fitted_model,
        hazard_ratios=hazard_ratios_df,
        survival_curves=survival_curves,