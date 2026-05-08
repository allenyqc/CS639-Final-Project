```python
"""
Sequential Experiment Monitoring with Interim Analyses
Implements group sequential testing with alpha spending functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Optional, Generator
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InterimResult:
    """Result of a single interim analysis."""
    analysis_number: int
    n_total: int
    n_treatment: int
    n_control: int
    p_treatment: float
    p_control: float
    effect_size: float          # p_treatment - p_control
    z_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    alpha_spent: float          # cumulative alpha spent so far
    alpha_boundary: float       # z-score boundary at this look
    futility_boundary: float    # z-score futility boundary
    information_fraction: float # t = n / N_max
    decision: str               # "continue", "stop_efficacy", "stop_futility"


@dataclass
class MonitoringResult:
    """Final result of the sequential monitoring process."""
    interim_results: list[InterimResult] = field(default_factory=list)
    final_decision: str = "continue"
    stopped_early: bool = False
    stopping_analysis: Optional[int] = None
    total_observations: int = 0
    final_p_value: Optional[float] = None
    final_effect_size: Optional[float] = None
    final_ci: Optional[tuple[float, float]] = None
    alpha_total: float = 0.05
    power_target: float = 0.80


# ─────────────────────────────────────────────────────────────────────────────
# Alpha spending functions
# ─────────────────────────────────────────────────────────────────────────────

def alpha_spend_obrien_fleming(t: float, alpha: float = 0.05) -> float:
    """
    O'Brien-Fleming alpha spending function.
    Conservative early on, liberal late.
    alpha_spent(t) = 2 * (1 - Phi(z_alpha/2 / sqrt(t)))
    """
    if t <= 0:
        return 0.0
    z_alpha = norm.ppf(1 - alpha / 2)
    return 2 * (1 - norm.cdf(z_alpha / np.sqrt(t)))


def alpha_spend_pocock(t: float, alpha: float = 0.05) -> float:
    """
    Pocock alpha spending function.
    Constant boundaries (approximately).
    alpha_spent(t) = alpha * ln(1 + (e - 1) * t)
    """
    if t <= 0:
        return 0.0
    return alpha * np.log(1 + (np.e - 1) * t)


def alpha_spend_power(t: float, alpha: float = 0.05, rho: float = 3.0) -> float:
    """
    Power family alpha spending function.
    alpha_spent(t) = alpha * t^rho
    rho=1: linear; rho=3: conservative early
    """
    if t <= 0:
        return 0.0
    return alpha * (t ** rho)


ALPHA_SPENDING_FUNCTIONS = {
    "obrien_fleming": alpha_spend_obrien_fleming,
    "pocock": alpha_spend_pocock,
    "power": alpha_spend_power,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core monitoring class
# ─────────────────────────────────────────────────────────────────────────────

class SequentialMonitor:
    """
    Sequential experiment monitor with interim analyses.

    Parameters
    ----------
    max_n : int
        Maximum total sample size (both arms combined).
    interim_every : int
        Perform an interim analysis every this many observations.
    alpha : float
        Overall Type I error rate (two-sided).
    beta : float
        Type II error rate (1 - power).
    spending_function : str
        One of "obrien_fleming", "pocock", "power".
    futility_spending_function : str or None
        Spending function for futility boundaries. If None, uses beta-spending
        mirroring the efficacy function.
    futility_binding : bool
        If True, crossing the futility boundary *stops* the trial (binding).
        If False, it is advisory only.
    min_n_per_arm : int
        Minimum observations per arm before any interim analysis is performed.
    """

    def __init__(
        self,
        max_n: int = 10_000,
        interim_every: int = 1_000,
        alpha: float = 0.05,
        beta: float = 0.20,
        spending_function: str = "obrien_fleming",
        futility_spending_function: Optional[str] = None,
        futility_binding: bool = False,
        min_n_per_arm: int = 50,
    ):
        self.max_n = max_n
        self.interim_every = interim_every
        self.alpha = alpha
        self.beta = beta
        self.futility_binding = futility_binding
        self.min_n_per_arm = min_n_per_arm

        if spending_function not in ALPHA_SPENDING_FUNCTIONS:
            raise ValueError(f"spending_function must be one of {list(ALPHA_SPENDING_FUNCTIONS)}")
        self.spend_fn = ALPHA_SPENDING_FUNCTIONS[spending_function]
        self.spending_function_name = spending_function

        fut_fn_name = futility_spending_function or spending_function
        if fut_fn_name not in ALPHA_SPENDING_FUNCTIONS:
            raise ValueError(f"futility_spending_function must be one of {list(ALPHA_SPENDING_FUNCTIONS)}")
        self.futility_spend_fn = ALPHA_SPENDING_FUNCTIONS[fut_fn_name]

        # Internal state
        self._treatment_outcomes: list[int] = []
        self._control_outcomes: list[int] = []
        self._next_interim_at: int = interim_every
        self._result = MonitoringResult(alpha_total=alpha, power_target=1 - beta)
        self._stopped = False

    # ── Data ingestion ────────────────────────────────────────────────────────

    def add_observation(self, group: str, outcome: int) -> Optional[InterimResult]:
        """
        Add a single observation.

        Parameters
        ----------
        group : str
            "treatment" or "control"
        outcome : int
            Binary outcome (0 or 1).

        Returns
        -------
        InterimResult if an interim analysis was triggered, else None.
        """
        if self._stopped:
            warnings.warn("Experiment already stopped; ignoring new observations.")
            return None

        if group == "treatment":
            self._treatment_outcomes.append(int(outcome))
        elif group == "control":
            self._control_outcomes.append(int(outcome))
        else:
            raise ValueError("group must be 'treatment' or 'control'")

        n_total = len(self._treatment_outcomes) + len(self._control_outcomes)

        if n_total >= self._next_interim_at:
            self._next_interim_at += self.interim_every
            return self._run_interim()

        return None

    def add_batch(
        self,
        groups: list[str],
        outcomes: list[int],
    ) -> list[InterimResult]:
        """
        Add a batch of observations, returning all interim results triggered.
        """
        triggered = []
        for g, o in zip(groups, outcomes):
            result = self.add_observation(g, o)
            if result is not None:
                triggered.append(result)
                if self._stopped:
                    break
        return triggered

    def stream_observations(
        self,
        data_generator: Generator[tuple[str, int], None, None],
    ) -> MonitoringResult:
        """
        Consume a generator of (group, outcome) tuples until exhausted or stopped.

        Returns the complete MonitoringResult.
        """
        for group, outcome in data_generator:
            self.add_observation(group, outcome)
            if self._stopped:
                break

        return self.finalize()

    # ── Interim analysis ──────────────────────────────────────────────────────

    def _run_interim(self) -> Optional[InterimResult]:
        """Perform an interim analysis on current data."""
        n_t = len(self._treatment_outcomes)
        n_c = len(self._control_outcomes)

        if n_t < self.min_n_per_arm or n_c < self.min_n_per_arm:
            return None  # Not enough data yet

        n_total = n_t + n_c
        analysis_number = len(self._result.interim_results) + 1

        # Proportions
        p_t = np.mean(self._treatment_outcomes) if n_t > 0 else 0.0
        p_c = np.mean(self._control_outcomes) if n_c > 0 else 0.0
        effect = p_t - p_c

        # Pooled proportion for z-test
        p_pool = (sum(self._treatment_outcomes) + sum(self._control_outcomes)) / n_total
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))

        if se < 1e-10:
            z = 0.0
        else:
            z = effect / se

        p_value = 2 * (1 - norm.cdf(abs(z)))

        # Confidence interval (Wald)
        se_diff = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
        z_ci = norm.ppf(0.975)
        ci_lower = effect - z_ci * se_diff
        ci_upper = effect + z_ci * se_diff

        # Information fraction
        t = min(n_total / self.max_n, 1.0)

        # Alpha spending: cumulative alpha spent up to this look
        alpha_spent_cum = self.spend_fn(t, self.alpha)

        # Efficacy boundary: find z such that P(|Z| > z) = incremental alpha
        prev_alpha_spent = (
            self._result.interim_results[-1].alpha_spent
            if self._result.interim_results
            else 0.0
        )
        incremental_alpha = max(alpha_spent_cum - prev_alpha_spent, 1e-10)
        # Two-sided boundary
        alpha_boundary_z = norm.ppf(1 - incremental_alpha / 2)

        # Futility boundary (beta spending, inner wedge)
        beta_spent_cum = self.futility_spend_fn(t, self.beta)
        prev_beta_spent = 0.0
        if self._result.interim_results:
            # Approximate: use 1 - (1 - beta_spent) as proxy
            prev_beta_spent = self.futility_spend_fn(
                self._result.interim_results[-1].information_fraction, self.beta
            )
        incremental_beta = max(beta_spent_cum - prev_beta_spent, 1e-10)
        # Futility: stop if |z| < futility_z (non-rejection region)
        futility_z = norm.ppf(1 - incremental_beta / 2)
        # Ensure futility < efficacy
        futility_z = min(futility_z, alpha_boundary_z - 0.1)

        # Decision
        decision = "continue"
        if abs(z) >= alpha_boundary_z:
            decision = "stop_efficacy"
        elif self.futility_binding and abs(z) <= futility_z:
            decision = "stop_futility"

        interim = InterimResult(
            analysis_number=analysis_number,
            n_total=n_total,
            n_treatment=n_t,
            n_control=n_c,
            p_treatment=p_t,
            p_control=p_c,
            effect_size=effect,
            z_statistic=z,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            alpha_spent=alpha_spent_cum,
            alpha_boundary=alpha_boundary_z,
            futility_boundary=futility_z,
            information_fraction=t,
            decision=decision,
        )

        self._result.interim_results.append(interim)
        self._result.total_observations = n_total

        if decision in ("stop_efficacy", "stop_futility"):
            self._stopped = True
            self._result.stopped_early = True
            self._result.final_decision = decision
            self._result.stopping_analysis = analysis_number
            self._result.final_p_value = p_value
            self._result.final_effect_size = effect
            self._result.final_ci = (ci_lower, ci_upper)

        return interim

    # ── Finalization ──────────────────────────────────────────────────────────

    def finalize(self) -> MonitoringResult:
        """
        Run a final analysis on all accumulated data (if not already stopped).
        """
        if not self._stopped:
            # Force a final analysis
            saved_next = self._next_interim_at
            self._next_interim_at = 0  # trigger immediately
            result = self._run_interim()
            self._next_interim_at = saved_next

            if result is not None and not self._stopped:
                # Determine final decision based on nominal alpha
                last = self._result.interim_results[-1]
                if last.p_value <= self.alpha:
                    self._result.final_decision = "reject_null"
                else:
                    self._result.final_decision = "fail_to_reject"
                self._result.final_p_value = last.p_value
                self._result.final_effect_size = last.effect_size
                self._result.final_ci = (last.ci_lower, last.ci_upper)

        self._result.total_observations = (
            len(self._treatment_outcomes) + len(self._control_outcomes)
        )
        return self._result

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_monitoring(
        self,
        figsize: tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Produce a comprehensive monitoring plot with:
        - Z-statistic trajectory vs efficacy/futility boundaries
        - Cumulative alpha spent
        - Effect size with confidence intervals
        - P-value trajectory
        """
        results = self._result.interim_results
        if not results:
            raise ValueError("No interim analyses have been performed yet.")

        analyses = [r.analysis_number for r in results]
        z_stats = [r.z_statistic for r in results]
        eff_bounds = [r.alpha_boundary for r in results]
        fut_bounds = [r.futility_boundary for r in results]
        alpha_spent = [r.alpha_spent for r in results]
        effects = [r.effect_size for r in results]
        ci_lowers = [r.ci_lower for r in results]
        ci_uppers = [r.ci_upper for r in results]
        p_values = [r.p_value for r in results]
        info_fracs = [r.information_fraction for r in results]
        n_totals = [r.n_total for r in results]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Sequential Experiment Monitoring\n"
            f"Spending: {self.spending_function_name.replace('_', ' ').title()} | "
            f"α={self.alpha} | β={self.beta}",
            fontsize=14,
            fontweight="bold",
        )

        # ── Panel 1: Z-statistic vs boundaries ───────────────────────────────
        ax1 = axes[0, 0]
        ax1.fill_between(
            analyses,
            eff_bounds,
            [max(eff_bounds) * 1.3] * len(analyses),
            alpha=0.15,
            color="green",
            label="Efficacy region",
        )
        ax1.fill_between(
            analyses,
            [-max(eff_bounds) * 1.3] * len(analyses),
            [-b for b in eff_bounds],
            alpha=0.15,
            color="green",
        )
        ax1.fill_between(
            analyses,
            [-b for b in fut_bounds],
            fut_bounds,
            alpha=0.15,
            color="orange",
            label="Futility region",
        )

        ax1.plot(analyses, eff_bounds, "g--", linewidth=2, label="Efficacy boundary (+)")
        ax1.plot(analyses, [-b for b in eff_bounds], "g--", linewidth=2)
        ax1.plot(analyses, fut_bounds, "orange", linestyle=":", linewidth=2, label="Futility boundary (+)")
        ax1.plot(analyses, [-b for b in fut_bounds], "orange", linestyle=":", linewidth=2)
        ax1.plot(analyses, z_stats, "b-o", linewidth=2, markersize=6, label="Z-statistic", zorder=5)
        ax1.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

        # Mark stopping point
        for r in results:
            if r.decision in ("stop_efficacy", "stop_futility"):
                color = "green" if r.decision == "stop_efficacy" else "orange"
                ax1.axvline(r.analysis_number, color=color, linestyle="--", alpha=0.7)
                ax1.scatter(
                    [r.analysis_number], [r.z_statistic],
                    color=color, s=150, zorder=10,
                    marker="*", label=f"Stopped ({r.decision.replace('_', ' ')})"
                )

        ax1.set_xlabel("Interim Analysis Number")
        ax1.set_ylabel("Z-Statistic")
        ax1.set_title("Test Statistic vs Stopping Boundaries")
        ax1.legend(fontsize=8, loc="upper left")
        ax1.grid(True, alpha=0.3)

        # ── Panel 2: Cumulative alpha spent ───────────────────────────────────
        ax2 = axes[0, 1]
        t_range = np.linspace(0.01, 1.0, 200)
        alpha_curve = [self.spend_fn(t, self.alpha) for t in t_range]
        ax2.plot(t_range, alpha_curve, "r-", linewidth=2, label=f"α-spending ({self.spending_function_name})", alpha=0.7)
        ax2.plot(info_fracs, alpha_spent, "b-o", linewidth=2, markersize=6, label="Cumulative α spent")
        ax2.axhline(self.alpha, color="red", linestyle="--", linewidth=1.5, label=f"Total α = {self.alpha}")
        ax2.fill_between(t_range, alpha_curve, self.alpha, alpha=0.1, color="red")

        ax2.set_xlabel("Information Fraction (t = n/N_max)")
        ax2.set_ylabel("Cumulative Alpha Spent")
        ax2.set_title("Alpha Spending Over Time")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, self.alpha * 1.2)

        # ── Panel 3: Effect size with CI ──────────────────────────────────────
        ax3 = axes[1, 0]
        ax3.fill_between(analyses, ci_lowers, ci_uppers, alpha=0.2,