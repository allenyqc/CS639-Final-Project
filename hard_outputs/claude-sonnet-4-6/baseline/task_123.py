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

warnings.filterwarnings('ignore')


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
    z_score: float
    p_value: float
    ci_lower: float
    ci_upper: float
    alpha_spent_cumulative: float
    efficacy_boundary: float    # z-score boundary for stopping for efficacy
    futility_boundary: float    # z-score boundary for stopping for futility
    decision: str               # "continue", "stop_efficacy", "stop_futility"
    information_fraction: float # t = n / max_n


@dataclass
class MonitoringResult:
    """Final result of the sequential monitoring process."""
    interim_results: list[InterimResult] = field(default_factory=list)
    final_decision: str = "continue"
    stopping_analysis: Optional[int] = None
    total_observations: int = 0
    final_p_value: Optional[float] = None
    final_effect_size: Optional[float] = None
    final_ci: Optional[tuple[float, float]] = None
    alpha_total: float = 0.05
    power_achieved: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Alpha spending functions
# ─────────────────────────────────────────────────────────────────────────────

def alpha_spend_obrien_fleming(t: float, alpha: float = 0.05) -> float:
    """
    O'Brien-Fleming alpha spending function.
    Conservative early on, liberal near the end.
    alpha_spent(t) = 2 * (1 - Phi(z_alpha/2 / sqrt(t)))
    """
    if t <= 0:
        return 0.0
    z_alpha = norm.ppf(1 - alpha / 2)
    return 2 * (1 - norm.cdf(z_alpha / np.sqrt(t)))


def alpha_spend_pocock(t: float, alpha: float = 0.05) -> float:
    """
    Pocock alpha spending function.
    Spends alpha more uniformly across analyses.
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
# Boundary computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_efficacy_boundary(
    alpha_increment: float,
    two_sided: bool = True,
) -> float:
    """
    Convert incremental alpha to a z-score boundary.
    Uses the Haybittle-Peto approximation for simplicity.
    """
    if alpha_increment <= 0:
        return np.inf
    if two_sided:
        return norm.ppf(1 - alpha_increment / 2)
    return norm.ppf(1 - alpha_increment)


def compute_futility_boundary(
    beta_spending: float,
    two_sided: bool = True,
) -> float:
    """
    Compute futility (non-binding) boundary.
    Returns the z-score below which we stop for futility.
    """
    if beta_spending <= 0:
        return -np.inf
    if two_sided:
        return norm.ppf(beta_spending / 2)
    return norm.ppf(beta_spending)


# ─────────────────────────────────────────────────────────────────────────────
# Core sequential monitor
# ─────────────────────────────────────────────────────────────────────────────

class SequentialExperimentMonitor:
    """
    Monitors a streaming A/B experiment with group sequential testing.

    Parameters
    ----------
    max_n : int
        Maximum planned sample size (both arms combined).
    interim_interval : int
        Number of new observations between interim analyses.
    alpha : float
        Overall Type I error rate (two-sided).
    beta : float
        Type II error rate (1 - power) for futility spending.
    alpha_spending : str
        Alpha spending function: "obrien_fleming", "pocock", or "power".
    two_sided : bool
        Whether to use two-sided tests.
    min_n_per_arm : int
        Minimum observations per arm before first interim analysis.
    """

    def __init__(
        self,
        max_n: int = 10_000,
        interim_interval: int = 1_000,
        alpha: float = 0.05,
        beta: float = 0.20,
        alpha_spending: str = "obrien_fleming",
        two_sided: bool = True,
        min_n_per_arm: int = 100,
    ):
        self.max_n = max_n
        self.interim_interval = interim_interval
        self.alpha = alpha
        self.beta = beta
        self.two_sided = two_sided
        self.min_n_per_arm = min_n_per_arm

        if alpha_spending not in ALPHA_SPENDING_FUNCTIONS:
            raise ValueError(
                f"alpha_spending must be one of {list(ALPHA_SPENDING_FUNCTIONS)}"
            )
        self._spend_fn = ALPHA_SPENDING_FUNCTIONS[alpha_spending]
        self._beta_spend_fn = ALPHA_SPENDING_FUNCTIONS["obrien_fleming"]

        # Internal state
        self._treatment_outcomes: list[int] = []
        self._control_outcomes: list[int] = []
        self._next_interim_at: int = interim_interval
        self._alpha_spent_so_far: float = 0.0
        self._beta_spent_so_far: float = 0.0
        self._result = MonitoringResult(alpha_total=alpha)
        self._stopped: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def add_observation(
        self,
        group: str,          # "treatment" or "control"
        outcome: int,        # 1 = success, 0 = failure
    ) -> Optional[InterimResult]:
        """
        Add a single observation. Returns an InterimResult if an interim
        analysis was triggered, otherwise returns None.
        """
        if self._stopped:
            return None

        if group == "treatment":
            self._treatment_outcomes.append(outcome)
        elif group == "control":
            self._control_outcomes.append(outcome)
        else:
            raise ValueError("group must be 'treatment' or 'control'")

        n_total = len(self._treatment_outcomes) + len(self._control_outcomes)

        # Check if we should run an interim analysis
        if n_total >= self._next_interim_at:
            self._next_interim_at += self.interim_interval
            return self._run_interim_analysis()

        # Final analysis at max_n
        if n_total >= self.max_n:
            return self._run_interim_analysis(final=True)

        return None

    def add_batch(
        self,
        groups: list[str],
        outcomes: list[int],
    ) -> list[InterimResult]:
        """
        Add a batch of observations. Returns all interim results triggered.
        """
        results = []
        for g, o in zip(groups, outcomes):
            r = self.add_observation(g, o)
            if r is not None:
                results.append(r)
        return results

    def stream_data(
        self,
        data_generator: Generator[tuple[str, int], None, None],
    ) -> MonitoringResult:
        """
        Consume a generator of (group, outcome) tuples until the experiment
        ends (stopped early or data exhausted). Returns the full monitoring
        result.
        """
        for group, outcome in data_generator:
            self.add_observation(group, outcome)
            if self._stopped:
                break

        # Run final analysis if not already stopped
        if not self._stopped:
            self._run_interim_analysis(final=True)

        return self._result

    def run_on_dataset(
        self,
        groups: np.ndarray,
        outcomes: np.ndarray,
    ) -> MonitoringResult:
        """
        Run the full monitoring procedure on a pre-collected dataset.
        Simulates sequential arrival of data.
        """
        for g, o in zip(groups, outcomes):
            self.add_observation(g, o)
            if self._stopped:
                break

        if not self._stopped:
            self._run_interim_analysis(final=True)

        return self._result

    @property
    def result(self) -> MonitoringResult:
        return self._result

    # ── Internal analysis ─────────────────────────────────────────────────────

    def _run_interim_analysis(self, final: bool = False) -> Optional[InterimResult]:
        n_t = len(self._treatment_outcomes)
        n_c = len(self._control_outcomes)

        if n_t < self.min_n_per_arm or n_c < self.min_n_per_arm:
            return None

        n_total = n_t + n_c
        t = min(n_total / self.max_n, 1.0)   # information fraction

        # Proportions
        p_t = np.mean(self._treatment_outcomes) if n_t > 0 else 0.0
        p_c = np.mean(self._control_outcomes) if n_c > 0 else 0.0
        delta = p_t - p_c

        # Pooled proportion for z-test
        p_pool = (sum(self._treatment_outcomes) + sum(self._control_outcomes)) / n_total
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))

        if se < 1e-10:
            z = 0.0
        else:
            z = delta / se

        # Two-sided p-value
        p_value = 2 * (1 - norm.cdf(abs(z)))

        # Confidence interval (Wald)
        se_ci = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
        z_crit = norm.ppf(1 - self.alpha / 2)
        ci_lower = delta - z_crit * se_ci
        ci_upper = delta + z_crit * se_ci

        # Alpha spending
        alpha_spent_total = self._spend_fn(t, self.alpha)
        alpha_increment = max(alpha_spent_total - self._alpha_spent_so_far, 0.0)
        self._alpha_spent_so_far = alpha_spent_total

        # Beta spending (futility)
        beta_spent_total = self._beta_spend_fn(t, self.beta)
        beta_increment = max(beta_spent_total - self._beta_spent_so_far, 0.0)
        self._beta_spent_so_far = beta_spent_total

        # Boundaries
        eff_boundary = compute_efficacy_boundary(alpha_increment, self.two_sided)
        fut_boundary = compute_futility_boundary(beta_increment, self.two_sided)

        # Decision
        analysis_number = len(self._result.interim_results) + 1
        if final:
            # At final analysis use full alpha
            remaining_alpha = self.alpha - (self._alpha_spent_so_far - alpha_increment)
            eff_boundary = compute_efficacy_boundary(remaining_alpha, self.two_sided)
            fut_boundary = 0.0  # No futility at final

        if abs(z) >= eff_boundary:
            decision = "stop_efficacy"
        elif abs(z) <= abs(fut_boundary) and not final:
            decision = "stop_futility"
        else:
            decision = "continue" if not final else "complete"

        interim = InterimResult(
            analysis_number=analysis_number,
            n_total=n_total,
            n_treatment=n_t,
            n_control=n_c,
            p_treatment=p_t,
            p_control=p_c,
            effect_size=delta,
            z_score=z,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            alpha_spent_cumulative=alpha_spent_total,
            efficacy_boundary=eff_boundary,
            futility_boundary=fut_boundary,
            decision=decision,
            information_fraction=t,
        )

        self._result.interim_results.append(interim)
        self._result.total_observations = n_total

        if decision in ("stop_efficacy", "stop_futility", "complete"):
            self._stopped = True
            self._result.final_decision = decision
            self._result.stopping_analysis = analysis_number
            self._result.final_p_value = p_value
            self._result.final_effect_size = delta
            self._result.final_ci = (ci_lower, ci_upper)

        return interim

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_monitoring(
        self,
        figsize: tuple[int, int] = (14, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Produce a comprehensive monitoring plot with:
        - Test statistic trajectory vs stopping boundaries
        - Cumulative alpha spent
        - Effect size with confidence intervals
        - P-value trajectory
        """
        results = self._result.interim_results
        if not results:
            raise ValueError("No interim analyses have been performed yet.")

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            "Sequential Experiment Monitoring Dashboard",
            fontsize=15, fontweight="bold", y=1.01,
        )

        analyses = [r.analysis_number for r in results]
        n_totals = [r.n_total for r in results]
        z_scores = [r.z_score for r in results]
        eff_bounds = [r.efficacy_boundary for r in results]
        fut_bounds = [r.futility_boundary for r in results]
        alpha_spent = [r.alpha_spent_cumulative for r in results]
        effects = [r.effect_size for r in results]
        ci_lowers = [r.ci_lower for r in results]
        ci_uppers = [r.ci_upper for r in results]
        p_values = [r.p_value for r in results]
        info_fracs = [r.information_fraction for r in results]

        # ── Panel 1: Z-score trajectory vs boundaries ─────────────────────────
        ax1 = axes[0, 0]
        ax1.plot(n_totals, z_scores, "b-o", linewidth=2, markersize=5,
                 label="Z-statistic", zorder=5)
        ax1.plot(n_totals, eff_bounds, "r--", linewidth=2, label="Efficacy boundary (+)")
        ax1.plot(n_totals, [-b for b in eff_bounds], "r--", linewidth=2,
                 label="Efficacy boundary (−)")

        # Futility boundaries (only where finite)
        fut_pos = [abs(b) for b in fut_bounds]
        ax1.plot(n_totals, fut_pos, "g:", linewidth=2, label="Futility boundary (+)")
        ax1.plot(n_totals, [-b for b in fut_pos], "g:", linewidth=2,
                 label="Futility boundary (−)")

        # Shade stopping regions
        max_bound = max(max(eff_bounds) * 1.3, max(abs(z) for z in z_scores) * 1.3, 4)
        ax1.fill_between(n_totals, eff_bounds, [max_bound] * len(n_totals),
                         alpha=0.1, color="red", label="Stop for efficacy")
        ax1.fill_between(n_totals, [-max_bound] * len(n_totals),
                         [-b for b in eff_bounds],
                         alpha=0.1, color="red")
        ax1.fill_between(n_totals, [-b for b in fut_pos], fut_pos,
                         alpha=0.08, color="green", label="Stop for futility")

        # Mark stopping point
        for r in results:
            if r.decision in ("stop_efficacy", "stop_futility", "complete"):
                color = "red" if r.decision == "stop_efficacy" else (
                    "green" if r.decision == "stop_futility" else "blue"
                )
                ax1.axvline(r.n_total, color=color, linestyle="-.", linewidth=2,
                            alpha=0.7, label=f"Stopped: {r.decision}")
                break

        ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax1.set_xlabel("Total Observations", fontsize=11)
        ax1.set_ylabel("Z-Statistic", fontsize=11)
        ax1.set_title("Test Statistic vs Stopping Boundaries", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=7, loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-max_bound, max_bound)

        # ── Panel 2: Cumulative alpha spent ───────────────────────────────────
        ax2 = axes[0, 1]
        ax2.plot(info_fracs, alpha_spent, "purple", linewidth=2, marker="s",
                 markersize=5, label="Alpha spent (cumulative)")
        ax2.axhline(self.alpha, color="red", linestyle="--", linewidth=2,
                    label=f"Total α = {self.alpha}")

        # Reference spending curves
        t_range = np.linspace(0.01, 1.0, 200)
        ax2.plot(t_range,
                 [alpha_spend_obrien_fleming(t, self.alpha) for t in t_range],
                 "r:", linewidth=1.5, alpha=0.6, label="O'Brien-Fleming")
        ax2.plot(t_range,
                 [alpha_spend_pocock(t, self.alpha) for t in t_range],
                 "b:", linewidth=1.5, alpha=0.6, label="Pocock")

        ax2.set_xlabel("Information Fraction (t)", fontsize=11)
        ax2.set_ylabel("Cumulative Alpha Spent", fontsize=11)
        ax2.set_title("Alpha Spending Over Time", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, self.alpha * 1.1)