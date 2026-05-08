```python
"""
Sequential Experiment Monitoring Module with Interim Analyses.

Implements group sequential testing with alpha spending functions,
stopping rules for efficacy/futility, and comprehensive monitoring plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import warnings
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations & Data Classes
# ---------------------------------------------------------------------------

class StoppingDecision(Enum):
    CONTINUE = "continue"
    STOP_EFFICACY = "stop_efficacy"
    STOP_FUTILITY = "stop_futility"
    MAX_SAMPLE_REACHED = "max_sample_reached"


class AlphaSpendingFunction(Enum):
    OBRIEN_FLEMING = "obrien_fleming"   # conservative early, liberal late
    POCOCK = "pocock"                   # equal spending at each look
    HWANG_SHIH_DECANI = "hwang_shih_decani"  # flexible via gamma parameter


@dataclass
class InterimResult:
    look_number: int
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
    alpha_spent_cumulative: float
    efficacy_boundary: float    # z-score boundary for stopping for efficacy
    futility_boundary: float    # z-score boundary for stopping for futility
    decision: StoppingDecision
    information_fraction: float  # t = n_current / n_max


@dataclass
class MonitoringConfig:
    max_n: int = 10_000
    interim_every: int = 1_000
    alpha: float = 0.05                 # two-sided overall alpha
    beta: float = 0.20                  # type-II error (power = 1 - beta)
    spending_function: AlphaSpendingFunction = AlphaSpendingFunction.OBRIEN_FLEMING
    gamma: float = -4.0                 # for Hwang-Shih-DeCani only
    futility_spending: bool = True      # whether to apply futility boundary
    futility_beta_fraction: float = 0.5 # fraction of beta to spend for futility
    min_n_per_arm: int = 50             # minimum observations before first look
    two_sided: bool = True


# ---------------------------------------------------------------------------
# Alpha Spending Functions
# ---------------------------------------------------------------------------

def alpha_spent_obrien_fleming(t: float, alpha: float) -> float:
    """O'Brien-Fleming alpha spending: 2*(1 - Phi(z_alpha/2 / sqrt(t)))."""
    if t <= 0:
        return 0.0
    z = norm.ppf(1 - alpha / 2)
    return 2 * (1 - norm.cdf(z / np.sqrt(t)))


def alpha_spent_pocock(t: float, alpha: float) -> float:
    """Pocock alpha spending: alpha * ln(1 + (e-1)*t)."""
    if t <= 0:
        return 0.0
    return alpha * np.log(1 + (np.e - 1) * t)


def alpha_spent_hwang_shih_decani(t: float, alpha: float, gamma: float = -4.0) -> float:
    """Hwang-Shih-DeCani alpha spending function."""
    if t <= 0:
        return 0.0
    if abs(gamma) < 1e-10:
        return alpha * t
    return alpha * (1 - np.exp(-gamma * t)) / (1 - np.exp(-gamma))


def compute_alpha_spent(
    t: float,
    alpha: float,
    spending_fn: AlphaSpendingFunction,
    gamma: float = -4.0,
) -> float:
    """Dispatch to the correct alpha spending function."""
    if spending_fn == AlphaSpendingFunction.OBRIEN_FLEMING:
        return alpha_spent_obrien_fleming(t, alpha)
    elif spending_fn == AlphaSpendingFunction.POCOCK:
        return alpha_spent_pocock(t, alpha)
    elif spending_fn == AlphaSpendingFunction.HWANG_SHIH_DECANI:
        return alpha_spent_hwang_shih_decani(t, alpha, gamma)
    else:
        raise ValueError(f"Unknown spending function: {spending_fn}")


# ---------------------------------------------------------------------------
# Boundary Computation
# ---------------------------------------------------------------------------

def efficacy_boundary_from_alpha_increment(
    alpha_increment: float,
    two_sided: bool = True,
) -> float:
    """Convert incremental alpha to a z-score boundary."""
    if alpha_increment <= 0:
        return np.inf
    if two_sided:
        return norm.ppf(1 - alpha_increment / 2)
    return norm.ppf(1 - alpha_increment)


def futility_boundary_from_beta_increment(
    beta_increment: float,
    two_sided: bool = True,
) -> float:
    """
    Non-binding futility boundary (inner wedge).
    Returns the z-score below which we stop for futility.
    """
    if beta_increment <= 0:
        return -np.inf
    if two_sided:
        return norm.ppf(beta_increment / 2)
    return norm.ppf(beta_increment)


# ---------------------------------------------------------------------------
# Core Test Statistic
# ---------------------------------------------------------------------------

def compute_z_statistic(
    n_t: int, x_t: int,
    n_c: int, x_c: int,
) -> Tuple[float, float, float, float, float, float]:
    """
    Two-proportion z-test (unpooled for CI, pooled for test statistic).

    Returns
    -------
    z_stat, p_value, p_t, p_c, ci_lower, ci_upper
    """
    if n_t == 0 or n_c == 0:
        return 0.0, 1.0, 0.0, 0.0, -1.0, 1.0

    p_t = x_t / n_t
    p_c = x_c / n_c
    diff = p_t - p_c

    # Pooled proportion for test statistic
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))

    if se_pool < 1e-12:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = diff / se_pool
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    # Unpooled SE for confidence interval
    se_unpooled = np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z_95 = norm.ppf(0.975)
    ci_lower = diff - z_95 * se_unpooled
    ci_upper = diff + z_95 * se_unpooled

    return z_stat, p_value, p_t, p_c, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Sequential Monitor
# ---------------------------------------------------------------------------

class SequentialExperimentMonitor:
    """
    Monitors a streaming A/B experiment with group sequential testing.

    Parameters
    ----------
    config : MonitoringConfig
        Experiment configuration.

    Usage
    -----
    monitor = SequentialExperimentMonitor(config)
    for batch in data_stream:
        result = monitor.add_observations(batch)
        if result and result.decision != StoppingDecision.CONTINUE:
            break
    final = monitor.finalize()
    monitor.plot()
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self._reset()

    def _reset(self):
        self._data: List[Dict[str, Any]] = []   # raw observations
        self._history: List[InterimResult] = []
        self._look_count: int = 0
        self._stopped: bool = False
        self._final_decision: Optional[StoppingDecision] = None
        self._alpha_spent_prev: float = 0.0
        self._beta_spent_prev: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_observations(
        self,
        observations: pd.DataFrame,
    ) -> Optional[InterimResult]:
        """
        Ingest a batch of observations and perform an interim analysis
        if the cumulative count crosses the next interim threshold.

        Parameters
        ----------
        observations : pd.DataFrame
            Must contain columns: 'treatment' (0/1) and 'outcome' (0/1).

        Returns
        -------
        InterimResult if an interim analysis was triggered, else None.
        """
        if self._stopped:
            logger.warning("Experiment already stopped. Ignoring new observations.")
            return None

        required_cols = {"treatment", "outcome"}
        if not required_cols.issubset(observations.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # Validate values
        if not observations["treatment"].isin([0, 1]).all():
            raise ValueError("'treatment' column must contain only 0 or 1.")
        if not observations["outcome"].isin([0, 1]).all():
            raise ValueError("'outcome' column must contain only 0 or 1.")

        self._data.extend(observations.to_dict("records"))
        n_total = len(self._data)

        # Check if we've reached a new interim threshold
        next_look_n = (self._look_count + 1) * self.config.interim_every
        if n_total >= next_look_n and n_total >= self.config.min_n_per_arm * 2:
            return self._perform_interim_analysis()

        # Check max sample size
        if n_total >= self.config.max_n:
            return self._perform_interim_analysis(force=True)

        return None

    def finalize(self) -> Dict[str, Any]:
        """
        Perform a final analysis on all accumulated data (if not already stopped).

        Returns
        -------
        dict with keys: history, decision, final_result, summary_stats
        """
        if not self._stopped and len(self._data) > 0:
            result = self._perform_interim_analysis(force=True)
            if result and result.decision == StoppingDecision.CONTINUE:
                self._final_decision = StoppingDecision.MAX_SAMPLE_REACHED
                self._history[-1] = InterimResult(
                    **{**self._history[-1].__dict__,
                       "decision": StoppingDecision.MAX_SAMPLE_REACHED}
                )

        final_result = self._history[-1] if self._history else None
        summary = self._build_summary(final_result)
        return {
            "history": self._history,
            "decision": self._final_decision or StoppingDecision.MAX_SAMPLE_REACHED,
            "final_result": final_result,
            "summary_stats": summary,
        }

    def plot(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Produce a monitoring plot with:
          - Z-statistic trajectory vs efficacy/futility boundaries
          - Cumulative alpha spent
          - Effect size (proportion difference) with CI
          - Sample size over time
        """
        if not self._history:
            raise RuntimeError("No interim analyses performed yet.")

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, :])   # Z-statistic (full width)
        ax2 = fig.add_subplot(gs[1, 0])   # Alpha spent
        ax3 = fig.add_subplot(gs[1, 1])   # Effect size

        looks = [r.look_number for r in self._history]
        z_stats = [r.z_statistic for r in self._history]
        eff_bounds = [r.efficacy_boundary for r in self._history]
        fut_bounds = [r.futility_boundary for r in self._history]
        alpha_spent = [r.alpha_spent_cumulative for r in self._history]
        effects = [r.effect_size for r in self._history]
        ci_lowers = [r.ci_lower for r in self._history]
        ci_uppers = [r.ci_upper for r in self._history]
        n_totals = [r.n_total for r in self._history]

        # ---- Panel 1: Z-statistic vs boundaries ----
        ax1.plot(looks, z_stats, "b-o", linewidth=2, markersize=6, label="Z-statistic", zorder=5)
        ax1.plot(looks, eff_bounds, "r--", linewidth=1.5, label="Efficacy boundary (+)")
        ax1.plot(looks, [-b for b in eff_bounds], "r--", linewidth=1.5, label="Efficacy boundary (−)")
        if self.config.futility_spending:
            ax1.plot(looks, fut_bounds, "g:", linewidth=1.5, label="Futility boundary (+)")
            ax1.plot(looks, [-b for b in fut_bounds], "g:", linewidth=1.5, label="Futility boundary (−)")
        ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        # Shade stopping regions
        max_look = max(looks) if looks else 1
        ax1.fill_between(
            [min(looks), max_look],
            [max(eff_bounds)] * 2,
            [max(eff_bounds) * 1.5] * 2,
            alpha=0.1, color="red", label="Stop for efficacy",
        )
        ax1.fill_between(
            [min(looks), max_look],
            [-max(eff_bounds) * 1.5] * 2,
            [-max(eff_bounds)] * 2,
            alpha=0.1, color="red",
        )

        # Mark stopping point
        for r in self._history:
            if r.decision in (StoppingDecision.STOP_EFFICACY, StoppingDecision.STOP_FUTILITY):
                color = "red" if r.decision == StoppingDecision.STOP_EFFICACY else "orange"
                ax1.axvline(r.look_number, color=color, linewidth=2, linestyle="--",
                            label=f"Stopped: {r.decision.value}")
                break

        ax1.set_xlabel("Interim Look Number", fontsize=11)
        ax1.set_ylabel("Z-Statistic", fontsize=11)
        ax1.set_title("Sequential Monitoring: Z-Statistic vs Stopping Boundaries", fontsize=13, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Annotate n at each look
        for i, (look, z, n) in enumerate(zip(looks, z_stats, n_totals)):
            if i % max(1, len(looks) // 5) == 0:
                ax1.annotate(f"n={n}", (look, z), textcoords="offset points",
                             xytext=(0, 8), fontsize=7, ha="center", color="navy")

        # ---- Panel 2: Cumulative alpha spent ----
        ax2.plot(looks, alpha_spent, "purple", linewidth=2, marker="s", markersize=5)
        ax2.axhline(self.config.alpha, color="red", linestyle="--", linewidth=1.5,
                    label=f"Total α = {self.config.alpha}")
        ax2.fill_between(looks, 0, alpha_spent, alpha=0.2, color="purple")
        ax2.set_xlabel("Interim Look Number", fontsize=10)
        ax2.set_ylabel("Cumulative α Spent", fontsize=10)
        ax2.set_title("Alpha Spending", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, self.config.alpha * 1.1)

        # ---- Panel 3: Effect size with CI ----
        ax3.plot(looks, effects, "b-o", linewidth=2, markersize=5, label="Effect size (p_T − p_C)")
        ax3.fill_between(looks, ci_lowers, ci_uppers, alpha=0.2, color="blue", label="95% CI")
        ax3.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax3.set_xlabel("Interim Look Number", fontsize=10)
        ax3.set_ylabel("Proportion Difference", fontsize=10)
        ax3.set_title("Effect Size Trajectory", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Overall title
        decision_str = (self._final_decision or StoppingDecision.CONTINUE).value.replace("_", " ").title()
        fig.suptitle(
            f"Sequential Experiment Monitor  |  Decision: {decision_str}  |  "
            f"Spending: {self.config.spending_function.value}",
            fontsize=13, fontweight="bold", y=1.01,
        )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _perform_interim_analysis(self, force: bool = False) -> Optional[InterimResult]:
        """Run an interim analysis on all data collected so far."""
        df = pd.DataFrame(self._data)
        n_total = len(df)

        if n_total < self.config.min_n_per_arm * 2 and not force:
            return None

        self._look_count += 1
        t = min(n_total / self.config.max_n, 1.0)  # information fraction

        # Split by arm
        treat = df[df["treatment"] == 1]
        ctrl = df[df["treatment"] == 0]
        n_t, n_c = len(treat), len(ctrl)
        x_t = int(treat["outcome"].sum())
        x_c = int(ctrl["outcome"].sum())

        # Test statistic
        z_stat, p_value, p_t, p_c, ci_lower, ci_upper = compute_z_statistic(n_t, x_t, n_c, x_c)

        # Alpha spending
        alpha_spent_cum = compute_alpha_spent(
            t, self.config.alpha, self.config.spending_function, self.config.gamma
        )
        alpha_increment = max(alpha_spent_cum - self._alpha_spent_prev, 0.0)
        self._alpha_spent_prev = alpha_spent_cum

        # Efficacy boundary
        eff_boundary = efficacy_boundary_from_alpha_increment(
            alpha_increment, self.config.two_sided
        )

        # Futility boundary (beta spending)
        if self.config.futility_spending:
            beta_spent_cum = compute_alpha_spent(
                t,
                self.config.beta * self.config.futility_beta_fraction,
                self.config.spending_function,
                self.config.gamma,
            )
            beta_increment = max(beta_