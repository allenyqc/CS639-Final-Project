```python
"""
Sequential Experiment Monitoring Module with Interim Analyses

Implements group sequential testing with alpha spending functions for
A/B experiment monitoring, including efficacy and futility stopping rules.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Optional plotting dependency
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available; plotting disabled.", ImportWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Enumerations and data classes
# ---------------------------------------------------------------------------

class StoppingDecision(Enum):
    CONTINUE = "continue"
    STOP_EFFICACY = "stop_efficacy"
    STOP_FUTILITY = "stop_futility"
    COMPLETED = "completed"


class AlphaSpendingFunction(Enum):
    """Supported alpha-spending functions."""
    OBRIEN_FLEMING = "obrien_fleming"   # O'Brien-Fleming (conservative early)
    POCOCK = "pocock"                   # Pocock (constant boundary)
    HWANG_SHIH_DECANI = "hwang_shih_decani"  # Hwang-Shih-DeCani family


@dataclass
class InterimResult:
    """Result of a single interim analysis."""
    analysis_index: int
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
    efficacy_boundary: float    # z-score threshold for stopping for efficacy
    futility_boundary: float    # z-score threshold for stopping for futility
    decision: StoppingDecision
    information_fraction: float  # t = n_current / n_max


@dataclass
class MonitoringConfig:
    """Configuration for sequential monitoring."""
    max_n: int = 10_000
    interim_every: int = 1_000
    alpha: float = 0.05                 # overall one-sided alpha
    beta: float = 0.20                  # overall beta (power = 1 - beta)
    alpha_spending: AlphaSpendingFunction = AlphaSpendingFunction.OBRIEN_FLEMING
    futility_spending: AlphaSpendingFunction = AlphaSpendingFunction.OBRIEN_FLEMING
    gamma: float = -4.0                 # HSD gamma parameter (if used)
    min_n_per_arm: int = 50             # minimum before any interim
    two_sided: bool = False             # one-sided by default
    confidence_level: float = 0.95


@dataclass
class MonitoringHistory:
    """Accumulated history of all interim analyses."""
    interim_results: list[InterimResult] = field(default_factory=list)
    final_decision: StoppingDecision = StoppingDecision.CONTINUE
    stopping_analysis: Optional[int] = None
    total_observations_used: int = 0
    config: Optional[MonitoringConfig] = None


# ---------------------------------------------------------------------------
# Alpha-spending functions
# ---------------------------------------------------------------------------

def _alpha_spent_obrien_fleming(t: float, alpha: float) -> float:
    """
    O'Brien-Fleming alpha-spending function.
    alpha(t) = 2 * (1 - Phi(z_{alpha/2} / sqrt(t)))
    """
    if t <= 0:
        return 0.0
    z_alpha = stats.norm.ppf(1.0 - alpha)
    return 2.0 * (1.0 - stats.norm.cdf(z_alpha / np.sqrt(t)))


def _alpha_spent_pocock(t: float, alpha: float) -> float:
    """Pocock alpha-spending: alpha(t) = alpha * ln(1 + (e-1)*t)."""
    if t <= 0:
        return 0.0
    return alpha * np.log(1.0 + (np.e - 1.0) * t)


def _alpha_spent_hsd(t: float, alpha: float, gamma: float = -4.0) -> float:
    """Hwang-Shih-DeCani alpha-spending function."""
    if t <= 0:
        return 0.0
    if abs(gamma) < 1e-10:
        return alpha * t
    return alpha * (1.0 - np.exp(-gamma * t)) / (1.0 - np.exp(-gamma))


def compute_alpha_spent(
    t: float,
    alpha: float,
    spending_fn: AlphaSpendingFunction,
    gamma: float = -4.0,
) -> float:
    """Dispatch to the appropriate alpha-spending function."""
    if spending_fn == AlphaSpendingFunction.OBRIEN_FLEMING:
        return _alpha_spent_obrien_fleming(t, alpha)
    elif spending_fn == AlphaSpendingFunction.POCOCK:
        return _alpha_spent_pocock(t, alpha)
    elif spending_fn == AlphaSpendingFunction.HWANG_SHIH_DECANI:
        return _alpha_spent_hsd(t, alpha, gamma)
    else:
        raise ValueError(f"Unknown spending function: {spending_fn}")


# ---------------------------------------------------------------------------
# Boundary computation
# ---------------------------------------------------------------------------

def _efficacy_boundary_z(
    alpha_increment: float,
    two_sided: bool = False,
) -> float:
    """
    Convert incremental alpha spend to a z-score boundary.
    For one-sided: z = Phi^{-1}(1 - alpha_increment)
    For two-sided: z = Phi^{-1}(1 - alpha_increment/2)
    """
    if alpha_increment <= 0:
        return np.inf
    if two_sided:
        return stats.norm.ppf(1.0 - alpha_increment / 2.0)
    return stats.norm.ppf(1.0 - alpha_increment)


def _futility_boundary_z(
    beta_increment: float,
) -> float:
    """
    Futility boundary (non-binding): stop if |z| < futility_z.
    Uses beta-spending analogously.
    """
    if beta_increment <= 0:
        return -np.inf
    return stats.norm.ppf(beta_increment)


# ---------------------------------------------------------------------------
# Core statistical computations
# ---------------------------------------------------------------------------

def _compute_z_score_and_ci(
    n_treat: int,
    n_ctrl: int,
    successes_treat: int,
    successes_ctrl: int,
    confidence_level: float = 0.95,
) -> tuple[float, float, float, float, float, float]:
    """
    Compute z-score, p-value, and CI for difference in proportions.

    Returns
    -------
    p_treat, p_ctrl, effect_size, z_score, p_value, ci_lower, ci_upper
    """
    p_treat = successes_treat / n_treat if n_treat > 0 else 0.0
    p_ctrl = successes_ctrl / n_ctrl if n_ctrl > 0 else 0.0
    effect = p_treat - p_ctrl

    # Pooled proportion for z-test
    p_pool = (successes_treat + successes_ctrl) / (n_treat + n_ctrl)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_treat + 1 / n_ctrl))

    if se_pool < 1e-12:
        z = 0.0
        p_val = 1.0
    else:
        z = effect / se_pool
        p_val = 1.0 - stats.norm.cdf(z)  # one-sided

    # Confidence interval (unpooled SE)
    se_unpooled = np.sqrt(
        p_treat * (1 - p_treat) / n_treat + p_ctrl * (1 - p_ctrl) / n_ctrl
    )
    z_ci = stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0)
    ci_lower = effect - z_ci * se_unpooled
    ci_upper = effect + z_ci * se_unpooled

    return p_treat, p_ctrl, effect, z, p_val, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Sequential monitor class
# ---------------------------------------------------------------------------

class SequentialExperimentMonitor:
    """
    Monitors a streaming A/B experiment with group-sequential stopping rules.

    Parameters
    ----------
    config : MonitoringConfig
        Experiment configuration.

    Usage
    -----
    >>> monitor = SequentialExperimentMonitor(config)
    >>> for batch in data_stream:
    ...     decision = monitor.ingest(batch)
    ...     if decision != StoppingDecision.CONTINUE:
    ...         break
    >>> history = monitor.get_history()
    >>> monitor.plot()
    """

    def __init__(self, config: Optional[MonitoringConfig] = None) -> None:
        self._cfg = config or MonitoringConfig()
        self._history = MonitoringHistory(config=self._cfg)
        self._buffer: list[dict] = []
        self._n_analyses_done: int = 0
        self._alpha_spent_so_far: float = 0.0
        self._beta_spent_so_far: float = 0.0
        self._stopped: bool = False
        self._total_n: int = 0

        # Accumulators (treatment=1, control=0)
        self._counts: dict[int, dict[str, int]] = {
            0: {"n": 0, "successes": 0},
            1: {"n": 0, "successes": 0},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, data: pd.DataFrame) -> StoppingDecision:
        """
        Ingest a new batch of observations and run interim analyses as needed.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns:
              - 'arm'     : int, 1 = treatment, 0 = control
              - 'outcome' : int/float, 1 = success, 0 = failure

        Returns
        -------
        StoppingDecision
        """
        if self._stopped:
            logger.warning("Experiment already stopped; ignoring new data.")
            return self._history.final_decision

        self._validate_data(data)
        self._buffer.append(data)
        self._total_n += len(data)

        # Update running accumulators
        for arm in [0, 1]:
            arm_data = data[data["arm"] == arm]
            self._counts[arm]["n"] += len(arm_data)
            self._counts[arm]["successes"] += int(arm_data["outcome"].sum())

        # Check if we should run an interim analysis
        n_analyses_expected = self._total_n // self._cfg.interim_every
        while self._n_analyses_done < n_analyses_expected:
            decision = self._run_interim_analysis()
            if decision in (StoppingDecision.STOP_EFFICACY, StoppingDecision.STOP_FUTILITY):
                self._stopped = True
                self._history.final_decision = decision
                self._history.stopping_analysis = self._n_analyses_done
                self._history.total_observations_used = self._total_n
                return decision

        # Check if we've reached max_n
        if self._total_n >= self._cfg.max_n:
            decision = self._run_final_analysis()
            self._stopped = True
            self._history.final_decision = decision
            self._history.total_observations_used = self._total_n
            return decision

        return StoppingDecision.CONTINUE

    def get_history(self) -> MonitoringHistory:
        """Return the full monitoring history."""
        return self._history

    def get_summary_df(self) -> pd.DataFrame:
        """Return interim results as a tidy DataFrame."""
        if not self._history.interim_results:
            return pd.DataFrame()
        records = []
        for r in self._history.interim_results:
            records.append({
                "analysis_index": r.analysis_index,
                "n_total": r.n_total,
                "n_treatment": r.n_treatment,
                "n_control": r.n_control,
                "p_treatment": r.p_treatment,
                "p_control": r.p_control,
                "effect_size": r.effect_size,
                "z_score": r.z_score,
                "p_value": r.p_value,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
                "alpha_spent_cumulative": r.alpha_spent_cumulative,
                "efficacy_boundary": r.efficacy_boundary,
                "futility_boundary": r.futility_boundary,
                "decision": r.decision.value,
                "information_fraction": r.information_fraction,
            })
        return pd.DataFrame(records)

    def plot(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Optional[object]:
        """
        Produce a monitoring plot: z-score trajectory vs stopping boundaries.

        Parameters
        ----------
        save_path : str, optional
            File path to save the figure (e.g., 'monitoring.png').
        show : bool
            Whether to call plt.show().

        Returns
        -------
        matplotlib Figure or None
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib is not installed; cannot produce plot.")
            return None

        df = self.get_summary_df()
        if df.empty:
            logger.warning("No interim results to plot.")
            return None

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # --- Top panel: z-score trajectory ---
        ax1 = axes[0]
        ax1.plot(
            df["information_fraction"],
            df["z_score"],
            marker="o",
            color="steelblue",
            linewidth=2,
            label="Test statistic (z)",
        )
        ax1.plot(
            df["information_fraction"],
            df["efficacy_boundary"],
            linestyle="--",
            color="red",
            linewidth=1.5,
            label="Efficacy boundary",
        )
        ax1.plot(
            df["information_fraction"],
            df["futility_boundary"],
            linestyle=":",
            color="orange",
            linewidth=1.5,
            label="Futility boundary",
        )
        ax1.axhline(0, color="gray", linewidth=0.8, linestyle="-")
        ax1.fill_between(
            df["information_fraction"],
            df["futility_boundary"],
            df["efficacy_boundary"],
            alpha=0.08,
            color="green",
            label="Continue region",
        )

        # Mark stopping point if applicable
        if self._history.stopping_analysis is not None:
            stop_idx = self._history.stopping_analysis - 1
            if 0 <= stop_idx < len(df):
                stop_row = df.iloc[stop_idx]
                ax1.axvline(
                    stop_row["information_fraction"],
                    color="purple",
                    linewidth=2,
                    linestyle="-.",
                    label=f"Stopped: {self._history.final_decision.value}",
                )

        ax1.set_ylabel("Z-score", fontsize=12)
        ax1.set_title("Sequential Experiment Monitoring", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # --- Bottom panel: cumulative alpha spent ---
        ax2 = axes[1]
        ax2.plot(
            df["information_fraction"],
            df["alpha_spent_cumulative"],
            marker="s",
            color="darkgreen",
            linewidth=2,
            label="Cumulative α spent",
        )
        ax2.axhline(
            self._cfg.alpha,
            color="red",
            linewidth=1.5,
            linestyle="--",
            label=f"Total α = {self._cfg.alpha}",
        )
        ax2.set_xlabel("Information fraction (t = n / N_max)", fontsize=12)
        ax2.set_ylabel("Cumulative α spent", fontsize=12)
        ax2.set_title("Alpha Spending", fontsize=12)
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, self._cfg.alpha * 1.2)

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Plot saved to %s", save_path)

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_data(self, data: pd.DataFrame) -> None:
        required = {"arm", "outcome"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")
        invalid_arms = set(data["arm"].unique()) - {0, 1}
        if invalid_arms:
            raise ValueError(f"'arm' column must contain only 0 or 1; found {invalid_arms}")

    def _run_interim_analysis(self) -> StoppingDecision:
        """Run one interim analysis and record the result."""
        self._n_analyses_done += 1
        n_treat = self._counts[1]["n"]
        n_ctrl = self._counts[0]["n"]
        n_total = n_treat + n_ctrl

        # Skip if not enough data per arm
        if n_treat < self._cfg.min_n_per_arm or n_ctrl < self._cfg.min_n_per_arm:
            logger.debug(
                "Analysis %d skipped: insufficient per-arm data (%d, %d).",
                self._n_analyses_done, n_treat, n_ctrl,
            )
            return StoppingDecision.CONTINUE

        t = min(n_total / self._cfg.max_n, 1.0)  # information fraction

        # Compute cumulative alpha spent up to t
        alpha_cum = compute_alpha_spent(
            t, self._cfg.alpha, self._cfg.alpha_spending, self._cfg.gamma
        )
        alpha_increment = max(alpha_cum - self._alpha_spent_so_far, 0.0)
        self._alpha_spent_so_far = alpha_cum

        # Compute cumulative beta spent up to t (for futility)
        beta_cum = compute_alpha_spent(
            t, self._cfg.beta, self._cfg.futility_spending, self._cfg.gamma
        )
        beta_increment = max(beta_cum - self._beta_spent_so_far, 0.0)
        self._beta_spent_so_far = beta_cum

        # Boundaries
        eff_boundary = _efficacy_boundary_z(alpha_increment, self._cfg.two_sided)
        fut_boundary = _futility_boundary_z(beta_increment)

        # Test statistic
        p_treat, p_ctrl, effect, z, p_val, ci_lo, ci_hi = _compute_z_score_and_ci(
            n_treat,
            n_ctrl,
            self._counts[1]["successes"],
            self._counts[0]["successes"],
            self._cfg.confidence_level,
        )

        # Stopping decision
        if z >= eff_boundary:
            decision = StoppingDecision.STOP_EFFICACY
        elif z <= fut_boundary:
            decision = StoppingDecision.STOP_FUTILITY
        else:
            decision = StoppingDecision.CONTINUE

        result = InterimResult(
            analysis_index=self._n_analyses_done,
            n_total=n_total,
            n_treatment=n_treat,
            n_control=n_ctrl