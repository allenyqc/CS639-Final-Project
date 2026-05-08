```python
"""
Statistical Power Analysis Module for ML Experiments

This module provides tools for:
1. Analytical power analysis for comparing two ML models
2. Monte Carlo simulation to empirically validate power estimates
3. Power curve visualization across different effect sizes and sample sizes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm, ttest_ind, wilcoxon
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PowerAnalysisResult:
    """Container for power analysis results."""
    required_sample_size: int
    analytical_power: float
    empirical_power: float
    effect_size: float
    alpha: float
    n_test_samples: int
    simulation_details: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "Statistical Power Analysis Results",
            "=" * 60,
            f"Effect Size (Cohen's d):   {self.effect_size:.4f}",
            f"Significance Level (α):    {self.alpha:.4f}",
            f"Test Samples Provided:     {self.n_test_samples}",
            f"Required Sample Size:      {self.required_sample_size}",
            f"Analytical Power:          {self.analytical_power:.4f}",
            f"Empirical Power:           {self.empirical_power:.4f}",
            f"Power Discrepancy:         {abs(self.analytical_power - self.empirical_power):.4f}",
            "=" * 60,
        ]
        if self.simulation_details:
            lines.append("Simulation Details:")
            for k, v in self.simulation_details.items():
                lines.append(f"  {k}: {v}")
            lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analytical power functions
# ---------------------------------------------------------------------------

def cohens_d_to_power(d: float, n: int, alpha: float = 0.05, two_tailed: bool = True) -> float:
    """
    Compute statistical power for a two-sample t-test given Cohen's d.

    Parameters
    ----------
    d : float
        Cohen's d effect size.
    n : int
        Sample size per group.
    alpha : float
        Significance level.
    two_tailed : bool
        Whether to use a two-tailed test.

    Returns
    -------
    float
        Statistical power (0–1).
    """
    if n <= 1:
        return 0.0
    # Non-centrality parameter for two-sample t-test
    ncp = d * np.sqrt(n / 2)
    df = 2 * n - 2
    if two_tailed:
        t_crit = stats.t.ppf(1 - alpha / 2, df)
    else:
        t_crit = stats.t.ppf(1 - alpha, df)

    # Power = P(|T| > t_crit | ncp)
    power = (
        1
        - stats.nct.cdf(t_crit, df, ncp)
        + stats.nct.cdf(-t_crit, df, ncp)
    )
    return float(np.clip(power, 0.0, 1.0))


def required_sample_size(
    d: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    two_tailed: bool = True,
    max_n: int = 100_000,
) -> int:
    """
    Find the minimum sample size per group to achieve *target_power*.

    Uses a simple linear search (fast enough for typical ranges).

    Parameters
    ----------
    d : float
        Cohen's d effect size.
    alpha : float
        Significance level.
    target_power : float
        Desired power (default 0.80).
    two_tailed : bool
        Whether to use a two-tailed test.
    max_n : int
        Upper bound for the search.

    Returns
    -------
    int
        Required sample size per group.
    """
    if abs(d) < 1e-9:
        raise ValueError("Effect size must be non-zero to compute required sample size.")

    for n in range(2, max_n + 1):
        if cohens_d_to_power(abs(d), n, alpha, two_tailed) >= target_power:
            return n

    warnings.warn(
        f"Could not reach target power {target_power} within max_n={max_n}. "
        "Returning max_n.",
        RuntimeWarning,
    )
    return max_n


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def _generate_model_errors(
    n: int,
    mean_error: float,
    std_error: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate synthetic per-sample absolute errors for one model.

    Errors are drawn from a half-normal distribution (always positive)
    shifted by *mean_error*.  This mimics realistic model error distributions.
    """
    raw = rng.normal(loc=mean_error, scale=std_error, size=n)
    return raw


def run_monte_carlo_simulation(
    effect_size: float,
    n_samples: int,
    alpha: float = 0.05,
    n_iterations: int = 1_000,
    test: str = "ttest",
    seed: Optional[int] = 42,
    apply_bonferroni: bool = False,
    n_comparisons: int = 1,
) -> Dict:
    """
    Empirically estimate statistical power via Monte Carlo simulation.

    In each iteration:
      - Generate synthetic prediction errors for Model A (baseline) and
        Model B (improved by *effect_size* standard deviations).
      - Run the specified statistical test.
      - Record whether the result is significant.

    Parameters
    ----------
    effect_size : float
        Cohen's d — the standardised mean difference between models.
    n_samples : int
        Number of test samples per iteration.
    alpha : float
        Significance level (per test).
    n_iterations : int
        Number of Monte Carlo iterations (≥ 1000 recommended).
    test : str
        Statistical test to use: "ttest" (default) or "wilcoxon".
    seed : int, optional
        Random seed for reproducibility.
    apply_bonferroni : bool
        If True, apply Bonferroni correction using *n_comparisons*.
    n_comparisons : int
        Number of simultaneous comparisons (for Bonferroni correction).

    Returns
    -------
    dict
        Simulation details including empirical power, p-values, etc.
    """
    if n_iterations < 1_000:
        warnings.warn(
            "n_iterations < 1000 may yield unreliable empirical power estimates.",
            UserWarning,
        )

    rng = np.random.default_rng(seed)

    # Effective alpha after multiple-comparison correction
    effective_alpha = alpha / n_comparisons if apply_bonferroni else alpha

    # Model parameters
    # Model A: mean=0, std=1  (baseline)
    # Model B: mean=effect_size, std=1  (better model has higher mean score)
    std = 1.0
    mean_a = 0.0
    mean_b = effect_size * std  # Cohen's d = (mu_b - mu_a) / pooled_std

    p_values: List[float] = []
    significant: List[bool] = []

    for _ in range(n_iterations):
        scores_a = rng.normal(loc=mean_a, scale=std, size=n_samples)
        scores_b = rng.normal(loc=mean_b, scale=std, size=n_samples)

        if test == "ttest":
            _, p_val = ttest_ind(scores_b, scores_a, equal_var=True)
        elif test == "wilcoxon":
            # Wilcoxon signed-rank test requires paired samples
            _, p_val = wilcoxon(scores_b - scores_a, alternative="greater")
        else:
            raise ValueError(f"Unknown test '{test}'. Choose 'ttest' or 'wilcoxon'.")

        p_values.append(float(p_val))
        significant.append(p_val < effective_alpha)

    p_values_arr = np.array(p_values)
    empirical_power = float(np.mean(significant))

    return {
        "empirical_power": empirical_power,
        "n_iterations": n_iterations,
        "n_significant": int(np.sum(significant)),
        "mean_p_value": float(np.mean(p_values_arr)),
        "median_p_value": float(np.median(p_values_arr)),
        "std_p_value": float(np.std(p_values_arr)),
        "effective_alpha": effective_alpha,
        "test_used": test,
        "bonferroni_applied": apply_bonferroni,
        "n_comparisons": n_comparisons,
        "p_values": p_values_arr,  # kept for diagnostics
    }


# ---------------------------------------------------------------------------
# Power curve plotting
# ---------------------------------------------------------------------------

def plot_power_curves(
    effect_sizes: Optional[List[float]] = None,
    alpha: float = 0.05,
    target_power: float = 0.80,
    n_range: Optional[Tuple[int, int]] = None,
    n_points: int = 100,
    highlight_n: Optional[int] = None,
    simulation_details: Optional[Dict] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot power curves as a function of sample size for multiple effect sizes.

    Parameters
    ----------
    effect_sizes : list of float, optional
        Effect sizes to plot (default: [0.2, 0.5, 0.8, 1.0]).
    alpha : float
        Significance level.
    target_power : float
        Target power line to draw.
    n_range : tuple (min_n, max_n), optional
        Range of sample sizes to plot.
    n_points : int
        Number of sample-size points on the x-axis.
    highlight_n : int, optional
        Highlight a specific sample size (e.g., the required n).
    simulation_details : dict, optional
        If provided, overlay the empirical power estimate.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if effect_sizes is None:
        effect_sizes = [0.2, 0.5, 0.8, 1.0]

    if n_range is None:
        # Auto-range: from 5 to the n needed for d=0.2 at 95% power
        max_n = max(required_sample_size(0.2, alpha, 0.95), 200)
        n_range = (5, max_n)

    ns = np.linspace(n_range[0], n_range[1], n_points, dtype=int)
    ns = np.unique(ns)  # remove duplicates from int conversion

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ---- Panel 1: Power curves ----
    ax1 = fig.add_subplot(gs[0, :])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(effect_sizes)))

    for d, color in zip(effect_sizes, colors):
        powers = [cohens_d_to_power(abs(d), int(n), alpha) for n in ns]
        ax1.plot(ns, powers, label=f"d = {d:.1f}", color=color, linewidth=2)

    ax1.axhline(target_power, color="red", linestyle="--", linewidth=1.5,
                label=f"Target power = {target_power:.0%}")
    ax1.axhline(alpha, color="gray", linestyle=":", linewidth=1.2,
                label=f"α = {alpha:.2f}")

    if highlight_n is not None:
        ax1.axvline(highlight_n, color="orange", linestyle="-.", linewidth=1.5,
                    label=f"Required n = {highlight_n}")

    if simulation_details is not None:
        emp_power = simulation_details.get("empirical_power")
        n_sim = simulation_details.get("n_samples_used")
        if emp_power is not None and n_sim is not None:
            ax1.scatter([n_sim], [emp_power], color="red", zorder=5, s=100,
                        label=f"Empirical power = {emp_power:.3f}")

    ax1.set_xlabel("Sample Size per Group", fontsize=12)
    ax1.set_ylabel("Statistical Power", fontsize=12)
    ax1.set_title(f"Power Curves (α = {alpha:.3f})", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Required n vs effect size ----
    ax2 = fig.add_subplot(gs[1, 0])
    d_range = np.linspace(0.1, 2.0, 80)
    req_ns = [required_sample_size(d, alpha, target_power) for d in d_range]
    ax2.plot(d_range, req_ns, color="steelblue", linewidth=2)
    ax2.set_xlabel("Effect Size (Cohen's d)", fontsize=11)
    ax2.set_ylabel("Required Sample Size", fontsize=11)
    ax2.set_title(f"Required n for {target_power:.0%} Power", fontsize=11, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: P-value distribution from simulation ----
    ax3 = fig.add_subplot(gs[1, 1])
    if simulation_details is not None and "p_values" in simulation_details:
        p_vals = simulation_details["p_values"]
        ax3.hist(p_vals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax3.axvline(simulation_details.get("effective_alpha", alpha),
                    color="red", linestyle="--", linewidth=1.5,
                    label=f"α = {simulation_details.get('effective_alpha', alpha):.4f}")
        ax3.set_xlabel("p-value", fontsize=11)
        ax3.set_ylabel("Frequency", fontsize=11)
        ax3.set_title("P-value Distribution (Monte Carlo)", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No simulation data\navailable",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=12)
        ax3.set_title("P-value Distribution", fontsize=11, fontweight="bold")

    fig.suptitle("Statistical Power Analysis for ML Model Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info("Power curve figure saved to '%s'.", save_path)

    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_experiment_power(
    effect_size: float,
    alpha: float = 0.05,
    n_test_samples: int = 1_000,
    target_power: float = 0.80,
    n_mc_iterations: int = 2_000,
    test: str = "ttest",
    n_comparisons: int = 1,
    apply_bonferroni: bool = False,
    seed: Optional[int] = 42,
    plot: bool = True,
    save_plot: Optional[str] = None,
    effect_sizes_for_plot: Optional[List[float]] = None,
) -> PowerAnalysisResult:
    """
    Full power analysis pipeline for an ML experiment comparing two models.

    Parameters
    ----------
    effect_size : float
        Expected Cohen's d between the two models' performance distributions.
    alpha : float
        Significance level (Type I error rate).
    n_test_samples : int
        Number of test samples available for the comparison.
    target_power : float
        Desired statistical power (default 0.80).
    n_mc_iterations : int
        Number of Monte Carlo iterations (≥ 1000).
    test : str
        Statistical test: "ttest" or "wilcoxon".
    n_comparisons : int
        Number of simultaneous comparisons (for multiple-testing correction).
    apply_bonferroni : bool
        Apply Bonferroni correction if True.
    seed : int, optional
        Random seed for reproducibility.
    plot : bool
        Whether to generate and display power curves.
    save_plot : str, optional
        Path to save the power curve figure.
    effect_sizes_for_plot : list of float, optional
        Effect sizes to include in the power curve plot.

    Returns
    -------
    PowerAnalysisResult
        Dataclass containing all analysis results.

    Notes
    -----
    Best practices enforced:
    - Train/test split must be done BEFORE any preprocessing (caller's responsibility).
    - This module only analyses test-set predictions; it never touches training data.
    - Multiple-comparison correction (Bonferroni) is applied when n_comparisons > 1.
    - No credentials or secrets are used or required.
    """
    logger.info("Starting power analysis: d=%.3f, α=%.3f, n=%d", effect_size, alpha, n_test_samples)

    # ------------------------------------------------------------------
    # 1. Analytical power for the given test-set size
    # ------------------------------------------------------------------
    analytical_power_at_n = cohens_d_to_power(abs(effect_size), n_test_samples, alpha)
    logger.info("Analytical power at n=%d: %.4f", n_test_samples, analytical_power_at_n)

    # ------------------------------------------------------------------
    # 2. Required sample size for target power
    # ------------------------------------------------------------------
    req_n = required_sample_size(abs(effect_size), alpha, target_power)
    logger.info("Required sample size for %.0f%% power: %d", target_power * 100, req_n)

    if n_test_samples < req_n:
        warnings.warn(
            f"Available test samples ({n_test_samples}) < required ({req_n}) "
            f"for {target_power:.0%} power. "
            f"Analytical power at n={n_test_samples} is {analytical_power_at_n:.4f}.",
            UserWarning,
        )

    # ------------------------------------------------------------------
    # 3. Monte Carlo simulation
    # ------------------------------------------------------------------
    logger.info("Running Monte Carlo simulation (%d iterations)...", n_mc_iterations)
    sim = run_monte_carlo_simulation(
        effect_size=effect_size,
        n_samples=n_test_samples,
        alpha=alpha,
        n_iterations=n_mc_iterations,
        test=test,
        seed=seed,
        apply_bonferroni=apply_bonferroni,
        n_comparisons=n_comparisons,
    )
    empirical_power = sim["empirical_power"]
    sim["n_samples_used"] = n_test_samples  # store for plotting
    logger.info("Empirical power: %.4f", empirical_power)

    # ------------------------------------------------------------------
    # 4. Compare analytical vs empirical
    # ------------------------------------------------------------------
    discrepancy = abs(analytical_power_at_n - empirical_power)
    logger.info("Power discrepancy (analytical vs empirical): %.4f", discrepancy)

    #