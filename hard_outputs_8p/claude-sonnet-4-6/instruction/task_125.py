```python
"""
Statistical Power Analysis Module for ML Experiments.

This module provides tools to:
1. Compute required sample sizes for comparing two ML models
2. Run Monte Carlo simulations to empirically validate power estimates
3. Plot power curves as a function of sample size for different effect sizes
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm


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
    n_simulations: int
    simulation_details: dict = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for a planned ML experiment."""
    effect_size: float          # Cohen's d or equivalent standardised effect
    alpha: float = 0.05         # Significance level
    n_test_samples: int = 1000  # Number of test samples available
    target_power: float = 0.80  # Desired statistical power
    n_simulations: int = 1000   # Monte Carlo iterations
    two_tailed: bool = True     # Two-tailed test flag
    random_seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Analytical power functions
# ---------------------------------------------------------------------------

def _analytical_power(n: int, effect_size: float, alpha: float,
                       two_tailed: bool = True) -> float:
    """
    Compute analytical power for a two-sample z-test (or t-test approximation).

    Parameters
    ----------
    n : int
        Sample size *per group*.
    effect_size : float
        Standardised effect size (Cohen's d).
    alpha : float
        Significance level.
    two_tailed : bool
        Whether the test is two-tailed.

    Returns
    -------
    float
        Statistical power in [0, 1].
    """
    if n <= 0:
        return 0.0
    alpha_adj = alpha / 2 if two_tailed else alpha
    z_alpha = norm.ppf(1 - alpha_adj)
    # Non-centrality parameter for two independent samples of equal size
    ncp = effect_size * np.sqrt(n / 2)
    power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return float(np.clip(power, 0.0, 1.0))


def compute_required_sample_size(effect_size: float,
                                  alpha: float = 0.05,
                                  target_power: float = 0.80,
                                  two_tailed: bool = True,
                                  max_n: int = 100_000) -> int:
    """
    Binary-search for the minimum sample size (per group) that achieves
    *target_power* for the given effect size and significance level.

    Parameters
    ----------
    effect_size : float
        Standardised effect size (Cohen's d).
    alpha : float
        Significance level.
    target_power : float
        Desired power (default 0.80).
    two_tailed : bool
        Two-tailed test flag.
    max_n : int
        Upper bound for the search.

    Returns
    -------
    int
        Required sample size per group.
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    if not (0 < target_power < 1):
        raise ValueError("target_power must be in (0, 1).")

    lo, hi = 2, max_n
    while lo < hi:
        mid = (lo + hi) // 2
        if _analytical_power(mid, effect_size, alpha, two_tailed) >= target_power:
            hi = mid
        else:
            lo = mid + 1

    if _analytical_power(lo, effect_size, alpha, two_tailed) < target_power:
        warnings.warn(
            f"Could not reach target power {target_power:.2f} within max_n={max_n}. "
            "Consider increasing max_n or reducing target_power.",
            UserWarning,
            stacklevel=2,
        )
    return int(lo)


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def _generate_synthetic_predictions(n: int,
                                     effect_size: float,
                                     rng: np.random.Generator
                                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic error scores for two models.

    Model A errors ~ N(0, 1); Model B errors ~ N(-effect_size, 1)
    (lower error = better model B).

    Parameters
    ----------
    n : int
        Number of test samples.
    effect_size : float
        Standardised mean difference.
    rng : np.random.Generator
        NumPy random generator (seeded externally).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Paired error arrays for model A and model B.
    """
    errors_a = rng.standard_normal(n)
    errors_b = rng.standard_normal(n) - effect_size  # B is systematically better
    return errors_a, errors_b


def run_monte_carlo_simulation(n_per_group: int,
                                effect_size: float,
                                alpha: float = 0.05,
                                n_simulations: int = 1000,
                                two_tailed: bool = True,
                                random_seed: Optional[int] = 42
                                ) -> dict:
    """
    Empirically estimate statistical power via Monte Carlo simulation.

    In each iteration:
      - Generate synthetic predictions from two models with the given effect size.
      - Run a paired t-test (appropriate for comparing two models on the same
        test set, which is the standard ML comparison setup).
      - Record whether the result is significant.

    Parameters
    ----------
    n_per_group : int
        Number of test samples per iteration.
    effect_size : float
        Standardised effect size (Cohen's d).
    alpha : float
        Significance level.
    n_simulations : int
        Number of Monte Carlo iterations (>= 1000 recommended).
    two_tailed : bool
        Two-tailed test flag.
    random_seed : int or None
        Seed for reproducibility.

    Returns
    -------
    dict
        Simulation details including empirical power, p-values, and effect sizes.
    """
    if n_simulations < 1000:
        warnings.warn(
            "n_simulations < 1000 may yield unreliable empirical power estimates.",
            UserWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(random_seed)
    p_values: list[float] = []
    observed_effects: list[float] = []
    significant: list[bool] = []

    for _ in range(n_simulations):
        errors_a, errors_b = _generate_synthetic_predictions(
            n_per_group, effect_size, rng
        )
        # Paired t-test: compare per-sample errors of model A vs model B
        diff = errors_a - errors_b
        t_stat, p_val = stats.ttest_1samp(diff, popmean=0)

        if not two_tailed:
            # One-tailed: test whether model B is better (diff > 0)
            p_val = p_val / 2 if t_stat > 0 else 1 - p_val / 2

        p_values.append(float(p_val))
        obs_effect = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
        observed_effects.append(obs_effect)
        significant.append(bool(p_val < alpha))

    p_arr = np.array(p_values)
    eff_arr = np.array(observed_effects)
    empirical_power = float(np.mean(significant))

    # 95% Wilson confidence interval for empirical power
    n_sig = int(np.sum(significant))
    ci_lo, ci_hi = _wilson_ci(n_sig, n_simulations, confidence=0.95)

    return {
        "empirical_power": empirical_power,
        "power_ci_95": (ci_lo, ci_hi),
        "n_significant": n_sig,
        "n_simulations": n_simulations,
        "p_values": p_arr,
        "observed_effect_sizes": eff_arr,
        "mean_observed_effect": float(np.mean(eff_arr)),
        "std_observed_effect": float(np.std(eff_arr, ddof=1)),
        "median_p_value": float(np.median(p_arr)),
    }


def _wilson_ci(n_success: int, n_total: int, confidence: float = 0.95
               ) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    centre = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
    return (float(max(0.0, centre - margin)), float(min(1.0, centre + margin)))


# ---------------------------------------------------------------------------
# Power curve plotting
# ---------------------------------------------------------------------------

def plot_power_curves(effect_sizes: Optional[list[float]] = None,
                       alpha: float = 0.05,
                       two_tailed: bool = True,
                       n_range: Optional[tuple[int, int]] = None,
                       target_power: float = 0.80,
                       highlight_n: Optional[int] = None,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot analytical power curves as a function of sample size for multiple
    effect sizes.

    Parameters
    ----------
    effect_sizes : list[float] or None
        Effect sizes to plot. Defaults to [0.2, 0.5, 0.8] (small/medium/large).
    alpha : float
        Significance level.
    two_tailed : bool
        Two-tailed test flag.
    n_range : tuple[int, int] or None
        (min_n, max_n) for the x-axis. Auto-determined if None.
    target_power : float
        Horizontal reference line for target power.
    highlight_n : int or None
        Vertical line marking a specific sample size (e.g. required n).
    save_path : str or None
        If provided, save the figure to this path instead of displaying.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if effect_sizes is None:
        effect_sizes = [0.2, 0.5, 0.8]

    if n_range is None:
        # Auto-range: from 5 to the n needed for power=0.99 for the smallest effect
        min_es = min(effect_sizes)
        max_n_needed = compute_required_sample_size(
            min_es, alpha=alpha, target_power=0.99, two_tailed=two_tailed
        )
        n_range = (5, max(max_n_needed, 200))

    n_values = np.arange(n_range[0], n_range[1] + 1, max(1, (n_range[1] - n_range[0]) // 500))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(effect_sizes)))  # type: ignore[attr-defined]

    for es, color in zip(effect_sizes, colors):
        powers = [_analytical_power(int(n), es, alpha, two_tailed) for n in n_values]
        label_str = f"d = {es:.2f}"
        ax.plot(n_values, powers, label=label_str, color=color, linewidth=2)

        # Mark the required n for target_power
        req_n = compute_required_sample_size(
            es, alpha=alpha, target_power=target_power, two_tailed=two_tailed
        )
        req_power = _analytical_power(req_n, es, alpha, two_tailed)
        ax.scatter([req_n], [req_power], color=color, s=60, zorder=5)

    # Reference lines
    ax.axhline(target_power, color="red", linestyle="--", linewidth=1.5,
               label=f"Target power = {target_power:.0%}")
    ax.axhline(alpha, color="grey", linestyle=":", linewidth=1.2,
               label=f"α = {alpha}")

    if highlight_n is not None:
        ax.axvline(highlight_n, color="orange", linestyle="-.", linewidth=1.5,
                   label=f"Required n = {highlight_n}")

    ax.set_xlabel("Sample Size per Group", fontsize=13)
    ax.set_ylabel("Statistical Power", fontsize=13)
    title_tail = "Two-tailed" if two_tailed else "One-tailed"
    ax.set_title(f"Power Curves — {title_tail} Test (α = {alpha})", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(n_range[0], n_range[1])
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_power_analysis(config: ExperimentConfig,
                        plot: bool = True,
                        save_plot_path: Optional[str] = None
                        ) -> PowerAnalysisResult:
    """
    Perform a full statistical power analysis for an ML experiment.

    Steps
    -----
    1. Compute the required sample size analytically.
    2. Compute the analytical power at the available sample size.
    3. Run a Monte Carlo simulation to empirically validate the power estimate.
    4. Optionally plot power curves.
    5. Return a structured result object.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    plot : bool
        Whether to generate and display/save power curves.
    save_plot_path : str or None
        Path to save the plot (None = display interactively).

    Returns
    -------
    PowerAnalysisResult
    """
    # ------------------------------------------------------------------ #
    # 1. Analytical required sample size
    # ------------------------------------------------------------------ #
    required_n = compute_required_sample_size(
        effect_size=config.effect_size,
        alpha=config.alpha,
        target_power=config.target_power,
        two_tailed=config.two_tailed,
    )

    # ------------------------------------------------------------------ #
    # 2. Analytical power at the *available* sample size
    # ------------------------------------------------------------------ #
    analytical_power = _analytical_power(
        n=config.n_test_samples,
        effect_size=config.effect_size,
        alpha=config.alpha,
        two_tailed=config.two_tailed,
    )

    # ------------------------------------------------------------------ #
    # 3. Monte Carlo simulation at the available sample size
    # ------------------------------------------------------------------ #
    sim_details = run_monte_carlo_simulation(
        n_per_group=config.n_test_samples,
        effect_size=config.effect_size,
        alpha=config.alpha,
        n_simulations=config.n_simulations,
        two_tailed=config.two_tailed,
        random_seed=config.random_seed,
    )
    empirical_power = sim_details["empirical_power"]

    # ------------------------------------------------------------------ #
    # 4. Summary comparison
    # ------------------------------------------------------------------ #
    sim_details["analytical_power"] = analytical_power
    sim_details["power_discrepancy"] = abs(analytical_power - empirical_power)
    sim_details["required_sample_size"] = required_n
    sim_details["available_sample_size"] = config.n_test_samples
    sim_details["adequately_powered"] = config.n_test_samples >= required_n

    _print_summary(config, required_n, analytical_power, empirical_power, sim_details)

    # ------------------------------------------------------------------ #
    # 5. Power curves
    # ------------------------------------------------------------------ #
    if plot:
        effect_sizes_to_plot = sorted(
            set([0.2, 0.5, 0.8, config.effect_size])
        )
        plot_power_curves(
            effect_sizes=effect_sizes_to_plot,
            alpha=config.alpha,
            two_tailed=config.two_tailed,
            target_power=config.target_power,
            highlight_n=required_n,
            save_path=save_plot_path,
        )

    return PowerAnalysisResult(
        required_sample_size=required_n,
        analytical_power=analytical_power,
        empirical_power=empirical_power,
        effect_size=config.effect_size,
        alpha=config.alpha,
        n_simulations=config.n_simulations,
        simulation_details=sim_details,
    )


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def _print_summary(config: ExperimentConfig,
                    required_n: int,
                    analytical_power: float,
                    empirical_power: float,
                    sim_details: dict) -> None:
    ci_lo, ci_hi = sim_details["power_ci_95"]
    print("=" * 60)
    print("  Statistical Power Analysis — ML Experiment Comparison")
    print("=" * 60)
    print(f"  Effect size (Cohen's d)   : {config.effect_size:.3f}")
    print(f"  Significance level (α)    : {config.alpha:.3f}")
    print(f"  Target power              : {config.target_power:.0%}")
    print(f"  Test type                 : {'Two-tailed' if config.two_tailed else 'One-tailed'}")
    print("-" * 60)
    print(f"  Required sample size/grp  : {required_n:,}")
    print(f"  Available test samples    : {config.n_test_samples:,}")
    adequacy = "✓ Adequately powered" if sim_details["adequately_powered"] else "✗ Under-powered"
    print(f"  Power adequacy            : {adequacy}")
    print("-" * 60)
    print(f"  Analytical power          : {analytical_power:.4f}")
    print(f"  Empirical power (MC)      : {empirical_power:.4f}  "
          f"[95% CI: {ci_lo:.4f} – {ci_hi:.4f}]")
    print(f"  Power discrepancy         : {sim_details['power_discrepancy']:.4f}")
    print(f"  MC iterations             : {config.n_simulations:,}")
    print(f"  Significant iterations    : {sim_details['n_significant']:,}")
    print(f"  Mean observed effect size : {sim_details['mean_observed_effect']:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Convenience wrapper for quick usage
# ---------------------------------------------------------------------------

def analyze_experiment(effect_size: float = 0.5,
                        alpha: float = 0.05,
                        n_test_samples: int = 200,
                        target_power: float = 0.80,
                        n_simulations: int = 2000,
                        two_tailed: bool = True,
                        random_seed: Optional[int] = 42,
                        plot: bool = True,
                        save_plot_