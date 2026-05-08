```python
"""
Statistical Power Analysis Module for ML Experiments

This module performs statistical power analysis for experiments comparing two ML models.
It provides analytical power calculations, Monte Carlo simulations, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentParams:
    """Parameters for the planned ML experiment."""
    effect_size: float          # Cohen's d (standardised mean difference)
    alpha: float = 0.05         # significance level
    n_test_samples: int = 1000  # number of test samples per iteration
    target_power: float = 0.80  # desired statistical power
    n_simulations: int = 1000   # Monte Carlo iterations
    two_tailed: bool = True     # two-tailed vs one-tailed test
    random_seed: Optional[int] = 42


@dataclass
class PowerAnalysisResult:
    """Results from the power analysis."""
    required_sample_size: int
    analytical_power: float
    empirical_power: float
    effect_size: float
    alpha: float
    n_simulations: int
    simulation_details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Statistical Power Analysis Results",
            "=" * 60,
            f"  Effect size (Cohen's d)  : {self.effect_size:.4f}",
            f"  Significance level (α)   : {self.alpha:.4f}",
            f"  Required sample size     : {self.required_sample_size}",
            f"  Analytical power         : {self.analytical_power:.4f}  "
            f"({self.analytical_power * 100:.1f}%)",
            f"  Empirical power          : {self.empirical_power:.4f}  "
            f"({self.empirical_power * 100:.1f}%)",
            f"  Power discrepancy        : "
            f"{abs(self.analytical_power - self.empirical_power):.4f}",
            f"  Monte Carlo iterations   : {self.n_simulations:,}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core analytical functions
# ─────────────────────────────────────────────────────────────────────────────

def _analytical_power(n: float, effect_size: float, alpha: float,
                      two_tailed: bool = True) -> float:
    """
    Compute analytical power for a two-sample t-test.

    Parameters
    ----------
    n           : sample size per group
    effect_size : Cohen's d
    alpha       : significance level
    two_tailed  : whether the test is two-tailed

    Returns
    -------
    power : float in [0, 1]
    """
    df = 2 * n - 2
    nc = effect_size * np.sqrt(n / 2)          # non-centrality parameter

    if two_tailed:
        t_crit = stats.t.ppf(1 - alpha / 2, df)
    else:
        t_crit = stats.t.ppf(1 - alpha, df)

    # Power = P(|T| > t_crit | H1) using non-central t distribution
    nct = stats.nct(df=df, nc=nc)
    if two_tailed:
        power = nct.sf(t_crit) + nct.cdf(-t_crit)
    else:
        power = nct.sf(t_crit)

    return float(np.clip(power, 0.0, 1.0))


def compute_required_sample_size(effect_size: float,
                                 alpha: float = 0.05,
                                 target_power: float = 0.80,
                                 two_tailed: bool = True,
                                 n_min: int = 2,
                                 n_max: int = 100_000) -> int:
    """
    Find the minimum sample size (per group) that achieves *target_power*.

    Uses Brent's root-finding method on the analytical power function.

    Returns
    -------
    n : int – required sample size per group
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive.")

    def objective(n: float) -> float:
        return _analytical_power(n, effect_size, alpha, two_tailed) - target_power

    # Quick boundary checks
    if objective(n_max) < 0:
        raise ValueError(
            f"Cannot achieve {target_power:.0%} power even with n={n_max:,}. "
            "Consider a larger effect size."
        )
    if objective(n_min) >= 0:
        return n_min

    n_opt = brentq(objective, n_min, n_max, xtol=0.5)
    return int(np.ceil(n_opt))


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo simulation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_model_errors(n: int, mean: float, std: float = 1.0,
                           rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate synthetic per-sample errors for one model.

    We model each model's per-sample absolute error as a normal distribution.
    Model A has mean=0 (baseline); Model B has mean=effect_size (better/worse).
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(loc=mean, scale=std, size=n)


def run_monte_carlo_simulation(n_samples: int,
                               effect_size: float,
                               alpha: float = 0.05,
                               n_simulations: int = 1000,
                               two_tailed: bool = True,
                               random_seed: Optional[int] = 42) -> dict:
    """
    Empirically estimate power via Monte Carlo simulation.

    In each iteration:
      1. Generate n_samples errors for Model A ~ N(0, 1)
      2. Generate n_samples errors for Model B ~ N(effect_size, 1)
      3. Run a paired t-test (or independent samples t-test) on the errors
      4. Record whether p < alpha

    Returns
    -------
    dict with keys:
        empirical_power   : float
        p_values          : np.ndarray of shape (n_simulations,)
        significant_count : int
        effect_sizes_obs  : np.ndarray – observed Cohen's d per iteration
    """
    rng = np.random.default_rng(random_seed)

    p_values = np.empty(n_simulations)
    effect_sizes_obs = np.empty(n_simulations)

    for i in range(n_simulations):
        errors_a = _generate_model_errors(n_samples, mean=0.0, rng=rng)
        errors_b = _generate_model_errors(n_samples, mean=effect_size, rng=rng)

        # Paired t-test (models evaluated on the same test samples)
        t_stat, p_val = stats.ttest_rel(errors_a, errors_b)

        if not two_tailed:
            p_val = p_val / 2  # one-tailed

        p_values[i] = p_val

        # Observed Cohen's d
        diff = errors_a - errors_b
        obs_d = diff.mean() / (diff.std(ddof=1) + 1e-12)
        effect_sizes_obs[i] = obs_d

    significant_count = int((p_values < alpha).sum())
    empirical_power = significant_count / n_simulations

    return {
        "empirical_power": empirical_power,
        "p_values": p_values,
        "significant_count": significant_count,
        "effect_sizes_obs": effect_sizes_obs,
        "n_simulations": n_simulations,
        "n_samples": n_samples,
        "alpha": alpha,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_power_curves(effect_sizes: list[float] = None,
                      alpha: float = 0.05,
                      two_tailed: bool = True,
                      n_range: tuple[int, int] = (5, 500),
                      target_power: float = 0.80,
                      simulation_result: Optional[dict] = None,
                      params: Optional[ExperimentParams] = None,
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot power curves as a function of sample size for different effect sizes.

    Optionally overlays empirical power from Monte Carlo simulation.

    Parameters
    ----------
    effect_sizes     : list of Cohen's d values to plot
    alpha            : significance level
    two_tailed       : two-tailed test flag
    n_range          : (min_n, max_n) range for x-axis
    target_power     : horizontal reference line
    simulation_result: output of run_monte_carlo_simulation (optional overlay)
    params           : ExperimentParams used in the analysis (for annotation)
    save_path        : if given, save figure to this path

    Returns
    -------
    fig : matplotlib Figure
    """
    if effect_sizes is None:
        effect_sizes = [0.2, 0.5, 0.8, 1.0, 1.5]

    n_values = np.arange(n_range[0], n_range[1] + 1)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(effect_sizes)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Statistical Power Analysis for ML Model Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── Left panel: Power curves ──────────────────────────────────────────────
    ax = axes[0]
    for d, color in zip(effect_sizes, colors):
        powers = [_analytical_power(n, d, alpha, two_tailed) for n in n_values]
        ax.plot(n_values, powers, label=f"d = {d:.1f}", color=color, linewidth=2)

    ax.axhline(target_power, color="red", linestyle="--", linewidth=1.5,
               label=f"Target power = {target_power:.0%}")
    ax.axhline(alpha, color="gray", linestyle=":", linewidth=1,
               label=f"α = {alpha}")

    # Mark required sample sizes on the target-power line
    for d, color in zip(effect_sizes, colors):
        try:
            n_req = compute_required_sample_size(d, alpha, target_power, two_tailed)
            if n_range[0] <= n_req <= n_range[1]:
                ax.axvline(n_req, color=color, linestyle=":", alpha=0.5)
                ax.scatter([n_req], [target_power], color=color, s=60, zorder=5)
        except ValueError:
            pass

    # Overlay empirical power if provided
    if simulation_result is not None and params is not None:
        ax.scatter(
            [simulation_result["n_samples"]],
            [simulation_result["empirical_power"]],
            marker="*", s=200, color="crimson", zorder=10,
            label=f"Empirical power (n={simulation_result['n_samples']}): "
                  f"{simulation_result['empirical_power']:.3f}"
        )

    ax.set_xlabel("Sample Size per Group", fontsize=12)
    ax.set_ylabel("Statistical Power", fontsize=12)
    ax.set_title("Power vs. Sample Size", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(n_range)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Right panel: Simulation diagnostics ──────────────────────────────────
    ax2 = axes[1]

    if simulation_result is not None:
        p_values = simulation_result["p_values"]
        n_sim = simulation_result["n_simulations"]

        # Histogram of p-values
        ax2.hist(p_values, bins=50, color="steelblue", edgecolor="white",
                 alpha=0.8, density=True, label="p-value distribution")
        ax2.axvline(alpha, color="red", linestyle="--", linewidth=2,
                    label=f"α = {alpha}")

        # Shade significant region
        ax2.axvspan(0, alpha, alpha=0.15, color="red", label="Rejection region")

        emp_power = simulation_result["empirical_power"]
        ax2.set_title(
            f"Monte Carlo Simulation Results\n"
            f"Empirical Power = {emp_power:.3f}  "
            f"({simulation_result['significant_count']:,}/{n_sim:,} significant)",
            fontsize=11
        )
        ax2.set_xlabel("p-value", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Annotate with key stats
        if params is not None:
            analytical_pwr = _analytical_power(
                simulation_result["n_samples"], params.effect_size,
                params.alpha, params.two_tailed
            )
            textstr = (
                f"Effect size d = {params.effect_size:.2f}\n"
                f"n per group   = {simulation_result['n_samples']}\n"
                f"Analytical    = {analytical_pwr:.3f}\n"
                f"Empirical     = {emp_power:.3f}\n"
                f"Δ             = {abs(analytical_pwr - emp_power):.4f}"
            )
            props = dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
            ax2.text(0.62, 0.97, textstr, transform=ax2.transAxes,
                     fontsize=9, verticalalignment="top", bbox=props)
    else:
        # Fallback: show power vs effect size for fixed n
        if params is not None:
            n_fixed = params.n_test_samples
        else:
            n_fixed = 100
        d_range = np.linspace(0.05, 2.0, 200)
        powers = [_analytical_power(n_fixed, d, alpha, two_tailed) for d in d_range]
        ax2.plot(d_range, powers, color="steelblue", linewidth=2)
        ax2.axhline(target_power, color="red", linestyle="--",
                    label=f"Target = {target_power:.0%}")
        ax2.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
        ax2.set_ylabel("Statistical Power", fontsize=12)
        ax2.set_title(f"Power vs. Effect Size (n = {n_fixed})", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_power_analysis(params: ExperimentParams,
                       plot: bool = True,
                       save_path: Optional[str] = None) -> PowerAnalysisResult:
    """
    Full power analysis pipeline.

    Steps
    -----
    1. Compute required sample size analytically.
    2. Compute analytical power at the required sample size.
    3. Run Monte Carlo simulation to empirically validate power.
    4. Compare analytical vs empirical power.
    5. Plot power curves (optional).
    6. Return PowerAnalysisResult.

    Parameters
    ----------
    params    : ExperimentParams
    plot      : whether to display/save the power curve figure
    save_path : path to save the figure (None = display only)

    Returns
    -------
    PowerAnalysisResult
    """
    print("\n" + "=" * 60)
    print("  Running Statistical Power Analysis")
    print("=" * 60)
    print(f"  Effect size (Cohen's d) : {params.effect_size}")
    print(f"  Significance level (α)  : {params.alpha}")
    print(f"  Target power            : {params.target_power:.0%}")
    print(f"  Test samples available  : {params.n_test_samples:,}")
    print(f"  Monte Carlo iterations  : {params.n_simulations:,}")
    print("=" * 60)

    # ── Step 1: Required sample size ─────────────────────────────────────────
    print("\n[1/4] Computing required sample size...")
    n_required = compute_required_sample_size(
        effect_size=params.effect_size,
        alpha=params.alpha,
        target_power=params.target_power,
        two_tailed=params.two_tailed,
    )
    print(f"      Required n per group : {n_required}")

    # ── Step 2: Analytical power at required n ────────────────────────────────
    print("\n[2/4] Computing analytical power...")
    analytical_power = _analytical_power(
        n=n_required,
        effect_size=params.effect_size,
        alpha=params.alpha,
        two_tailed=params.two_tailed,
    )
    print(f"      Analytical power     : {analytical_power:.4f} "
          f"({analytical_power * 100:.1f}%)")

    # Also report power at the user-specified n_test_samples
    power_at_user_n = _analytical_power(
        n=params.n_test_samples,
        effect_size=params.effect_size,
        alpha=params.alpha,
        two_tailed=params.two_tailed,
    )
    print(f"      Power at n={params.n_test_samples:,}       : "
          f"{power_at_user_n:.4f} ({power_at_user_n * 100:.1f}%)")

    # ── Step 3: Monte Carlo simulation ────────────────────────────────────────
    print(f"\n[3/4] Running Monte Carlo simulation "
          f"({params.n_simulations:,} iterations)...")
    sim_result = run_monte_carlo_simulation(
        n_samples=n_required,
        effect_size=params.effect_size,
        alpha=params.alpha,
        n_simulations=params.n_simulations,
        two_tailed=params.two_tailed,
        random_seed=params.random_seed,
    )
    empirical_power = sim_result["empirical_power"]
    print(f"      Empirical power      : {empirical_power:.4f} "
          f"({empirical_power * 100:.1f}%)")
    print(f"      Significant runs     : "
          f"{sim_result['significant_count']:,}/{params.n_simulations:,}")

    # ── Step 4: Compare analytical vs empirical ───────────────────────────────
    print("\n[4/4] Comparing analytical vs empirical power...")
    discrepancy = abs(analytical_power - empirical_power)
    print(f"      |Analytical