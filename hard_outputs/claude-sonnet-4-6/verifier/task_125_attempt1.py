```python
"""
Statistical Power Analysis Module for ML Experiments
Compares two models using McNemar's test or proportion z-test
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower, zt_ind_solve_power
import warnings
from dataclasses import dataclass, field
from typing import Optional
import math

warnings.filterwarnings('ignore')


@dataclass
class PowerAnalysisResult:
    """Container for power analysis results"""
    required_sample_size: int
    analytical_power: float
    empirical_power: float
    effect_size: float
    significance_level: float
    n_test_samples: int
    n_simulations: int
    simulation_p_values: list = field(default_factory=list)
    simulation_significant: list = field(default_factory=list)
    power_curve_data: dict = field(default_factory=dict)

    def __str__(self):
        lines = [
            "=" * 60,
            "STATISTICAL POWER ANALYSIS RESULTS",
            "=" * 60,
            f"Effect Size (Cohen's h):    {self.effect_size:.4f}",
            f"Significance Level (alpha): {self.significance_level:.4f}",
            f"Number of Test Samples:     {self.n_test_samples}",
            f"Number of Simulations:      {self.n_simulations}",
            "-" * 60,
            f"Required Sample Size:       {self.required_sample_size}",
            f"Analytical Power:           {self.analytical_power:.4f} ({self.analytical_power*100:.1f}%)",
            f"Empirical Power:            {self.empirical_power:.4f} ({self.empirical_power*100:.1f}%)",
            f"Power Difference:           {abs(self.analytical_power - self.empirical_power):.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def cohens_h(p1: float, p2: float) -> float:
    """
    Compute Cohen's h effect size for two proportions.
    
    Args:
        p1: Proportion for model 1
        p2: Proportion for model 2
    
    Returns:
        Cohen's h effect size
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return abs(phi1 - phi2)


def analytical_power(effect_size: float, n_samples: int, alpha: float = 0.05) -> float:
    """
    Compute analytical power for a two-proportion z-test.
    
    Args:
        effect_size: Cohen's h effect size
        n_samples: Number of samples per group
        alpha: Significance level
    
    Returns:
        Statistical power
    """
    z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed test
    z_beta = effect_size * np.sqrt(n_samples) - z_alpha
    power = norm.cdf(z_beta)
    return float(np.clip(power, 0, 1))


def required_sample_size(effect_size: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """
    Compute required sample size for desired power.
    
    Args:
        effect_size: Cohen's h effect size
        alpha: Significance level
        power: Desired statistical power (default 0.80)
    
    Returns:
        Required sample size (per group)
    """
    if effect_size <= 0:
        raise ValueError("Effect size must be positive")
    
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n = ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))


def generate_synthetic_predictions(
    n_samples: int,
    accuracy_model1: float,
    accuracy_model2: float,
    random_state: Optional[int] = None
) -> tuple:
    """
    Generate synthetic binary predictions for two models.
    
    Args:
        n_samples: Number of test samples
        accuracy_model1: Expected accuracy of model 1
        accuracy_model2: Expected accuracy of model 2
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (predictions_model1, predictions_model2, true_labels)
    """
    rng = np.random.RandomState(random_state)
    
    # Generate true labels (balanced binary classification)
    true_labels = rng.randint(0, 2, size=n_samples)
    
    # Generate predictions based on accuracy
    # Model 1 predictions
    correct_mask1 = rng.random(n_samples) < accuracy_model1
    pred1 = np.where(correct_mask1, true_labels, 1 - true_labels)
    
    # Model 2 predictions
    correct_mask2 = rng.random(n_samples) < accuracy_model2
    pred2 = np.where(correct_mask2, true_labels, 1 - true_labels)
    
    return pred1, pred2, true_labels


def mcnemar_test(pred1: np.ndarray, pred2: np.ndarray, true_labels: np.ndarray) -> tuple:
    """
    Perform McNemar's test to compare two classifiers.
    
    Args:
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        true_labels: True labels
    
    Returns:
        Tuple of (test_statistic, p_value)
    """
    correct1 = (pred1 == true_labels).astype(int)
    correct2 = (pred2 == true_labels).astype(int)
    
    # Contingency table
    # b: model1 correct, model2 wrong
    # c: model1 wrong, model2 correct
    b = np.sum((correct1 == 1) & (correct2 == 0))
    c = np.sum((correct1 == 0) & (correct2 == 1))
    
    if b + c == 0:
        return 0.0, 1.0
    
    # McNemar's test statistic with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return float(chi2), float(p_value)


def proportion_z_test(pred1: np.ndarray, pred2: np.ndarray, true_labels: np.ndarray) -> tuple:
    """
    Perform two-proportion z-test to compare model accuracies.
    
    Args:
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        true_labels: True labels
    
    Returns:
        Tuple of (z_statistic, p_value)
    """
    n = len(true_labels)
    acc1 = np.mean(pred1 == true_labels)
    acc2 = np.mean(pred2 == true_labels)
    
    # Pooled proportion
    p_pool = (acc1 * n + acc2 * n) / (2 * n)
    
    if p_pool == 0 or p_pool == 1:
        return 0.0, 1.0
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n + 1/n))
    
    if se == 0:
        return 0.0, 1.0
    
    z = (acc1 - acc2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed
    
    return float(z), float(p_value)


def run_monte_carlo_simulation(
    effect_size: float,
    n_samples: int,
    alpha: float = 0.05,
    n_simulations: int = 1000,
    base_accuracy: float = 0.75,
    test_method: str = 'proportion_z',
    random_seed: int = 42
) -> tuple:
    """
    Run Monte Carlo simulation to empirically estimate statistical power.
    
    Args:
        effect_size: Cohen's h effect size
        n_samples: Number of test samples per simulation
        alpha: Significance level
        n_simulations: Number of Monte Carlo iterations
        base_accuracy: Baseline accuracy for model 1
        test_method: Statistical test to use ('proportion_z' or 'mcnemar')
        random_seed: Base random seed
    
    Returns:
        Tuple of (empirical_power, p_values, significant_flags)
    """
    # Convert Cohen's h to accuracy difference
    phi1 = 2 * np.arcsin(np.sqrt(base_accuracy))
    phi2 = phi1 + effect_size  # Add effect size
    accuracy_model2 = np.sin(phi2 / 2) ** 2
    accuracy_model2 = float(np.clip(accuracy_model2, 0.01, 0.99))
    
    p_values = []
    significant = []
    
    rng = np.random.RandomState(random_seed)
    
    for i in range(n_simulations):
        seed = rng.randint(0, 2**31 - 1)
        
        pred1, pred2, true_labels = generate_synthetic_predictions(
            n_samples=n_samples,
            accuracy_model1=base_accuracy,
            accuracy_model2=accuracy_model2,
            random_state=seed
        )
        
        if test_method == 'mcnemar':
            _, p_val = mcnemar_test(pred1, pred2, true_labels)
        else:
            _, p_val = proportion_z_test(pred1, pred2, true_labels)
        
        p_values.append(p_val)
        significant.append(p_val < alpha)
    
    empirical_power = float(np.mean(significant))
    
    return empirical_power, p_values, significant


def compute_power_curves(
    effect_sizes: list,
    sample_sizes: list,
    alpha: float = 0.05
) -> dict:
    """
    Compute power curves for different effect sizes and sample sizes.
    
    Args:
        effect_sizes: List of effect sizes to analyze
        sample_sizes: List of sample sizes to evaluate
        alpha: Significance level
    
    Returns:
        Dictionary with power values for each effect size
    """
    power_curves = {}
    
    for es in effect_sizes:
        powers = [analytical_power(es, n, alpha) for n in sample_sizes]
        power_curves[es] = powers
    
    return power_curves


def plot_power_curves(
    power_curves: dict,
    sample_sizes: list,
    alpha: float = 0.05,
    target_power: float = 0.80,
    current_n: Optional[int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot power curves as a function of sample size for different effect sizes.
    
    Args:
        power_curves: Dictionary of {effect_size: [power_values]}
        sample_sizes: List of sample sizes
        alpha: Significance level used
        target_power: Target power level to highlight
        current_n: Current/required sample size to mark on plot
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color palette
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(power_curves)))
    
    # --- Plot 1: Power Curves ---
    ax1 = axes[0]
    
    for (es, powers), color in zip(power_curves.items(), colors):
        label = f"h = {es:.2f}"
        ax1.plot(sample_sizes, powers, color=color, linewidth=2.5, label=label)
    
    # Target power line
    ax1.axhline(y=target_power, color='red', linestyle='--', linewidth=1.5,
                label=f'Target Power = {target_power:.0%}', alpha=0.8)
    
    # Alpha line
    ax1.axhline(y=alpha, color='gray', linestyle=':', linewidth=1.5,
                label=f'α = {alpha}', alpha=0.6)
    
    # Mark current sample size
    if current_n is not None:
        ax1.axvline(x=current_n, color='orange', linestyle='-.', linewidth=2,
                    label=f'Required n = {current_n}', alpha=0.9)
    
    ax1.set_xlabel('Sample Size (n)', fontsize=13)
    ax1.set_ylabel('Statistical Power', fontsize=13)
    ax1.set_title(f'Power Curves by Effect Size\n(α = {alpha})', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(min(sample_sizes), max(sample_sizes))
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(sample_sizes, target_power, 1.0, alpha=0.05, color='green',
                     label='_nolegend_')
    ax1.fill_between(sample_sizes, 0, target_power, alpha=0.05, color='red',
                     label='_nolegend_')
    
    # Annotate required sample sizes
    for (es, powers), color in zip(power_curves.items(), colors):
        # Find where power crosses target
        for j, (n, p) in enumerate(zip(sample_sizes, powers)):
            if p >= target_power:
                ax1.annotate(f'n={n}', xy=(n, target_power),
                             xytext=(n + max(sample_sizes)*0.02, target_power - 0.05),
                             fontsize=7, color=color, alpha=0.8)
                break
    
    # --- Plot 2: Required Sample Size vs Effect Size ---
    ax2 = axes[1]
    
    effect_range = np.linspace(0.05, 1.0, 200)
    req_sizes = [required_sample_size(es, alpha, target_power) for es in effect_range]
    
    ax2.plot(effect_range, req_sizes, color='steelblue', linewidth=2.5)
    ax2.fill_between(effect_range, req_sizes, alpha=0.15, color='steelblue')
    
    # Mark specific effect sizes from power curves
    for (es, _), color in zip(power_curves.items(), colors):
        n_req = required_sample_size(es, alpha, target_power)
        ax2.scatter([es], [n_req], color=color, s=100, zorder=5,
                    label=f'h={es:.2f}: n={n_req}')
        ax2.annotate(f'  n={n_req}', xy=(es, n_req), fontsize=9, color=color)
    
    ax2.set_xlabel("Effect Size (Cohen's h)", fontsize=13)
    ax2.set_ylabel('Required Sample Size', fontsize=13)
    ax2.set_title(f'Required Sample Size vs Effect Size\n(Power = {target_power:.0%}, α = {alpha})',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(0.05, 1.0)
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Power curves saved to: {save_path}")
    
    return fig


def plot_simulation_results(
    p_values: list,
    alpha: float = 0.05,
    analytical_power: float = None,
    empirical_power: float = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Monte Carlo simulation results.
    
    Args:
        p_values: List of p-values from simulations
        alpha: Significance level
        analytical_power: Analytical power estimate
        empirical_power: Empirical power from simulation
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Plot 1: P-value Distribution ---
    ax1 = axes[0]
    
    p_values_arr = np.array(p_values)
    n_sig = np.sum(p_values_arr < alpha)
    n_total = len(p_values_arr)
    
    ax1.hist(p_values_arr, bins=50, color='steelblue', edgecolor='white',
             alpha=0.8, density=True)
    ax1.axvline(x=alpha, color='red', linestyle='--', linewidth=2,
                label=f'α = {alpha}')
    
    # Shade significant region
    ax1.axvspan(0, alpha, alpha=0.2, color='red', label=f'Significant ({n_sig}/{n_total})')
    
    ax1.set_xlabel('P-value', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution of P-values\n(Monte Carlo Simulation)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Power Comparison ---
    ax2 = axes[1]
    
    methods = ['Analytical\nPower', 'Empirical\nPower\n(Monte Carlo)']
    values = [analytical_power or 0, empirical_power or 0]
    colors_bar = ['#2196F3', '#4CAF50']
    
    bars = ax2.bar(methods, values, color=colors_bar, width=0.5, edgecolor='white',
                   linewidth=1.5, alpha=0.85)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.4f}\n({val*100:.1f}%)',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.axhline(y=0.80, color='orange', linestyle='--', linewidth=2,
                label='Target Power (80%)', alpha=0.9)
    ax2.axhline(y=alpha, color='gray', linestyle=':', linewidth=1.5,
                label=f'α = {alpha}', alpha=0.7)
    
    ax2.set_ylabel('Power', fontsize=12)
    ax2.set_title('Analytical vs Empirical Power', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add difference annotation
    if analytical_power and empirical_power:
        diff = abs(analytical_power - empirical_power)
        ax2.annotate(f'Difference: {diff:.4f}',
                     xy=(0.5, 0.05), xycoords='axes fraction',
                     ha='center', fontsize=10, color='gray',
                     style='italic')
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt