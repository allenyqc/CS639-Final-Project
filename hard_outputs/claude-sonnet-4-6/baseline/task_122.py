```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
from typing import Tuple, Optional
import warnings


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUC-ROC score."""
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def jackknife_auc_differences(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray
) -> np.ndarray:
    """
    Compute jackknife estimates of AUC differences for BCa acceleration.
    Leave-one-out jackknife resampling.
    """
    n = len(y_true)
    jack_diffs = np.zeros(n)
    
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        y_true_jack = y_true[mask]
        y_pred_a_jack = y_pred_a[mask]
        y_pred_b_jack = y_pred_b[mask]
        
        auc_a = compute_auc(y_true_jack, y_pred_a_jack)
        auc_b = compute_auc(y_true_jack, y_pred_b_jack)
        jack_diffs[i] = auc_a - auc_b
    
    return jack_diffs


def bca_confidence_interval(
    bootstrap_diffs: np.ndarray,
    observed_diff: float,
    jackknife_diffs: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute BCa (Bias-Corrected and Accelerated) confidence interval.
    
    Parameters
    ----------
    bootstrap_diffs : array of bootstrap difference estimates
    observed_diff : observed difference on original data
    jackknife_diffs : jackknife estimates for acceleration computation
    alpha : significance level (default 0.05 for 95% CI)
    
    Returns
    -------
    (lower, upper) confidence interval bounds
    """
    n_boot = len(bootstrap_diffs)
    
    # Bias correction factor z0
    prop_less = np.mean(bootstrap_diffs < observed_diff)
    if prop_less == 0:
        prop_less = 1 / (2 * n_boot)
    elif prop_less == 1:
        prop_less = 1 - 1 / (2 * n_boot)
    
    z0 = stats.norm.ppf(prop_less)
    
    # Acceleration factor a (using jackknife)
    jack_mean = np.mean(jackknife_diffs)
    jack_deviations = jack_mean - jackknife_diffs
    
    numerator = np.sum(jack_deviations ** 3)
    denominator = 6 * (np.sum(jack_deviations ** 2) ** 1.5)
    
    if denominator == 0:
        a = 0.0
        warnings.warn("Acceleration factor denominator is zero; using a=0 (standard BC interval).")
    else:
        a = numerator / denominator
    
    # Adjusted quantiles
    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)
    
    def adjusted_quantile(z_alpha_val):
        num = z0 + z_alpha_val
        denom = 1 - a * (z0 + z_alpha_val)
        if denom == 0:
            return np.nan
        return stats.norm.cdf(z0 + num / denom)
    
    q_low = adjusted_quantile(z_alpha_low)
    q_high = adjusted_quantile(z_alpha_high)
    
    # Clamp quantiles to valid range
    q_low = np.clip(q_low, 0.001, 0.999)
    q_high = np.clip(q_high, 0.001, 0.999)
    
    ci_lower = np.quantile(bootstrap_diffs, q_low)
    ci_upper = np.quantile(bootstrap_diffs, q_high)
    
    return ci_lower, ci_upper


def paired_bootstrap_pvalue(
    bootstrap_diffs: np.ndarray,
    observed_diff: float
) -> float:
    """
    Compute two-sided p-value using paired bootstrap test.
    
    Under H0: true difference = 0, we shift the bootstrap distribution
    to be centered at 0 and compute the probability of observing a
    difference as extreme as the observed one.
    
    Parameters
    ----------
    bootstrap_diffs : bootstrap distribution of differences
    observed_diff : observed difference on original data
    
    Returns
    -------
    p-value (two-sided)
    """
    # Shift bootstrap distribution to be centered at 0 (under H0)
    shifted_diffs = bootstrap_diffs - np.mean(bootstrap_diffs)
    
    # Two-sided p-value
    p_value = np.mean(np.abs(shifted_diffs) >= np.abs(observed_diff))
    
    # Ensure p-value is not exactly 0
    n_boot = len(bootstrap_diffs)
    p_value = max(p_value, 1.0 / n_boot)
    
    return p_value


def plot_bootstrap_distribution(
    bootstrap_diffs: np.ndarray,
    observed_diff: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the bootstrap distribution of AUC differences.
    
    Parameters
    ----------
    bootstrap_diffs : bootstrap distribution of differences
    observed_diff : observed difference on original data
    ci_lower, ci_upper : 95% BCa confidence interval bounds
    p_value : computed p-value
    save_path : optional path to save the figure
    
    Returns
    -------
    matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Bootstrap Distribution of AUC Differences (Model A - Model B)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    # --- Plot 1: Histogram with CI and observed difference ---
    ax1 = axes[0]
    
    n_bins = min(50, int(np.sqrt(len(bootstrap_diffs))))
    counts, bin_edges, patches = ax1.hist(
        bootstrap_diffs, bins=n_bins, density=True,
        color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5,
        label='Bootstrap Distribution'
    )
    
    # Color the CI region
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if ci_lower <= left_edge <= ci_upper:
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.8)
    
    # Overlay KDE
    kde = stats.gaussian_kde(bootstrap_diffs)
    x_range = np.linspace(bootstrap_diffs.min(), bootstrap_diffs.max(), 300)
    ax1.plot(x_range, kde(x_range), 'navy', linewidth=2, label='KDE')
    
    # Vertical lines
    ax1.axvline(observed_diff, color='red', linewidth=2.5, linestyle='-',
                label=f'Observed Diff = {observed_diff:.4f}')
    ax1.axvline(ci_lower, color='green', linewidth=2, linestyle='--',
                label=f'95% BCa CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    ax1.axvline(ci_upper, color='green', linewidth=2, linestyle='--')
    ax1.axvline(0, color='gray', linewidth=1.5, linestyle=':', alpha=0.8,
                label='H₀: Difference = 0')
    
    ax1.set_xlabel('AUC Difference (A - B)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Bootstrap Distribution with BCa CI', fontsize=12)
    ax1.legend(fontsize=9, loc='upper left')
    
    # Add p-value annotation
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    color = "red" if p_value < 0.05 else "gray"
    ax1.text(0.98, 0.95, f'p-value = {p_value:.4f}\n({significance})',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             color=color)
    
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Cumulative Distribution ---
    ax2 = axes[1]
    
    sorted_diffs = np.sort(bootstrap_diffs)
    cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
    
    ax2.plot(sorted_diffs, cumulative, color='steelblue', linewidth=2,
             label='Empirical CDF')
    
    # Shade CI region
    ci_mask = (sorted_diffs >= ci_lower) & (sorted_diffs <= ci_upper)
    if ci_mask.any():
        ax2.fill_betweenx(
            cumulative[ci_mask],
            sorted_diffs[ci_mask],
            alpha=0.3, color='green', label='95% BCa CI Region'
        )
    
    ax2.axvline(observed_diff, color='red', linewidth=2.5, linestyle='-',
                label=f'Observed = {observed_diff:.4f}')
    ax2.axvline(ci_lower, color='green', linewidth=2, linestyle='--',
                label=f'CI Lower = {ci_lower:.4f}')
    ax2.axvline(ci_upper, color='darkgreen', linewidth=2, linestyle='--',
                label=f'CI Upper = {ci_upper:.4f}')
    ax2.axvline(0, color='gray', linewidth=1.5, linestyle=':', alpha=0.8,
                label='H₀: Diff = 0')
    
    ax2.axhline(0.025, color='orange', linewidth=1, linestyle=':', alpha=0.7)
    ax2.axhline(0.975, color='orange', linewidth=1, linestyle=':', alpha=0.7,
                label='2.5% / 97.5% levels')
    
    ax2.set_xlabel('AUC Difference (A - B)', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Empirical CDF of Bootstrap Differences', fontsize=12)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def compare_models_bootstrap(
    y_true: np.ndarray,
    y_pred_model_a: np.ndarray,
    y_pred_model_b: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: Optional[int] = 42,
    plot: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """
    Compare two ML models using bootstrap resampling with BCa confidence intervals.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_pred_model_a : array-like of shape (n_samples,)
        Predicted probabilities from Model A.
    y_pred_model_b : array-like of shape (n_samples,)
        Predicted probabilities from Model B.
    n_bootstrap : int, default=2000
        Number of bootstrap iterations (minimum 1000 recommended).
    alpha : float, default=0.05
        Significance level for confidence interval (0.05 → 95% CI).
    random_state : int or None, default=42
        Random seed for reproducibility.
    plot : bool, default=True
        Whether to generate and display the bootstrap distribution plot.
    save_path : str or None, default=None
        Path to save the plot (e.g., 'bootstrap_comparison.png').
    verbose : bool, default=True
        Whether to print a summary of results.
    
    Returns
    -------
    dict with keys:
        - 'observed_diff': float, AUC_A - AUC_B on original test set
        - 'auc_a': float, AUC of Model A
        - 'auc_b': float, AUC of Model B
        - 'ci_lower': float, lower bound of BCa 95% CI
        - 'ci_upper': float, upper bound of BCa 95% CI
        - 'p_value': float, two-sided p-value from paired bootstrap test
        - 'bootstrap_diffs': np.ndarray, full bootstrap distribution
        - 'n_bootstrap': int, number of bootstrap iterations used
        - 'is_significant': bool, whether p_value < alpha
        - 'figure': matplotlib Figure or None
    """
    # Input validation
    y_true = np.asarray(y_true)
    y_pred_model_a = np.asarray(y_pred_model_a, dtype=float)
    y_pred_model_b = np.asarray(y_pred_model_b, dtype=float)
    
    if not (len(y_true) == len(y_pred_model_a) == len(y_pred_model_b)):
        raise ValueError("y_true, y_pred_model_a, and y_pred_model_b must have the same length.")
    
    if n_bootstrap < 1000:
        warnings.warn(f"n_bootstrap={n_bootstrap} is less than 1000. Results may be unreliable.")
    
    n_samples = len(y_true)
    rng = np.random.default_rng(random_state)
    
    # Step 1: Compute observed performance difference
    auc_a = compute_auc(y_true, y_pred_model_a)
    auc_b = compute_auc(y_true, y_pred_model_b)
    observed_diff = auc_a - auc_b
    
    if verbose:
        print(f"{'='*60}")
        print(f"  Model Comparison via Bootstrap Resampling")
        print(f"{'='*60}")
        print(f"  Original Test Set:")
        print(f"    AUC (Model A): {auc_a:.4f}")
        print(f"    AUC (Model B): {auc_b:.4f}")
        print(f"    Observed Difference (A - B): {observed_diff:.4f}")
        print(f"  Running {n_bootstrap} bootstrap iterations...")
    
    # Step 2: Bootstrap resampling
    bootstrap_diffs = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        
        y_true_boot = y_true[indices]
        y_pred_a_boot = y_pred_model_a[indices]
        y_pred_b_boot = y_pred_model_b[indices]
        
        # Skip if only one class present
        if len(np.unique(y_true_boot)) < 2:
            bootstrap_diffs[i] = observed_diff  # fallback
            continue
        
        auc_a_boot = compute_auc(y_true_boot, y_pred_a_boot)
        auc_b_boot = compute_auc(y_true_boot, y_pred_b_boot)
        bootstrap_diffs[i] = auc_a_boot - auc_b_boot
    
    # Remove NaN values if any
    valid_mask = ~np.isnan(bootstrap_diffs)
    if not valid_mask.all():
        warnings.warn(f"{(~valid_mask).sum()} bootstrap iterations produced NaN; these are excluded.")
        bootstrap_diffs = bootstrap_diffs[valid_mask]
    
    # Step 3: Compute jackknife estimates for BCa acceleration
    if verbose:
        print(f"  Computing jackknife estimates for BCa CI...")
    
    jackknife_diffs = jackknife_auc_differences(y_true, y_pred_model_a, y_pred_model_b)
    
    # Step 4: BCa confidence interval
    ci_lower, ci_upper = bca_confidence_interval(
        bootstrap_diffs, observed_diff, jackknife_diffs, alpha=alpha
    )
    
    # Step 5: Paired bootstrap p-value
    p_value = paired_bootstrap_pvalue(bootstrap_diffs, observed_diff)
    
    is_significant = p_value < alpha
    
    if verbose:
        print(f"\n  Results:")
        print(f"    Bootstrap Mean Difference: {np.mean(bootstrap_diffs):.4f}")
        print(f"    Bootstrap Std Difference:  {np.std(bootstrap_diffs):.4f}")
        print(f"    BCa {int((1-alpha)*100)}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"    P-value (two-sided):       {p_value:.4f}")
        print(f"    Statistically Significant: {is_significant} (α={alpha})")
        print(f"{'='*60}")
    
    # Step 6: Plot
    fig = None
    if plot:
        fig = plot_bootstrap_distribution(
            bootstrap_diffs, observed_diff, ci_lower, ci_upper, p_value, save_path
        )
        plt.show()
    
    return {
        'observed_diff': observed_diff,
        'auc_a': auc_a,
        'auc_b': auc_b,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'bootstrap_diffs': bootstrap_diffs,
        'n_bootstrap': len(bootstrap_diffs),
        'is_significant': is_significant,
        'figure': fig
    }


# ─── Demo / Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model A: Logistic Regression
    model_a = LogisticRegression(random_state=42, max_iter=1000)
    model_a.fit(X_train_scaled, y_train)
    y_pred_a = model