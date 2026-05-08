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
    Compute jackknife estimates of AUC difference for BCa interval.
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
    jackknife_diffs : jackknife estimates for acceleration
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
        warnings.warn("Acceleration factor is zero; falling back to bias-corrected CI.")
    else:
        a = numerator / denominator
    
    # Adjusted quantiles
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)
    
    def adjusted_quantile(z_alpha_val):
        numerator_adj = z0 + z_alpha_val
        denominator_adj = 1 - a * (z0 + z_alpha_val)
        if denominator_adj == 0:
            return z_alpha_val
        return stats.norm.cdf(z0 + numerator_adj / denominator_adj)
    
    alpha_lo = adjusted_quantile(z_alpha_lo)
    alpha_hi = adjusted_quantile(z_alpha_hi)
    
    # Clamp to valid range
    alpha_lo = np.clip(alpha_lo, 0.001, 0.999)
    alpha_hi = np.clip(alpha_hi, 0.001, 0.999)
    
    lower = np.quantile(bootstrap_diffs, alpha_lo)
    upper = np.quantile(bootstrap_diffs, alpha_hi)
    
    return lower, upper


def bootstrap_model_comparison(
    y_true: np.ndarray,
    y_pred_model_a: np.ndarray,
    y_pred_model_b: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = 42,
    plot: bool = True,
    plot_title: str = "Bootstrap Distribution of AUC Differences (Model A - Model B)",
    save_path: Optional[str] = None
) -> dict:
    """
    Compare two ML models using bootstrap resampling.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_pred_model_a : array-like of shape (n_samples,)
        Predicted probabilities from Model A.
    y_pred_model_b : array-like of shape (n_samples,)
        Predicted probabilities from Model B.
    n_bootstrap : int, default=1000
        Number of bootstrap iterations.
    alpha : float, default=0.05
        Significance level for confidence interval and hypothesis test.
    random_state : int or None, default=42
        Random seed for reproducibility.
    plot : bool, default=True
        Whether to plot the bootstrap distribution.
    plot_title : str
        Title for the plot.
    save_path : str or None
        If provided, save the plot to this path.
    
    Returns
    -------
    dict with keys:
        - 'observed_difference': float, AUC_A - AUC_B on original data
        - 'auc_model_a': float, AUC of Model A
        - 'auc_model_b': float, AUC of Model B
        - 'confidence_interval': tuple (lower, upper), BCa 95% CI
        - 'p_value': float, two-sided p-value from paired bootstrap test
        - 'bootstrap_distribution': np.ndarray of bootstrap differences
        - 'n_bootstrap': int, number of bootstrap iterations
        - 'significant': bool, whether difference is significant at alpha level
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_model_a)
    y_pred_b = np.asarray(y_pred_model_b)
    
    # Validate inputs
    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("All input arrays must have the same length.")
    if len(y_true) < 10:
        raise ValueError("Need at least 10 samples for meaningful bootstrap analysis.")
    if n_bootstrap < 1000:
        warnings.warn(f"n_bootstrap={n_bootstrap} is less than 1000. Results may be unreliable.")
    
    n = len(y_true)
    rng = np.random.default_rng(random_state)
    
    # -------------------------------------------------------------------------
    # Step 1: Observed performance on original test set
    # -------------------------------------------------------------------------
    auc_a_obs = compute_auc(y_true, y_pred_a)
    auc_b_obs = compute_auc(y_true, y_pred_b)
    observed_diff = auc_a_obs - auc_b_obs
    
    print(f"Original Test Set Results:")
    print(f"  AUC Model A: {auc_a_obs:.4f}")
    print(f"  AUC Model B: {auc_b_obs:.4f}")
    print(f"  Observed Difference (A - B): {observed_diff:.4f}")
    
    # -------------------------------------------------------------------------
    # Step 2: Bootstrap resampling
    # -------------------------------------------------------------------------
    bootstrap_diffs = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        y_true_boot = y_true[indices]
        y_pred_a_boot = y_pred_a[indices]
        y_pred_b_boot = y_pred_b[indices]
        
        # Skip if only one class present
        if len(np.unique(y_true_boot)) < 2:
            bootstrap_diffs[i] = np.nan
            continue
        
        auc_a_boot = compute_auc(y_true_boot, y_pred_a_boot)
        auc_b_boot = compute_auc(y_true_boot, y_pred_b_boot)
        bootstrap_diffs[i] = auc_a_boot - auc_b_boot
    
    # Remove NaN values
    valid_mask = ~np.isnan(bootstrap_diffs)
    n_valid = valid_mask.sum()
    if n_valid < n_bootstrap * 0.9:
        warnings.warn(f"Only {n_valid}/{n_bootstrap} bootstrap samples were valid.")
    bootstrap_diffs_clean = bootstrap_diffs[valid_mask]
    
    # -------------------------------------------------------------------------
    # Step 3: BCa Confidence Interval
    # -------------------------------------------------------------------------
    jackknife_diffs = jackknife_auc_differences(y_true, y_pred_a, y_pred_b)
    ci_lower, ci_upper = bca_confidence_interval(
        bootstrap_diffs_clean, observed_diff, jackknife_diffs, alpha=alpha
    )
    
    print(f"\nBCa {int((1 - alpha) * 100)}% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # -------------------------------------------------------------------------
    # Step 4: Paired Bootstrap Test (p-value)
    # -------------------------------------------------------------------------
    # Under H0: true difference = 0
    # Shift bootstrap distribution to be centered at 0
    bootstrap_diffs_shifted = bootstrap_diffs_clean - np.mean(bootstrap_diffs_clean)
    
    # Two-sided p-value: proportion of shifted bootstrap diffs as extreme as observed
    p_value = np.mean(np.abs(bootstrap_diffs_shifted) >= np.abs(observed_diff))
    
    # Ensure p-value is not exactly 0
    if p_value == 0:
        p_value = 1 / n_valid
    
    significant = p_value < alpha
    
    print(f"P-value (two-sided paired bootstrap test): {p_value:.4f}")
    print(f"Statistically significant at alpha={alpha}: {significant}")
    
    # -------------------------------------------------------------------------
    # Step 5: Plot bootstrap distribution
    # -------------------------------------------------------------------------
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram of bootstrap differences
        n_bins = min(50, int(np.sqrt(n_valid)))
        counts, bin_edges, patches = ax.hist(
            bootstrap_diffs_clean,
            bins=n_bins,
            color='steelblue',
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5,
            label='Bootstrap Differences'
        )
        
        # Color the CI region
        ci_mask = (bin_edges[:-1] >= ci_lower) & (bin_edges[1:] <= ci_upper)
        for patch, in_ci in zip(patches, ci_mask):
            if in_ci:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.9)
        
        # Vertical lines
        ax.axvline(
            observed_diff, color='red', linewidth=2.5,
            linestyle='-', label=f'Observed Diff = {observed_diff:.4f}'
        )
        ax.axvline(
            ci_lower, color='darkorange', linewidth=2,
            linestyle='--', label=f'BCa 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]'
        )
        ax.axvline(ci_upper, color='darkorange', linewidth=2, linestyle='--')
        ax.axvline(0, color='black', linewidth=1.5, linestyle=':', alpha=0.7, label='No Difference (0)')
        
        # Annotations
        ax.set_xlabel('AUC Difference (Model A - Model B)', fontsize=13)
        ax.set_ylabel('Frequency', fontsize=13)
        ax.set_title(plot_title, fontsize=14, fontweight='bold')
        
        # Stats box
        stats_text = (
            f"AUC A: {auc_a_obs:.4f}\n"
            f"AUC B: {auc_b_obs:.4f}\n"
            f"Observed Δ: {observed_diff:.4f}\n"
            f"BCa 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
            f"p-value: {p_value:.4f}\n"
            f"Significant: {significant}\n"
            f"n_bootstrap: {n_valid}"
        )
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        ax.text(
            0.02, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props
        )
        
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
    
    # -------------------------------------------------------------------------
    # Step 6: Return results
    # -------------------------------------------------------------------------
    results = {
        'observed_difference': observed_diff,
        'auc_model_a': auc_a_obs,
        'auc_model_b': auc_b_obs,
        'confidence_interval': (ci_lower, ci_upper),
        'p_value': p_value,
        'bootstrap_distribution': bootstrap_diffs_clean,
        'n_bootstrap': n_valid,
        'significant': significant,
        'alpha': alpha,
        'bootstrap_mean': np.mean(bootstrap_diffs_clean),
        'bootstrap_std': np.std(bootstrap_diffs_clean),
    }
    
    return results


# =============================================================================
# Example usage / demo
# =============================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("Bootstrap Model Comparison Demo")
    print("=" * 60)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train two models
    model_a = RandomForestClassifier(n_estimators=100, random_state=42)
    model_b = LogisticRegression(max_iter=1000, random_state=42)
    
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)
    
    y_pred_a = model_a.predict_proba(X_test)[:, 1]
    y_pred_b = model_b.predict_proba(X_test)[:, 1]
    
    # Run comparison
    results = bootstrap_model_comparison(
        y_true=y_test,
        y_pred_model_a=y_pred_a,
        y_pred_model_b=y_pred_b,
        n_bootstrap=2000,
        alpha=0.05,
        random_state=42,
        plot=True,
        plot_title="Bootstrap Comparison: Random Forest vs Logistic Regression"
    )
    
    print("\n" + "=" * 60)
    print("Summary of Results:")
    print("=" * 60)
    for key, value in results.items():
        if key == 'bootstrap_distribution':
            print(f"  {key}: array of shape {value.shape}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")