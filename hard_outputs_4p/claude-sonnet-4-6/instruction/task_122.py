"""
bootstrap_model_comparison.py

A module for comparing two ML models using bootstrap resampling with BCa
confidence intervals and paired bootstrap hypothesis testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
from typing import Tuple, Dict, Optional
import warnings


# ---------------------------------------------------------------------------
# Core metric helper
# ---------------------------------------------------------------------------

def _auc_diff(y_true: np.ndarray,
              y_pred_a: np.ndarray,
              y_pred_b: np.ndarray,
              indices: Optional[np.ndarray] = None) -> float:
    """Return AUC_A - AUC_B for the given (optionally resampled) indices."""
    if indices is not None:
        y_true  = y_true[indices]
        y_pred_a = y_pred_a[indices]
        y_pred_b = y_pred_b[indices]

    # Guard against degenerate bootstrap samples (only one class present)
    try:
        auc_a = roc_auc_score(y_true, y_pred_a)
        auc_b = roc_auc_score(y_true, y_pred_b)
    except ValueError:
        return np.nan

    return auc_a - auc_b


# ---------------------------------------------------------------------------
# BCa confidence interval
# ---------------------------------------------------------------------------

def _bca_ci(observed: float,
            bootstrap_dist: np.ndarray,
            y_true: np.ndarray,
            y_pred_a: np.ndarray,
            y_pred_b: np.ndarray,
            alpha: float = 0.05) -> Tuple[float, float]:
    """
    Bias-corrected and accelerated (BCa) confidence interval.

    Parameters
    ----------
    observed      : point estimate on the original sample
    bootstrap_dist: 1-D array of bootstrap replicate statistics
    y_true        : ground-truth labels
    y_pred_a/b    : model predictions
    alpha         : significance level (default 0.05 → 95 % CI)

    Returns
    -------
    (lower, upper) confidence interval
    """
    valid = bootstrap_dist[~np.isnan(bootstrap_dist)]
    n_boot = len(valid)

    if n_boot == 0:
        raise ValueError("All bootstrap samples were degenerate (single class).")

    # --- Bias-correction factor z0 ---
    prop_less = np.mean(valid < observed)
    # Clamp to avoid infinite z0
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(prop_less)

    # --- Acceleration factor a (jackknife) ---
    n = len(y_true)
    jack_vals = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jack_vals[i] = _auc_diff(y_true[mask], y_pred_a[mask], y_pred_b[mask])

    # Replace NaN jackknife values with the observed statistic
    jack_vals = np.where(np.isnan(jack_vals), observed, jack_vals)

    jack_mean = np.mean(jack_vals)
    num   = np.sum((jack_mean - jack_vals) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack_vals) ** 2) ** 1.5)

    a = num / denom if denom != 0 else 0.0

    # --- Adjusted quantiles ---
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    def adjusted_quantile(z_alpha_side: float) -> float:
        num_   = z0 + z_alpha_side
        denom_ = 1.0 - a * (z0 + z_alpha_side)
        adj_z  = z0 + num_ / denom_
        return float(stats.norm.cdf(adj_z))

    q_lo = adjusted_quantile(z_alpha_lo)
    q_hi = adjusted_quantile(z_alpha_hi)

    # Clamp quantiles to [0, 1]
    q_lo = np.clip(q_lo, 0.0, 1.0)
    q_hi = np.clip(q_hi, 0.0, 1.0)

    lower = float(np.percentile(valid, 100 * q_lo))
    upper = float(np.percentile(valid, 100 * q_hi))
    return lower, upper


# ---------------------------------------------------------------------------
# Paired bootstrap p-value
# ---------------------------------------------------------------------------

def _paired_bootstrap_pvalue(observed: float,
                              bootstrap_dist: np.ndarray) -> float:
    """
    Two-sided paired bootstrap p-value.

    Under H0 the true difference is 0.  We shift the bootstrap distribution
    to be centred at 0 and count how often the shifted replicates are at
    least as extreme as the observed statistic.
    """
    valid = bootstrap_dist[~np.isnan(bootstrap_dist)]
    if len(valid) == 0:
        return np.nan

    # Shift distribution to be centred at 0 (null hypothesis)
    shifted = valid - np.mean(valid)

    # Two-sided: proportion of shifted replicates with |value| >= |observed|
    p_value = np.mean(np.abs(shifted) >= np.abs(observed))
    return float(p_value)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_bootstrap_distribution(bootstrap_dist: np.ndarray,
                                  observed: float,
                                  ci: Tuple[float, float],
                                  p_value: float,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """Plot the bootstrap distribution of AUC differences."""
    valid = bootstrap_dist[~np.isnan(bootstrap_dist)]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(valid, bins=60, color="steelblue", edgecolor="white",
            alpha=0.8, label="Bootstrap replicates")

    ax.axvline(observed, color="crimson", linewidth=2.0,
               label=f"Observed diff = {observed:.4f}")
    ax.axvline(ci[0], color="darkorange", linewidth=1.5, linestyle="--",
               label=f"95 % BCa CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    ax.axvline(ci[1], color="darkorange", linewidth=1.5, linestyle="--")
    ax.axvline(0.0,   color="black",      linewidth=1.0, linestyle=":",
               label="No difference (0)")

    ax.set_xlabel("AUC(Model A) − AUC(Model B)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Bootstrap Distribution of AUC Difference\n"
        f"p-value = {p_value:.4f}  |  n_boot = {len(valid):,}",
        fontsize=13
    )
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_models(
    y_true: np.ndarray,
    y_pred_model_a: np.ndarray,
    y_pred_model_b: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: Optional[int] = 42,
    plot: bool = True,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Compare two ML models using bootstrap resampling.

    Parameters
    ----------
    y_true          : Ground-truth binary labels (shape [n_samples]).
    y_pred_model_a  : Predicted probabilities / scores from Model A.
    y_pred_model_b  : Predicted probabilities / scores from Model B.
    n_bootstrap     : Number of bootstrap iterations (≥ 1000 recommended).
    alpha           : Significance level for the confidence interval.
    random_state    : Seed for reproducibility.
    plot            : Whether to display the bootstrap distribution plot.
    save_path       : Optional file path to save the figure.

    Returns
    -------
    dict with keys:
        observed_diff   – AUC_A − AUC_B on the original test set
        ci_lower        – Lower bound of the BCa confidence interval
        ci_upper        – Upper bound of the BCa confidence interval
        p_value         – Two-sided paired bootstrap p-value
        bootstrap_dist  – NumPy array of bootstrap replicate differences
        auc_a           – AUC of Model A on the original test set
        auc_b           – AUC of Model B on the original test set
        n_bootstrap     – Number of bootstrap iterations requested
        n_valid_boot    – Number of non-degenerate bootstrap samples used

    Notes
    -----
    * All data passed here must already be the **held-out test set** — never
      pass training data or validation data used for model selection.
    * Thresholds, calibration, and hyperparameters must have been fixed
      before calling this function.
    * If multiple comparisons are performed, apply Bonferroni or
      Benjamini-Hochberg correction to the returned p-values externally.
    """
    # ------------------------------------------------------------------ #
    # Input validation
    # ------------------------------------------------------------------ #
    y_true         = np.asarray(y_true,         dtype=float)
    y_pred_model_a = np.asarray(y_pred_model_a, dtype=float)
    y_pred_model_b = np.asarray(y_pred_model_b, dtype=float)

    if not (y_true.shape == y_pred_model_a.shape == y_pred_model_b.shape):
        raise ValueError(
            "y_true, y_pred_model_a, and y_pred_model_b must have the same shape."
        )
    if y_true.ndim != 1:
        raise ValueError("Arrays must be 1-D.")
    if n_bootstrap < 1000:
        warnings.warn(
            f"n_bootstrap={n_bootstrap} is below the recommended minimum of 1000.",
            UserWarning,
            stacklevel=2,
        )

    n = len(y_true)
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------ #
    # Observed statistics on the original test set
    # ------------------------------------------------------------------ #
    auc_a = roc_auc_score(y_true, y_pred_model_a)
    auc_b = roc_auc_score(y_true, y_pred_model_b)
    observed_diff = auc_a - auc_b

    # ------------------------------------------------------------------ #
    # Bootstrap resampling
    # ------------------------------------------------------------------ #
    bootstrap_dist = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)          # sample with replacement
        bootstrap_dist[i] = _auc_diff(y_true, y_pred_model_a, y_pred_model_b, idx)

    n_valid = int(np.sum(~np.isnan(bootstrap_dist)))
    if n_valid < n_bootstrap * 0.9:
        warnings.warn(
            f"More than 10 % of bootstrap samples were degenerate "
            f"({n_bootstrap - n_valid} / {n_bootstrap}). "
            "Results may be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------ #
    # BCa confidence interval
    # ------------------------------------------------------------------ #
    ci_lower, ci_upper = _bca_ci(
        observed_diff, bootstrap_dist,
        y_true, y_pred_model_a, y_pred_model_b,
        alpha=alpha,
    )

    # ------------------------------------------------------------------ #
    # Paired bootstrap p-value
    # ------------------------------------------------------------------ #
    p_value = _paired_bootstrap_pvalue(observed_diff, bootstrap_dist)

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    if plot:
        fig = _plot_bootstrap_distribution(
            bootstrap_dist,
            observed_diff,
            (ci_lower, ci_upper),
            p_value,
            save_path=save_path,
        )
        plt.show()

    # ------------------------------------------------------------------ #
    # Return results
    # ------------------------------------------------------------------ #
    return {
        "observed_diff":  observed_diff,
        "ci_lower":       ci_lower,
        "ci_upper":       ci_upper,
        "p_value":        p_value,
        "bootstrap_dist": bootstrap_dist,
        "auc_a":          auc_a,
        "auc_b":          auc_b,
        "n_bootstrap":    n_bootstrap,
        "n_valid_boot":   n_valid,
    }


# ---------------------------------------------------------------------------
# Bonferroni / BH correction helpers (for multiple comparisons)
# ---------------------------------------------------------------------------

def bonferroni_correction(p_values: np.ndarray) -> np.ndarray:
    """Apply Bonferroni correction to an array of p-values."""
    p_values = np.asarray(p_values, dtype=float)
    return np.minimum(p_values * len(p_values), 1.0)


def benjamini_hochberg_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction.

    Returns adjusted p-values (same length as input).
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    adjusted = p_values * n / ranks
    # Enforce monotonicity (step-down)
    adjusted_sorted = adjusted[order]
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted[order] = adjusted_sorted
    return np.minimum(adjusted, 1.0)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate a test set — in real usage this must be a genuinely held-out
    # partition that was never touched during training or model selection.
    _rng = np.random.default_rng(0)
    _n   = 500

    _y_true = _rng.integers(0, 2, size=_n)

    # Model A: slightly better
    _y_pred_a = np.clip(
        _y_true * 0.6 + _rng.normal(0, 0.25, _n), 0, 1
    )
    # Model B: slightly worse
    _y_pred_b = np.clip(
        _y_true * 0.5 + _rng.normal(0, 0.30, _n), 0, 1
    )

    results = compare_models(
        y_true          = _y_true,
        y_pred_model_a  = _y_pred_a,
        y_pred_model_b  = _y_pred_b,
        n_bootstrap     = 2000,
        alpha           = 0.05,
        random_state    = 42,
        plot            = True,
    )

    print("\n=== Model Comparison Results ===")
    print(f"  AUC (Model A)       : {results['auc_a']:.4f}")
    print(f"  AUC (Model B)       : {results['auc_b']:.4f}")
    print(f"  Observed difference : {results['observed_diff']:.4f}")
    print(f"  95 % BCa CI         : [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    print(f"  p-value (two-sided) : {results['p_value']:.4f}")
    print(f"  Valid bootstrap runs: {results['n_valid_boot']} / {results['n_bootstrap']}")

    # Example: multiple comparison correction
    p_vals = np.array([results["p_value"], 0.03, 0.12])
    print("\nBonferroni-corrected p-values:", bonferroni_correction(p_vals))
    print("BH-corrected p-values        :", benjamini_hochberg_correction(p_vals))