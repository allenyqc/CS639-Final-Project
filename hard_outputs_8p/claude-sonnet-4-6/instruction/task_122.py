"""
Bootstrap model comparison module.

Compares two ML models using bootstrap resampling with BCa confidence intervals.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------

class BootstrapComparisonResult(NamedTuple):
    """Container for bootstrap comparison results."""

    observed_difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    bootstrap_differences: np.ndarray
    n_bootstrap: int
    alpha: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return AUC or None if it cannot be computed (e.g. single class)."""
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _jackknife_differences(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> np.ndarray:
    """
    Compute leave-one-out jackknife estimates of (AUC_A - AUC_B).

    Used for the BCa acceleration parameter.
    """
    n = len(y_true)
    jack_diffs: list[float] = []

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        yt = y_true[mask]
        pa = y_pred_a[mask]
        pb = y_pred_b[mask]

        auc_a = _safe_auc(yt, pa)
        auc_b = _safe_auc(yt, pb)

        if auc_a is None or auc_b is None:
            # Skip samples that produce degenerate splits
            continue

        jack_diffs.append(auc_a - auc_b)

    return np.array(jack_diffs)


def _bca_ci(
    observed: float,
    bootstrap_stats: np.ndarray,
    jackknife_stats: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Compute BCa (bias-corrected and accelerated) confidence interval.

    Parameters
    ----------
    observed:
        The statistic computed on the original sample.
    bootstrap_stats:
        Array of bootstrap replicate statistics.
    jackknife_stats:
        Array of jackknife leave-one-out statistics.
    alpha:
        Significance level (default 0.05 → 95 % CI).

    Returns
    -------
    (lower, upper) confidence interval bounds.
    """
    n_boot = len(bootstrap_stats)

    # --- Bias-correction factor z0 ---
    prop_less = np.mean(bootstrap_stats < observed)
    # Clamp to avoid infinite z0
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_less)

    # --- Acceleration factor a ---
    jack_mean = np.mean(jackknife_stats)
    num = np.sum((jack_mean - jackknife_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)

    if den == 0.0:
        a = 0.0
    else:
        a = num / den

    # --- Adjusted quantiles ---
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    def _adjusted_quantile(z_alpha: float) -> float:
        num_inner = z0 + z_alpha
        denom_inner = 1.0 - a * (z0 + z_alpha)
        if denom_inner == 0.0:
            return z_alpha
        return float(stats.norm.cdf(z0 + num_inner / denom_inner))

    q_lo = _adjusted_quantile(z_alpha_lo)
    q_hi = _adjusted_quantile(z_alpha_hi)

    # Clamp quantiles to valid range
    q_lo = np.clip(q_lo, 0.0, 1.0)
    q_hi = np.clip(q_hi, 0.0, 1.0)

    ci_lower = float(np.quantile(bootstrap_stats, q_lo))
    ci_upper = float(np.quantile(bootstrap_stats, q_hi))

    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compare_models_bootstrap(
    y_true: np.ndarray,
    y_pred_model_a: np.ndarray,
    y_pred_model_b: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = 42,
    plot: bool = True,
    plot_path: str | None = None,
) -> BootstrapComparisonResult:
    """
    Compare two ML models via bootstrap resampling of AUC difference.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0/1).
    y_pred_model_a:
        Predicted probabilities / scores from model A.
    y_pred_model_b:
        Predicted probabilities / scores from model B.
    n_bootstrap:
        Number of bootstrap iterations (≥ 1000 recommended).
    alpha:
        Significance level for the confidence interval (default 0.05).
    random_state:
        Seed for reproducibility.  Pass None for non-deterministic runs.
    plot:
        Whether to display / save the bootstrap distribution plot.
    plot_path:
        If provided, save the figure to this path instead of showing it.

    Returns
    -------
    BootstrapComparisonResult
        Named tuple with observed_difference, ci_lower, ci_upper, p_value,
        bootstrap_differences, n_bootstrap, alpha.

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or fewer than 2 samples.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_model_a, dtype=float)
    y_pred_b = np.asarray(y_pred_model_b, dtype=float)

    if y_true.ndim != 1 or y_pred_a.ndim != 1 or y_pred_b.ndim != 1:
        raise ValueError("All input arrays must be 1-D.")

    n = len(y_true)
    if n != len(y_pred_a) or n != len(y_pred_b):
        raise ValueError(
            f"Array length mismatch: y_true={n}, "
            f"y_pred_a={len(y_pred_a)}, y_pred_b={len(y_pred_b)}."
        )

    if n < 2:
        raise ValueError("At least 2 samples are required.")

    if n_bootstrap < 1000:
        warnings.warn(
            f"n_bootstrap={n_bootstrap} is below the recommended minimum of 1000.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Observed difference on the original test set
    # ------------------------------------------------------------------
    auc_a_obs = _safe_auc(y_true, y_pred_a)
    auc_b_obs = _safe_auc(y_true, y_pred_b)

    if auc_a_obs is None or auc_b_obs is None:
        raise ValueError(
            "AUC cannot be computed on the original data "
            "(possibly only one class present)."
        )

    observed_diff = auc_a_obs - auc_b_obs

    # ------------------------------------------------------------------
    # Bootstrap resampling
    # ------------------------------------------------------------------
    rng = np.random.default_rng(random_state)
    boot_diffs: list[float] = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        yt_b = y_true[indices]
        pa_b = y_pred_a[indices]
        pb_b = y_pred_b[indices]

        auc_a_b = _safe_auc(yt_b, pa_b)
        auc_b_b = _safe_auc(yt_b, pb_b)

        if auc_a_b is None or auc_b_b is None:
            # Degenerate bootstrap sample — skip
            continue

        boot_diffs.append(auc_a_b - auc_b_b)

    boot_diffs_arr = np.array(boot_diffs)

    if len(boot_diffs_arr) < 100:
        raise RuntimeError(
            "Too few valid bootstrap samples were obtained "
            f"({len(boot_diffs_arr)}).  Check your data for class imbalance."
        )

    # ------------------------------------------------------------------
    # BCa confidence interval
    # ------------------------------------------------------------------
    jack_diffs = _jackknife_differences(y_true, y_pred_a, y_pred_b)

    if len(jack_diffs) < 2:
        warnings.warn(
            "Jackknife produced fewer than 2 valid estimates; "
            "falling back to percentile CI (acceleration = 0).",
            UserWarning,
            stacklevel=2,
        )
        jack_diffs = np.full(n, observed_diff)  # a = 0 fallback

    ci_lower, ci_upper = _bca_ci(
        observed=observed_diff,
        bootstrap_stats=boot_diffs_arr,
        jackknife_stats=jack_diffs,
        alpha=alpha,
    )

    # ------------------------------------------------------------------
    # Paired bootstrap p-value
    # ------------------------------------------------------------------
    # Under H0: true difference = 0.
    # Shift the bootstrap distribution to be centred at 0, then compute
    # the proportion of shifted replicates at least as extreme as the
    # observed difference (two-tailed).
    shifted = boot_diffs_arr - np.mean(boot_diffs_arr)
    p_value = float(
        np.mean(np.abs(shifted) >= np.abs(observed_diff))
    )
    # Ensure p-value is never exactly 0 (use 1/n_boot as minimum)
    p_value = max(p_value, 1.0 / len(boot_diffs_arr))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if plot:
        _plot_bootstrap_distribution(
            boot_diffs=boot_diffs_arr,
            observed_diff=observed_diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            alpha=alpha,
            plot_path=plot_path,
        )

    return BootstrapComparisonResult(
        observed_difference=observed_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        bootstrap_differences=boot_diffs_arr,
        n_bootstrap=len(boot_diffs_arr),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _plot_bootstrap_distribution(
    boot_diffs: np.ndarray,
    observed_diff: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    alpha: float,
    plot_path: str | None,
) -> None:
    """Render the bootstrap distribution of AUC differences."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        boot_diffs,
        bins=60,
        color="steelblue",
        edgecolor="white",
        alpha=0.75,
        label="Bootstrap differences",
    )

    # Confidence interval shading
    ci_mask = (boot_diffs >= ci_lower) & (boot_diffs <= ci_upper)
    ax.hist(
        boot_diffs[ci_mask],
        bins=60,
        color="cornflowerblue",
        edgecolor="white",
        alpha=0.55,
        label=f"{int((1 - alpha) * 100)}% BCa CI",
    )

    # Vertical lines
    ax.axvline(
        observed_diff,
        color="crimson",
        linewidth=2.0,
        linestyle="-",
        label=f"Observed diff = {observed_diff:.4f}",
    )
    ax.axvline(
        ci_lower,
        color="darkorange",
        linewidth=1.5,
        linestyle="--",
        label=f"CI lower = {ci_lower:.4f}",
    )
    ax.axvline(
        ci_upper,
        color="darkorange",
        linewidth=1.5,
        linestyle="--",
        label=f"CI upper = {ci_upper:.4f}",
    )
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle=":", label="No difference")

    ax.set_xlabel("AUC(Model A) − AUC(Model B)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Bootstrap Distribution of AUC Difference\n"
        f"p-value = {p_value:.4f}  |  n_boot = {len(boot_diffs):,}",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    if plot_path is not None:
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Convenience summary printer
# ---------------------------------------------------------------------------

def print_comparison_summary(result: BootstrapComparisonResult) -> None:
    """Pretty-print the comparison result."""
    ci_pct = int((1 - result.alpha) * 100)
    sig = "YES" if result.p_value < result.alpha else "NO"

    print("=" * 55)
    print("  Bootstrap Model Comparison Summary")
    print("=" * 55)
    print(f"  Observed AUC difference (A − B) : {result.observed_difference:+.6f}")
    print(f"  {ci_pct}% BCa confidence interval    : "
          f"[{result.ci_lower:+.6f}, {result.ci_upper:+.6f}]")
    print(f"  Paired bootstrap p-value         : {result.p_value:.6f}")
    print(f"  Significant at α={result.alpha}           : {sig}")
    print(f"  Valid bootstrap iterations       : {result.n_bootstrap:,}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _rng = np.random.default_rng(0)
    _n = 500

    # Simulate ground truth
    _y_true = _rng.integers(0, 2, size=_n)

    # Model A: slightly better
    _y_pred_a = np.clip(
        _y_true * 0.6 + _rng.normal(0, 0.25, _n), 0.0, 1.0
    )
    # Model B: slightly worse
    _y_pred_b = np.clip(
        _y_true * 0.5 + _rng.normal(0, 0.30, _n), 0.0, 1.0
    )

    result = compare_models_bootstrap(
        y_true=_y_true,
        y_pred_model_a=_y_pred_a,
        y_pred_model_b=_y_pred_b,
        n_bootstrap=2000,
        alpha=0.05,
        random_state=42,
        plot=True,
        plot_path=None,
    )

    print_comparison_summary(result)