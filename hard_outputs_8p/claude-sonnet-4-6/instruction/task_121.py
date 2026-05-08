```python
"""
Comprehensive A/B/C/D Test Analysis Module for E-Commerce Experiments.

This module performs rigorous statistical analysis of multi-variant experiments,
including conversion rate analysis, revenue analysis, pairwise comparisons,
effect sizes, and multiple testing corrections.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"variant", "converted", "revenue"}
VALID_VARIANTS = {"A", "B", "C", "D"}
CONTROL_VARIANT = "A"
ALPHA = 0.05  # family-wise significance level before correction


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame has the required structure."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    invalid_variants = set(df["variant"].unique()) - VALID_VARIANTS
    if invalid_variants:
        raise ValueError(
            f"Unexpected variant labels found: {invalid_variants}. "
            f"Allowed labels are {VALID_VARIANTS}."
        )

    if df["converted"].isin([0, 1]).sum() != len(df):
        raise ValueError("Column 'converted' must contain only 0 or 1 values.")

    if not pd.api.types.is_numeric_dtype(df["revenue"]):
        raise TypeError("Column 'revenue' must be numeric.")

    if CONTROL_VARIANT not in df["variant"].values:
        raise ValueError(
            f"Control variant '{CONTROL_VARIANT}' not found in the data."
        )


# ---------------------------------------------------------------------------
# Per-variant descriptive statistics
# ---------------------------------------------------------------------------

def _wilson_confidence_interval(
    successes: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute the Wilson score confidence interval for a proportion.

    Parameters
    ----------
    successes : int
        Number of successes (conversions).
    n : int
        Total number of observations.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    if n == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _mean_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Compute the mean and its confidence interval using the t-distribution.

    Parameters
    ----------
    data : np.ndarray
        Array of numeric values.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    tuple[float, float, float]
        Mean, lower bound, upper bound.
    """
    n = len(data)
    if n < 2:
        mean = float(np.mean(data)) if n == 1 else float("nan")
        return mean, float("nan"), float("nan")

    mean = float(np.mean(data))
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, mean - h, mean + h


def compute_per_variant_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-variant descriptive statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns 'variant', 'converted', 'revenue'.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by variant with descriptive statistics.
    """
    records = []
    for variant, group in df.groupby("variant", sort=True):
        n = len(group)
        conversions = int(group["converted"].sum())
        conversion_rate = conversions / n if n > 0 else float("nan")

        cr_lower, cr_upper = _wilson_confidence_interval(conversions, n)

        revenue_vals = group["revenue"].to_numpy(dtype=float)
        mean_rev, rev_lower, rev_upper = _mean_confidence_interval(revenue_vals)
        std_rev = float(np.std(revenue_vals, ddof=1)) if n > 1 else float("nan")

        records.append(
            {
                "variant": variant,
                "n": n,
                "conversions": conversions,
                "conversion_rate": conversion_rate,
                "cr_ci_lower": cr_lower,
                "cr_ci_upper": cr_upper,
                "mean_revenue": mean_rev,
                "std_revenue": std_rev,
                "rev_ci_lower": rev_lower,
                "rev_ci_upper": rev_upper,
            }
        )

    return pd.DataFrame(records).set_index("variant")


# ---------------------------------------------------------------------------
# Effect size helpers
# ---------------------------------------------------------------------------

def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for two independent groups.

    Parameters
    ----------
    group1 : np.ndarray
        First group values.
    group2 : np.ndarray
        Second group values.

    Returns
    -------
    float
        Cohen's d value.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def _odds_ratio(
    conversions_a: int,
    n_a: int,
    conversions_b: int,
    n_b: int,
) -> tuple[float, float, float]:
    """
    Compute the odds ratio and its 95% confidence interval.

    Parameters
    ----------
    conversions_a : int
        Conversions in group A (reference).
    n_a : int
        Total observations in group A.
    conversions_b : int
        Conversions in group B (treatment).
    n_b : int
        Total observations in group B.

    Returns
    -------
    tuple[float, float, float]
        Odds ratio, lower CI, upper CI.
    """
    non_conv_a = n_a - conversions_a
    non_conv_b = n_b - conversions_b

    # Add 0.5 continuity correction if any cell is zero
    if 0 in (conversions_a, non_conv_a, conversions_b, non_conv_b):
        conversions_a += 0.5
        non_conv_a += 0.5
        conversions_b += 0.5
        non_conv_b += 0.5

    or_val = (conversions_b / non_conv_b) / (conversions_a / non_conv_a)

    # Log-based CI
    log_or = np.log(or_val)
    se_log_or = np.sqrt(
        1 / conversions_a + 1 / non_conv_a + 1 / conversions_b + 1 / non_conv_b
    )
    z = stats.norm.ppf(0.975)
    ci_lower = float(np.exp(log_or - z * se_log_or))
    ci_upper = float(np.exp(log_or + z * se_log_or))

    return float(or_val), ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Pairwise statistical tests
# ---------------------------------------------------------------------------

def _run_pairwise_tests(
    df: pd.DataFrame,
    variant_a: str,
    variant_b: str,
) -> dict:
    """
    Run chi-squared test for conversion and Welch's t-test for revenue
    between two variants.

    Parameters
    ----------
    df : pd.DataFrame
        Full experiment DataFrame.
    variant_a : str
        Reference variant label.
    variant_b : str
        Treatment variant label.

    Returns
    -------
    dict
        Dictionary with test statistics, p-values, and effect sizes.
    """
    group_a = df[df["variant"] == variant_a]
    group_b = df[df["variant"] == variant_b]

    n_a, n_b = len(group_a), len(group_b)
    conv_a = int(group_a["converted"].sum())
    conv_b = int(group_b["converted"].sum())

    # --- Chi-squared test for conversion ---
    contingency = np.array(
        [[conv_a, n_a - conv_a], [conv_b, n_b - conv_b]]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi2_stat, chi2_p, chi2_dof, _ = chi2_contingency(
            contingency, correction=False
        )

    or_val, or_lower, or_upper = _odds_ratio(conv_a, n_a, conv_b, n_b)

    # --- Welch's t-test for revenue ---
    rev_a = group_a["revenue"].to_numpy(dtype=float)
    rev_b = group_b["revenue"].to_numpy(dtype=float)

    if len(rev_a) >= 2 and len(rev_b) >= 2:
        t_stat, t_p = ttest_ind(rev_b, rev_a, equal_var=False)
    else:
        t_stat, t_p = float("nan"), float("nan")

    cohen_d = _cohens_d(rev_b, rev_a)

    return {
        "variant_reference": variant_a,
        "variant_treatment": variant_b,
        "n_reference": n_a,
        "n_treatment": n_b,
        "conv_rate_reference": conv_a / n_a if n_a > 0 else float("nan"),
        "conv_rate_treatment": conv_b / n_b if n_b > 0 else float("nan"),
        "chi2_stat": float(chi2_stat),
        "chi2_p_value": float(chi2_p),
        "odds_ratio": or_val,
        "odds_ratio_ci_lower": or_lower,
        "odds_ratio_ci_upper": or_upper,
        "mean_rev_reference": float(np.mean(rev_a)) if len(rev_a) > 0 else float("nan"),
        "mean_rev_treatment": float(np.mean(rev_b)) if len(rev_b) > 0 else float("nan"),
        "t_stat": float(t_stat),
        "t_p_value": float(t_p),
        "cohens_d": float(cohen_d),
    }


# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------

def _apply_bonferroni_correction(
    p_values: np.ndarray, alpha: float = ALPHA
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Bonferroni correction to an array of p-values.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Corrected p-values and boolean significance flags.
    """
    n_tests = len(p_values)
    corrected = np.minimum(p_values * n_tests, 1.0)
    significant = corrected < alpha
    return corrected, significant


def _apply_bh_correction(
    p_values: np.ndarray, alpha: float = ALPHA
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values.
    alpha : float
        False discovery rate threshold.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Adjusted p-values and boolean significance flags.
    """
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)

    order = np.argsort(p_values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)

    adjusted = np.minimum(p_values * n / ranks, 1.0)
    # Enforce monotonicity (cumulative minimum from largest rank)
    adjusted_sorted = adjusted[order]
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted[order] = adjusted_sorted

    significant = adjusted < alpha
    return adjusted, significant


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_abcd_analysis(
    df: pd.DataFrame,
    alpha: float = ALPHA,
    correction_method: str = "bh",
) -> dict[str, pd.DataFrame]:
    """
    Perform a comprehensive A/B/C/D test analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'variant' (A/B/C/D), 'converted' (0/1),
        and 'revenue' (numeric).
    alpha : float
        Significance level (default 0.05).
    correction_method : str
        Multiple testing correction method: 'bonferroni' or 'bh'
        (Benjamini-Hochberg). Default is 'bh'.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys:
        - 'descriptive_stats': per-variant summary statistics
        - 'vs_control': pairwise comparisons against control (variant A)
        - 'all_pairwise': all pairwise comparisons
        - 'significance_summary': which variants beat control significantly
    """
    _validate_dataframe(df)

    if correction_method not in ("bonferroni", "bh"):
        raise ValueError(
            f"correction_method must be 'bonferroni' or 'bh', "
            f"got '{correction_method}'."
        )

    # -----------------------------------------------------------------------
    # 1. Per-variant descriptive statistics
    # -----------------------------------------------------------------------
    desc_stats = compute_per_variant_stats(df)

    # -----------------------------------------------------------------------
    # 2. Comparisons vs. control (variant A)
    # -----------------------------------------------------------------------
    treatment_variants = sorted(
        [v for v in df["variant"].unique() if v != CONTROL_VARIANT]
    )

    vs_control_records = []
    for treatment in treatment_variants:
        record = _run_pairwise_tests(df, CONTROL_VARIANT, treatment)
        vs_control_records.append(record)

    vs_control_df = pd.DataFrame(vs_control_records)

    # Apply multiple testing correction to vs-control comparisons
    if len(vs_control_df) > 0:
        chi2_raw = vs_control_df["chi2_p_value"].to_numpy(dtype=float)
        t_raw = vs_control_df["t_p_value"].to_numpy(dtype=float)

        if correction_method == "bonferroni":
            chi2_adj, chi2_sig = _apply_bonferroni_correction(chi2_raw, alpha)
            t_adj, t_sig = _apply_bonferroni_correction(t_raw, alpha)
        else:
            chi2_adj, chi2_sig = _apply_bh_correction(chi2_raw, alpha)
            t_adj, t_sig = _apply_bh_correction(t_raw, alpha)

        vs_control_df["chi2_p_adjusted"] = chi2_adj
        vs_control_df["chi2_significant"] = chi2_sig
        vs_control_df["t_p_adjusted"] = t_adj
        vs_control_df["t_significant"] = t_sig
        vs_control_df["correction_method"] = correction_method

    # -----------------------------------------------------------------------
    # 3. All pairwise comparisons
    # -----------------------------------------------------------------------
    all_variants = sorted(df["variant"].unique())
    all_pairs = list(combinations(all_variants, 2))

    all_pairwise_records = []
    for var_a, var_b in all_pairs:
        record = _run_pairwise_tests(df, var_a, var_b)
        all_pairwise_records.append(record)

    all_pairwise_df = pd.DataFrame(all_pairwise_records)

    # Apply multiple testing correction to all pairwise comparisons
    if len(all_pairwise_df) > 0:
        chi2_raw_all = all_pairwise_df["chi2_p_value"].to_numpy(dtype=float)
        t_raw_all = all_pairwise_df["t_p_value"].to_numpy(dtype=float)

        if correction_method == "bonferroni":
            chi2_adj_all, chi2_sig_all = _apply_bonferroni_correction(
                chi2_raw_all, alpha
            )
            t_adj_all, t_sig_all = _apply_bonferroni_correction(t_raw_all, alpha)
        else:
            chi2_adj_all, chi2_sig_all = _apply_bh_correction(chi2_raw_all, alpha)
            t_adj_all, t_sig_all = _apply_bh_correction(t_raw_all, alpha)

        all_pairwise_df["chi2_p_adjusted"] = chi2_adj_all
        all_pairwise_df["chi2_significant"] = chi2_sig_all
        all_pairwise_df["t_p_adjusted"] = t_adj_all
        all_pairwise_df["t_significant"] = t_sig_all
        all_pairwise_df["correction_method"] = correction_method

    # -----------------------------------------------------------------------
    # 4. Significance summary: which variants beat control?
    # -----------------------------------------------------------------------
    summary_records = []
    for _, row in vs_control_df.iterrows():
        treatment = row["variant_treatment"]
        conv_rate_ctrl = row["conv_rate_reference"]
        conv_rate_trt = row["conv_rate_treatment"]
        mean_rev_ctrl = row["mean_rev_reference"]
        mean_rev_trt = row["mean_rev_treatment"]

        better_conversion = (
            bool(row["chi2_significant"])
            and conv_rate_trt > conv_rate_ctrl
        )
        better_revenue = (
            bool(row["t_significant"])
            and mean_rev_trt > mean_rev_ctrl
        )

        summary_records.append(
            {
                "variant": treatment,
                "beats_control_conversion": better_conversion,
                "beats_control_revenue": better_revenue,
                "beats_control_both": better_conversion and better_revenue,
                "conversion_lift_pct": (
                    (conv_rate_trt - conv_rate_ctrl) / conv_rate_ctrl * 100
                    if conv_rate_ctrl > 0
                    else float("nan")
                ),
                "revenue_lift_pct": (
                    (mean_rev_trt - mean_rev_ctrl) / mean_rev_ctrl * 100
                    if mean_rev_ctrl > 0
                    else float("nan")
                ),
                "odds