"""
Comprehensive A/B/C/D Test Analysis Module for E-Commerce Experiments.

This module provides statistical analysis for multi-variant experiments,
including conversion rate analysis, revenue analysis, pairwise comparisons,
effect sizes, and multiple testing corrections.
"""

import warnings
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame has the required columns and values."""
    required_cols = {"variant", "converted", "revenue"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    valid_variants = {"A", "B", "C", "D"}
    found_variants = set(df["variant"].unique())
    invalid = found_variants - valid_variants
    if invalid:
        raise ValueError(f"Invalid variant labels found: {invalid}. Expected subset of {valid_variants}.")

    if "A" not in found_variants:
        raise ValueError("Control variant 'A' must be present in the data.")

    if not df["converted"].isin([0, 1]).all():
        raise ValueError("Column 'converted' must contain only 0 or 1.")

    if df["revenue"].isnull().any():
        raise ValueError("Column 'revenue' contains null values.")


def _confidence_interval_proportion(
    successes: int, n: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _confidence_interval_mean(
    values: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float]:
    """t-distribution confidence interval for the mean."""
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    se = stats.sem(values)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    mean = np.mean(values)
    return (mean - t_crit * se, mean + t_crit * se)


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return np.nan
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def _odds_ratio(
    conv_treatment: int,
    n_treatment: int,
    conv_control: int,
    n_control: int,
) -> Tuple[float, float, float]:
    """
    Compute odds ratio and its 95 % CI (Woolf logit method).

    Returns (odds_ratio, ci_lower, ci_upper).
    """
    a = conv_treatment
    b = n_treatment - conv_treatment
    c = conv_control
    d = n_control - conv_control

    # Add 0.5 continuity correction if any cell is zero
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_val = (a * d) / (b * c)
    log_or = np.log(or_val)
    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf(0.975)
    ci_lower = np.exp(log_or - z * se_log_or)
    ci_upper = np.exp(log_or + z * se_log_or)
    return or_val, ci_lower, ci_upper


def _apply_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Bonferroni correction; return list of significance flags."""
    m = len(p_values)
    if m == 0:
        return []
    threshold = alpha / m
    return [p <= threshold for p in p_values]


def _apply_benjamini_hochberg(
    p_values: List[float], alpha: float = 0.05
) -> List[bool]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Returns a boolean list indicating which hypotheses are rejected.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        if p <= (rank / m) * alpha:
            rejected[orig_idx] = True
        else:
            # Once a p-value exceeds the threshold, stop (step-up procedure)
            break
    return rejected


# ---------------------------------------------------------------------------
# Per-variant summary
# ---------------------------------------------------------------------------

def compute_variant_summary(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Compute per-variant descriptive statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns 'variant', 'converted', 'revenue'.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by variant.
    """
    records = []
    for variant, group in df.groupby("variant"):
        n = len(group)
        conversions = int(group["converted"].sum())
        conv_rate = conversions / n if n > 0 else np.nan
        conv_ci = _confidence_interval_proportion(conversions, n, alpha)

        revenue = group["revenue"].values
        mean_rev = np.mean(revenue)
        rev_ci = _confidence_interval_mean(revenue, alpha)

        records.append(
            {
                "variant": variant,
                "n": n,
                "conversions": conversions,
                "conversion_rate": conv_rate,
                "conv_ci_lower": conv_ci[0],
                "conv_ci_upper": conv_ci[1],
                "mean_revenue": mean_rev,
                "revenue_ci_lower": rev_ci[0],
                "revenue_ci_upper": rev_ci[1],
                "revenue_std": np.std(revenue, ddof=1) if n > 1 else np.nan,
            }
        )

    summary = pd.DataFrame(records).set_index("variant").sort_index()
    return summary


# ---------------------------------------------------------------------------
# Pairwise statistical tests
# ---------------------------------------------------------------------------

def _chi2_conversion_test(
    group_a: pd.DataFrame, group_b: pd.DataFrame
) -> Tuple[float, float]:
    """Chi-squared test for difference in conversion rates."""
    conv_a = int(group_a["converted"].sum())
    non_a = len(group_a) - conv_a
    conv_b = int(group_b["converted"].sum())
    non_b = len(group_b) - conv_b

    contingency = np.array([[conv_a, non_a], [conv_b, non_b]])
    # Avoid degenerate tables
    if contingency.min() < 0 or contingency.sum() == 0:
        return np.nan, np.nan

    chi2, p, _, _ = chi2_contingency(contingency, correction=True)
    return float(chi2), float(p)


def _ttest_revenue(
    group_a: pd.DataFrame, group_b: pd.DataFrame
) -> Tuple[float, float]:
    """Welch's t-test for difference in mean revenue."""
    rev_a = group_a["revenue"].values
    rev_b = group_b["revenue"].values
    if len(rev_a) < 2 or len(rev_b) < 2:
        return np.nan, np.nan
    t_stat, p = ttest_ind(rev_b, rev_a, equal_var=False)
    return float(t_stat), float(p)


def run_pairwise_tests(
    df: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Run pairwise chi-squared (conversion) and t-test (revenue) comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    pairs : list of (str, str), optional
        Specific pairs to compare. If None, all combinations are used.

    Returns
    -------
    pd.DataFrame
        Raw (uncorrected) test statistics and p-values for each pair.
    """
    variants = sorted(df["variant"].unique())
    if pairs is None:
        pairs = list(combinations(variants, 2))

    groups = {v: df[df["variant"] == v] for v in variants}
    records = []

    for v1, v2 in pairs:
        if v1 not in groups or v2 not in groups:
            continue
        g1, g2 = groups[v1], groups[v2]

        chi2_stat, p_conv = _chi2_conversion_test(g1, g2)
        t_stat, p_rev = _ttest_revenue(g1, g2)

        # Effect sizes
        or_val, or_lo, or_hi = _odds_ratio(
            int(g2["converted"].sum()),
            len(g2),
            int(g1["converted"].sum()),
            len(g1),
        )
        d = _cohens_d(g2["revenue"].values, g1["revenue"].values)

        records.append(
            {
                "variant_1": v1,
                "variant_2": v2,
                "chi2_stat": chi2_stat,
                "p_conversion_raw": p_conv,
                "t_stat": t_stat,
                "p_revenue_raw": p_rev,
                "odds_ratio": or_val,
                "odds_ratio_ci_lower": or_lo,
                "odds_ratio_ci_upper": or_hi,
                "cohens_d": d,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------

def apply_multiple_testing_correction(
    pairwise_df: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "benjamini-hochberg",
) -> pd.DataFrame:
    """
    Apply multiple testing correction to raw p-values.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Output of `run_pairwise_tests`.
    alpha : float
        Family-wise error rate / FDR threshold.
    method : str
        'bonferroni' or 'benjamini-hochberg'.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with corrected significance flags.
    """
    result = pairwise_df.copy()

    for col_raw, col_sig in [
        ("p_conversion_raw", "sig_conversion"),
        ("p_revenue_raw", "sig_revenue"),
    ]:
        p_vals = result[col_raw].tolist()
        if method == "bonferroni":
            flags = _apply_bonferroni(p_vals, alpha)
        elif method in ("benjamini-hochberg", "bh", "fdr"):
            flags = _apply_benjamini_hochberg(p_vals, alpha)
        else:
            raise ValueError(f"Unknown correction method: {method!r}")
        result[col_sig] = flags

    return result


# ---------------------------------------------------------------------------
# Control vs. treatment analysis
# ---------------------------------------------------------------------------

def control_vs_treatments(
    df: pd.DataFrame,
    control: str = "A",
    alpha: float = 0.05,
    correction: str = "benjamini-hochberg",
) -> pd.DataFrame:
    """
    Compare each non-control variant against the control.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    control : str
        Label of the control variant (default 'A').
    alpha : float
        Significance level.
    correction : str
        Multiple testing correction method.

    Returns
    -------
    pd.DataFrame
        Summary of control vs. each treatment variant.
    """
    treatments = [v for v in sorted(df["variant"].unique()) if v != control]
    pairs = [(control, t) for t in treatments]

    raw = run_pairwise_tests(df, pairs=pairs)
    corrected = apply_multiple_testing_correction(raw, alpha=alpha, method=correction)

    # Rename for clarity: variant_2 is always the treatment
    corrected = corrected.rename(
        columns={"variant_1": "control", "variant_2": "treatment"}
    )
    return corrected


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def run_ab_analysis(
    df: pd.DataFrame,
    control: str = "A",
    alpha: float = 0.05,
    correction: str = "benjamini-hochberg",
) -> Dict[str, pd.DataFrame]:
    """
    Run a comprehensive A/B/C/D test analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'variant' (A/B/C/D), 'converted' (0/1),
        and 'revenue' (numeric).
    control : str
        Label of the control variant.
    alpha : float
        Significance level for tests and confidence intervals.
    correction : str
        Multiple testing correction: 'bonferroni' or 'benjamini-hochberg'.

    Returns
    -------
    dict with keys:
        'variant_summary'       – per-variant descriptive statistics
        'control_vs_treatments' – corrected tests vs. control
        'all_pairwise'          – corrected tests for all pairs
        'significant_variants'  – variants significantly better than control
    """
    _validate_dataframe(df)

    # 1. Per-variant summary
    summary = compute_variant_summary(df, alpha=alpha)

    # 2. Control vs. treatments
    ctrl_vs_treat = control_vs_treatments(
        df, control=control, alpha=alpha, correction=correction
    )

    # 3. All pairwise comparisons
    all_pairs_raw = run_pairwise_tests(df)
    all_pairs = apply_multiple_testing_correction(
        all_pairs_raw, alpha=alpha, method=correction
    )

    # 4. Identify variants significantly better than control on EITHER metric
    sig_variants = ctrl_vs_treat[
        ctrl_vs_treat["sig_conversion"] | ctrl_vs_treat["sig_revenue"]
    ]["treatment"].tolist()

    sig_df = pd.DataFrame(
        {
            "variant": sig_variants,
            "sig_conversion": ctrl_vs_treat.set_index("treatment").loc[
                sig_variants, "sig_conversion"
            ].values
            if sig_variants
            else [],
            "sig_revenue": ctrl_vs_treat.set_index("treatment").loc[
                sig_variants, "sig_revenue"
            ].values
            if sig_variants
            else [],
        }
    )

    return {
        "variant_summary": summary,
        "control_vs_treatments": ctrl_vs_treat,
        "all_pairwise": all_pairs,
        "significant_variants": sig_df,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_analysis_report(results: Dict[str, pd.DataFrame]) -> None:
    """Print a human-readable summary of the A/B/C/D analysis results."""
    sep = "=" * 70

    print(sep)
    print("VARIANT SUMMARY")
    print(sep)
    print(results["variant_summary"].to_string())

    print(f"\n{sep}")
    print("CONTROL vs. TREATMENTS (with multiple-testing correction)")
    print(sep)
    print(results["control_vs_treatments"].to_string(index=False))

    print(f"\n{sep}")
    print("ALL PAIRWISE COMPARISONS (with multiple-testing correction)")
    print(sep)
    print(results["all_pairwise"].to_string(index=False))

    print(f"\n{sep}")
    print("VARIANTS SIGNIFICANTLY BETTER THAN CONTROL")
    print(sep)
    sig = results["significant_variants"]
    if sig.empty:
        print("No variant is statistically significantly better than control.")
    else:
        print(sig.to_string(index=False))
    print(sep)


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    n_per_variant = 500
    variants = []
    converted = []
    revenue = []

    params = {
        "A": {"conv_p": 0.10, "rev_mean": 50.0, "rev_std": 20.0},
        "B": {"conv_p": 0.12, "rev_mean": 55.0, "rev_std": 22.0},
        "C": {"conv_p": 0.10, "rev_mean": 51.0, "rev_std": 19.0},
        "D": {"conv_p": 0.15, "rev_mean": 65.0, "rev_std": 25.0},
    }

    for variant, p in params.items():
        conv = rng.binomial(1, p["conv_p"], n_per_variant)
        rev = rng.normal(p["rev_mean"], p["rev_std"], n_per_variant).clip(0)
        variants.extend([variant] * n_per_variant)
        converted.extend(conv.tolist())
        revenue.extend(rev.tolist())

    df_example = pd.DataFrame(
        {"variant": variants, "converted": converted, "revenue": revenue}
    )

    results = run_ab_analysis(
        df_example,
        control="A",
        alpha=0.05,
        correction="benjamini-hochberg",
    )
    print_analysis_report(results)