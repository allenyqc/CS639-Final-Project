```python
"""
Comprehensive A/B/C/D Test Analysis Module for E-Commerce Experiments
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, norm
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for a sample mean."""
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return (mean - h, mean + h)


def compute_proportion_ci(successes, n, confidence=0.95):
    """Compute Wilson score confidence interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    z = norm.ppf((1 + confidence) / 2.)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denominator
    return (max(0, center - margin), min(1, center + margin))


def cohens_d(group1, group2):
    """Compute Cohen's d effect size for two independent groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (mean2 - mean1) / pooled_std


def odds_ratio(conv_a, n_a, conv_b, n_b):
    """Compute odds ratio comparing group B to group A."""
    if n_a == 0 or n_b == 0:
        return np.nan
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    if p_a == 0 or p_a == 1 or p_b == 0 or p_b == 1:
        # Apply Haldane-Anscombe correction
        p_a = (conv_a + 0.5) / (n_a + 1)
        p_b = (conv_b + 0.5) / (n_b + 1)
    odds_a = p_a / (1 - p_a)
    odds_b = p_b / (1 - p_b)
    if odds_a == 0:
        return np.nan
    return odds_b / odds_a


def odds_ratio_ci(conv_a, n_a, conv_b, n_b, confidence=0.95):
    """Compute confidence interval for odds ratio using log method."""
    or_val = odds_ratio(conv_a, n_a, conv_b, n_b)
    if np.isnan(or_val) or or_val <= 0:
        return (np.nan, np.nan)
    # Use Haldane-Anscombe correction for stability
    a = conv_a + 0.5
    b = n_a - conv_a + 0.5
    c = conv_b + 0.5
    d = n_b - conv_b + 0.5
    log_or = np.log(or_val)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = norm.ppf((1 + confidence) / 2.)
    return (np.exp(log_or - z * se_log_or), np.exp(log_or + z * se_log_or))


def interpret_cohens_d(d):
    """Interpret Cohen's d magnitude."""
    if np.isnan(d):
        return "N/A"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_odds_ratio(or_val):
    """Interpret odds ratio magnitude."""
    if np.isnan(or_val):
        return "N/A"
    if or_val < 1:
        return f"lower odds ({or_val:.3f}x)"
    elif or_val < 1.5:
        return "negligible increase"
    elif or_val < 2.0:
        return "moderate increase"
    else:
        return "large increase"


def run_chi_squared_test(conv_a, n_a, conv_b, n_b):
    """Run chi-squared test for conversion rates."""
    contingency_table = np.array([
        [conv_a, n_a - conv_a],
        [conv_b, n_b - conv_b]
    ])
    # Check for zero cells
    if np.any(contingency_table < 0):
        return np.nan, np.nan, np.nan
    try:
        chi2, p_val, dof, expected = chi2_contingency(contingency_table, correction=True)
        return chi2, p_val, dof
    except Exception:
        return np.nan, np.nan, np.nan


def run_ttest(group_a, group_b):
    """Run Welch's t-test for revenue comparison."""
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan, np.nan
    try:
        t_stat, p_val = ttest_ind(group_a, group_b, equal_var=False)
        return t_stat, p_val
    except Exception:
        return np.nan, np.nan


def apply_bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction to p-values."""
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests if n_tests > 0 else alpha
    return corrected_alpha, [p <= corrected_alpha for p in p_values]


def compute_variant_stats(df, variant_name):
    """Compute per-variant statistics."""
    variant_data = df[df['variant'] == variant_name]
    n = len(variant_data)
    conversions = variant_data['converted'].sum()
    conv_rate = conversions / n if n > 0 else 0
    conv_ci = compute_proportion_ci(conversions, n)
    revenue = variant_data['revenue']
    mean_rev = revenue.mean()
    rev_ci = compute_confidence_interval(revenue.values)
    return {
        'variant': variant_name,
        'n': n,
        'conversions': int(conversions),
        'conversion_rate': conv_rate,
        'conv_ci_lower': conv_ci[0],
        'conv_ci_upper': conv_ci[1],
        'mean_revenue': mean_rev,
        'rev_ci_lower': rev_ci[0],
        'rev_ci_upper': rev_ci[1],
        'std_revenue': revenue.std(),
        'median_revenue': revenue.median(),
        'total_revenue': revenue.sum(),
        'revenue_data': revenue.values
    }


def run_abcd_analysis(df, alpha=0.05, control='A'):
    """
    Run comprehensive A/B/C/D test analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'variant' (A/B/C/D), 'converted' (0/1), 'revenue'
    alpha : float
        Significance level (default 0.05)
    control : str
        Control variant name (default 'A')

    Returns
    -------
    dict containing:
        - 'variant_stats': per-variant statistics DataFrame
        - 'vs_control': comparisons against control DataFrame
        - 'pairwise': all pairwise comparisons DataFrame
        - 'summary': overall summary dict
        - 'significant_variants': list of variants significantly better than control
    """
    # ── Validate input ──────────────────────────────────────────────────────────
    required_cols = {'variant', 'converted', 'revenue'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    df = df.copy()
    df['converted'] = df['converted'].astype(int)
    df['revenue'] = df['revenue'].astype(float)

    variants = sorted(df['variant'].unique().tolist())
    if control not in variants:
        raise ValueError(f"Control variant '{control}' not found in data. "
                         f"Available variants: {variants}")

    print(f"{'='*70}")
    print(f"  A/B/C/D TEST ANALYSIS")
    print(f"  Variants: {variants} | Control: {control} | Alpha: {alpha}")
    print(f"  Total observations: {len(df):,}")
    print(f"{'='*70}\n")

    # ── Per-variant statistics ──────────────────────────────────────────────────
    variant_stats = {}
    for v in variants:
        variant_stats[v] = compute_variant_stats(df, v)

    stats_rows = []
    for v in variants:
        s = variant_stats[v]
        stats_rows.append({
            'variant': s['variant'],
            'n': s['n'],
            'conversions': s['conversions'],
            'conversion_rate': round(s['conversion_rate'], 4),
            'conv_ci_lower': round(s['conv_ci_lower'], 4),
            'conv_ci_upper': round(s['conv_ci_upper'], 4),
            'mean_revenue': round(s['mean_revenue'], 4),
            'rev_ci_lower': round(s['rev_ci_lower'], 4),
            'rev_ci_upper': round(s['rev_ci_upper'], 4),
            'std_revenue': round(s['std_revenue'], 4),
            'median_revenue': round(s['median_revenue'], 4),
            'total_revenue': round(s['total_revenue'], 4),
        })
    variant_stats_df = pd.DataFrame(stats_rows).set_index('variant')

    # ── Comparisons vs control ──────────────────────────────────────────────────
    ctrl = variant_stats[control]
    treatment_variants = [v for v in variants if v != control]

    vs_control_rows = []
    conv_p_values = []
    rev_p_values = []

    for v in treatment_variants:
        trt = variant_stats[v]

        # Chi-squared for conversion
        chi2, p_conv, dof = run_chi_squared_test(
            ctrl['conversions'], ctrl['n'],
            trt['conversions'], trt['n']
        )

        # Welch's t-test for revenue
        t_stat, p_rev = run_ttest(ctrl['revenue_data'], trt['revenue_data'])

        # Effect sizes
        or_val = odds_ratio(ctrl['conversions'], ctrl['n'],
                            trt['conversions'], trt['n'])
        or_ci = odds_ratio_ci(ctrl['conversions'], ctrl['n'],
                              trt['conversions'], trt['n'])
        d = cohens_d(ctrl['revenue_data'], trt['revenue_data'])

        # Relative differences
        conv_lift = ((trt['conversion_rate'] - ctrl['conversion_rate']) /
                     ctrl['conversion_rate'] * 100
                     if ctrl['conversion_rate'] > 0 else np.nan)
        rev_lift = ((trt['mean_revenue'] - ctrl['mean_revenue']) /
                    ctrl['mean_revenue'] * 100
                    if ctrl['mean_revenue'] != 0 else np.nan)

        conv_p_values.append(p_conv if not np.isnan(p_conv) else 1.0)
        rev_p_values.append(p_rev if not np.isnan(p_rev) else 1.0)

        vs_control_rows.append({
            'variant': v,
            'control': control,
            # Conversion metrics
            'conv_rate_control': round(ctrl['conversion_rate'], 4),
            'conv_rate_variant': round(trt['conversion_rate'], 4),
            'conv_rate_lift_pct': round(conv_lift, 2) if not np.isnan(conv_lift) else np.nan,
            'chi2_stat': round(chi2, 4) if not np.isnan(chi2) else np.nan,
            'p_value_conversion': round(p_conv, 6) if not np.isnan(p_conv) else np.nan,
            'odds_ratio': round(or_val, 4) if not np.isnan(or_val) else np.nan,
            'or_ci_lower': round(or_ci[0], 4) if not np.isnan(or_ci[0]) else np.nan,
            'or_ci_upper': round(or_ci[1], 4) if not np.isnan(or_ci[1]) else np.nan,
            'or_interpretation': interpret_odds_ratio(or_val),
            # Revenue metrics
            'mean_rev_control': round(ctrl['mean_revenue'], 4),
            'mean_rev_variant': round(trt['mean_revenue'], 4),
            'rev_lift_pct': round(rev_lift, 2) if not np.isnan(rev_lift) else np.nan,
            't_stat': round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
            'p_value_revenue': round(p_rev, 6) if not np.isnan(p_rev) else np.nan,
            'cohens_d': round(d, 4) if not np.isnan(d) else np.nan,
            'cohens_d_interpretation': interpret_cohens_d(d),
        })

    # Apply Bonferroni correction
    bonf_alpha_conv, conv_sig_bonf = apply_bonferroni_correction(conv_p_values, alpha)
    bonf_alpha_rev, rev_sig_bonf = apply_bonferroni_correction(rev_p_values, alpha)

    for i, row in enumerate(vs_control_rows):
        row['sig_conversion_raw'] = row['p_value_conversion'] <= alpha if not np.isnan(row['p_value_conversion']) else False
        row['sig_revenue_raw'] = row['p_value_revenue'] <= alpha if not np.isnan(row['p_value_revenue']) else False
        row['sig_conversion_bonferroni'] = conv_sig_bonf[i]
        row['sig_revenue_bonferroni'] = rev_sig_bonf[i]
        row['bonferroni_alpha_conv'] = round(bonf_alpha_conv, 6)
        row['bonferroni_alpha_rev'] = round(bonf_alpha_rev, 6)
        # Overall significance: both conversion and revenue significant (raw)
        row['significantly_better_than_control'] = (
            row['sig_conversion_raw'] and
            row['sig_revenue_raw'] and
            row['conv_rate_lift_pct'] > 0 and
            row['rev_lift_pct'] > 0
        )

    vs_control_df = pd.DataFrame(vs_control_rows).set_index('variant')

    # ── All pairwise comparisons ────────────────────────────────────────────────
    pairwise_rows = []
    all_pairs = list(combinations(variants, 2))

    for v1, v2 in all_pairs:
        s1 = variant_stats[v1]
        s2 = variant_stats[v2]

        chi2, p_conv, dof = run_chi_squared_test(
            s1['conversions'], s1['n'],
            s2['conversions'], s2['n']
        )
        t_stat, p_rev = run_ttest(s1['revenue_data'], s2['revenue_data'])
        or_val = odds_ratio(s1['conversions'], s1['n'],
                            s2['conversions'], s2['n'])
        or_ci = odds_ratio_ci(s1['conversions'], s1['n'],
                              s2['conversions'], s2['n'])
        d = cohens_d(s1['revenue_data'], s2['revenue_data'])

        conv_lift = ((s2['conversion_rate'] - s1['conversion_rate']) /
                     s1['conversion_rate'] * 100
                     if s1['conversion_rate'] > 0 else np.nan)
        rev_lift = ((s2['mean_revenue'] - s1['mean_revenue']) /
                    s1['mean_revenue'] * 100
                    if s1['mean_revenue'] != 0 else np.nan)

        pairwise_rows.append({
            'variant_1': v1,
            'variant_2': v2,
            'comparison': f"{v1} vs {v2}",
            'conv_rate_v1': round(s1['conversion_rate'], 4),
            'conv_rate_v2': round(s2['conversion_rate'], 4),
            'conv_lift_pct': round(conv_lift, 2) if not np.isnan(conv_lift) else np.nan,
            'chi2_stat': round(chi2, 4) if not np.isnan(chi2) else np.nan,
            'p_value_conversion': round(p_conv, 6) if not np.isnan(p_conv) else np.nan,
            'sig_conversion': p_conv <= alpha if not np.isnan(p_conv) else False,
            'odds_ratio': round(or_val, 4) if not np.isnan(or_val) else np.nan,
            'or_ci_lower': round(or_ci[0], 4) if not np.isnan(or_ci[0]) else np.nan,
            'or_ci_upper': round(or_ci[1], 4) if not np.isnan(or_ci[1]) else np.nan,
            'mean_rev_v1': round(s1['mean_revenue'], 4),
            'mean_rev_v2': round(s2['mean_revenue'], 4),
            'rev_lift_pct': round(rev_lift, 2) if not np.isnan(rev_lift) else np.nan,
            't_stat': round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
            'p_value_revenue': round(p_rev, 6) if not np.isnan(p_rev) else np.nan,
            'sig_revenue': p_rev <= alpha if not np.isnan(p_rev) else False,
            'cohens_d': round(d, 4) if not np.isnan(d) else np.nan,
            'cohens_d_interpretation': interpret_cohens_d(d),
        })

    pairwise_df = pd.DataFrame(pairwise_rows).set_index('comparison')

    # ── Significant variants ────────────────────────────────────────────────────
    significant_variants = []
    if len(vs_control_rows) > 0:
        sig_df = vs_control_df[vs_control_df['significantly_better_than_control'] == True]
        significant_variants = sig_df.index.tolist()

    # ── Summary ─────────────────────────────────────────────────────────────────
    best_conv_variant = variant_stats_df['conversion_rate'].idxmax()
    best_rev_variant = variant_stats_df['mean_revenue'].idxmax()

    summary = {
        'total_observations': len(df),
        'variants': variants,
        'control': control,
        'alpha': alpha,