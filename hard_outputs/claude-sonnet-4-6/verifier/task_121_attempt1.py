```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, norm
from itertools import combinations
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """Compute confidence interval for the mean of a dataset."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return (mean - h, mean + h)


def compute_proportion_ci(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """Compute Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf((1 + confidence) / 2.)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def odds_ratio(conv_a: int, n_a: int, conv_b: int, n_b: int) -> tuple:
    """
    Compute odds ratio and its 95% confidence interval.
    Returns (odds_ratio, ci_lower, ci_upper).
    """
    # Add 0.5 continuity correction if any cell is zero
    a = conv_a + 0.5
    b = (n_a - conv_a) + 0.5
    c = conv_b + 0.5
    d = (n_b - conv_b) + 0.5

    or_val = (a * d) / (b * c)
    log_or = np.log(or_val)
    se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
    z = norm.ppf(0.975)
    ci_lower = np.exp(log_or - z * se_log_or)
    ci_upper = np.exp(log_or + z * se_log_or)
    return (or_val, ci_lower, ci_upper)


def apply_bonferroni_correction(p_values: list, alpha: float = 0.05) -> list:
    """Apply Bonferroni correction to a list of p-values."""
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values], corrected_alpha


def run_abcd_analysis(
    df: pd.DataFrame,
    control_variant: str = 'A',
    alpha: float = 0.05,
    confidence: float = 0.95,
    apply_correction: bool = True
) -> dict:
    """
    Perform comprehensive A/B/C/D test analysis for an e-commerce experiment.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
        - 'variant': str, one of A/B/C/D
        - 'converted': int, 0 or 1
        - 'revenue': float, revenue per user
    control_variant : str
        The control variant label (default 'A')
    alpha : float
        Significance level (default 0.05)
    confidence : float
        Confidence level for intervals (default 0.95)
    apply_correction : bool
        Whether to apply Bonferroni correction for multiple comparisons

    Returns
    -------
    dict with keys:
        - 'variant_summary': DataFrame with per-variant metrics
        - 'control_comparisons': DataFrame with pairwise comparisons vs control
        - 'all_pairwise': DataFrame with all pairwise comparisons
        - 'significant_variants': list of variants significantly better than control
        - 'analysis_metadata': dict with analysis parameters
    """
    # ── Validate input ──────────────────────────────────────────────────────────
    required_cols = {'variant', 'converted', 'revenue'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    df = df.copy()
    df['variant'] = df['variant'].astype(str).str.upper()
    df['converted'] = df['converted'].astype(int)
    df['revenue'] = df['revenue'].astype(float)

    variants = sorted(df['variant'].unique().tolist())
    if control_variant not in variants:
        raise ValueError(f"Control variant '{control_variant}' not found in data. "
                         f"Available variants: {variants}")

    # ── Per-variant summary ─────────────────────────────────────────────────────
    summary_rows = []
    variant_data = {}

    for variant in variants:
        vdf = df[df['variant'] == variant]
        n = len(vdf)
        conversions = vdf['converted'].sum()
        conv_rate = conversions / n if n > 0 else 0.0
        revenue_vals = vdf['revenue'].values
        mean_rev = np.mean(revenue_vals)
        median_rev = np.median(revenue_vals)
        std_rev = np.std(revenue_vals, ddof=1) if n > 1 else 0.0

        conv_ci = compute_proportion_ci(conversions, n, confidence)
        rev_ci = compute_confidence_interval(revenue_vals, confidence) if n > 1 else (mean_rev, mean_rev)

        summary_rows.append({
            'variant': variant,
            'n_users': n,
            'n_conversions': conversions,
            'conversion_rate': conv_rate,
            'conv_ci_lower': conv_ci[0],
            'conv_ci_upper': conv_ci[1],
            'mean_revenue': mean_rev,
            'median_revenue': median_rev,
            'std_revenue': std_rev,
            'rev_ci_lower': rev_ci[0],
            'rev_ci_upper': rev_ci[1],
            'total_revenue': vdf['revenue'].sum(),
        })

        variant_data[variant] = {
            'n': n,
            'conversions': conversions,
            'revenue': revenue_vals,
            'conv_rate': conv_rate,
            'mean_rev': mean_rev,
        }

    variant_summary = pd.DataFrame(summary_rows).set_index('variant')

    # ── Helper: pairwise statistical tests ─────────────────────────────────────
    def run_pairwise_tests(var_a: str, var_b: str) -> dict:
        """Run chi-squared and t-test between two variants."""
        da = variant_data[var_a]
        db = variant_data[var_b]

        # Chi-squared test for conversion rates
        contingency = np.array([
            [da['conversions'], da['n'] - da['conversions']],
            [db['conversions'], db['n'] - db['conversions']]
        ])
        # Avoid degenerate tables
        if contingency.min() < 0 or contingency.sum() == 0:
            chi2_stat, chi2_p = np.nan, np.nan
        else:
            try:
                chi2_stat, chi2_p, _, _ = chi2_contingency(contingency, correction=True)
            except Exception:
                chi2_stat, chi2_p = np.nan, np.nan

        # Odds ratio (variant_b vs variant_a as reference)
        or_val, or_ci_lower, or_ci_upper = odds_ratio(
            da['conversions'], da['n'],
            db['conversions'], db['n']
        )

        # Two-sample t-test for revenue
        if len(da['revenue']) > 1 and len(db['revenue']) > 1:
            t_stat, t_p = ttest_ind(db['revenue'], da['revenue'], equal_var=False)
        else:
            t_stat, t_p = np.nan, np.nan

        # Cohen's d (variant_b relative to variant_a)
        d = cohens_d(db['revenue'], da['revenue'])

        # Relative lifts
        conv_lift = (db['conv_rate'] - da['conv_rate']) / da['conv_rate'] if da['conv_rate'] > 0 else np.nan
        rev_lift = (db['mean_rev'] - da['mean_rev']) / abs(da['mean_rev']) if da['mean_rev'] != 0 else np.nan

        return {
            'variant_a': var_a,
            'variant_b': var_b,
            'conv_rate_a': da['conv_rate'],
            'conv_rate_b': db['conv_rate'],
            'conv_lift_pct': conv_lift * 100 if not np.isnan(conv_lift) else np.nan,
            'chi2_stat': chi2_stat,
            'chi2_p_value': chi2_p,
            'odds_ratio': or_val,
            'or_ci_lower': or_ci_lower,
            'or_ci_upper': or_ci_upper,
            'mean_rev_a': da['mean_rev'],
            'mean_rev_b': db['mean_rev'],
            'rev_lift_pct': rev_lift * 100 if not np.isnan(rev_lift) else np.nan,
            't_stat': t_stat,
            't_p_value': t_p,
            'cohens_d': d,
        }

    # ── Control comparisons ─────────────────────────────────────────────────────
    treatment_variants = [v for v in variants if v != control_variant]
    control_rows = []

    for tv in treatment_variants:
        row = run_pairwise_tests(control_variant, tv)
        control_rows.append(row)

    control_comparisons = pd.DataFrame(control_rows)

    # Apply multiple comparison correction to control comparisons
    if not control_comparisons.empty and apply_correction:
        n_tests = len(control_comparisons)
        corrected_alpha = alpha / n_tests

        control_comparisons['corrected_alpha'] = corrected_alpha
        control_comparisons['conv_significant'] = (
            control_comparisons['chi2_p_value'] < corrected_alpha
        )
        control_comparisons['rev_significant'] = (
            control_comparisons['t_p_value'] < corrected_alpha
        )
    elif not control_comparisons.empty:
        control_comparisons['corrected_alpha'] = alpha
        control_comparisons['conv_significant'] = (
            control_comparisons['chi2_p_value'] < alpha
        )
        control_comparisons['rev_significant'] = (
            control_comparisons['t_p_value'] < alpha
        )

    # Determine overall significance: significant on at least one metric
    if not control_comparisons.empty:
        control_comparisons['overall_significant'] = (
            control_comparisons['conv_significant'] | control_comparisons['rev_significant']
        )
        # Better than control: significant AND positive lift
        control_comparisons['better_than_control'] = (
            control_comparisons['overall_significant'] &
            (control_comparisons['conv_lift_pct'] > 0)
        )

    # ── All pairwise comparisons ────────────────────────────────────────────────
    all_pairs = list(combinations(variants, 2))
    pairwise_rows = []

    for var_a, var_b in all_pairs:
        row = run_pairwise_tests(var_a, var_b)
        pairwise_rows.append(row)

    all_pairwise = pd.DataFrame(pairwise_rows)

    # Apply Bonferroni correction to all pairwise
    if not all_pairwise.empty and apply_correction:
        n_pairs = len(all_pairwise)
        corrected_alpha_pairs = alpha / n_pairs
        all_pairwise['corrected_alpha'] = corrected_alpha_pairs
        all_pairwise['conv_significant'] = all_pairwise['chi2_p_value'] < corrected_alpha_pairs
        all_pairwise['rev_significant'] = all_pairwise['t_p_value'] < corrected_alpha_pairs
    elif not all_pairwise.empty:
        all_pairwise['corrected_alpha'] = alpha
        all_pairwise['conv_significant'] = all_pairwise['chi2_p_value'] < alpha
        all_pairwise['rev_significant'] = all_pairwise['t_p_value'] < alpha

    if not all_pairwise.empty:
        all_pairwise['overall_significant'] = (
            all_pairwise['conv_significant'] | all_pairwise['rev_significant']
        )

    # ── Significant variants better than control ────────────────────────────────
    significant_variants = []
    if not control_comparisons.empty and 'better_than_control' in control_comparisons.columns:
        sig_mask = control_comparisons['better_than_control']
        significant_variants = control_comparisons.loc[sig_mask, 'variant_b'].tolist()

    # ── Effect size interpretation ──────────────────────────────────────────────
    def interpret_cohens_d(d: float) -> str:
        if np.isnan(d):
            return 'N/A'
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'

    def interpret_odds_ratio(or_val: float) -> str:
        if np.isnan(or_val):
            return 'N/A'
        if or_val < 0.5:
            return 'large negative'
        elif or_val < 0.8:
            return 'moderate negative'
        elif or_val < 1.25:
            return 'negligible'
        elif or_val < 2.0:
            return 'moderate positive'
        else:
            return 'large positive'

    if not control_comparisons.empty:
        control_comparisons['cohens_d_interpretation'] = control_comparisons['cohens_d'].apply(interpret_cohens_d)
        control_comparisons['or_interpretation'] = control_comparisons['odds_ratio'].apply(interpret_odds_ratio)

    if not all_pairwise.empty:
        all_pairwise['cohens_d_interpretation'] = all_pairwise['cohens_d'].apply(interpret_cohens_d)
        all_pairwise['or_interpretation'] = all_pairwise['odds_ratio'].apply(interpret_odds_ratio)

    # ── Metadata ────────────────────────────────────────────────────────────────
    metadata = {
        'control_variant': control_variant,
        'treatment_variants': treatment_variants,
        'alpha': alpha,
        'confidence_level': confidence,
        'bonferroni_correction_applied': apply_correction,
        'n_control_comparisons': len(treatment_variants),
        'n_pairwise_comparisons': len(all_pairs),
        'total_users': len(df),
        'variants_analyzed': variants,
    }

    return {
        'variant_summary': variant_summary,
        'control_comparisons': control_comparisons,
        'all_pairwise': all_pairwise,
        'significant_variants': significant_variants,
        'analysis_metadata': metadata,
    }


def print_analysis_report(results: dict) -> None:
    """Pretty-print the analysis results."""
    meta = results['analysis_metadata']
    print("=" * 70)
    print("A/B/C/D TEST ANALYSIS REPORT")
    print("=" * 70)
    print(f"Control variant      : {meta['control_variant']}")
    print(f"Treatment variants   : {meta['treatment_variants']}")
    print(f"Total users          : {meta['total_users']:,}")
    print(f"Significance level   : α = {meta['alpha']}")
    print(f"Confidence level     : {meta['confidence_level'] * 100:.0f}%")
    print(f"Bonferroni correction: {'Yes' if meta['bonferroni_correction_applied'] else 'No'}")

    print("\n" + "─" * 70)
    print("PER-VARIANT SUMMARY")
    print("─" * 70)
    summary = results['variant_summary']
    print(summary[[
        'n_users', 'n_conversions', 'conversion_rate',
        'conv_ci_lower', 'conv_ci_upper',
        'mean_revenue', 'rev_ci_lower', 'rev_ci_upper',
        'total_revenue'
    ]].to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "─" * 70)
    print("COMPARISONS VS CONTROL")
    print("─" * 70)
    ctrl = results['control_comparisons']
    if not ctrl.empty:
        display_cols = [
            'variant_b', 'conv_lift_pct', 'chi2_p_value', 'conv_significant',
            'odds_ratio', 'or_interpretation',
            'rev_lift_pct', 't_p_value', 'rev_significant',
            'cohens_d', 'cohens_d_interpretation',
            'overall_significant', 'better_than_control'
        ]
        available = [c for c in display_cols if c in ctrl.columns]
        print(ctrl[available].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "─" * 70)
    print("ALL PAIRWISE COMPARISONS")
    print("─" * 70)
    pw = results['all_pairwise']
    if not pw.empty:
        display_cols = [
            'variant_a', 'variant_b',
            'conv_lift_pct', 'chi2_p_value', 'conv_significant',
            'odds_ratio', 'rev_lift_pct', 't_p_value', 'rev_significant',
            'cohens_d', 'overall_significant'
        ]
        available = [c for c in display_cols if c in pw.columns]
        print(pw[available].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "─" * 70)
    print("CONCLUSION")
    print("─" * 70)
    sig = results['significant_variants']
    if sig:
        print(f"✓ Variants significantly better than control ({meta['control_variant']}): {sig}")
    else:
        print(f"✗ No variants are statistically significantly better than control ({meta['control_variant']})")
    print("=" * 70)


# ── Demo / self-test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rng = np.random.default_rng(42)

    def generate_variant_data(n, conv_prob, mean_rev, std_rev, label):
        converted = rng.binomial(1, conv_prob, n)
        revenue