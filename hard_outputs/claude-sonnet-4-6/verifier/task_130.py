```python
"""
Medical Dataset Anonymization Pipeline for ML Research

This module implements a comprehensive data anonymization pipeline that:
- Removes direct identifiers
- Generalizes quasi-identifiers
- Verifies and enforces k-anonymity
- Trains ML models on anonymized data
- Reports privacy and utility metrics
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import hashlib
import logging

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DIRECT_IDENTIFIERS = ['name', 'insurance_id', 'physician_name']

QUASI_IDENTIFIERS = ['age_bracket', 'zip_prefix', 'gender']

SENSITIVE_ATTRIBUTES = ['diagnosis', 'treatment', 'outcome']

AGE_BRACKETS = [
    (0,   17,  '0-17'),
    (18,  29,  '18-29'),
    (30,  44,  '30-44'),
    (45,  59,  '45-59'),
    (60,  74,  '60-74'),
    (75,  89,  '75-89'),
    (90,  150, '90+'),
]

BROADER_AGE_BRACKETS = [
    (0,   17,  'Minor'),
    (18,  64,  'Adult'),
    (65,  150, 'Senior'),
]

SUPPRESSION_MARKER = '*'


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Remove Direct Identifiers
# ─────────────────────────────────────────────────────────────────────────────

def remove_direct_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that directly identify individuals.

    Parameters
    ----------
    df : pd.DataFrame
        Raw patient DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with direct identifiers removed.
    """
    cols_to_drop = [c for c in DIRECT_IDENTIFIERS if c in df.columns]
    logger.info("Removing direct identifiers: %s", cols_to_drop)
    return df.drop(columns=cols_to_drop, errors='ignore').copy()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Generalise Quasi-Identifiers
# ─────────────────────────────────────────────────────────────────────────────

def _age_from_dob(dob: Any) -> Optional[int]:
    """Compute age in years from a date-of-birth value."""
    if pd.isna(dob):
        return None
    if isinstance(dob, (int, float)):
        return int(dob)
    if isinstance(dob, (datetime, date)):
        today = date.today()
        born = dob.date() if isinstance(dob, datetime) else dob
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    # Try parsing string
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y'):
        try:
            born = datetime.strptime(str(dob), fmt).date()
            today = date.today()
            return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        except ValueError:
            continue
    return None


def _assign_age_bracket(age: Optional[int], brackets: List[Tuple]) -> str:
    """Map an integer age to the appropriate bracket label."""
    if age is None:
        return 'Unknown'
    for lo, hi, label in brackets:
        if lo <= age <= hi:
            return label
    return 'Unknown'


def generalise_quasi_identifiers(
    df: pd.DataFrame,
    broad: bool = False
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Generalise quasi-identifiers and compute information-loss metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after direct-identifier removal.
    broad : bool
        If True, use broader age brackets (for additional generalisation).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        Generalised DataFrame and a dict of per-column information-loss scores.
    """
    df = df.copy()
    info_loss: Dict[str, float] = {}
    brackets = BROADER_AGE_BRACKETS if broad else AGE_BRACKETS

    # ── Date of birth → age bracket ──────────────────────────────────────────
    if 'date_of_birth' in df.columns:
        ages = df['date_of_birth'].apply(_age_from_dob)
        df['age_bracket'] = ages.apply(lambda a: _assign_age_bracket(a, brackets))
        # Information loss: fraction of distinct values collapsed
        n_original = ages.dropna().nunique()
        n_generalised = df['age_bracket'].nunique()
        info_loss['date_of_birth'] = (
            1.0 - (n_generalised / n_original) if n_original > 0 else 0.0
        )
        df.drop(columns=['date_of_birth'], inplace=True)
        logger.info("Generalised date_of_birth → age_bracket (%s brackets)", n_generalised)

    # ── ZIP code → first 3 digits ─────────────────────────────────────────────
    if 'zip_code' in df.columns:
        original_zips = df['zip_code'].astype(str).str.strip()
        df['zip_prefix'] = original_zips.str[:3].where(
            original_zips.str.len() >= 3, other='***'
        )
        n_original = original_zips.nunique()
        n_generalised = df['zip_prefix'].nunique()
        info_loss['zip_code'] = (
            1.0 - (n_generalised / n_original) if n_original > 0 else 0.0
        )
        df.drop(columns=['zip_code'], inplace=True)
        logger.info("Generalised zip_code → zip_prefix (%s unique prefixes)", n_generalised)

    # ── Gender: keep as-is (already categorical) ─────────────────────────────
    if 'gender' in df.columns:
        info_loss['gender'] = 0.0  # no generalisation applied

    return df, info_loss


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – K-Anonymity Verification
# ─────────────────────────────────────────────────────────────────────────────

def compute_k_anonymity(df: pd.DataFrame, qi_cols: List[str]) -> Tuple[int, pd.Series]:
    """
    Compute the k-anonymity level of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    qi_cols : List[str]
        Quasi-identifier column names present in df.

    Returns
    -------
    Tuple[int, pd.Series]
        (k value, Series of equivalence-class sizes)
    """
    present_qi = [c for c in qi_cols if c in df.columns]
    if not present_qi:
        logger.warning("No quasi-identifier columns found; k-anonymity undefined.")
        return 0, pd.Series(dtype=int)

    group_sizes = df.groupby(present_qi, dropna=False).size()
    k = int(group_sizes.min()) if len(group_sizes) > 0 else 0
    return k, group_sizes


def verify_k_anonymity(
    df: pd.DataFrame,
    k: int,
    qi_cols: List[str]
) -> Tuple[bool, int, pd.Series]:
    """
    Verify whether the DataFrame satisfies k-anonymity.

    Returns
    -------
    Tuple[bool, int, pd.Series]
        (satisfied, achieved_k, equivalence_class_sizes)
    """
    achieved_k, group_sizes = compute_k_anonymity(df, qi_cols)
    satisfied = achieved_k >= k
    logger.info(
        "K-anonymity check: required k=%d, achieved k=%d → %s",
        k, achieved_k, "PASS" if satisfied else "FAIL"
    )
    return satisfied, achieved_k, group_sizes


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Enforce K-Anonymity (Further Generalisation + Suppression)
# ─────────────────────────────────────────────────────────────────────────────

def _suppress_small_groups(
    df: pd.DataFrame,
    qi_cols: List[str],
    k: int
) -> Tuple[pd.DataFrame, int]:
    """
    Suppress (remove) equivalence classes with fewer than k records.

    Returns
    -------
    Tuple[pd.DataFrame, int]
        (filtered DataFrame, number of records suppressed)
    """
    present_qi = [c for c in qi_cols if c in df.columns]
    group_sizes = df.groupby(present_qi, dropna=False).transform('size')
    mask = group_sizes >= k
    suppressed = int((~mask).sum())
    logger.info("Suppressing %d records from small equivalence classes.", suppressed)
    return df[mask].copy(), suppressed


def enforce_k_anonymity(
    df: pd.DataFrame,
    k: int,
    qi_cols: List[str],
    max_suppression_rate: float = 0.20
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enforce k-anonymity through iterative generalisation and suppression.

    Strategy:
      1. Try broader age brackets.
      2. Suppress ZIP prefix to first 1 digit.
      3. Suppress gender.
      4. Suppress small equivalence classes (up to max_suppression_rate).

    Parameters
    ----------
    df : pd.DataFrame
    k : int
        Required k-anonymity level.
    qi_cols : List[str]
    max_suppression_rate : float
        Maximum fraction of records that may be suppressed.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (anonymised DataFrame, enforcement metadata)
    """
    n_original = len(df)
    metadata: Dict[str, Any] = {
        'steps_applied': [],
        'records_suppressed': 0,
        'suppression_rate': 0.0,
    }

    satisfied, achieved_k, _ = verify_k_anonymity(df, k, qi_cols)
    if satisfied:
        metadata['final_k'] = achieved_k
        return df, metadata

    # ── Step A: Broader age brackets ─────────────────────────────────────────
    if 'age_bracket' in df.columns:
        logger.info("Applying broader age brackets to improve k-anonymity.")
        df = df.copy()
        # Re-derive age from bracket midpoints is not possible; apply broader
        # mapping directly on existing brackets
        narrow_to_broad = {
            '0-17': 'Minor',
            '18-29': 'Adult', '30-44': 'Adult', '45-59': 'Adult',
            '60-74': 'Senior', '75-89': 'Senior', '90+': 'Senior',
            'Unknown': 'Unknown',
        }
        df['age_bracket'] = df['age_bracket'].map(
            lambda x: narrow_to_broad.get(x, x)
        )
        metadata['steps_applied'].append('broader_age_brackets')
        satisfied, achieved_k, _ = verify_k_anonymity(df, k, qi_cols)
        if satisfied:
            metadata['final_k'] = achieved_k
            return df, metadata

    # ── Step B: Truncate ZIP prefix to 1 digit ────────────────────────────────
    if 'zip_prefix' in df.columns:
        logger.info("Truncating zip_prefix to 1 digit.")
        df = df.copy()
        df['zip_prefix'] = df['zip_prefix'].astype(str).str[:1].where(
            df['zip_prefix'].astype(str) != '***', other='*'
        )
        metadata['steps_applied'].append('zip_prefix_1digit')
        satisfied, achieved_k, _ = verify_k_anonymity(df, k, qi_cols)
        if satisfied:
            metadata['final_k'] = achieved_k
            return df, metadata

    # ── Step C: Suppress gender ───────────────────────────────────────────────
    if 'gender' in df.columns:
        logger.info("Suppressing gender column.")
        df = df.copy()
        df['gender'] = SUPPRESSION_MARKER
        metadata['steps_applied'].append('gender_suppressed')
        satisfied, achieved_k, _ = verify_k_anonymity(df, k, qi_cols)
        if satisfied:
            metadata['final_k'] = achieved_k
            return df, metadata

    # ── Step D: Record-level suppression ─────────────────────────────────────
    logger.info("Applying record-level suppression for remaining violations.")
    df_suppressed, n_suppressed = _suppress_small_groups(df, qi_cols, k)
    suppression_rate = n_suppressed / n_original if n_original > 0 else 0.0

    if suppression_rate <= max_suppression_rate:
        metadata['steps_applied'].append('record_suppression')
        metadata['records_suppressed'] = n_suppressed
        metadata['suppression_rate'] = suppression_rate
        df = df_suppressed
    else:
        logger.warning(
            "Suppression rate %.2f%% exceeds limit %.2f%%. "
            "Accepting best-effort k-anonymity.",
            suppression_rate * 100, max_suppression_rate * 100
        )
        metadata['steps_applied'].append('record_suppression_partial')
        metadata['records_suppressed'] = n_suppressed
        metadata['suppression_rate'] = suppression_rate

    _, achieved_k, _ = verify_k_anonymity(df, k, qi_cols)
    metadata['final_k'] = achieved_k
    return df, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Privacy Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_privacy_metrics(
    original_df: pd.DataFrame,
    anonymised_df: pd.DataFrame,
    qi_cols: List[str],
    info_loss: Dict[str, float],
    enforcement_meta: Dict[str, Any],
    k_required: int,
) -> Dict[str, Any]:
    """
    Compute comprehensive privacy metrics.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - k_required, k_achieved
        - k_anonymity_satisfied
        - equivalence_class_stats (min/max/mean/median size)
        - information_loss (per column + aggregate)
        - suppression_rate
        - records_retained
        - enforcement_steps
    """
    achieved_k, group_sizes = compute_k_anonymity(anonymised_df, qi_cols)

    eq_stats: Dict[str, float] = {}
    if len(group_sizes) > 0:
        eq_stats = {
            'min_size':    float(group_sizes.min()),
            'max_size':    float(group_sizes.max()),
            'mean_size':   float(group_sizes.mean()),
            'median_size': float(group_sizes.median()),
            'num_classes': int(len(group_sizes)),
        }

    # Aggregate information loss (mean of per-column losses)
    agg_info_loss = float(np.mean(list(info_loss.values()))) if info_loss else 0.0

    metrics = {
        'k_required':             k_required,
        'k_achieved':             achieved_k,
        'k_anonymity_satisfied':  achieved_k >= k_required,
        'equivalence_class_stats': eq_stats,
        'information_loss_per_column': info_loss,
        'information_loss_aggregate':  agg_info_loss,
        'suppression_rate':       enforcement_meta.get('suppression_rate', 0.0),
        'records_original':       len(original_df),
        'records_retained':       len(anonymised_df),
        'enforcement_steps':      enforcement_meta.get('steps_applied', []),
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Prepare Features for ML
# ─────────────────────────────────────────────────────────────────────────────

def prepare_ml_features(
    df: pd.DataFrame,
    target_col: str = 'outcome',
    qi_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], LabelEncoder]:
    """
    Encode anonymised data for ML training.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    qi_cols : Optional[List[str]]
        Quasi-identifier columns to include as features.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, List[str], LabelEncoder]
        (X, y, feature_names, target_encoder)
    """
    if qi_cols is None:
        qi_cols = QUASI_IDENTIFIERS

    df = df.copy()

    # ── Target ───────────────────────────────────────────────────────────────
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    target_enc = LabelEncoder()
    y = pd.Series(
        target_enc.fit_transform(df[target_col].astype(str)),
        index=df.index,
        name=target_col
    )

    # ── Features ─────────────────────────────────────────────────────────────
    feature_cols = [
        c for c in df.columns
        if c != target_col
        and c not in DIRECT_IDENTIFIERS
        and c not in ['date_of_birth', 'zip_code']
    ]

    X = df[feature_cols].copy()

    # Encode categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Fill any remaining NaNs
    X = X.fillna(-1)

    logger.info("ML features prepared: %d columns, %d rows.", len(feature_cols), len(X))
    return X, y, feature_cols, target_enc


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 – Train Classifiers
# ─────────────────────────────────────────────────────────────────────────────

def train_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_encoder: LabelEncoder,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Train multiple classifiers and evaluate them.

    Parameters
    ----------
    X_train, y_train : training data