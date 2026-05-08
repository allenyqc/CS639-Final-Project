```python
"""
Medical Dataset Anonymization Pipeline for ML Research.

This module implements a HIPAA-inspired data anonymization pipeline that:
- Removes direct identifiers
- Generalizes quasi-identifiers
- Verifies and enforces k-anonymity
- Trains a classifier on anonymized data
- Reports utility and privacy metrics
"""

import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECT_IDENTIFIERS = ["name", "insurance_id", "physician_name"]
QUASI_IDENTIFIERS = ["age_bracket", "zip3", "gender"]
SENSITIVE_ATTRIBUTES = ["diagnosis", "treatment", "outcome"]

AGE_BRACKETS = [0, 18, 30, 45, 60, 75, 120]
AGE_LABELS = ["0-17", "18-29", "30-44", "45-59", "60-74", "75+"]


# ---------------------------------------------------------------------------
# Step 1 – Remove direct identifiers
# ---------------------------------------------------------------------------


def remove_direct_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are direct identifiers."""
    cols_to_drop = [c for c in DIRECT_IDENTIFIERS if c in df.columns]
    return df.drop(columns=cols_to_drop)


# ---------------------------------------------------------------------------
# Step 2 – Generalise quasi-identifiers
# ---------------------------------------------------------------------------


def _compute_age(dob_series: pd.Series) -> pd.Series:
    """Convert date-of-birth strings/dates to integer age."""
    today = datetime.today()
    dob = pd.to_datetime(dob_series, errors="coerce")
    age = (today - dob).dt.days // 365
    return age


def generalize_quasi_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generalise quasi-identifiers:
      - date_of_birth  → age_bracket  (e.g. '30-44')
      - zip_code       → zip3         (first 3 digits)
      - gender         → kept as-is (already categorical)
    """
    df = df.copy()

    if "date_of_birth" in df.columns:
        age = _compute_age(df["date_of_birth"])
        df["age_bracket"] = pd.cut(
            age,
            bins=AGE_BRACKETS,
            labels=AGE_LABELS,
            right=False,
        ).astype(str)
        df.drop(columns=["date_of_birth"], inplace=True)

    if "zip_code" in df.columns:
        df["zip3"] = df["zip_code"].astype(str).str[:3]
        df.drop(columns=["zip_code"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Step 3 – k-anonymity verification
# ---------------------------------------------------------------------------


def compute_k_anonymity(df: pd.DataFrame, quasi_ids: list[str]) -> int:
    """Return the minimum group size across all quasi-identifier combinations."""
    available_qi = [q for q in quasi_ids if q in df.columns]
    if not available_qi:
        return len(df)
    group_sizes = df.groupby(available_qi, observed=True).size()
    return int(group_sizes.min()) if len(group_sizes) > 0 else 0


def check_k_anonymity(df: pd.DataFrame, k: int, quasi_ids: list[str]) -> bool:
    """Return True if the dataset satisfies k-anonymity."""
    return compute_k_anonymity(df, quasi_ids) >= k


# ---------------------------------------------------------------------------
# Step 4 – Enforce k-anonymity via further generalisation / suppression
# ---------------------------------------------------------------------------


def _suppress_rare_groups(
    df: pd.DataFrame, quasi_ids: list[str], k: int
) -> pd.DataFrame:
    """Remove records belonging to groups smaller than k."""
    available_qi = [q for q in quasi_ids if q in df.columns]
    if not available_qi:
        return df
    group_sizes = df.groupby(available_qi, observed=True).transform("size")
    return df[group_sizes >= k].copy()


def _coarsen_age_bracket(df: pd.DataFrame) -> pd.DataFrame:
    """Merge age brackets into coarser bins."""
    coarse_map = {
        "0-17": "0-29",
        "18-29": "0-29",
        "30-44": "30-59",
        "45-59": "30-59",
        "60-74": "60+",
        "75+": "60+",
    }
    if "age_bracket" in df.columns:
        df = df.copy()
        df["age_bracket"] = df["age_bracket"].map(coarse_map).fillna(df["age_bracket"])
    return df


def _coarsen_zip3(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce zip3 to first 2 digits."""
    if "zip3" in df.columns:
        df = df.copy()
        df["zip3"] = df["zip3"].astype(str).str[:2]
    return df


def enforce_k_anonymity(
    df: pd.DataFrame,
    k: int = 5,
    quasi_ids: list[str] | None = None,
    max_iterations: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Iteratively apply generalisation and suppression until k-anonymity is met.

    Returns the anonymised DataFrame and a dict with privacy metrics.
    """
    if quasi_ids is None:
        quasi_ids = QUASI_IDENTIFIERS

    original_size = len(df)
    iterations_used = 0
    generalisation_steps: list[str] = []

    for iteration in range(max_iterations):
        current_k = compute_k_anonymity(df, quasi_ids)
        if current_k >= k:
            break

        iterations_used = iteration + 1

        if iteration == 0:
            df = _coarsen_age_bracket(df)
            generalisation_steps.append("coarsen_age_bracket")
        elif iteration == 1:
            df = _coarsen_zip3(df)
            generalisation_steps.append("coarsen_zip3")
        else:
            # Last resort: suppress rare groups
            df = _suppress_rare_groups(df, quasi_ids, k)
            generalisation_steps.append("suppress_rare_groups")

    final_k = compute_k_anonymity(df, quasi_ids)
    records_removed = original_size - len(df)
    information_loss = 1.0 - (len(df) / original_size) if original_size > 0 else 0.0

    privacy_metrics = {
        "k_anonymity_achieved": final_k,
        "k_target": k,
        "k_satisfied": final_k >= k,
        "records_removed": records_removed,
        "records_remaining": len(df),
        "information_loss_ratio": round(information_loss, 4),
        "generalisation_steps": generalisation_steps,
        "iterations_used": iterations_used,
    }

    return df, privacy_metrics


# ---------------------------------------------------------------------------
# Step 5 – Compute information loss (additional utility metric)
# ---------------------------------------------------------------------------


def compute_information_loss(
    original_df: pd.DataFrame, anonymised_df: pd.DataFrame
) -> dict[str, float]:
    """
    Estimate information loss by comparing cardinality of quasi-identifiers
    before and after generalisation.
    """
    metrics: dict[str, float] = {}

    # Cardinality reduction for zip_code → zip3
    if "zip_code" in original_df.columns and "zip3" in anonymised_df.columns:
        orig_card = original_df["zip_code"].nunique()
        anon_card = anonymised_df["zip3"].nunique()
        metrics["zip_cardinality_reduction"] = round(
            1.0 - (anon_card / orig_card) if orig_card > 0 else 0.0, 4
        )

    # Cardinality reduction for date_of_birth → age_bracket
    if "date_of_birth" in original_df.columns and "age_bracket" in anonymised_df.columns:
        orig_card = original_df["date_of_birth"].nunique()
        anon_card = anonymised_df["age_bracket"].nunique()
        metrics["dob_cardinality_reduction"] = round(
            1.0 - (anon_card / orig_card) if orig_card > 0 else 0.0, 4
        )

    return metrics


# ---------------------------------------------------------------------------
# Step 6 – ML pipeline (train on anonymised data)
# ---------------------------------------------------------------------------


def build_ml_pipeline(
    categorical_features: list[str],
    numerical_features: list[str],
) -> Pipeline:
    """Build a scikit-learn Pipeline with preprocessing and a RandomForest."""

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )
    return pipeline


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str = "outcome",
    test_size: float = 0.2,
    k_folds: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Split data FIRST, then fit preprocessing and model ONLY on training data.
    Evaluate on the held-out test set.

    Returns the anonymised DataFrame (unchanged) and model results dict.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].astype(str))
    X = df.drop(columns=[target_col])

    # -----------------------------------------------------------------------
    # BEST PRACTICE: Split BEFORE any fitting
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # Identify feature types from training data only
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    pipeline = build_ml_pipeline(categorical_features, numerical_features)

    # -----------------------------------------------------------------------
    # BEST PRACTICE: Hyperparameter search via CV on training data only
    # -----------------------------------------------------------------------
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_leaf": [1, 5],
    }

    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        refit=True,  # refit best model on full training set
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # -----------------------------------------------------------------------
    # BEST PRACTICE: Evaluate ONLY on held-out test set
    # -----------------------------------------------------------------------
    y_pred = best_model.predict(X_test)
    y_proba = (
        best_model.predict_proba(X_test)
        if hasattr(best_model, "predict_proba")
        else None
    )

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    roc_auc: float | None = None
    if y_proba is not None and len(le.classes_) == 2:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    elif y_proba is not None and len(le.classes_) > 2:
        roc_auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="weighted"
        )

    report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    model_results = {
        "best_params": grid_search.best_params_,
        "cv_best_f1_weighted": round(grid_search.best_score_, 4),
        "test_accuracy": round(accuracy, 4),
        "test_f1_weighted": round(f1, 4),
        "test_roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "classification_report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "target_classes": list(le.classes_),
        "feature_importances": _extract_feature_importances(
            best_model, categorical_features, numerical_features
        ),
    }

    return df, model_results


def _extract_feature_importances(
    pipeline: Pipeline,
    categorical_features: list[str],
    numerical_features: list[str],
) -> dict[str, float]:
    """Extract feature importances from the fitted pipeline."""
    try:
        clf = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Get feature names after one-hot encoding
        cat_transformer = preprocessor.named_transformers_.get("cat")
        if cat_transformer is not None:
            ohe = cat_transformer.named_steps["onehot"]
            cat_names = list(ohe.get_feature_names_out(categorical_features))
        else:
            cat_names = []

        all_feature_names = cat_names + numerical_features
        importances = clf.feature_importances_

        return {
            name: round(float(imp), 6)
            for name, imp in zip(all_feature_names, importances)
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def anonymize_and_train(
    df: pd.DataFrame,
    k: int = 5,
    target_col: str = "outcome",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """
    Full anonymization + ML pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw patient DataFrame with expected columns.
    k : int
        Minimum k for k-anonymity (default 5).
    target_col : str
        Column to predict (default 'outcome').
    test_size : float
        Fraction of data for the test split (default 0.2).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    anonymised_df : pd.DataFrame
        The anonymised dataset.
    privacy_metrics : dict
        k-anonymity level, information loss, suppression stats.
    model_results : dict
        Classifier performance on the held-out test set.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # --- Anonymisation steps ------------------------------------------------

    # 1. Remove direct identifiers
    df_anon = remove_direct_identifiers(df)

    # 2. Generalise quasi-identifiers
    df_anon = generalize_quasi_identifiers(df_anon)

    # 3. Compute additional information loss before enforcement
    info_loss = compute_information_loss(df, df_anon)

    # 4. Enforce k-anonymity
    df_anon, privacy_metrics = enforce_k_anonymity(
        df_anon, k=k, quasi_ids=QUASI_IDENTIFIERS
    )
    privacy_metrics["information_loss_details"] = info_loss

    # --- ML pipeline --------------------------------------------------------

    if target_col not in df_anon.columns:
        raise ValueError(
            f"Target column '{target_col}' was removed during anonymisation."
        )

    _, model_results = train_and_evaluate(
        df_anon,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    return df_anon, privacy_metrics, model_results


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------


def _generate_synthetic_data(n: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic patient DataFrame for testing."""
    rng = np.random.default_rng(random_state)

    first_names = ["Alice", "Bob", "Carol", "David", "Eve"]
    last_names = ["Smith", "Jones", "Brown", "Taylor", "Wilson"]
    genders = ["M", "F", "Other"]
    diagnoses = ["Hypertension", "Diabetes", "Asthma", "Arthritis", "COPD"]
    treatments = ["Medication", "Surgery", "Therapy", "Lifestyle", "Monitoring"]
    outcomes = ["Improved", "Stable", "Deteriorated"]

    dobs = pd.date_range("1940-01-01", "2005-12-31", periods=n)
    dobs = dobs[rng.integers(0, n, size=n)]

    data = {
        "name": [
            f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n)
        ],
        "date_of_birth": dobs.strftime("%Y-%m-%d"),
        "zip_code": [str(rng.integers(10000, 99999)) for _ in range(n)],
        "gender": rng.choice(genders, size=n),
        "diagnosis": rng.choice(diagnoses, size=n),
        "treatment": rng.choice(treatments, size=n),
        "outcome": rng.choice(outcomes, size=n, p=[0.5, 0.3, 0.2]),
        "insurance_id": [f"INS{rng.integers(100000, 999999)}" for _ in range(n)],
        "physician_name": [
            f"Dr. {rng.choice(last_names)}" for _ in range(n)
        ],
    }
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("=" * 60)
    print("Medical Dataset Anonymization Pipeline – Demo")
    print("=" * 60)

    raw_df = _generate_synthetic_data(n=500)
    print(f"\nRaw dataset shape: {raw_df.shape}")
    print(f"Columns: {list(raw_df.columns)}\n")

    anonymised_df, privacy_metrics, model_results = anonymize_and_train(
        raw_df, k=5, target_col="outcome", test_size=0.2, random_state=42
    )

    print("── Privacy Metrics