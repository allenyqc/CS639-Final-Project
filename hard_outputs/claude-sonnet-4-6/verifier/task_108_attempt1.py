import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_sample_weight
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def engineer_features(df: pd.DataFrame, top_diag_codes: list = None, fit: bool = True):
    """
    Engineer features from raw EHR DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.
    top_diag_codes : list, optional
        Pre-computed top-20 diagnosis codes (used during transform/test phase).
    fit : bool
        If True, compute top_diag_codes from data (training phase).

    Returns
    -------
    features : pd.DataFrame
        Engineered feature matrix.
    top_diag_codes : list
        Top-20 diagnosis codes used for one-hot encoding.
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # 1. Parse dates
    # ------------------------------------------------------------------ #
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df = df.sort_values(['patient_id', 'admission_date']).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 2. Patient history features
    # ------------------------------------------------------------------ #
    df['prior_admissions'] = (
        df.groupby('patient_id').cumcount()          # 0-indexed count of prior rows
    )

    df['days_since_last_admission'] = (
        df.groupby('patient_id')['admission_date']
        .diff()
        .dt.days
        .fillna(-1)                                  # -1 signals first admission
    )

    # ------------------------------------------------------------------ #
    # 3. Interaction feature
    # ------------------------------------------------------------------ #
    df['los_x_procedures'] = df['length_of_stay'] * df['num_procedures']

    # ------------------------------------------------------------------ #
    # 4. Gender encoding
    # ------------------------------------------------------------------ #
    df['gender_encoded'] = (df['gender'].str.upper() == 'M').astype(int)

    # ------------------------------------------------------------------ #
    # 5. One-hot encoding of top-20 diagnosis codes
    # ------------------------------------------------------------------ #
    # Explode comma-separated codes into individual tokens
    all_codes_series = (
        df['diagnosis_codes']
        .fillna('')
        .str.split(',')
        .explode()
        .str.strip()
    )

    if fit:
        code_counts = Counter(all_codes_series[all_codes_series != ''])
        top_diag_codes = [code for code, _ in code_counts.most_common(20)]

    for code in top_diag_codes:
        col_name = f'diag_{code.replace(".", "_")}'
        df[col_name] = df['diagnosis_codes'].fillna('').apply(
            lambda x: int(code in [c.strip() for c in x.split(',')])
        )

    # ------------------------------------------------------------------ #
    # 6. Assemble final feature columns
    # ------------------------------------------------------------------ #
    base_features = [
        'length_of_stay',
        'num_procedures',
        'num_medications',
        'age',
        'gender_encoded',
        'prior_admissions',
        'days_since_last_admission',
        'los_x_procedures',
    ]
    diag_features = [f'diag_{c.replace(".", "_")}' for c in top_diag_codes]
    feature_cols = base_features + diag_features

    features = df[feature_cols].copy()
    return features, top_diag_codes


def train_readmission_model(df: pd.DataFrame,
                            test_size: float = 0.2,
                            random_state: int = 42,
                            n_estimators: int = 300,
                            learning_rate: float = 0.05,
                            max_depth: int = 4,
                            subsample: float = 0.8,
                            n_calibration_bins: int = 10):
    """
    Train a Gradient Boosting model to predict 30-day hospital readmission.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns:
        'patient_id', 'admission_date', 'diagnosis_codes',
        'length_of_stay', 'num_procedures', 'num_medications',
        'age', 'gender', 'readmitted_30d'.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Random seed for reproducibility.
    n_estimators : int
        Number of boosting stages.
    learning_rate : float
        Shrinkage parameter.
    max_depth : int
        Maximum depth of individual trees.
    subsample : float
        Fraction of samples used per tree.
    n_calibration_bins : int
        Number of bins for calibration curve.

    Returns
    -------
    results : dict with keys:
        'model'              – trained GradientBoostingClassifier
        'metrics'            – dict of AUC, F1, Precision, Recall
        'calibration_data'   – dict with 'fraction_of_positives' and 'mean_predicted_value'
        'feature_importances'– pd.Series sorted descending
        'roc_curve'          – dict with 'fpr', 'tpr', 'thresholds'
        'top_diag_codes'     – list of top-20 ICD codes used
    """

    # ------------------------------------------------------------------ #
    # Validate required columns
    # ------------------------------------------------------------------ #
    required_cols = {
        'patient_id', 'admission_date', 'diagnosis_codes',
        'length_of_stay', 'num_procedures', 'num_medications',
        'age', 'gender', 'readmitted_30d'
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # ------------------------------------------------------------------ #
    # Train / test split  (split on rows, stratified by target)
    # ------------------------------------------------------------------ #
    target = df['readmitted_30d'].astype(int)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=target
    )

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)

    y_train = target.loc[train_idx].reset_index(drop=True)
    y_test  = target.loc[test_idx].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Feature engineering
    # ------------------------------------------------------------------ #
    X_train, top_diag_codes = engineer_features(df_train, fit=True)
    X_test,  _              = engineer_features(df_test,  top_diag_codes=top_diag_codes, fit=False)

    feature_names = X_train.columns.tolist()

    # ------------------------------------------------------------------ #
    # Sample weights to handle class imbalance
    # (GradientBoostingClassifier does not support class_weight natively,
    #  so we use sample_weight computed from class frequencies)
    # ------------------------------------------------------------------ #
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # ------------------------------------------------------------------ #
    # Model training
    # ------------------------------------------------------------------ #
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #
    y_prob  = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    # ------------------------------------------------------------------ #
    # Evaluation metrics
    # ------------------------------------------------------------------ #
    auc       = roc_auc_score(y_test, y_prob)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)

    metrics = {
        'AUC':       round(auc, 4),
        'F1':        round(f1, 4),
        'Precision': round(precision, 4),
        'Recall':    round(recall, 4),
    }

    # ------------------------------------------------------------------ #
    # Calibration curve
    # ------------------------------------------------------------------ #
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_prob, n_bins=n_calibration_bins, strategy='uniform'
    )
    calibration_data = {
        'fraction_of_positives': fraction_of_positives.tolist(),
        'mean_predicted_value':  mean_predicted_value.tolist(),
    }

    # ------------------------------------------------------------------ #
    # ROC curve
    # ------------------------------------------------------------------ #
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_data = {
        'fpr':        fpr.tolist(),
        'tpr':        tpr.tolist(),
        'thresholds': thresholds.tolist(),
    }

    # ------------------------------------------------------------------ #
    # Feature importances
    # ------------------------------------------------------------------ #
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    # ------------------------------------------------------------------ #
    # Return results
    # ------------------------------------------------------------------ #
    results = {
        'model':               model,
        'metrics':             metrics,
        'calibration_data':    calibration_data,
        'feature_importances': importances,
        'roc_curve':           roc_data,
        'top_diag_codes':      top_diag_codes,
    }

    return results


def predict(model, top_diag_codes: list, new_df: pd.DataFrame) -> np.ndarray:
    """
    Generate readmission probability predictions for new data.

    Parameters
    ----------
    model : trained GradientBoostingClassifier
    top_diag_codes : list
        Top-20 ICD codes returned by train_readmission_model.
    new_df : pd.DataFrame
        New patient visit data (same schema as training data, without 'readmitted_30d').

    Returns
    -------
    probabilities : np.ndarray of shape (n_samples,)
        Predicted probability of 30-day readmission.
    """
    X_new, _ = engineer_features(new_df, top_diag_codes=top_diag_codes, fit=False)
    return model.predict_proba(X_new)[:, 1]


# --------------------------------------------------------------------------- #
# Demo / smoke-test
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    np.random.seed(0)
    N = 2000

    icd_pool = [
        'I10', 'E11', 'J44', 'N18', 'I50', 'Z87', 'M54',
        'K21', 'F32', 'I25', 'J18', 'E78', 'G47', 'R05',
        'Z96', 'I48', 'E14', 'N39', 'J06', 'K92', 'L89',
        'M17', 'C34', 'I63', 'Z79'
    ]

    def random_codes():
        k = np.random.randint(1, 6)
        return ','.join(np.random.choice(icd_pool, k, replace=False))

    patient_ids = np.random.randint(1, 400, size=N)
    base_dates  = pd.date_range('2018-01-01', periods=N, freq='12h')

    demo_df = pd.DataFrame({
        'patient_id':     patient_ids,
        'admission_date': base_dates,
        'diagnosis_codes': [random_codes() for _ in range(N)],
        'length_of_stay':  np.random.randint(1, 20, N),
        'num_procedures':  np.random.randint(0, 10, N),
        'num_medications': np.random.randint(1, 15, N),
        'age':             np.random.randint(18, 90, N),
        'gender':          np.random.choice(['M', 'F'], N),
        'readmitted_30d':  np.random.binomial(1, 0.2, N),
    })

    results = train_readmission_model(demo_df, n_estimators=100)

    print("=== Metrics ===")
    for k, v in results['metrics'].items():
        print(f"  {k}: {v}")

    print("\n=== Top-10 Feature Importances ===")
    print(results['feature_importances'].head(10).to_string())

    print("\n=== Calibration Data (first 5 bins) ===")
    cal = results['calibration_data']
    for fop, mpv in zip(cal['fraction_of_positives'][:5],
                        cal['mean_predicted_value'][:5]):
        print(f"  mean_pred={mpv:.3f}  frac_pos={fop:.3f}")

    print("\n=== Sample Predictions on 5 new rows ===")
    sample_new = demo_df.head(5).drop(columns=['readmitted_30d'])
    probs = predict(results['model'], results['top_diag_codes'], sample_new)
    print(probs)