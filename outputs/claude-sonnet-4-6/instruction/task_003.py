```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def tune_and_evaluate_random_forest(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    scoring: str = "f1_weighted",
    param_grid: dict | None = None,
    verbose: int = 1,
) -> dict:
    """
    Tune hyperparameters for a Random Forest classifier using GridSearchCV,
    then evaluate the best model on a held-out test set.

    Best-practice guarantees
    ------------------------
    * Train/test split happens FIRST — before any preprocessing or feature engineering.
    * The StandardScaler is fitted ONLY on X_train (inside a Pipeline).
    * GridSearchCV uses only training data for cross-validation.
    * The test set is touched exactly once — for final evaluation.
    * Evaluation metrics (F1, AUC, MCC) are appropriate for imbalanced datasets.

    Parameters
    ----------
    X            : Feature matrix (array-like or DataFrame).
    y            : Target vector (array-like or Series).
    test_size    : Fraction of data reserved for the final test set.
    random_state : Seed for reproducibility.
    cv_folds     : Number of stratified CV folds used during grid search.
    scoring      : Metric used to rank hyperparameter combinations.
                   Recommended: 'f1_weighted', 'roc_auc', or 'matthews_corrcoef'.
    param_grid   : Custom hyperparameter grid. A sensible default is used if None.
    verbose      : Verbosity level passed to GridSearchCV.

    Returns
    -------
    results : dict with keys
        'best_params'      – best hyperparameters found by GridSearchCV
        'best_cv_score'    – best cross-validated score on the training set
        'test_f1'          – weighted F1 on the test set
        'test_auc'         – macro-averaged ROC-AUC on the test set
        'test_mcc'         – Matthews Correlation Coefficient on the test set
        'pipeline'         – fitted Pipeline (scaler + best RandomForest)
        'X_test'           – held-out features (for downstream inspection)
        'y_test'           – held-out labels
        'y_pred'           – predictions on the test set
        'y_proba'          – predicted probabilities on the test set
    """

    # ------------------------------------------------------------------
    # 0. Encode labels if necessary
    # ------------------------------------------------------------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(np.asarray(y).ravel())
    n_classes = len(le.classes_)

    # ------------------------------------------------------------------
    # 1. Train / test split — FIRST, before any preprocessing
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,          # preserve class distribution
    )

    print(f"Dataset split: {len(X_train)} train samples | {len(X_test)} test samples")
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Training class distribution: { {le.classes_[k]: v for k, v in zip(unique, counts)} }")

    # ------------------------------------------------------------------
    # 2. Build a Pipeline (scaler fitted ONLY on training data)
    # ------------------------------------------------------------------
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),          # fit_transform on train, transform on test
            ("clf", RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",           # handles class imbalance
            )),
        ]
    )

    # ------------------------------------------------------------------
    # 3. Define hyperparameter grid
    # ------------------------------------------------------------------
    if param_grid is None:
        param_grid = {
            "clf__n_estimators":      [100, 300, 500],
            "clf__max_depth":         [None, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf":  [1, 2, 4],
            "clf__max_features":      ["sqrt", "log2"],
        }

    # ------------------------------------------------------------------
    # 4. GridSearchCV — uses ONLY training data
    # ------------------------------------------------------------------
    cv_strategy = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_strategy,
        refit=True,          # refit best model on the full training set
        n_jobs=-1,
        verbose=verbose,
        return_train_score=False,
    )

    print(f"\nRunning GridSearchCV with {cv_folds}-fold stratified CV "
          f"(scoring='{scoring}') …")
    grid_search.fit(X_train, y_train)

    best_params    = grid_search.best_params_
    best_cv_score  = grid_search.best_score_
    best_pipeline  = grid_search.best_estimator_   # already refitted on full X_train

    print(f"\nBest CV {scoring}: {best_cv_score:.4f}")
    print(f"Best hyperparameters:\n  {best_params}")

    # ------------------------------------------------------------------
    # 5. Final evaluation on the held-out test set (touched ONCE)
    # ------------------------------------------------------------------
    y_pred  = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)

    # F1 (weighted handles imbalance)
    test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (macro, one-vs-rest for multi-class)
    auc_kwargs = dict(multi_class="ovr", average="macro") if n_classes > 2 else {}
    test_auc = roc_auc_score(
        y_test,
        y_proba if n_classes > 2 else y_proba[:, 1],
        **auc_kwargs,
    )

    # Matthews Correlation Coefficient
    test_mcc = matthews_corrcoef(y_test, y_pred)

    print("\n" + "=" * 55)
    print("TEST SET EVALUATION (held-out, never used for tuning)")
    print("=" * 55)
    print(f"  Weighted F1 : {test_f1:.4f}")
    print(f"  ROC-AUC     : {test_auc:.4f}")
    print(f"  MCC         : {test_mcc:.4f}")
    print("\nDetailed classification report:")
    print(
        classification_report(
            y_test, y_pred,
            target_names=[str(c) for c in le.classes_],
            zero_division=0,
        )
    )

    # ------------------------------------------------------------------
    # 6. Confusion matrix plot
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[str(