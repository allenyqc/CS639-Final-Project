import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)


def train_and_evaluate_multiclass(X, y, random_state=42):
    """
    Train and evaluate a multi-class classifier on a 10-class dataset.

    Best practices applied:
      - Train/test split is performed BEFORE any preprocessing/feature engineering.
      - Preprocessors are fit ONLY on training data (via Pipeline + CV on train).
      - Hyperparameters are tuned with cross-validation on training data only.
      - Test set is used exactly once, at the end, for final evaluation.
      - Macro-F1 and MCC are reported (robust to class imbalance).
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # 1) Split BEFORE any feature engineering / preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    # 2) Build a pipeline so preprocessing is fit ONLY on training folds.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",  # helps with class imbalance
            random_state=random_state,
        )),
    ])

    # 3) Hyperparameter tuning via CV on TRAINING data only.
    param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",  # appropriate for (potentially imbalanced) multi-class
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # 4) Final evaluation on the held-out test set (used only once).
    y_pred = best_model.predict(X_test)

    metrics = {
        "best_params": grid.best_params_,
        "cv_best_f1_macro": grid.best_score_,
        "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
        "test_f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "test_mcc": matthews_corrcoef(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return best_model, metrics


if __name__ == "__main__":
    from sklearn.datasets import load_digits

    data = load_digits()  # 10-class dataset (digits 0-9)
    model, results = train_and_evaluate_multiclass(data.data, data.target)

    print("Best params       :", results["best_params"])
    print(f"CV best F1 (macro): {results['cv_best_f1_macro']:.4f}")
    print(f"Test F1 (macro)   : {results['test_f1_macro']:.4f}")
    print(f"Test F1 (weighted): {results['test_f1_weighted']:.4f}")
    print(f"Test MCC          : {results['test_mcc']:.4f}")
    print("\nClassification report:\n", results["classification_report"])
    print("Confusion matrix:\n", results["confusion_matrix"])