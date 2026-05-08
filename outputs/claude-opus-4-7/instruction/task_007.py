import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    classification_report,
)


def select_features_and_train(X, y, k=10, test_size=0.2, random_state=42):
    """
    Select the top-k most informative features using mutual information,
    train a decision tree on the selected features, and report metrics.

    Best practices applied:
      - Train/test split is performed BEFORE any feature engineering.
      - Scaler and feature selector are fit ONLY on training data.
      - For imbalanced datasets, F1/AUC/MCC are reported in addition to accuracy.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # 1) Split FIRST, before any preprocessing or feature selection
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2) Build a pipeline so scaler + selector are fit ONLY on training data
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "selector",
                SelectKBest(
                    score_func=lambda X_, y_: mutual_info_classif(
                        X_, y_, random_state=random_state
                    ),
                    k=min(k, X_train.shape[1]),
                ),
            ),
            (
                "clf",
                DecisionTreeClassifier(random_state=random_state),
            ),
        ]
    )

    # 3) Fit on training data only
    pipeline.fit(X_train, y_train)

    # 4) Evaluate on the held-out test set
    y_pred = pipeline.predict(X_test)

    # Detect imbalance
    classes, counts = np.unique(y_train, return_counts=True)
    imbalance_ratio = counts.max() / counts.min()
    is_imbalanced = imbalance_ratio > 1.5
    is_binary = len(classes) == 2

    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(
        y_test, y_pred, average="binary" if is_binary else "macro"
    )

    auc = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test)
            if is_binary:
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = roc_auc_score(y_test, proba, multi_class="ovr")
        except Exception:
            auc = None

    # Report
    print(f"Class distribution (train): {dict(zip(classes, counts))}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f} "
          f"({'imbalanced' if is_imbalanced else 'balanced'})")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test F1 ({'binary' if is_binary else 'macro'}): {f1:.4f}")
    if auc is not None:
        print(f"Test ROC-AUC: {auc:.4f}")
    print(f"Test MCC: {mcc:.4f}")

    if is_imbalanced:
        print("\nNOTE: Dataset is imbalanced — prefer F1/AUC/MCC over accuracy.")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Selected feature indices (relative to original X)
    selected_idx = pipeline.named_steps["selector"].get_support(indices=True)

    return {
        "pipeline": pipeline,
        "selected_features": selected_idx,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "mcc": mcc,
    }


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    results = select_features_and_train(data.data, data.target, k=10)
    print("\nSelected feature indices:", results["selected_features"])
    print("Selected feature names:",
          [data.feature_names[i] for i in results["selected_features"]])