import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    classification_report
)


def select_features_and_train(
    X=None,
    y=None,
    k: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = 5,
):
    """
    Select the top-k most informative features using mutual information,
    train a Decision Tree on the selected features, and report test metrics.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), optional
        Feature matrix. Defaults to the breast-cancer dataset.
    y : array-like of shape (n_samples,), optional
        Target vector. Defaults to the breast-cancer dataset.
    k : int
        Number of top features to select (default 10).
    test_size : float
        Fraction of data reserved for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.
    max_depth : int
        Maximum depth of the decision tree.

    Returns
    -------
    results : dict
        Dictionary containing selected feature indices, the trained pipeline
        objects, and all evaluation metrics.
    """
    # ------------------------------------------------------------------ #
    # 0. Load default dataset if none provided
    # ------------------------------------------------------------------ #
    if X is None or y is None:
        data = load_breast_cancer(as_frame=False)
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        print(f"Using breast-cancer dataset: {X.shape[0]} samples, "
              f"{X.shape[1]} features, 2 classes.\n")
    else:
        X = np.asarray(X)
        y = np.asarray(y)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # ------------------------------------------------------------------ #
    # 1. Train / test split — BEFORE any preprocessing or feature selection
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # preserve class balance
    )
    print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}\n")

    # ------------------------------------------------------------------ #
    # 2. Scale features — fit ONLY on training data
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
    X_test_scaled  = scaler.transform(X_test)        # transform only

    # ------------------------------------------------------------------ #
    # 3. Feature selection via mutual information — fit ONLY on training data
    # ------------------------------------------------------------------ #
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)  # fit + transform
    X_test_sel  = selector.transform(X_test_scaled)                # transform only

    selected_indices = selector.get_support(indices=True)
    selected_names   = [feature_names[i] for i in selected_indices]
    mi_scores        = selector.scores_[selected_indices]

    print(f"Top-{k} features selected by mutual information:")
    for rank, (name, score) in enumerate(
        sorted(zip(selected_names, mi_scores), key=lambda x: -x[1]), start=1
    ):
        print(f"  {rank:2d}. {name:<35s}  MI score = {score:.4f}")
    print()

    # ------------------------------------------------------------------ #
    # 4. Train Decision Tree — on selected training features only
    # ------------------------------------------------------------------ #
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",   # handles class imbalance
    )
    clf.fit(X_train_sel, y_train)

    # ------------------------------------------------------------------ #
    # 5. Evaluate on the held-out test set
    # ------------------------------------------------------------------ #
    y_pred      = clf.predict(X_test_sel)
    y_pred_prob = clf.predict_proba(X_test_sel)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    auc  = roc_auc_score(y_test, y_pred_prob)
    mcc  = matthews_corrcoef(y_test, y_pred)

    print("=" * 50)
    print("TEST SET METRICS")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (wtd)  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  MCC       : {mcc:.4f}")
    print()
    print("Detailed classification report:")
    print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------ #
    # 6. Return all artefacts for downstream use
    # ------------------------------------------------------------------ #
    return {
        "scaler"           : scaler,
        "selector"         : selector,
        "classifier"       : clf,
        "selected_indices" : selected_indices.tolist(),
        "selected_names"   : selected_names,
        "mi_scores"        : mi_scores.tolist(),
        "metrics": {
            "accuracy" : acc,
            "f1"       : f1,
            "roc_auc"  : auc,
            "mcc"      : mcc,
        },
    }


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    results = select_features_and_train(k=10, max_depth=5)
    print("\nFinal test accuracy:", round(results["metrics"]["accuracy"], 4))