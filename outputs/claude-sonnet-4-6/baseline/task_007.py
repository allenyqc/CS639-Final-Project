import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def select_features_and_train(
    X=None,
    y=None,
    k=10,
    test_size=0.2,
    random_state=42,
    max_depth=None,
    criterion="gini",
):
    """
    Selects the top-k most informative features using mutual information,
    trains a decision tree on the selected features, and reports test accuracy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), optional
        Feature matrix. If None, the breast cancer dataset is used.
    y : array-like of shape (n_samples,), optional
        Target vector. If None, the breast cancer dataset labels are used.
    k : int, default=10
        Number of top features to select.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random state for reproducibility.
    max_depth : int or None, default=None
        Maximum depth of the decision tree.
    criterion : str, default='gini'
        The function to measure the quality of a split ('gini' or 'entropy').

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'test_accuracy': float, accuracy on the test set
        - 'selected_feature_indices': ndarray, indices of selected features
        - 'mutual_info_scores': ndarray, MI scores for all features
        - 'pipeline': fitted sklearn Pipeline
        - 'X_train_selected': ndarray, training data after feature selection
        - 'X_test_selected': ndarray, test data after feature selection
    """
    # Load default dataset if none provided
    if X is None or y is None:
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        print(f"Using Breast Cancer dataset: {X.shape[0]} samples, {X.shape[1]} features")
    else:
        X = np.array(X)
        y = np.array(y)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        print(f"Using provided dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Validate k
    if k > X.shape[1]:
        print(f"Warning: k={k} > number of features ({X.shape[1]}). Setting k={X.shape[1]}.")
        k = X.shape[1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Build pipeline: feature selection + decision tree
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    clf = DecisionTreeClassifier(
        max_depth=max_depth, criterion=criterion, random_state=random_state
    )
    pipeline = Pipeline([("selector", selector), ("classifier", clf)])

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Extract feature selection details
    mi_scores = pipeline.named_steps["selector"].scores_
    selected_mask = pipeline.named_steps["selector"].get_support()
    selected_indices = np.where(selected_mask)[0]

    # Transform data for inspection
    X_train_selected = pipeline.named_steps["selector"].transform(X_train)
    X_test_selected = pipeline.named_steps["selector"].transform(X_test)

    # Report results
    print("\n" + "=" * 60)
    print(f"Top-{k} Features Selected by Mutual Information:")
    print("=" * 60)
    sorted_selected = sorted(
        zip(selected_indices, mi_scores[selected_indices]),
        key=lambda x: x[1],
        reverse=True,
    )
    for rank, (idx, score) in enumerate(sorted_selected, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"  {rank:2d}. {name:<35s} MI Score: {score:.4f}")

    print("\n" + "=" * 60)
    print(f"Decision Tree Configuration:")
    print(f"  Criterion : {criterion}")
    print(f"  Max Depth : {max_depth if max_depth else 'None (unlimited)'}")
    print(f"\nTest Accuracy : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("=" * 60)

    return {
        "test_accuracy": test_accuracy,
        "selected_feature_indices": selected_indices,
        "mutual_info_scores": mi_scores,
        "pipeline": pipeline,
        "X_train_selected": X_train_selected,
        "X_test_selected": X_test_selected,
    }


def compare_with_all_features(X=None, y=None, k=10, random_state=42):
    """
    Compares decision tree performance using top-k MI features vs. all features.

    Parameters
    ----------
    X, y : array-like, optional
        Dataset. Defaults to breast cancer dataset.
    k : int, default=10
        Number of features to select.
    random_state : int, default=42
        Random state for reproducibility.
    """
    if X is None or y is None:
        data = load_breast_cancer()
        X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Train on all features
    clf_all = DecisionTreeClassifier(random_state=random_state)
    clf_all.fit(X_train, y_train)
    acc_all = accuracy_score(y_test, clf_all.predict(X_test))

    # Train on selected features
    results = select_features_and_train(X, y, k=k, random_state=random_state)
    acc_selected = results["test_accuracy"]

    print("\n" + "=" * 60)
    print("Comparison: All Features vs. MI-Selected Features")
    print("=" * 60)
    print(f"  All features ({X.shape[1]:3d})  -> Accuracy: {acc_all:.4f} ({acc_all*100:.2f}%)")
    print(f"  Top-{k} MI features      -> Accuracy: {acc_selected:.4f} ({acc_selected*100:.2f}%)")
    diff = acc_selected - acc_all
    direction = "improvement" if diff >= 0 else "decrease"
    print(f"  Difference              : {diff:+.4f} ({direction})")
    print("=" * 60)

    return {"accuracy_all_features": acc_all, "accuracy_selected_features": acc_selected}


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(">>> Running feature selection + decision tree on Breast Cancer dataset\n")
    results = select_features_and_train(k=10)

    print("\n>>> Comparing MI-selected features vs. all features\n")
    compare_with_all_features(k=10)