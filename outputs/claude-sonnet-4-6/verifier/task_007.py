import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def select_features_and_train(X, y, k=10, test_size=0.2, random_state=42):
    """
    Selects the top-k most informative features using mutual information,
    trains a decision tree on the selected features, and reports test accuracy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    k : int, default=10
        Number of top features to select.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    dict
        A dictionary containing:
        - 'test_accuracy': float, accuracy on the test set
        - 'selected_feature_indices': list of int, indices of selected features
        - 'mutual_info_scores': ndarray, MI scores for all features
        - 'model': trained Pipeline (selector + decision tree)
    """
    X = np.array(X)
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Build pipeline: feature selection + decision tree
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    clf = DecisionTreeClassifier(random_state=random_state)

    pipeline = Pipeline([
        ("feature_selection", selector),
        ("classifier", clf),
    ])

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Extract selector info
    fitted_selector = pipeline.named_steps["feature_selection"]
    mi_scores = fitted_selector.scores_
    selected_indices = fitted_selector.get_support(indices=True).tolist()

    # Report results
    print(f"Number of original features : {X.shape[1]}")
    print(f"Number of selected features : {k}")
    print(f"Selected feature indices    : {selected_indices}")
    print(f"\nMutual Information Scores (all features):")
    for i, score in enumerate(mi_scores):
        marker = " <-- selected" if i in selected_indices else ""
        print(f"  Feature {i:3d}: MI = {score:.4f}{marker}")
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    return {
        "test_accuracy": test_accuracy,
        "selected_feature_indices": selected_indices,
        "mutual_info_scores": mi_scores,
        "model": pipeline,
    }


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load a sample dataset (breast cancer: 30 features, binary classification)
    data = load_breast_cancer()
    X, y = data.data, data.target

    print("=" * 60)
    print("Dataset: Breast Cancer Wisconsin")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print("=" * 60)

    results = select_features_and_train(X, y, k=10)

    print("\n" + "=" * 60)
    print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")
    print("=" * 60)