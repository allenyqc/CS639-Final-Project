from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def top10_mi_decision_tree(X, y, test_size=0.2, random_state=42):
    """
    Select top-10 features by mutual information, train a decision tree,
    and return the test accuracy along with the fitted pipeline and selected features.

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target labels.
    test_size : float, default=0.2
        Proportion of data to use for testing.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    accuracy : float
        Accuracy on the test set.
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline with feature selection and decision tree.
    selected_features : list
        Names or indices of the top-10 selected features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    k = min(10, X_train.shape[1])

    pipeline = Pipeline([
        ("select", SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=random_state),
            k=k,
        )),
        ("tree", DecisionTreeClassifier(random_state=random_state)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    selector = pipeline.named_steps["select"]
    mask = selector.get_support()
    if hasattr(X, "columns"):
        selected_features = list(X.columns[mask])
    else:
        selected_features = list(np.where(mask)[0])

    print(f"Selected top-{k} features: {selected_features}")
    print(f"Test accuracy: {accuracy:.4f}")

    return accuracy, pipeline, selected_features


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    acc, model, feats = top10_mi_decision_tree(data.data, data.target)