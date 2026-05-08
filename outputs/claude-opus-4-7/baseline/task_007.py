from typing import Tuple
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def top10_mi_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[float, np.ndarray, DecisionTreeClassifier]:
    """
    Select the top-10 most informative features using mutual information,
    train a decision tree classifier on them, and report the test accuracy.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    test_size : float, default=0.2
        Proportion of data to use for the test set.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    accuracy : float
        Test accuracy of the trained decision tree.
    selected_features : np.ndarray
        Indices of the selected top-10 features.
    model : DecisionTreeClassifier
        The trained decision tree model.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    k = min(10, X_train.shape[1])
    selector = SelectKBest(
        score_func=lambda X_, y_: mutual_info_classif(X_, y_, random_state=random_state),
        k=k,
    )
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    selected_features = selector.get_support(indices=True)

    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train_sel, y_train)

    y_pred = model.predict(X_test_sel)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Selected feature indices: {selected_features}")
    print(f"Test accuracy: {accuracy:.4f}")

    return accuracy, selected_features, model


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    top10_mi_decision_tree(data.data, data.target)