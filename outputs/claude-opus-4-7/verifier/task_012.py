import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def train_and_evaluate_nn(
    X,
    y,
    test_size=0.2,
    random_state=42,
    hidden_layer_sizes=(64, 32),
    max_iter=500,
):
    """
    Normalize numeric features to [0, 1], split into train/test,
    train a neural network, and return the test accuracy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    test_size : float, optional
        Proportion of the dataset to allocate to the test split.
    random_state : int, optional
        Random seed for reproducibility.
    hidden_layer_sizes : tuple, optional
        Sizes of hidden layers in the MLP.
    max_iter : int, optional
        Maximum number of training iterations.

    Returns
    -------
    float
        Accuracy on the test set.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    # Split first to avoid data leakage from the test set into the scaler.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalize numeric features to [0, 1].
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train neural network.
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate.
    y_pred = model.predict(X_test_scaled)
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    acc = train_and_evaluate_nn(data.data, data.target)
    print(f"Test accuracy: {acc:.4f}")