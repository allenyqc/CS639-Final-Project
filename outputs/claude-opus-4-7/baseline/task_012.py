import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def normalize_split_train_evaluate(
    X,
    y,
    test_size=0.2,
    random_state=42,
    hidden_layer_sizes=(64, 32),
    max_iter=300,
):
    """
    Normalize numeric features to [0, 1] using MinMaxScaler, split into train/test,
    train an MLP neural network, and return the test accuracy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target labels.
    test_size : float, default=0.2
        Fraction of data to use for the test set.
    random_state : int, default=42
        Random seed for reproducibility.
    hidden_layer_sizes : tuple, default=(64, 32)
        Hidden layer architecture for the MLP.
    max_iter : int, default=300
        Maximum number of training iterations.

    Returns
    -------
    test_accuracy : float
        Accuracy on the test set.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Split first to avoid leaking test statistics into the scaler
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalize numeric features to [0, 1]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train neural network
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    return test_accuracy


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    acc = normalize_split_train_evaluate(data.data, data.target)
    print(f"Test accuracy: {acc:.4f}")