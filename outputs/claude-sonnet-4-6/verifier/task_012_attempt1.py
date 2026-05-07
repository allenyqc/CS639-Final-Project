import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def train_neural_network_with_normalization(
    X,
    y,
    test_size=0.2,
    random_state=42,
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    verbose=False
):
    """
    Normalizes numeric features, splits data, trains a neural network, and returns test accuracy.

    Parameters:
    -----------
    X : array-like or pd.DataFrame
        Feature matrix
    y : array-like
        Target labels
    test_size : float, optional (default=0.2)
        Proportion of data to use for testing
    random_state : int, optional (default=42)
        Random seed for reproducibility
    hidden_layer_sizes : tuple, optional (default=(100, 50))
        Number of neurons in each hidden layer
    max_iter : int, optional (default=500)
        Maximum number of training iterations
    activation : str, optional (default='relu')
        Activation function ('relu', 'tanh', 'logistic', 'identity')
    solver : str, optional (default='adam')
        Optimization solver ('adam', 'sgd', 'lbfgs')
    learning_rate_init : float, optional (default=0.001)
        Initial learning rate
    early_stopping : bool, optional (default=True)
        Whether to use early stopping
    validation_fraction : float, optional (default=0.1)
        Fraction of training data for validation (used with early_stopping)
    n_iter_no_change : int, optional (default=10)
        Number of iterations with no improvement before stopping
    verbose : bool, optional (default=False)
        Whether to print training progress

    Returns:
    --------
    dict : Dictionary containing:
        - 'test_accuracy': float, accuracy on the test set
        - 'train_accuracy': float, accuracy on the training set
        - 'model': trained MLPClassifier
        - 'scaler': fitted MinMaxScaler
        - 'X_test_scaled': scaled test features
        - 'y_test': test labels
        - 'y_pred': predictions on test set
    """
    # Convert to numpy arrays if needed
    if isinstance(X, pd.DataFrame):
        # Select only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].values
        if verbose:
            print(f"Selected {len(numeric_cols)} numeric features: {list(numeric_cols)}")
    else:
        X_numeric = np.array(X, dtype=float)

    y = np.array(y)

    # Handle missing values by replacing with column means
    if np.any(np.isnan(X_numeric)):
        col_means = np.nanmean(X_numeric, axis=0)
        nan_indices = np.where(np.isnan(X_numeric))
        X_numeric[nan_indices] = np.take(col_means, nan_indices[1])
        if verbose:
            print("Warning: Missing values detected and replaced with column means.")

    # Split data into train and test sets BEFORE scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Normalize features using MinMaxScaler (fit only on training data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if verbose:
        print(f"Training set size: {X_train_scaled.shape}")
        print(f"Test set size: {X_test_scaled.shape}")
        print(f"Feature range after scaling - Min: {X_train_scaled.min():.4f}, Max: {X_train_scaled.max():.4f}")

    # Initialize and train the neural network
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction if early_stopping else 0.1,
        n_iter_no_change=n_iter_no_change,
        verbose=verbose
    )

    model.fit(X_train_scaled, y_train)

    if verbose:
        print(f"Training completed in {model.n_iter_} iterations.")

    # Evaluate the model
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    if verbose:
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy:  {test_accuracy:.4f}")

    return {
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'model': model,
        'scaler': scaler,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'y_pred': y_pred_test
    }


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine

    print("=" * 60)
    print("Example 1: Breast Cancer Dataset")
    print("=" * 60)
    data = load_breast_cancer()
    results = train_neural_network_with_normalization(
        data.data, data.target, verbose=True
    )
    print(f"Final Test Accuracy: {results['test_accuracy']:.4f}\n")

    print("=" * 60)
    print("Example 2: Iris Dataset")
    print("=" * 60)
    data = load_iris()
    results = train_neural_network_with_normalization(
        data.data, data.target,
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        verbose=True
    )
    print(f"Final Test Accuracy: {results['test_accuracy']:.4f}\n")

    print("=" * 60)
    print("Example 3: Wine Dataset (as DataFrame)")
    print("=" * 60)
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    results = train_neural_network_with_normalization(
        df, data.target,
        hidden_layer_sizes=(128, 64, 32),
        activation='tanh',
        verbose=True
    )
    print(f"Final Test Accuracy: {results['test_accuracy']:.4f}")