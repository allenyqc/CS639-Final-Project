import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def train_neural_network(
    data,
    target_column,
    test_size=0.2,
    random_state=42,
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    activation='relu',
    solver='adam',
    learning_rate_init=0.001
):
    """
    Normalizes numeric features, splits data, trains a neural network, and returns test accuracy.
    
    Parameters:
    -----------
    data : pd.DataFrame or np.ndarray
        Input dataset containing features and target
    target_column : str or int
        Column name (for DataFrame) or index (for ndarray) of the target variable
    test_size : float, optional (default=0.2)
        Proportion of dataset to include in the test split
    random_state : int, optional (default=42)
        Random state for reproducibility
    hidden_layer_sizes : tuple, optional (default=(100, 50))
        Number of neurons in each hidden layer
    max_iter : int, optional (default=500)
        Maximum number of iterations for training
    activation : str, optional (default='relu')
        Activation function for hidden layers
    solver : str, optional (default='adam')
        Solver for weight optimization
    learning_rate_init : float, optional (default=0.001)
        Initial learning rate
    
    Returns:
    --------
    float
        Test accuracy of the trained neural network
    """
    # Handle both DataFrame and ndarray inputs
    if isinstance(data, pd.DataFrame):
        X = data.drop(columns=[target_column])
        y = data[target_column].values
    elif isinstance(data, np.ndarray):
        if isinstance(target_column, int):
            X = np.delete(data, target_column, axis=1)
            y = data[:, target_column]
        else:
            raise ValueError("For ndarray input, target_column must be an integer index")
    else:
        raise TypeError("data must be a pandas DataFrame or numpy ndarray")
    
    # Convert to DataFrame for easier handling of numeric columns
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Identify numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if not numeric_columns:
        raise ValueError("No numeric features found in the dataset")
    
    # Normalize numeric features using MinMaxScaler
    scaler = MinMaxScaler()
    X_numeric_scaled = scaler.fit_transform(X[numeric_columns])
    X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=numeric_columns, index=X.index)
    
    # Combine scaled numeric features with non-numeric features (if any)
    if non_numeric_columns:
        X_processed = pd.concat([X_numeric_scaled, X[non_numeric_columns]], axis=1)
    else:
        X_processed = X_numeric_scaled
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Initialize and train the neural network
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        learning_rate_init=learning_rate_init,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    mlp.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = mlp.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Number of iterations: {mlp.n_iter_}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return test_accuracy


# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_breast_cancer, load_wine
    
    print("=" * 50)
    print("Test 1: Iris Dataset")
    print("=" * 50)
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    accuracy = train_neural_network(iris_df, target_column='target')
    print(f"Iris Test Accuracy: {accuracy:.4f}\n")
    
    print("=" * 50)
    print("Test 2: Breast Cancer Dataset")
    print("=" * 50)
    cancer = load_breast_cancer()
    cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    cancer_df['target'] = cancer.target
    accuracy = train_neural_network(
        cancer_df,
        target_column='target',
        hidden_layer_sizes=(64, 32),
        max_iter=300
    )
    print(f"Breast Cancer Test Accuracy: {accuracy:.4f}\n")
    
    print("=" * 50)
    print("Test 3: Wine Dataset")
    print("=" * 50)
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target
    accuracy = train_neural_network(
        wine_df,
        target_column='target',
        hidden_layer_sizes=(128, 64, 32),
        max_iter=500
    )
    print(f"Wine Test Accuracy: {accuracy:.4f}\n")
    
    print("=" * 50)
    print("Test 4: NumPy Array Input")
    print("=" * 50)
    data_array = np.column_stack([iris.data, iris.target])
    accuracy = train_neural_network(data_array, target_column=4)
    print(f"NumPy Array Test Accuracy: {accuracy:.4f}")