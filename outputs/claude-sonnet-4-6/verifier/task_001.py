import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains a StandardScaler and Logistic Regression model on the provided dataset.
    
    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix
    X_test : array-like of shape (n_samples, n_features)
        Test feature matrix
    y_train : array-like of shape (n_samples,)
        Training labels
    y_test : array-like of shape (n_samples,)
        Test labels
    
    Returns:
    --------
    float
        Test accuracy of the trained model
    """
    # Initialize and fit the StandardScaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using the fitted scaler (no fitting on test data)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate and return test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    return test_accuracy


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load sample dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate the model
    accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")