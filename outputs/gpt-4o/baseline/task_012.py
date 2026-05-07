from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def normalize_and_train_nn(X, y, test_size=0.2, random_state=42):
    # Normalize the features to the [0, 1] range
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_size, random_state=random_state)

    # Train a neural network
    nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=random_state)
    nn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = nn.predict(X_test)

    # Calculate and return the test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    return test_accuracy

# Example usage:
# X = np.random.rand(100, 10)  # 100 samples, 10 features
# y = np.random.randint(0, 2, 100)  # Binary target
# accuracy = normalize_and_train_nn(X, y)
# print(f"Test Accuracy: {accuracy}")