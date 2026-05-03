import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def normalize_and_train_nn(X, y, test_size=0.2, random_state=42):
    """
    Normalizes numeric features, splits data, trains a neural network, and returns test accuracy.

    Parameters:
    - X: np.ndarray, feature matrix
    - y: np.ndarray, target vector
    - test_size: float, proportion of data to use for testing
    - random_state: int, random seed for reproducibility

    Returns:
    - test_accuracy: float, accuracy of the model on the test set
    """
    # Split the data into train and test sets first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Normalize features to [0, 1] range using MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert target to one-hot encoding if it's a classification problem
    if len(np.unique(y)) > 2:  # Assuming multi-class classification
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    # Define a simple neural network model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(y_train.shape[1] if len(y_train.shape) > 1 else 1, activation='softmax' if len(y_train.shape) > 1 else 'sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy' if len(y_train.shape) > 1 else 'binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    if len(y_test.shape) > 1:  # Multi-class classification
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
    else:  # Binary classification
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        y_test_classes = y_test

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test_classes, y_pred_classes)
    return test_accuracy