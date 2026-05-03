import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def normalize_and_train(data, labels, test_size=0.2, random_state=42):
    """
    Normalizes numeric features, splits the data, trains a neural network, and returns test accuracy.

    Parameters:
    - data: np.ndarray, feature matrix
    - labels: np.ndarray, target labels
    - test_size: float, proportion of the dataset to include in the test split
    - random_state: int, random seed for reproducibility

    Returns:
    - test_accuracy: float, accuracy of the model on the test set
    """
    # Normalize features to [0, 1] range
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data_normalized, labels, test_size=test_size, random_state=random_state
    )

    # Convert labels to one-hot encoding if they are categorical
    num_classes = len(np.unique(labels))
    y_train_one_hot = to_categorical(y_train, num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes)

    # Define a simple neural network
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train_one_hot, epochs=20, batch_size=32, verbose=0)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    test_accuracy = accuracy_score(y_test, y_pred_classes)

    return test_accuracy