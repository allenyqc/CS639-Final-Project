import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def normalize_and_train_nn(data, target, test_size=0.2, random_state=42):
    # Normalize the data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_normalized, target, test_size=test_size, random_state=random_state)
    
    # Convert labels to categorical one-hot encoding
    y_train_categorical = to_categorical(y_train)
    y_test_categorical = to_categorical(y_test)
    
    # Define the neural network model
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(y_train_categorical.shape[1], activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train_categorical, epochs=50, batch_size=10, verbose=0)
    
    # Evaluate the model on the test set
    _, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
    
    return test_accuracy

# Example usage with the Iris dataset
iris = load_iris()
data, target = iris.data, iris.target
accuracy = normalize_and_train_nn(data, target)
print(f"Test Accuracy: {accuracy:.2f}")