import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def normalize_and_train_nn(X, y, test_size=0.2, random_state=42, epochs=50, batch_size=32):
    """
    Normalizes numeric features, splits data into train/test sets, trains a neural network, and evaluates it.

    Parameters:
    - X: numpy array or pandas DataFrame of features.
    - y: numpy array or pandas Series of labels.
    - test_size: float, proportion of the dataset to include in the test split.
    - random_state: int, random seed for reproducibility.
    - epochs: int, number of epochs to train the neural network.
    - batch_size: int, batch size for training.

    Returns:
    - f1: float, F1 score on the test set.
    """
    # Step 1: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Step 2: Normalize features using MinMaxScaler (fit only on training data)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Build a simple neural network
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Assuming binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Step 4: Train the model
    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Step 5: Evaluate the model on the test set
    y_pred_prob = model.predict(X_test_scaled).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Step 6: Calculate F1 score (preferred metric for imbalanced datasets)
    f1 = f1_score(y_test, y_pred)

    return f1