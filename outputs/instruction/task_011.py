import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def impute_and_train_knn(X, y, test_size=0.2, random_state=42, n_neighbors=5):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Impute missing values using the mean of the training data
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Standardize the features using the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Train a k-nearest-neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    
    # Predict on the test set
    y_pred = knn.predict(X_test_scaled)
    
    # Calculate and return the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Example usage:
# X = pd.DataFrame(...)  # Your feature data
# y = pd.Series(...)     # Your target data
# accuracy = impute_and_train_knn(X, y)
# print(f"Test Accuracy: {accuracy}")