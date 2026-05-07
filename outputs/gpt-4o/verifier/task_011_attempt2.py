import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def impute_and_train_knn(data, target_column, n_neighbors=5, test_size=0.2, random_state=None):
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Impute missing values with column mean on training data only
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Initialize and train the k-nearest-neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_imputed, y_train)
    
    # Predict on the test set
    y_pred = knn.predict(X_test_imputed)
    
    # Calculate and return the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy