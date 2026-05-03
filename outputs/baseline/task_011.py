import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def impute_and_train_knn(data, target_column, n_neighbors=5, test_size=0.2, random_state=42):
    """
    Imputes missing values using column mean, trains a k-NN classifier, and returns test accuracy.
    
    Parameters:
        data (pd.DataFrame): The dataset containing features and target.
        target_column (str): The name of the target column.
        n_neighbors (int): Number of neighbors for k-NN. Default is 5.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Random state for reproducibility. Default is 42.
    
    Returns:
        float: Test accuracy of the k-NN classifier.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Impute missing values with column mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=test_size, random_state=random_state
    )
    
    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = knn.predict(X_test)
    
    # Calculate and return test accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy