import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def impute_and_train_knn(data, target_column, test_size=0.2, n_neighbors=5, random_state=42):
    """
    Imputes missing values using column mean, trains a k-NN classifier, and returns test accuracy.

    Parameters:
        data (pd.DataFrame): The input dataset.
        target_column (str): The name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        n_neighbors (int): Number of neighbors for k-NN.
        random_state (int): Random state for reproducibility.

    Returns:
        float: Test accuracy of the k-NN classifier.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Impute missing values using column mean
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Train a k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_imputed, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test_imputed)

    # Calculate and return test accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy