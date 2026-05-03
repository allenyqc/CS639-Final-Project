from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

def handle_imbalance_and_train(X, y, test_size=0.2, random_state=42):
    """
    Oversamples the minority class using SMOTE, trains a logistic regression classifier,
    and reports the test F1-score.

    Parameters:
    - X: pd.DataFrame or np.ndarray, feature matrix
    - y: pd.Series or np.ndarray, target vector
    - test_size: float, proportion of the dataset to include in the test split
    - random_state: int, random seed for reproducibility

    Returns:
    - f1: float, F1-score on the test set
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Apply SMOTE to oversample the minority class
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train a logistic regression model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the F1-score
    f1 = f1_score(y_test, y_pred)
    
    return f1

# Example usage:
# Assuming `data` is a pandas DataFrame with features and `target` is the target column
# X = data.drop(columns=['target'])
# y = data['target']
# f1 = handle_imbalance_and_train(X, y)
# print(f"Test F1-score: {f1}")