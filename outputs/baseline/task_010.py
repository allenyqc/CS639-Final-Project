from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import numpy as np

def train_with_smote(X, y, test_size=0.2, random_state=42):
    """
    Oversamples the minority class using SMOTE, trains a logistic regression classifier,
    and reports the test F1-score.

    Parameters:
    - X: numpy array or pandas DataFrame, feature matrix.
    - y: numpy array or pandas Series, target vector.
    - test_size: float, proportion of the dataset to include in the test split.
    - random_state: int, random seed for reproducibility.

    Returns:
    - f1: float, F1-score on the test set.
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Apply SMOTE to oversample the minority class in the training set
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train a logistic regression classifier
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train_resampled, y_train_resampled)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Calculate the F1-score
    f1 = f1_score(y_test, y_pred)

    return f1