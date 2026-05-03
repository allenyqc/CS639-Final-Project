import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

def train_logistic_regression_and_optimize_threshold(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Trains a logistic regression classifier, finds the optimal classification threshold
    that maximizes the F1-score on a validation set, and reports the final test F1-score
    at that threshold.

    Parameters:
    - X: np.ndarray, feature matrix
    - y: np.ndarray, target vector
    - test_size: float, proportion of the dataset to include in the test split
    - val_size: float, proportion of the training set to include in the validation split
    - random_state: int, random seed for reproducibility

    Returns:
    - optimal_threshold: float, the threshold that maximizes F1-score on the validation set
    - test_f1: float, the F1-score on the test set at the optimal threshold
    """
    # Split the data into training+validation and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Further split the training+validation set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=random_state)

    # Train a logistic regression classifier
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)

    # Get predicted probabilities for the validation set
    y_val_probs = model.predict_proba(X_val)[:, 1]

    # Compute precision, recall, and thresholds on the validation set
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)

    # Compute F1-scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Find the threshold that maximizes the F1-score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Get predicted probabilities for the test set
    y_test_probs = model.predict_proba(X_test)[:, 1]

    # Compute the final F1-score on the test set using the optimal threshold
    y_test_pred_optimal = (y_test_probs >= optimal_threshold).astype(int)
    test_f1 = f1_score(y_test, y_test_pred_optimal)

    return optimal_threshold, test_f1