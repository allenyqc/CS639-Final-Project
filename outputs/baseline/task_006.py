from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

def train_logistic_regression_and_optimize_f1(X, y, test_size=0.2, random_state=42):
    """
    Trains a logistic regression classifier, finds the optimal classification threshold
    that maximizes F1-score, and reports the final test F1-score at that threshold.

    Parameters:
    - X: Features (numpy array or pandas DataFrame)
    - y: Labels (numpy array or pandas Series)
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Random seed for reproducibility

    Returns:
    - optimal_threshold: The threshold that maximizes F1-score
    - test_f1: The F1-score on the test set at the optimal threshold
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train a logistic regression classifier
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)

    # Get predicted probabilities for the test set
    y_probs = model.predict_proba(X_test)[:, 1]

    # Find the optimal threshold that maximizes F1-score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = f1_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]

    # Calculate the final F1-score on the test set at the optimal threshold
    y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
    test_f1 = f1_score(y_test, y_pred_optimal)

    return optimal_threshold, test_f1