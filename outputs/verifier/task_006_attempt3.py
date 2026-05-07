from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import numpy as np

def train_logistic_regression_and_optimize_f1(X, y, test_size=0.2, validation_size=0.2, random_state=None):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Further split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size / (1 - test_size), random_state=random_state)
    
    # Train a logistic regression classifier
    clf = LogisticRegression(solver='liblinear', random_state=random_state)
    clf.fit(X_train, y_train)
    
    # Get predicted probabilities for the validation set
    y_val_probs = clf.predict_proba(X_val)[:, 1]
    
    # Calculate precision-recall curve on the validation set
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find the threshold that maximizes the F1 score
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    
    # Get predicted probabilities for the test set
    y_test_probs = clf.predict_proba(X_test)[:, 1]
    
    # Report the final test F1 at the best threshold
    y_pred_best_threshold = (y_test_probs >= best_threshold).astype(int)
    final_f1_score = f1_score(y_test, y_pred_best_threshold)
    
    return best_threshold, final_f1_score

# Example usage:
# X, y = ... # Load or generate your data here
# best_threshold, final_f1 = train_logistic_regression_and_optimize_f1(X, y)
# print(f"Best Threshold: {best_threshold}, Final F1 Score: {final_f1}")