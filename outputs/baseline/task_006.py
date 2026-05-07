from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
import numpy as np

def train_and_evaluate_logistic_regression(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a logistic regression classifier
    clf = LogisticRegression(solver='liblinear', random_state=42)
    clf.fit(X_train, y_train)
    
    # Get predicted probabilities for the test set
    y_probs = clf.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find the threshold that maximizes F1 score
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_f1_score = f1_scores[best_threshold_index]
    
    # Report the final test F1 at the best threshold
    y_pred_best_threshold = (y_probs >= best_threshold).astype(int)
    final_f1_score = f1_score(y_test, y_pred_best_threshold)
    
    return best_threshold, final_f1_score

# Example usage:
# X, y = ... # Load or generate your dataset
# best_threshold, final_f1 = train_and_evaluate_logistic_regression(X, y)
# print(f"Best Threshold: {best_threshold}, Final F1 Score: {final_f1}")