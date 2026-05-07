from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
import numpy as np

def train_logistic_regression_and_optimize_threshold(X, y, test_size=0.2, random_state=42):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train a logistic regression classifier
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    
    # Get predicted probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find the threshold that maximizes the F1 score
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_f1 = f1_scores[best_threshold_index]
    
    # Report the final test F1 at the best threshold
    y_pred_best_threshold = (y_probs >= best_threshold).astype(int)
    final_f1 = f1_score(y_test, y_pred_best_threshold)
    
    return best_threshold, final_f1

# Example usage:
# X, y = load_your_data()  # Replace with your data loading function
# best_threshold, final_f1 = train_logistic_regression_and_optimize_threshold(X, y)
# print(f"Best Threshold: {best_threshold}, Final F1 Score: {final_f1}")