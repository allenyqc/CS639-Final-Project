from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
import numpy as np

def train_logistic_regression_and_optimize_threshold(X, y):
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Train a logistic regression classifier
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    
    # Get the predicted probabilities for the validation set
    y_val_probs = model.predict_proba(X_val)[:, 1]
    
    # Calculate precision-recall pairs for different thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find the threshold that maximizes the F1 score
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    
    # Get the predicted probabilities for the test set
    y_test_probs = model.predict_proba(X_test)[:, 1]
    
    # Report the final test F1 score at the best threshold
    y_pred_best_threshold = (y_test_probs >= best_threshold).astype(int)
    final_f1_score = f1_score(y_test, y_pred_best_threshold)
    
    return best_threshold, final_f1_score

# Example usage:
# X, y = load_your_data()  # Replace with your data loading function
# best_threshold, final_f1 = train_logistic_regression_and_optimize_threshold(X, y)
# print(f"Best Threshold: {best_threshold}, Final F1 Score: {final_f1}")