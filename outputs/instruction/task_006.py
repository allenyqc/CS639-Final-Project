import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.datasets import make_classification

def train_logistic_regression_and_optimize_f1(X, y, test_size=0.2, random_state=42):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize and fit the scaler on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the logistic regression model
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Get the predicted probabilities for the test set
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find the threshold that maximizes the F1 score
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_f1_score = f1_scores[best_threshold_index]
    
    # Report the final test F1 score at the best threshold
    y_pred_best_threshold = (y_probs >= best_threshold).astype(int)
    final_f1_score = f1_score(y_test, y_pred_best_threshold)
    
    return best_threshold, final_f1_score

# Example usage with a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.7, 0.3], random_state=42)
best_threshold, final_f1_score = train_logistic_regression_and_optimize_f1(X, y)
print(f"Best Threshold: {best_threshold}, Final Test F1 Score: {final_f1_score}")