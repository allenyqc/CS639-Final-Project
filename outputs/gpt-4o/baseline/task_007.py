from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def select_top_features_and_train(X, y, test_size=0.2, random_state=42):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Compute mutual information scores
    mi_scores = mutual_info_classif(X_train, y_train, random_state=random_state)
    
    # Get indices of the top-10 features
    top_10_indices = np.argsort(mi_scores)[-10:]
    
    # Select the top-10 features
    X_train_top10 = X_train[:, top_10_indices]
    X_test_top10 = X_test[:, top_10_indices]
    
    # Train a decision tree classifier on the selected features
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train_top10, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test_top10)
    
    # Calculate and return the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Example usage:
# X, y = load_your_data()  # Replace with your data loading function
# accuracy = select_top_features_and_train(X, y)
# print(f"Test Accuracy: {accuracy:.2f}")