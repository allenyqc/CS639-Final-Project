from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def select_and_train_decision_tree(X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Compute mutual information scores
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    
    # Select top-10 features based on mutual information scores
    top_10_indices = np.argsort(mi_scores)[-10:]
    X_train_selected = X_train[:, top_10_indices]
    X_test_selected = X_test[:, top_10_indices]
    
    # Create a pipeline with a scaler and a decision tree classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Fit scaler only on training data
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train_selected, y_train)
    
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test_selected)
    
    # Calculate and return the test accuracy and F1 score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use weighted F1 for imbalanced datasets
    
    return accuracy, f1

# Example usage:
# X, y = load_your_data()  # Replace with your data loading function
# accuracy, f1 = select_and_train_decision_tree(X, y)
# print(f"Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}")