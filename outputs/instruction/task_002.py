import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline

def train_and_evaluate():
    # Create an imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, 
                               n_redundant=10, n_clusters_per_class=1, 
                               weights=[0.9, 0.1], flip_y=0, random_state=42)
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        stratify=y, random_state=42)
    
    # Create a pipeline with a scaler and a classifier
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', random_state=42))
    
    # Fit the model on the training data
    pipeline.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Print the performance metrics
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"MCC: {mcc:.4f}")

# Run the function
train_and_evaluate()