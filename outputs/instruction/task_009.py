import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.datasets import make_classification

def train_and_evaluate_classifier():
    # Step 1: Generate a synthetic 10-class dataset
    X, y = make_classification(
        n_samples=5000, 
        n_features=20, 
        n_informative=15, 
        n_classes=10, 
        n_clusters_per_class=1, 
        weights=[0.1] * 10,  # Simulate imbalance
        random_state=42
    )
    
    # Step 2: Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Step 3: Preprocess the data (fit scaler only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 4: Train a multi-class classifier
    clf = RandomForestClassifier(random_state=42, n_estimators=100)
    clf.fit(X_train_scaled, y_train)
    
    # Step 5: Evaluate the classifier using F1-score (macro)
    y_pred = clf.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1-Score: {f1:.4f}")
    
    # Optional: Print a detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Call the function
train_and_evaluate_classifier()