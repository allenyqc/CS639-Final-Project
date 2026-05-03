import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

def train_and_evaluate_binary_classifier():
    # Generate an imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                                n_redundant=5, n_clusters_per_class=2, weights=[0.9, 0.1], 
                                flip_y=0, random_state=42)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Initialize the classifier
    clf = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Call the function
train_and_evaluate_binary_classifier()