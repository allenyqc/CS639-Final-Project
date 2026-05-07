from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import numpy as np

def train_and_evaluate_classifier():
    # Create an imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, 
                               n_redundant=10, n_clusters_per_class=1, 
                               weights=[0.9, 0.1], flip_y=0, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Resample the training set to balance the classes
    X_train_majority = X_train[y_train == 0]
    y_train_majority = y_train[y_train == 0]
    X_train_minority = X_train[y_train == 1]
    y_train_minority = y_train[y_train == 1]

    X_train_minority_upsampled, y_train_minority_upsampled = resample(X_train_minority, y_train_minority,
                                                                      replace=True, 
                                                                      n_samples=len(y_train_majority), 
                                                                      random_state=42)

    X_train_balanced = np.vstack((X_train_majority, X_train_minority_upsampled))
    y_train_balanced = np.hstack((y_train_majority, y_train_minority_upsampled))

    # Train a RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_balanced, y_train_balanced)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Call the function to train and evaluate the classifier
train_and_evaluate_classifier()