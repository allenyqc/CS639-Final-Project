import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

def train_and_evaluate_classifier():
    # Create a synthetic 10-class dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
                               n_classes=10, n_clusters_per_class=1, random_state=42)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create a pipeline with a scaler and a classifier
    pipeline = make_pipeline(
        StandardScaler(),  # Fit only on training data
        RandomForestClassifier(random_state=42)
    )

    # Train the classifier
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the classifier using F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')

    return f1

# Example usage
f1_score_result = train_and_evaluate_classifier()
print(f"Weighted F1 Score: {f1_score_result:.4f}")