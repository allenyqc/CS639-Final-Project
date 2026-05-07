from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import numpy as np

def train_logistic_regression_with_smote(X, y, test_size=0.2, random_state=42):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale the features
    scaler = StandardScaler()
    X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train_resampled_scaled, y_train_resampled)

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate the F1-score
    f1 = f1_score(y_test, y_pred)

    return f1

# Example usage:
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10,
                           n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=42)

f1 = train_logistic_regression_with_smote(X, y)
print(f"Test F1-score: {f1:.4f}")