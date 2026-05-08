from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train a StandardScaler and Logistic Regression model and return test accuracy.

    Best practices applied:
    - StandardScaler is fit ONLY on X_train; X_test is only transformed.
    - No hyperparameter selection uses test data.
    - Note: For imbalanced datasets, prefer F1, AUC, or MCC over accuracy.
    """
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    return test_accuracy