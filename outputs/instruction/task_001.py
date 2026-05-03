from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains a StandardScaler and Logistic Regression model on the training data,
    and evaluates it using the F1 score on the test data.

    Parameters:
    - X_train: Training features
    - X_test: Test features
    - y_train: Training labels
    - y_test: Test labels

    Returns:
    - f1: F1 score on the test data
    """
    # Create a pipeline with StandardScaler and LogisticRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=42))
    ])
    
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = pipeline.predict(X_test)
    
    # Calculate the F1 score
    f1 = f1_score(y_test, y_pred)
    
    return f1