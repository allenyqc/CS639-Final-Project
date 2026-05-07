from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Create a pipeline with StandardScaler and LogisticRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic_regression', LogisticRegression())
    ])
    
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = pipeline.predict(X_test)
    
    # Calculate F1 score for imbalanced datasets
    f1 = f1_score(y_test, y_pred, average='binary')
    
    return f1

# Example usage:
# f1_score = train_and_evaluate(X_train, X_test, y_train, y_test)
# print("F1 Score:", f1_score)