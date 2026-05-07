from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler on the training data and transform both train and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize the Logistic Regression model
    model = LogisticRegression()
    
    # Fit the model on the scaled training data
    model.fit(X_train_scaled, y_train)
    
    # Predict on the scaled test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculate and return the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy