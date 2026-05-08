from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train a StandardScaler and Logistic Regression model and return test accuracy.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    X_test : array-like
        Test feature matrix.
    y_train : array-like
        Training target vector.
    y_test : array-like
        Test target vector.

    Returns
    -------
    float
        Test accuracy of the trained logistic regression model.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy