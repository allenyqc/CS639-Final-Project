from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains a StandardScaler and a logistic regression model on the training
    data, then evaluates the model on the test data.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix.
    X_test : array-like of shape (n_samples, n_features)
        Test feature matrix.
    y_train : array-like of shape (n_samples,)
        Training labels.
    y_test : array-like of shape (n_samples,)
        Test labels.

    Returns
    -------
    float
        The accuracy of the trained logistic regression model on the test set.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy