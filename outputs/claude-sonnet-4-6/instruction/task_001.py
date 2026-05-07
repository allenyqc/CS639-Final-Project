import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains a StandardScaler and Logistic Regression model on training data,
    then evaluates on test data.

    Best practices followed:
    - StandardScaler is fit ONLY on X_train, then applied to both X_train and X_test.
    - No hyperparameter tuning is done using test data.
    - The function returns test accuracy (suitable for balanced datasets).

    Parameters
    ----------
    X_train : array-like of shape (n_train_samples, n_features)
    X_test  : array-like of shape (n_test_samples, n_features)
    y_train : array-like of shape (n_train_samples,)
    y_test  : array-like of shape (n_test_samples,)

    Returns
    -------
    test_accuracy : float
        Accuracy score on the test set.
    """
    # ------------------------------------------------------------------ #
    # 1. Preprocessing — fit ONLY on training data, transform both splits #
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
    X_test_scaled  = scaler.transform(X_test)        # transform only on test

    # ------------------------------------------------------------------ #
    # 2. Model training                                                    #
    # ------------------------------------------------------------------ #
    model = LogisticRegression(
        max_iter=1000,       # enough iterations for convergence
        random_state=42,     # reproducibility
        solver="lbfgs",      # robust default solver
        class_weight="balanced",  # handles mild class imbalance gracefully
    )
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------ #
    # 3. Evaluation — test data is used ONLY here, never for tuning       #
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    return test_accuracy


# ---------------------------------------------------------------------- #
# Quick smoke-test with a synthetic dataset                               #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Step 1: create raw data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        random_state=42,
    )

    # Step 2: split BEFORE any feature engineering (best practice #4)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 3: train and evaluate
    accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")