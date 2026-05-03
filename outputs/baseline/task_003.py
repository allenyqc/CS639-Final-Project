from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

def tune_and_evaluate_random_forest(X, y):
    """
    Tunes hyperparameters for a Random Forest model using GridSearchCV and evaluates it on a test set.

    Parameters:
    X (pd.DataFrame or np.ndarray): Feature matrix.
    y (pd.Series or np.ndarray): Target vector.

    Returns:
    dict: A dictionary containing the best parameters, test accuracy, and classification report.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

    # Fit the model to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_rf = grid_search.best_estimator_

    # Evaluate the best model on the test set
    y_pred = best_rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Return the results
    return {
        'best_params': grid_search.best_params_,
        'test_accuracy': test_accuracy,
        'classification_report': class_report
    }