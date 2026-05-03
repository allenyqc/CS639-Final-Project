from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report

def tune_and_evaluate_rf(X, y, param_grid=None, test_size=0.2, random_state=42):
    """
    Tunes hyperparameters for a Random Forest model using GridSearchCV and evaluates the best model.

    Parameters:
    - X: Features (Pandas DataFrame or NumPy array)
    - y: Target variable (Pandas Series or NumPy array)
    - param_grid: Dictionary of hyperparameters to tune (default is None, uses a predefined grid)
    - test_size: Proportion of the dataset to include in the test split (default is 0.2)
    - random_state: Random state for reproducibility (default is 42)

    Returns:
    - best_model: The best Random Forest model from GridSearchCV
    - test_accuracy: Accuracy of the best model on the test set
    - classification_report_dict: Classification report as a dictionary
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Default hyperparameter grid if none is provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

    # Initialize the Random Forest model
    rf = RandomForestClassifier(random_state=random_state)

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Get the best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)

    return best_model, test_accuracy, classification_report_dict