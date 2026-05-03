from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def select_features_and_train(X, y, test_size=0.2, random_state=42):
    """
    Selects the top-10 most informative features using mutual information,
    trains a decision tree on the selected features, and reports the test accuracy.

    Parameters:
    - X: numpy array or pandas DataFrame, feature matrix
    - y: numpy array or pandas Series, target vector
    - test_size: float, proportion of the dataset to include in the test split
    - random_state: int, random seed for reproducibility

    Returns:
    - accuracy: float, test accuracy of the decision tree
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Compute mutual information scores
    mi_scores = mutual_info_classif(X_train, y_train, random_state=random_state)

    # Select the indices of the top-10 features
    top_10_indices = np.argsort(mi_scores)[-10:]

    # Reduce the feature set to the top-10 features
    X_train_selected = X_train[:, top_10_indices]
    X_test_selected = X_test[:, top_10_indices]

    # Train a decision tree classifier on the selected features
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train_selected, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test_selected)

    # Calculate and return the test accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy