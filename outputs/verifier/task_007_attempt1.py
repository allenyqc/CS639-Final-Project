import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def select_top_features_and_train(X, y, num_features=10, test_size=0.2, random_state=42):
    """
    Selects the top `num_features` most informative features using mutual information,
    trains a decision tree classifier on the selected features, and reports the test accuracy.
    
    Parameters:
        X (numpy.ndarray or pandas.DataFrame): Feature matrix.
        y (numpy.ndarray or pandas.Series): Target vector.
        num_features (int): Number of top features to select.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        float: Test accuracy of the decision tree classifier.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Compute mutual information between each feature and the target
    mi_scores = mutual_info_classif(X_train, y_train, random_state=random_state)
    
    # Get the indices of the top `num_features` features
    top_features_indices = np.argsort(mi_scores)[-num_features:]
    
    # Select the top features for training and testing
    X_train_selected = X_train[:, top_features_indices]
    X_test_selected = X_test[:, top_features_indices]
    
    # Train a decision tree classifier on the selected features
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train_selected, y_train)
    
    # Make predictions on the test set
    y_pred = clf.predict(X_test_selected)
    
    # Calculate and return the test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    return test_accuracy