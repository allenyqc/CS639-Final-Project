from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def select_and_train(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select the top-10 most informative features using mutual information
    selector = SelectKBest(score_func=mutual_info_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Train a decision tree classifier on the selected features
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_selected, y_train)

    # Predict on the test set and calculate accuracy
    y_pred = clf.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Example usage:
# X, y = your_dataset_features, your_dataset_labels
# accuracy = select_and_train(X, y)
# print(f"Test Accuracy: {accuracy:.2f}")