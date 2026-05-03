from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_and_evaluate_classifier():
    # Generate a synthetic 10-class dataset
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=15, 
        n_redundant=5, 
        n_classes=10, 
        random_state=42
    )
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the classifier
    clf = RandomForestClassifier(random_state=42)
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the classifier using a classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

# Call the function to train and evaluate the classifier
train_and_evaluate_classifier()