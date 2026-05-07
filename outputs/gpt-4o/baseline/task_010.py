from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def train_and_evaluate_with_smote():
    # Create a synthetic imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10,
                               n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0, random_state=42)
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Train a logistic regression classifier
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Calculate the F1-score
    f1 = f1_score(y_test, y_pred)
    
    return f1

# Example usage
f1_score_result = train_and_evaluate_with_smote()
print(f"Test F1-score: {f1_score_result:.4f}")