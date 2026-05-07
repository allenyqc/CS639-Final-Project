from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def select_top_features(X_train, y_train, num_features=10):
    mi = mutual_info_classif(X_train, y_train)
    top_features_indices = mi.argsort()[-num_features:][::-1]
    return top_features_indices

def train_decision_tree(X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Select top-10 most informative features using mutual information
    top_features_indices = select_top_features(X_train, y_train)
    X_train_selected = X_train[:, top_features_indices]
    X_test_selected = X_test[:, top_features_indices]
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Predict and evaluate using F1 score
    y_pred = clf.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return f1

# Example usage with a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=42)
f1 = train_decision_tree(X, y)
print(f"F1 Score: {f1:.4f}")