from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def train_svm_with_pca(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA to reduce to 10 principal components
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train an SVM classifier
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_pca, y_train)
    
    # Predict on the test set
    y_pred = svm.predict(X_test_pca)
    
    # Calculate F1 score as a measure of accuracy for imbalanced datasets
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return f1

# Example usage:
# X, y = load_your_data()  # Replace with your data loading function
# f1 = train_svm_with_pca(X, y)
# print(f"F1 Score: {f1}")