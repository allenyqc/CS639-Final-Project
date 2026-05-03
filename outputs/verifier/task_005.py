from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def pca_svm_classifier(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the training and testing datasets separately to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reduce to 10 principal components using PCA (fit only on training data)
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train an SVM classifier
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_pca, y_train)
    
    # Predict on the test set
    y_pred = svm.predict(X_test_pca)
    
    # Calculate and return the test accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy