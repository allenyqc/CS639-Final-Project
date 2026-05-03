from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def pca_svm_classifier(X, y):
    # Standardize the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reduce to 10 principal components using PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Train an SVM classifier
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = svm.predict(X_test)
    
    # Calculate and return the test accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy