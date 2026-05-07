from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

def train_svm_with_pca(X, y):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create a pipeline with StandardScaler, PCA, and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features by removing the mean and scaling to unit variance
        ('pca', PCA(n_components=10)),  # Reduce to 10 principal components
        ('svm', SVC(kernel='linear', random_state=42))  # SVM classifier with a linear kernel
    ])
    
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = pipeline.predict(X_test)
    
    # Calculate the F1 score as the evaluation metric
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return f1

# Example usage:
# X, y = load_your_data()
# f1 = train_svm_with_pca(X, y)
# print(f"F1 Score: {f1}")