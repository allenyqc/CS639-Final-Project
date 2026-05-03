import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

def train_svm_with_pca(X, y, test_size=0.2, random_state=42):
    """
    Reduces the dataset to 10 principal components using PCA, trains an SVM classifier,
    and evaluates the model using F1 score on the test set.

    Parameters:
    - X: np.ndarray, feature matrix
    - y: np.ndarray, target array
    - test_size: float, proportion of the dataset to include in the test split
    - random_state: int, random seed for reproducibility

    Returns:
    - f1: float, F1 score on the test set
    """
    # Step 1: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Step 2: Create a pipeline with scaling, PCA, and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('pca', PCA(n_components=10)),  # Reduce to 10 principal components
        ('svm', SVC(kernel='rbf', random_state=random_state))  # SVM classifier
    ])
    
    # Step 3: Fit the pipeline on training data
    pipeline.fit(X_train, y_train)
    
    # Step 4: Predict on the test set
    y_pred = pipeline.predict(X_test)
    
    # Step 5: Evaluate using F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')  # Weighted F1 for imbalanced datasets
    
    return f1