import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def mean_target_encoding_and_gradient_boosting(X_train, X_test, y_train, y_test):
    """
    Applies mean target encoding to categorical features, trains a gradient boosting
    classifier, and returns the test ROC-AUC score.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray - Training features
    X_test  : pd.DataFrame or np.ndarray - Test features
    y_train : pd.Series or np.ndarray    - Training labels
    y_test  : pd.Series or np.ndarray    - Test labels
    
    Returns:
    --------
    float : ROC-AUC score on the test set
    """
    # Convert to DataFrames if necessary
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test  = pd.Series(y_test).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    
    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # If no categorical columns found, try to infer from low-cardinality integer columns
    if not categorical_cols:
        for col in X_train.columns:
            if X_train[col].nunique() < 20 and X_train[col].dtype in ['int64', 'int32', 'object']:
                categorical_cols.append(col)
    
    X_train_encoded = X_train.copy()
    X_test_encoded  = X_test.copy()
    
    global_mean = y_train.mean()
    
    # Apply out-of-fold mean target encoding to training data to avoid leakage
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for col in categorical_cols:
        # Out-of-fold encoding for training set
        train_encoded = np.full(len(X_train), global_mean)
        
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val   = X_train.iloc[val_idx]
            
            # Compute mean encoding from fold training data
            encoding_map = y_fold_train.groupby(X_fold_train[col].values).mean()
            
            # Apply to validation fold
            train_encoded[val_idx] = X_fold_val[col].map(encoding_map).fillna(global_mean).values
        
        X_train_encoded[col] = train_encoded
        
        # For test set, use full training data encoding
        full_encoding_map = y_train.groupby(X_train[col].values).mean()
        X_test_encoded[col] = X_test[col].map(full_encoding_map).fillna(global_mean).values
    
    # Convert all columns to float
    X_train_encoded = X_train_encoded.astype(float)
    X_test_encoded  = X_test_encoded.astype(float)
    
    # Handle any remaining NaN values
    X_train_encoded = X_train_encoded.fillna(X_train_encoded.mean())
    X_test_encoded  = X_test_encoded.fillna(X_train_encoded.mean())
    
    # Train Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_clf.fit(X_train_encoded, y_train)
    
    # Predict probabilities on test set
    y_pred_proba = gb_clf.predict_proba(X_test_encoded)
    
    # Compute ROC-AUC score
    if y_pred_proba.shape[1] == 2:
        # Binary classification
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        # Multi-class classification
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    
    print(f"Categorical columns encoded: {categorical_cols}")
    print(f"Test ROC-AUC Score: {roc_auc:.4f}")
    
    return roc_auc


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Example 1: Purely numerical dataset
    print("=== Example 1: Numerical features ===")
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    score = mean_target_encoding_and_gradient_boosting(X_train, X_test, y_train, y_test)
    print(f"ROC-AUC: {score:.4f}\n")
    
    # Example 2: Dataset with categorical features
    print("=== Example 2: Mixed features with categorical columns ===")
    np.random.seed(42)
    n = 1000
    categories = ['A', 'B', 'C', 'D']
    
    df = pd.DataFrame({
        'cat_feature1': np.random.choice(categories, n),
        'cat_feature2': np.random.choice(['X', 'Y', 'Z'], n),
        'num_feature1': np.random.randn(n),
        'num_feature2': np.random.randn(n),
    })
    
    # Create a target that correlates with the categorical features
    cat_map = {'A': 0.8, 'B': 0.5, 'C': 0.3, 'D': 0.1}
    prob = df['cat_feature1'].map(cat_map) + 0.1 * df['num_feature1']
    prob = prob.clip(0, 1)
    y = (np.random.rand(n) < prob).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    score = mean_target_encoding_and_gradient_boosting(X_train, X_test, y_train, y_test)
    print(f"ROC-AUC: {score:.4f}")