import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def mean_target_encoding_and_train(X_train, X_test, y_train, y_test, 
                                    cat_columns=None, n_splits=5, 
                                    random_state=42):
    """
    Applies mean target encoding to categorical features, trains a gradient 
    boosting classifier, and returns the test ROC-AUC score.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    X_test : pd.DataFrame or np.ndarray
        Test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    cat_columns : list, optional
        List of categorical column names/indices to encode.
        If None, automatically detects object/category dtype columns.
    n_splits : int
        Number of folds for cross-validation during encoding
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    float
        ROC-AUC score on the test set
    """
    # Convert to DataFrame if necessary
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Detect categorical columns if not provided
    if cat_columns is None:
        cat_columns = X_train.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
    
    # Global mean for smoothing/fallback
    global_mean = y_train.mean()
    
    # Store encoding maps for test set
    encoding_maps = {}
    
    # Apply out-of-fold mean target encoding to training set
    X_train_encoded = X_train.copy()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for col in cat_columns:
        # Initialize encoded column with global mean
        X_train_encoded[col] = global_mean
        
        # Out-of-fold encoding to prevent target leakage
        for train_idx, val_idx in kf.split(X_train):
            # Compute mean target per category on training fold
            fold_map = (
                pd.Series(y_train[train_idx])
                .groupby(X_train[col].iloc[train_idx].values)
                .mean()
            )
            # Apply encoding to validation fold
            X_train_encoded.loc[
                X_train.index[val_idx], col
            ] = X_train[col].iloc[val_idx].map(fold_map).fillna(global_mean)
        
        # Compute full training set encoding map for test set
        full_map = (
            pd.Series(y_train)
            .groupby(X_train[col].values)
            .mean()
        )
        encoding_maps[col] = full_map
    
    # Apply encoding to test set using full training encoding map
    X_test_encoded = X_test.copy()
    for col in cat_columns:
        X_test_encoded[col] = X_test[col].map(encoding_maps[col]).fillna(global_mean)
    
    # Ensure all columns are numeric
    X_train_encoded = X_train_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test_encoded = X_test_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Train Gradient Boosting Classifier
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state
    )
    clf.fit(X_train_encoded, y_train)
    
    # Predict probabilities on test set
    y_pred_proba = clf.predict_proba(X_test_encoded)
    
    # Handle binary and multiclass cases
    if y_pred_proba.shape[1] == 2:
        # Binary classification
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        # Multiclass classification
        roc_auc = roc_auc_score(
            y_test, y_pred_proba, multi_class='ovr', average='macro'
        )
    
    return roc_auc


# Example usage and testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic dataset
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5, 
        random_state=42
    )
    
    # Convert to DataFrame and add a categorical feature
    X_df = pd.DataFrame(X, columns=[f'num_feat_{i}' for i in range(10)])
    categories = ['A', 'B', 'C', 'D', 'E']
    X_df['cat_feat_1'] = np.random.choice(categories, size=len(X_df))
    X_df['cat_feat_2'] = np.random.choice(['low', 'medium', 'high'], size=len(X_df))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Run the function
    roc_auc = mean_target_encoding_and_train(
        X_train, X_test, y_train, y_test,
        cat_columns=['cat_feat_1', 'cat_feat_2']
    )
    
    print(f"Test ROC-AUC Score: {roc_auc:.4f}")
    
    # Test with auto-detection of categorical columns
    roc_auc_auto = mean_target_encoding_and_train(
        X_train, X_test, y_train, y_test
    )
    print(f"Test ROC-AUC Score (auto-detect cats): {roc_auc_auto:.4f}")