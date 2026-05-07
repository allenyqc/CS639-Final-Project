import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def mean_target_encoding(X_train, X_test, y_train, categorical_feature):
    # Calculate mean target encoding for the training set
    mean_target = X_train.groupby(categorical_feature)[y_train.name].mean()
    X_train_encoded = X_train.copy()
    X_train_encoded[categorical_feature] = X_train[categorical_feature].map(mean_target)
    
    # Apply the same encoding to the test set
    X_test_encoded = X_test.copy()
    X_test_encoded[categorical_feature] = X_test[categorical_feature].map(mean_target)
    
    # Handle categories in test set not present in train set
    X_test_encoded[categorical_feature].fillna(y_train.mean(), inplace=True)
    
    return X_train_encoded, X_test_encoded

def train_and_evaluate(X_train, X_test, y_train, y_test, categorical_feature):
    # Apply mean target encoding
    X_train_encoded, X_test_encoded = mean_target_encoding(X_train, X_test, y_train, categorical_feature)
    
    # Train a Gradient Boosting Classifier
    model = GradientBoostingClassifier()
    model.fit(X_train_encoded, y_train)
    
    # Predict probabilities for the test set
    y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]
    
    # Calculate and return the ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return roc_auc