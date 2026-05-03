import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def mean_target_encoding_and_train(X_train, X_test, y_train, y_test, categorical_feature):
    # Mean target encoding
    mean_target = X_train.join(y_train).groupby(categorical_feature)[y_train.name].mean()
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    # Map the mean target encoding to the categorical feature
    X_train_encoded[categorical_feature] = X_train[categorical_feature].map(mean_target)
    X_test_encoded[categorical_feature] = X_test[categorical_feature].map(mean_target)
    
    # Handle unseen categories in the test set
    X_test_encoded[categorical_feature] = X_test_encoded[categorical_feature].fillna(y_train.mean())
    
    # Train Gradient Boosting Classifier
    model = GradientBoostingClassifier()
    model.fit(X_train_encoded, y_train)
    
    # Predict probabilities and calculate ROC-AUC score
    y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return roc_auc