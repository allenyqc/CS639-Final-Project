import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def mean_target_encoding_and_train(X_train, X_test, y_train, y_test, cat_feature):
    """
    Applies mean target encoding to a categorical feature, trains a gradient boosting classifier,
    and returns the test ROC-AUC score.

    Parameters:
    - X_train: pd.DataFrame, training features
    - X_test: pd.DataFrame, test features
    - y_train: pd.Series, training labels
    - y_test: pd.Series, test labels
    - cat_feature: str, name of the categorical feature to encode

    Returns:
    - roc_auc: float, ROC-AUC score on the test set
    """
    # Mean target encoding
    mean_target = X_train.join(y_train).groupby(cat_feature)[y_train.name].mean()
    X_train[cat_feature + "_mean_enc"] = X_train[cat_feature].map(mean_target)
    X_test[cat_feature + "_mean_enc"] = X_test[cat_feature].map(mean_target)
    
    # Fill NaN values in test set (if any category in test is not in train)
    X_test[cat_feature + "_mean_enc"].fillna(y_train.mean(), inplace=True)
    
    # Drop the original categorical feature
    X_train = X_train.drop(columns=[cat_feature])
    X_test = X_test.drop(columns=[cat_feature])
    
    # Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict probabilities for ROC-AUC calculation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return roc_auc