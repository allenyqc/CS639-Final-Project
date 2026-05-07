from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_feature):
        self.categorical_feature = categorical_feature
        self.target_means = None

    def fit(self, X, y):
        # Calculate mean target for each category
        self.target_means = y.groupby(X[self.categorical_feature]).mean()
        return self

    def transform(self, X):
        # Map the mean target to the categorical feature
        X_transformed = X.copy()
        X_transformed[self.categorical_feature] = X_transformed[self.categorical_feature].map(self.target_means)
        # Fill NaN values with the global mean
        global_mean = self.target_means.mean()
        X_transformed[self.categorical_feature].fillna(global_mean, inplace=True)
        return X_transformed

def train_and_evaluate(X_train, X_test, y_train, y_test, categorical_feature):
    # Apply mean target encoding
    encoder = MeanTargetEncoder(categorical_feature=categorical_feature)
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)

    # Train a gradient boosting classifier
    model = GradientBoostingClassifier()
    model.fit(X_train_encoded, y_train)

    # Predict probabilities and calculate ROC-AUC score
    y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return roc_auc