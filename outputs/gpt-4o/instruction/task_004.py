from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.target_means = {}

    def fit(self, X, y):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns
        for col in self.columns:
            self.target_means[col] = X.groupby(col)[y.name].mean()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            X_transformed[col] = X_transformed[col].map(self.target_means[col])
        return X_transformed

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Mean Target Encoding
    encoder = MeanTargetEncoder()
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)

    # Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_encoded, y_train)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]

    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return roc_auc