from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

def mean_target_encoding(X_train, X_test, y_train, categorical_feature):
    # Calculate mean target encoding for the training data
    target_mean = y_train.mean()
    means = X_train.groupby(categorical_feature)[y_train.name].mean()
    counts = X_train.groupby(categorical_feature)[y_train.name].count()
    smoothing = 1 / (1 + np.exp(-(counts - 1)))
    prior = target_mean
    means_smoothed = prior * (1 - smoothing) + means * smoothing

    # Map the means to the train and test set
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    X_train_encoded[categorical_feature] = X_train[categorical_feature].map(means_smoothed)
    X_test_encoded[categorical_feature] = X_test[categorical_feature].map(means_smoothed).fillna(prior)

    return X_train_encoded, X_test_encoded

def train_and_evaluate(X_train, X_test, y_train, y_test, categorical_feature):
    # Apply mean target encoding
    X_train_encoded, X_test_encoded = mean_target_encoding(X_train, X_test, y_train, categorical_feature)

    # Train a Gradient Boosting Classifier
    model = GradientBoostingClassifier()
    model.fit(X_train_encoded, y_train)

    # Predict probabilities for the test set
    y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]

    # Calculate ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return roc_auc