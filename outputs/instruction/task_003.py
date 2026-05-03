import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.pipeline import Pipeline

def tune_and_evaluate_rf(X, y, random_state=42):
    """
    Tunes hyperparameters for a Random Forest model using GridSearchCV and evaluates the best model on a test set.

    Parameters:
    - X: pd.DataFrame or np.ndarray, feature matrix
    - y: pd.Series or np.ndarray, target vector
    - random_state: int, random seed for reproducibility

    Returns:
    - dict containing evaluation metrics (F1, AUC, MCC) and the best model
    """
    # Step 1: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

    # Step 2: Define preprocessing and model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale features
        ('rf', RandomForestClassifier(random_state=random_state))  # Random Forest model
    ])

    # Step 3: Define hyperparameter grid for Random Forest
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }

    # Step 4: Perform GridSearchCV on training data
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',  # Use F1 score for hyperparameter tuning
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Step 5: Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Return evaluation metrics and the best model
    return {
        'f1_score': f1,
        'auc': auc,
        'mcc': mcc,
        'classification_report': classification_report(y_test, y_pred),
        'best_model': best_model
    }