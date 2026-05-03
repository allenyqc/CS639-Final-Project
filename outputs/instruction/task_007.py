import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

def select_top_features_and_train(X, y, random_state=42):
    """
    Selects the top-10 most informative features using mutual information,
    trains a decision tree on the selected features, and reports test metrics.

    Parameters:
    - X: pd.DataFrame, feature matrix
    - y: pd.Series or np.array, target labels
    - random_state: int, random seed for reproducibility

    Returns:
    - metrics: dict, containing F1 score, AUC, and MCC on the test set
    """
    # Step 1: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Step 2: Feature selection using mutual information
    mi_scores = mutual_info_classif(X_train, y_train, random_state=random_state)
    top_features_idx = np.argsort(mi_scores)[-10:]  # Indices of top-10 features
    X_train_selected = X_train.iloc[:, top_features_idx]
    X_test_selected = X_test.iloc[:, top_features_idx]

    # Step 3: Scale the features (fit scaler on training data only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Step 4: Train a decision tree classifier
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train_scaled, y_train)

    # Step 5: Evaluate the model on the test set
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics = {"F1 Score": f1, "AUC": auc, "MCC": mcc}
    return metrics