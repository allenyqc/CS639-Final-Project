import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

def train_and_evaluate_binary_classifier(X, y, random_state=42):
    """
    Train a binary classifier on an imbalanced dataset and report its performance.

    Parameters:
    - X: np.ndarray or pd.DataFrame, feature matrix
    - y: np.ndarray or pd.Series, target vector (binary labels)
    - random_state: int, random seed for reproducibility

    Returns:
    - None (prints performance metrics)
    """
    # Step 1: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

    # Step 2: Preprocessing (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Handle class imbalance using class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Step 4: Train the model
    model = RandomForestClassifier(random_state=random_state, class_weight=class_weight_dict)
    model.fit(X_train_scaled, y_train)

    # Step 5: Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Step 6: Evaluate performance
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)

    print("Performance Metrics:")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Example usage:
# Assuming you have an imbalanced dataset with features X and labels y
# X, y = ... (load your dataset here)
# train_and_evaluate_binary_classifier(X, y)