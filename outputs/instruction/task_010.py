import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

def train_and_evaluate_with_smote(X, y, test_size=0.2, random_state=42):
    """
    Oversamples the minority class using SMOTE, trains a logistic regression classifier,
    and reports the test F1-score.

    Parameters:
    - X: Features (numpy array or pandas DataFrame)
    - y: Target labels (numpy array or pandas Series)
    - test_size: Proportion of the dataset to include in the test split
    - random_state: Random seed for reproducibility

    Returns:
    - f1: F1-score on the test set
    """
    # Step 1: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Step 2: Apply SMOTE to the training data
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Step 3: Scale the features (fit scaler on training data only)
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    # Step 4: Train logistic regression model
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train_resampled, y_train_resampled)

    # Step 5: Make predictions and calculate F1-score
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    return f1