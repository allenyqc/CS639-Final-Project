import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve

def train_logistic_regression_with_f1(X, y, test_size=0.2, random_state=42):
    """
    Trains a logistic regression classifier, finds the optimal classification threshold
    to maximize F1-score, and reports the final test F1-score at that threshold.

    Parameters:
    - X: Features (numpy array or pandas DataFrame)
    - y: Labels (numpy array or pandas Series)
    - test_size: Proportion of data to use for testing
    - random_state: Random seed for reproducibility

    Returns:
    - optimal_threshold: The threshold that maximizes F1-score
    - test_f1: The F1-score on the test set at the optimal threshold
    """
    # Step 1: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Step 2: Preprocessing (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Train logistic regression model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Step 4: Find optimal threshold on training data
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)  # Avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Step 5: Evaluate on the test set using the optimal threshold
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    test_f1 = f1_score(y_test, y_test_pred)

    return optimal_threshold, test_f1