import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef


def train_and_evaluate_nn(
    X,
    y,
    test_size=0.2,
    random_state=42,
    hidden_layer_sizes=(64, 32),
    max_iter=200,
    imbalance_threshold=0.2,
):
    """
    Normalize numeric features to [0, 1], split into train/test,
    train an MLP, and return evaluation metrics.

    Best practices applied:
    - Train/test split is performed BEFORE any feature engineering.
    - MinMaxScaler is fit ONLY on training data, then applied to test data.
    - For imbalanced datasets, F1, AUC, and MCC are reported alongside accuracy.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Numeric feature matrix.
    y : array-like of shape (n_samples,)
        Target labels.
    test_size : float
        Fraction of samples used for testing.
    random_state : int
        Random seed for reproducibility.
    hidden_layer_sizes : tuple
        MLP architecture.
    max_iter : int
        Maximum training iterations.
    imbalance_threshold : float
        If minority class fraction is below this, treat as imbalanced.

    Returns
    -------
    results : dict
        Dictionary containing test accuracy and (when applicable)
        F1, AUC, and MCC scores.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # 1) Split BEFORE any feature engineering, stratified to preserve class ratios.
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # 2) Fit scaler ONLY on training data, then transform both splits.
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3) Train neural network.
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    # 4) Evaluate.
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    results = {"test_accuracy": test_accuracy}

    # Detect class imbalance from TRAINING data only (never use test for decisions).
    classes, counts = np.unique(y_train, return_counts=True)
    minority_fraction = counts.min() / counts.sum()
    is_imbalanced = (len(classes) > 1) and (minority_fraction < imbalance_threshold)

    if is_imbalanced:
        avg = "binary" if len(classes) == 2 else "macro"
        results["f1"] = f1_score(y_test, y_pred, average=avg)
        results["mcc"] = matthews_corrcoef(y_test, y_pred)
        try:
            if len(classes) == 2:
                proba = model.predict_proba(X_test_scaled)[:, 1]
                results["auc"] = roc_auc_score(y_test, proba)
            else:
                proba = model.predict_proba(X_test_scaled)
                results["auc"] = roc_auc_score(
                    y_test, proba, multi_class="ovr", average="macro"
                )
        except Exception:
            results["auc"] = None
        results["note"] = (
            "Imbalanced dataset detected; prefer F1/AUC/MCC over plain accuracy."
        )

    return results


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    metrics = train_and_evaluate_nn(data.data, data.target)
    print(metrics)