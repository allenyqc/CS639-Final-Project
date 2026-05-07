import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris  # default demo dataset


def train_neural_network_and_evaluate(
    X=None,
    y=None,
    test_size: float = 0.2,
    random_state: int = 42,
    hidden_layer_sizes: tuple = (100, 50),
    max_iter: int = 500,
    activation: str = "relu",
    solver: str = "adam",
    learning_rate_init: float = 1e-3,
) -> dict:
    """
    Normalizes numeric features, trains an MLP classifier, and returns evaluation metrics.

    Best-practice highlights
    ------------------------
    * Train/test split happens FIRST (before any preprocessing).
    * MinMaxScaler is fit ONLY on X_train, then applied to both splits.
    * Returns accuracy plus F1 / AUC so callers can choose the right metric
      for imbalanced problems.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), optional
        Feature matrix.  Defaults to the Iris dataset for demonstration.
    y : array-like of shape (n_samples,), optional
        Target vector.  Defaults to the Iris labels.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility.
    hidden_layer_sizes : tuple
        MLP architecture (neurons per hidden layer).
    max_iter : int
        Maximum training epochs.
    activation : str
        Activation function for hidden layers.
    solver : str
        Weight-optimisation algorithm.
    learning_rate_init : float
        Initial learning rate (used by 'adam' and 'sgd').

    Returns
    -------
    dict with keys:
        test_accuracy  – plain accuracy on the test set
        test_f1        – macro-averaged F1 (better for imbalanced data)
        test_auc       – macro-averaged one-vs-rest ROC-AUC
        model          – fitted MLPClassifier
        scaler         – fitted MinMaxScaler (for inference on new data)
    """
    from sklearn.metrics import f1_score, roc_auc_score

    # ------------------------------------------------------------------ #
    # 0. Load default data when none is provided
    # ------------------------------------------------------------------ #
    if X is None or y is None:
        data = load_iris(as_frame=False)
        X, y = data.data, data.target

    X = np.array(X, dtype=float)
    y = np.array(y)

    # ------------------------------------------------------------------ #
    # 1. Train / test split — BEFORE any preprocessing
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ------------------------------------------------------------------ #
    # 2. Scale features — fit ONLY on training data
    # ------------------------------------------------------------------ #
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
    X_test_scaled  = scaler.transform(X_test)        # transform only on test

    # ------------------------------------------------------------------ #
    # 3. Train the neural network
    # ------------------------------------------------------------------ #
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,          # uses an internal validation split
        n_iter_no_change=20,
    )
    model.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------ #
    # 4. Evaluate on the held-out test set
    # ------------------------------------------------------------------ #
    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    test_accuracy = accuracy_score(y_test, y_pred)

    # Macro F1 — robust to class imbalance
    test_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Macro OvR AUC — robust to class imbalance
    n_classes = len(np.unique(y))
    if n_classes == 2:
        test_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        test_auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        )

    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
    print(f"Test AUC (macro OvR): {test_auc:.4f}")

    return {
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_auc": test_auc,
        "model": model,
        "scaler": scaler,
    }


# ------------------------------------------------------------------ #
# Quick smoke-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    results = train_neural_network_and_evaluate()
    print("\nReturned dict keys:", list(results.keys()))
    print("Test accuracy:", results["test_accuracy"])