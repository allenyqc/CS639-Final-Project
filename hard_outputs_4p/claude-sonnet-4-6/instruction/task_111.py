"""
multi_layer_nn_training.py

A module that trains a multi-layer neural network with early stopping and
learning rate scheduling, following ML best practices.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import warnings
import copy

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Custom Neural Network with manual training loop (numpy-based)
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_deriv(a):
    return a * (1.0 - a)


def relu(z):
    return np.maximum(0, z)


def relu_deriv(a):
    return (a > 0).astype(float)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(y_pred_proba, y_true_onehot):
    eps = 1e-12
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred_proba + eps), axis=1))


class SimpleNeuralNetwork:
    """
    A simple fully-connected neural network implemented with NumPy.
    Supports:
      - Configurable hidden layers
      - ReLU activations (hidden) + Softmax (output)
      - Mini-batch SGD
      - Early stopping (patience-based on validation loss)
      - Learning rate halving every `lr_decay_epochs` epochs
    """

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        learning_rate=0.01,
        max_epochs=200,
        batch_size=32,
        patience=10,
        lr_decay_epochs=20,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr_decay_epochs = lr_decay_epochs
        self.random_state = random_state

        self.weights = []
        self.biases = []
        self.train_loss_history = []
        self.val_loss_history = []
        self.stopped_epoch = max_epochs
        self.classes_ = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_params(self, n_features, n_classes):
        rng = np.random.default_rng(self.random_state)
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_classes]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # He initialisation for ReLU layers
            scale = np.sqrt(2.0 / fan_in)
            W = rng.normal(0, scale, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X):
        activations = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            if i < len(self.weights) - 1:
                a = relu(z)
            else:
                a = softmax(z)
            activations.append(a)
        return activations  # activations[-1] is the output probabilities

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def _backward(self, activations, y_onehot, lr):
        m = y_onehot.shape[0]
        n_layers = len(self.weights)

        # Output layer gradient (softmax + cross-entropy combined)
        delta = activations[-1] - y_onehot  # (m, n_classes)

        for i in reversed(range(n_layers)):
            a_prev = activations[i]
            dW = (a_prev.T @ delta) / m
            db = delta.mean(axis=0, keepdims=True)

            if i > 0:
                # Propagate to previous layer
                delta = (delta @ self.weights[i].T) * relu_deriv(activations[i])

            # Update parameters
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db

    # ------------------------------------------------------------------
    # One-hot encoding helper
    # ------------------------------------------------------------------

    def _to_onehot(self, y):
        n_classes = len(self.classes_)
        onehot = np.zeros((len(y), n_classes))
        for idx, cls in enumerate(self.classes_):
            onehot[y == cls, idx] = 1.0
        return onehot

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train the network.

        Parameters
        ----------
        X_train, y_train : training data (already scaled)
        X_val, y_val     : validation data (already scaled)
        """
        self.classes_ = np.unique(y_train)
        n_features = X_train.shape[1]
        n_classes = len(self.classes_)

        self._init_params(n_features, n_classes)

        y_train_oh = self._to_onehot(y_train)
        y_val_oh = self._to_onehot(y_val)

        best_val_loss = np.inf
        best_weights = None
        best_biases = None
        no_improve_count = 0
        lr = self.learning_rate

        rng = np.random.default_rng(self.random_state)

        for epoch in range(1, self.max_epochs + 1):

            # ---- Learning rate schedule: halve every lr_decay_epochs ----
            if epoch > 1 and (epoch - 1) % self.lr_decay_epochs == 0:
                lr /= 2.0

            # ---- Mini-batch SGD ----
            indices = rng.permutation(len(X_train))
            X_shuf = X_train[indices]
            y_shuf = y_train_oh[indices]

            for start in range(0, len(X_train), self.batch_size):
                end = start + self.batch_size
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                activations = self._forward(Xb)
                self._backward(activations, yb, lr)

            # ---- Compute losses ----
            train_acts = self._forward(X_train)
            train_loss = cross_entropy_loss(train_acts[-1], y_train_oh)

            val_acts = self._forward(X_val)
            val_loss = cross_entropy_loss(val_acts[-1], y_val_oh)

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            # ---- Early stopping ----
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                self.stopped_epoch = epoch
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(best val loss={best_val_loss:.4f}, lr={lr:.6f})"
                )
                break
        else:
            self.stopped_epoch = self.max_epochs

        # Restore best weights
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X):
        activations = self._forward(X)
        return activations[-1]

    def predict(self, X):
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_neural_network(
    X,
    y,
    test_size=0.2,
    val_size=0.15,
    hidden_layer_sizes=(128, 64),
    learning_rate=0.01,
    max_epochs=300,
    batch_size=64,
    patience=10,
    lr_decay_epochs=20,
    random_state=42,
    plot=True,
):
    """
    Full training pipeline for a multi-layer neural network.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    test_size : float, fraction of data held out as final test set
    val_size  : float, fraction of *training* data used for validation
    hidden_layer_sizes : tuple of ints
    learning_rate : float
    max_epochs : int
    batch_size : int
    patience : int, early-stopping patience (epochs)
    lr_decay_epochs : int, halve LR every this many epochs
    random_state : int
    plot : bool, whether to plot loss curves

    Returns
    -------
    dict with keys:
        'model'         : trained SimpleNeuralNetwork
        'scaler'        : fitted StandardScaler
        'metrics'       : dict(accuracy, f1, auc, stopped_epoch)
        'loss_history'  : dict(train, val)
    """

    # ------------------------------------------------------------------
    # 1. Train / test split FIRST — before any preprocessing
    # ------------------------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ------------------------------------------------------------------
    # 2. Further split train into train + validation
    # ------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    print(
        f"Dataset sizes — train: {len(X_train)}, "
        f"val: {len(X_val)}, test: {len(X_test)}"
    )

    # ------------------------------------------------------------------
    # 3. Fit scaler ONLY on training data; transform val & test
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Build and train the model
    # ------------------------------------------------------------------
    model = SimpleNeuralNetwork(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=patience,
        lr_decay_epochs=lr_decay_epochs,
        random_state=random_state,
    )

    print("\nTraining …")
    model.fit(X_train_sc, y_train, X_val_sc, y_val)

    # ------------------------------------------------------------------
    # 5. Evaluate ONLY on the held-out test set
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    n_classes = len(np.unique(y))
    if n_classes == 2:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="weighted"
        )

    metrics = {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "auc": auc,
        "stopped_epoch": model.stopped_epoch,
    }

    print("\n=== Test-set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:>15s}: {v:.4f}" if isinstance(v, float) else f"  {k:>15s}: {v}")

    # ------------------------------------------------------------------
    # 6. Plot training / validation loss curves
    # ------------------------------------------------------------------
    loss_history = {
        "train": model.train_loss_history,
        "val": model.val_loss_history,
    }

    if plot:
        _plot_loss_curves(loss_history, model.stopped_epoch)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "loss_history": loss_history,
    }


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _plot_loss_curves(loss_history, stopped_epoch):
    epochs = range(1, len(loss_history["train"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, loss_history["train"], label="Training loss", linewidth=2)
    ax.plot(epochs, loss_history["val"], label="Validation loss", linewidth=2, linestyle="--")

    ax.axvline(
        x=stopped_epoch,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=f"Early stop (epoch {stopped_epoch})",
    )

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
    ax.set_title("Training & Validation Loss Curves", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)
    plt.show()
    print("Loss curve saved to loss_curves.png")


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate a synthetic classification dataset
    X_demo, y_demo = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=0,
    )

    results = train_neural_network(
        X_demo,
        y_demo,
        test_size=0.20,
        val_size=0.15,
        hidden_layer_sizes=(128, 64, 32),
        learning_rate=0.02,
        max_epochs=300,
        batch_size=64,
        patience=10,
        lr_decay_epochs=20,
        random_state=42,
        plot=True,
    )

    print("\nReturned keys:", list(results.keys()))
    print("Loss history length (train):", len(results["loss_history"]["train"]))