"""
Multi-layer Neural Network Training Module with Early Stopping and LR Scheduling.

Implements a training loop using numpy-based MLP with:
- Train/validation/test splits (no data leakage)
- Early stopping (patience=10)
- Learning rate decay (halve every 20 epochs)
- Loss curve tracking
- Comprehensive evaluation metrics
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    accuracy: float
    f1: float
    auc: float
    stopped_epoch: int
    best_val_loss: float


@dataclass
class LossHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Categorical cross-entropy loss."""
    eps = 1e-12
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


# ---------------------------------------------------------------------------
# Simple MLP implemented with numpy
# ---------------------------------------------------------------------------

class NumpyMLP:
    """
    A simple multi-layer perceptron implemented with numpy.

    Architecture: input -> [hidden layers with ReLU] -> output (softmax)
    """

    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.01,
        l2_lambda: float = 1e-4,
        random_state: Optional[int] = 42,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least input and output dimensions.")
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.rng = np.random.default_rng(random_state)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._init_params()

    def _init_params(self) -> None:
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # He initialisation for ReLU layers
            scale = np.sqrt(2.0 / fan_in)
            W = self.rng.normal(0, scale, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        pre_activations = []
        current = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            pre_activations.append(z)
            if i < len(self.weights) - 1:
                current = relu(z)
            else:
                current = softmax(z)
            activations.append(current)
        return activations, pre_activations

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def _backward(
        self,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
        y_true: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n = y_true.shape[0]
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        # Output layer gradient (softmax + cross-entropy combined)
        delta = activations[-1] - y_true  # (n, n_classes)

        for i in reversed(range(len(self.weights))):
            grad_weights[i] = (activations[i].T @ delta) / n + self.l2_lambda * self.weights[i]
            grad_biases[i] = delta.mean(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_grad(pre_activations[i - 1])

        return grad_weights, grad_biases

    def update_params(
        self,
        grad_weights: list[np.ndarray],
        grad_biases: list[np.ndarray],
    ) -> None:
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]

    def compute_loss(self, X: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = self.predict_proba(X)
        return cross_entropy_loss(y_pred, y_true)

    def partial_fit_batch(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> float:
        activations, pre_activations = self._forward(X_batch)
        loss = cross_entropy_loss(activations[-1], y_batch)
        grad_w, grad_b = self._backward(activations, pre_activations, y_batch)
        self.update_params(grad_w, grad_b)
        return loss


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    learning_rate: float = 0.01,
    max_epochs: int = 300,
    batch_size: int = 64,
    patience: int = 10,
    lr_decay_every: int = 20,
    lr_decay_factor: float = 0.5,
    l2_lambda: float = 1e-4,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    plot: bool = True,
) -> tuple[NumpyMLP, TrainingMetrics, LossHistory]:
    """
    Train a numpy MLP with early stopping and learning rate scheduling.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,) — integer class labels
    hidden_layer_sizes : tuple of ints defining hidden layer widths
    learning_rate : initial learning rate
    max_epochs : maximum number of training epochs
    batch_size : mini-batch size
    patience : early stopping patience (epochs without val improvement)
    lr_decay_every : halve LR every this many epochs
    lr_decay_factor : multiplicative factor for LR decay
    l2_lambda : L2 regularisation coefficient
    test_size : fraction of data held out as test set
    val_size : fraction of *training* data used for validation
    random_state : reproducibility seed
    plot : whether to plot loss curves

    Returns
    -------
    model : trained NumpyMLP
    metrics : TrainingMetrics on the test set
    history : LossHistory (train and val losses per epoch)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    # ------------------------------------------------------------------
    # 1. Split BEFORE any preprocessing (test set never touched again)
    # ------------------------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Separate validation set from training data
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=random_state,
        stratify=y_trainval,
    )

    # ------------------------------------------------------------------
    # 2. Fit scaler ONLY on training data
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 3. One-hot encode labels
    # ------------------------------------------------------------------
    lb = LabelBinarizer()
    y_train_oh = lb.fit_transform(y_train)
    y_val_oh = lb.transform(y_val)
    n_classes = y_train_oh.shape[1]

    # Handle binary case: LabelBinarizer returns (n, 1) for binary
    if n_classes == 1:
        y_train_oh = np.hstack([1 - y_train_oh, y_train_oh])
        y_val_oh = np.hstack([1 - y_val_oh, y_val_oh])
        n_classes = 2

    n_features = X_train_scaled.shape[1]
    layer_sizes = [n_features, *list(hidden_layer_sizes), n_classes]

    # ------------------------------------------------------------------
    # 4. Build model
    # ------------------------------------------------------------------
    model = NumpyMLP(
        layer_sizes=layer_sizes,
        learning_rate=learning_rate,
        l2_lambda=l2_lambda,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # 5. Training loop with early stopping and LR scheduling
    # ------------------------------------------------------------------
    history = LossHistory()
    best_val_loss = np.inf
    best_weights = [w.copy() for w in model.weights]
    best_biases = [b.copy() for b in model.biases]
    no_improve_count = 0
    stopped_epoch = max_epochs
    n_train = X_train_scaled.shape[0]
    rng = np.random.default_rng(random_state)

    for epoch in range(1, max_epochs + 1):
        # --- Learning rate schedule: halve every lr_decay_every epochs ---
        if epoch > 1 and (epoch - 1) % lr_decay_every == 0:
            model.learning_rate *= lr_decay_factor

        # --- Mini-batch SGD ---
        indices = rng.permutation(n_train)
        epoch_train_losses = []
        for start in range(0, n_train, batch_size):
            batch_idx = indices[start : start + batch_size]
            X_batch = X_train_scaled[batch_idx]
            y_batch = y_train_oh[batch_idx]
            batch_loss = model.partial_fit_batch(X_batch, y_batch)
            epoch_train_losses.append(batch_loss)

        train_loss = float(np.mean(epoch_train_losses))
        val_loss = model.compute_loss(X_val_scaled, y_val_oh)

        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)

        # --- Early stopping ---
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights = [w.copy() for w in model.weights]
            best_biases = [b.copy() for b in model.biases]
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            stopped_epoch = epoch
            print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
            break
    else:
        stopped_epoch = max_epochs

    # Restore best weights
    model.weights = best_weights
    model.biases = best_biases

    # ------------------------------------------------------------------
    # 6. Final evaluation on the HELD-OUT test set (only here)
    # ------------------------------------------------------------------
    y_pred_labels = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred_labels)

    # F1: macro-averaged (appropriate for multi-class; warns if imbalanced)
    f1 = f1_score(y_test, y_pred_labels, average="macro", zero_division=0)

    # AUC: handle binary vs multi-class
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(
                y_test,
                y_pred_proba,
                multi_class="ovr",
                average="macro",
            )
    except ValueError as exc:
        warnings.warn(f"AUC computation failed: {exc}. Setting AUC=NaN.")
        auc = float("nan")

    metrics = TrainingMetrics(
        accuracy=accuracy,
        f1=f1,
        auc=auc,
        stopped_epoch=stopped_epoch,
        best_val_loss=best_val_loss,
    )

    print("\n=== Test Set Evaluation ===")
    print(f"  Accuracy      : {accuracy:.4f}")
    print(f"  Macro F1      : {f1:.4f}")
    print(f"  AUC (macro)   : {auc:.4f}")
    print(f"  Stopped epoch : {stopped_epoch}")
    print(f"  Best val loss : {best_val_loss:.6f}")

    # ------------------------------------------------------------------
    # 7. Plot loss curves
    # ------------------------------------------------------------------
    if plot:
        _plot_loss_curves(history, stopped_epoch)

    return model, metrics, history


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _plot_loss_curves(history: LossHistory, stopped_epoch: int) -> None:
    epochs = range(1, len(history.train_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history.train_losses, label="Training loss", linewidth=2)
    ax.plot(epochs, history.val_losses, label="Validation loss", linewidth=2, linestyle="--")
    ax.axvline(
        x=stopped_epoch,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label=f"Early stop (epoch {stopped_epoch})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150)
    plt.show()
    print("Loss curve saved to loss_curves.png")


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Generate a synthetic classification dataset
    X_demo, y_demo = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=0,
    )

    trained_model, test_metrics, loss_history = train_mlp(
        X=X_demo,
        y=y_demo,
        hidden_layer_sizes=(128, 64, 32),
        learning_rate=0.05,
        max_epochs=300,
        batch_size=64,
        patience=10,
        lr_decay_every=20,
        lr_decay_factor=0.5,
        l2_lambda=1e-4,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        plot=True,
    )