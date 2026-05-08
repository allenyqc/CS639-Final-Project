```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    n = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(n), y_true] + 1e-15)
    return np.mean(log_likelihood)


class NeuralNetwork:
    """
    A simple multi-layer neural network implemented with NumPy.
    Supports multiple hidden layers, ReLU activations, and softmax output.
    """

    def __init__(self, layer_sizes, learning_rate=0.01, random_state=42):
        """
        Args:
            layer_sizes: list of ints, e.g. [input_dim, 64, 32, num_classes]
            learning_rate: initial learning rate
            random_state: seed for reproducibility
        """
        np.random.seed(random_state)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self._initialize_weights()

    def _initialize_weights(self):
        """He initialization for weights."""
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """
        Forward pass through the network.
        Returns activations at each layer and pre-activations (z values).
        """
        activations = [X]
        z_values = []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)
            a = relu(z)
            activations.append(a)

        # Output layer with softmax
        z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        z_values.append(z_out)
        a_out = softmax(z_out)
        activations.append(a_out)

        return activations, z_values

    def backward(self, X, y, activations, z_values):
        """
        Backpropagation to compute gradients.
        """
        n = X.shape[0]
        num_layers = len(self.weights)
        grad_w = [None] * num_layers
        grad_b = [None] * num_layers

        # Output layer gradient (cross-entropy + softmax combined)
        delta = activations[-1].copy()
        delta[range(n), y] -= 1
        delta /= n

        grad_w[-1] = activations[-2].T @ delta
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Hidden layers
        for i in range(num_layers - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * relu_derivative(z_values[i])
            grad_w[i] = activations[i].T @ delta
            grad_b[i] = np.sum(delta, axis=0, keepdims=True)

        return grad_w, grad_b

    def update_weights(self, grad_w, grad_b):
        """Gradient descent weight update."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_w[i]
            self.biases[i] -= self.learning_rate * grad_b[i]

    def predict_proba(self, X):
        """Return class probabilities."""
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X):
        """Return class predictions."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def compute_loss(self, X, y):
        """Compute cross-entropy loss."""
        proba = self.predict_proba(X)
        return cross_entropy_loss(proba, y)


def train_neural_network(
    X,
    y,
    hidden_layers=(64, 32),
    max_epochs=200,
    patience=10,
    lr_decay_every=20,
    lr_decay_factor=0.5,
    initial_lr=0.01,
    batch_size=32,
    test_size=0.2,
    val_size=0.15,
    random_state=42,
    use_sklearn=False,
    plot=True,
):
    """
    Train a multi-layer neural network with early stopping and LR scheduling.

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        hidden_layers: tuple of hidden layer sizes
        max_epochs: maximum number of training epochs
        patience: number of epochs without improvement before stopping
        lr_decay_every: halve LR every this many epochs
        lr_decay_factor: factor to multiply LR by (default 0.5 = halve)
        initial_lr: starting learning rate
        batch_size: mini-batch size
        test_size: fraction of data for test set
        val_size: fraction of training data for validation
        random_state: random seed
        use_sklearn: if True, use sklearn MLPClassifier (demonstrates early stopping)
        plot: if True, plot training/validation curves

    Returns:
        dict with keys: model, metrics, train_loss_history, val_loss_history,
                        stopped_epoch, scaler
    """
    print("=" * 60)
    print("Neural Network Training with Early Stopping & LR Scheduling")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Split data into train / validation / test
    # ------------------------------------------------------------------ #
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_trainval,
    )

    print(f"\nData splits:")
    print(f"  Train:      {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test:       {X_test.shape[0]} samples")

    # ------------------------------------------------------------------ #
    # 2. Feature scaling
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    num_classes = len(np.unique(y))
    print(f"\nNumber of classes: {num_classes}")

    # ------------------------------------------------------------------ #
    # 3. Choose training path
    # ------------------------------------------------------------------ #
    if use_sklearn:
        model, metrics, train_loss_history, val_loss_history, stopped_epoch = \
            _train_sklearn(
                X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
                hidden_layers, max_epochs, patience, initial_lr, random_state
            )
    else:
        model, metrics, train_loss_history, val_loss_history, stopped_epoch = \
            _train_numpy(
                X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
                hidden_layers, max_epochs, patience,
                lr_decay_every, lr_decay_factor, initial_lr,
                batch_size, num_classes, random_state
            )

    # ------------------------------------------------------------------ #
    # 4. Report metrics
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Final Test Set Metrics")
    print("=" * 60)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  AUC      : {metrics['auc']:.4f}")
    print(f"  Stopped at epoch: {stopped_epoch}")

    # ------------------------------------------------------------------ #
    # 5. Plot curves
    # ------------------------------------------------------------------ #
    if plot:
        _plot_curves(train_loss_history, val_loss_history, stopped_epoch)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "stopped_epoch": stopped_epoch,
    }


# ------------------------------------------------------------------ #
# NumPy training loop
# ------------------------------------------------------------------ #
def _train_numpy(
    X_train, y_train, X_val, y_val, X_test, y_test,
    hidden_layers, max_epochs, patience,
    lr_decay_every, lr_decay_factor, initial_lr,
    batch_size, num_classes, random_state
):
    n_features = X_train.shape[1]
    layer_sizes = [n_features] + list(hidden_layers) + [num_classes]

    model = NeuralNetwork(layer_sizes, learning_rate=initial_lr,
                          random_state=random_state)

    train_loss_history = []
    val_loss_history = []

    best_val_loss = np.inf
    best_weights = None
    best_biases = None
    epochs_no_improve = 0
    stopped_epoch = max_epochs

    n_train = X_train.shape[0]
    current_lr = initial_lr

    print(f"\nArchitecture: {layer_sizes}")
    print(f"Initial LR: {initial_lr}, Patience: {patience}, "
          f"LR decay every {lr_decay_every} epochs (factor {lr_decay_factor})")
    print(f"\nTraining for up to {max_epochs} epochs...\n")

    for epoch in range(1, max_epochs + 1):
        # ---- LR scheduling: halve every lr_decay_every epochs ----
        if epoch > 1 and (epoch - 1) % lr_decay_every == 0:
            current_lr *= lr_decay_factor
            model.learning_rate = current_lr
            print(f"  [Epoch {epoch}] LR decayed to {current_lr:.6f}")

        # ---- Mini-batch SGD ----
        indices = np.random.permutation(n_train)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            activations, z_values = model.forward(X_batch)
            grad_w, grad_b = model.backward(X_batch, y_batch, activations, z_values)
            model.update_weights(grad_w, grad_b)

        # ---- Compute losses ----
        train_loss = model.compute_loss(X_train, y_train)
        val_loss = model.compute_loss(X_val, y_val)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # ---- Early stopping ----
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights = [w.copy() for w in model.weights]
            best_biases = [b.copy() for b in model.biases]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stopped_epoch = epoch
                print(f"\n  Early stopping triggered at epoch {epoch}. "
                      f"Best val loss: {best_val_loss:.4f}")
                break

    # Restore best weights
    if best_weights is not None:
        model.weights = best_weights
        model.biases = best_biases

    # ---- Evaluate on test set ----
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = _compute_metrics(y_test, y_pred, y_proba)
    return model, metrics, train_loss_history, val_loss_history, stopped_epoch


# ------------------------------------------------------------------ #
# sklearn training path
# ------------------------------------------------------------------ #
def _train_sklearn(
    X_train, y_train, X_val, y_val, X_test, y_test,
    hidden_layers, max_epochs, patience, initial_lr, random_state
):
    """
    Use sklearn MLPClassifier with warm_start to simulate epoch-by-epoch
    training, enabling custom early stopping and LR decay tracking.
    """
    print(f"\nUsing sklearn MLPClassifier")
    print(f"Architecture: {hidden_layers}")

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=initial_lr,
        max_iter=1,          # one epoch at a time
        warm_start=True,     # keep weights between calls
        random_state=random_state,
        early_stopping=False,
    )

    train_loss_history = []
    val_loss_history = []

    best_val_loss = np.inf
    best_model_coefs = None
    best_model_intercepts = None
    epochs_no_improve = 0
    stopped_epoch = max_epochs
    current_lr = initial_lr

    for epoch in range(1, max_epochs + 1):
        # LR decay every 20 epochs
        if epoch > 1 and (epoch - 1) % 20 == 0:
            current_lr *= 0.5
            model.learning_rate_init = current_lr
            print(f"  [Epoch {epoch}] LR decayed to {current_lr:.6f}")

        model.fit(X_train, y_train)

        train_loss = model.loss_
        # Compute val loss manually
        val_proba = model.predict_proba(X_val)
        n = len(y_val)
        val_loss = -np.mean(np.log(val_proba[range(n), y_val] + 1e-15))

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_coefs = [c.copy() for c in model.coefs_]
            best_model_intercepts = [i.copy() for i in model.intercepts_]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stopped_epoch = epoch
                print(f"\n  Early stopping triggered at epoch {epoch}. "
                      f"Best val loss: {best_val_loss:.4f}")
                break

    # Restore best weights
    if best_model_coefs is not None:
        model.coefs_ = best_model_coefs
        model.intercepts_ = best_model_intercepts

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = _compute_metrics(y_test, y_pred, y_proba)
    return model, metrics, train_loss_history, val_loss_history, stopped_epoch


# ------------------------------------------------------------------ #
# Metrics helper
# ------------------------------------------------------------------ #
def _compute_metrics(y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
    num_classes = y_proba.shape[1]

    if num_classes == 2:
        f1 = f1_score(y_test, y_pred, average='binary')
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')

    return {"accuracy": accuracy, "f1": f1, "auc": auc}


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #
def _plot_curves(train_loss, val_loss, stopped_epoch):
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label='Training Loss', color='steelblue', linewidth=2)
    ax.plot(epochs, val_loss, label='Validation Loss', color='tomato', linewidth=2)
    ax.axvline(x=stopped_epoch, color='green', linestyle='--',
               linewidth=1.5, label=f'Early Stop (epoch {stopped_epoch})')

    # Mark best validation loss
    best_epoch = int(np.argmin(val_loss)) + 1
    ax.scatter([best_epoch], [min(val_loss)], color='gold', zorder=5,
               s=100, label=f'Best Val Loss (epoch {best_epoch})')

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=13)
    ax.set_title('Training & Validation Loss Curves', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=150)
    plt.show()
    print("\nLoss curve saved to 'loss_curves.png'")


# ------------------------------------------------------------------ #
# Demo / self-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    print("Generating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=3,
        random_state=42,
    )

    # ---- NumPy implementation ----
    print("\n" + "#" * 60)
    print("