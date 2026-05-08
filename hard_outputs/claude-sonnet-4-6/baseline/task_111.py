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


def cross_entropy_loss(y_pred, y_true, n_classes):
    m = y_true.shape[0]
    y_one_hot = np.zeros((m, n_classes))
    y_one_hot[np.arange(m), y_true.astype(int)] = 1
    log_pred = np.log(y_pred + 1e-15)
    loss = -np.sum(y_one_hot * log_pred) / m
    return loss


class SimpleMLPClassifier:
    """
    A simple multi-layer neural network classifier implemented with NumPy.
    Supports early stopping and learning rate decay scheduling.
    """

    def __init__(self, hidden_layer_sizes=(100, 50), learning_rate=0.01,
                 max_epochs=500, patience=10, lr_decay_every=20,
                 lr_decay_factor=0.5, batch_size=32, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr_decay_every = lr_decay_every
        self.lr_decay_factor = lr_decay_factor
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.stopped_epoch = max_epochs
        self.n_classes = None
        self.is_fitted = False

    def _initialize_weights(self, layer_sizes):
        np.random.seed(self.random_state)
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # He initialization
            scale = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

    def _forward_pass(self, X):
        activations = [X]
        pre_activations = []
        current = X

        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            current = relu(z)
            activations.append(current)

        # Output layer
        z_out = current @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z_out)
        if self.n_classes == 2:
            output = sigmoid(z_out)
        else:
            output = softmax(z_out)
        activations.append(output)

        return activations, pre_activations

    def _backward_pass(self, activations, pre_activations, y):
        m = y.shape[0]
        n_layers = len(self.weights)
        grad_weights = [None] * n_layers
        grad_biases = [None] * n_layers

        y_one_hot = np.zeros((m, self.n_classes))
        y_one_hot[np.arange(m), y.astype(int)] = 1

        # Output layer gradient
        delta = activations[-1] - y_one_hot

        for i in reversed(range(n_layers)):
            grad_weights[i] = activations[i].T @ delta / m
            grad_biases[i] = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_derivative(pre_activations[i - 1])

        return grad_weights, grad_biases

    def _update_weights(self, grad_weights, grad_biases):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]

    def fit(self, X_train, y_train, X_val, y_val):
        self.n_classes = len(np.unique(y_train))
        n_features = X_train.shape[1]

        # Build layer sizes
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [self.n_classes]
        self._initialize_weights(layer_sizes)

        best_val_loss = np.inf
        patience_counter = 0
        best_weights = None
        best_biases = None
        self.learning_rate = self.initial_learning_rate

        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        n_samples = X_train.shape[0]

        for epoch in range(self.max_epochs):
            # Learning rate decay schedule
            if epoch > 0 and epoch % self.lr_decay_every == 0:
                self.learning_rate *= self.lr_decay_factor
                print(f"  [LR Decay] Epoch {epoch}: Learning rate -> {self.learning_rate:.6f}")

            self.learning_rates.append(self.learning_rate)

            # Mini-batch training
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                activations, pre_activations = self._forward_pass(X_batch)
                grad_weights, grad_biases = self._backward_pass(activations, pre_activations, y_batch)
                self._update_weights(grad_weights, grad_biases)

            # Compute training loss
            train_activations, _ = self._forward_pass(X_train)
            train_loss = cross_entropy_loss(train_activations[-1], y_train, self.n_classes)
            self.train_losses.append(train_loss)

            # Compute validation loss
            val_activations, _ = self._forward_pass(X_val)
            val_loss = cross_entropy_loss(val_activations[-1], y_val, self.n_classes)
            self.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.max_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"LR: {self.learning_rate:.6f}")

            # Early stopping check
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                self.stopped_epoch = epoch + 1
                print(f"\n  [Early Stopping] Triggered at epoch {self.stopped_epoch}. "
                      f"Best val loss: {best_val_loss:.4f}")
                break
        else:
            self.stopped_epoch = self.max_epochs

        # Restore best weights
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        activations, _ = self._forward_pass(X)
        proba = activations[-1]
        if self.n_classes == 2 and proba.shape[1] == 2:
            return proba
        elif self.n_classes == 2 and proba.shape[1] == 1:
            return np.hstack([1 - proba, proba])
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def train_neural_network(X, y, hidden_layer_sizes=(100, 50), learning_rate=0.01,
                          max_epochs=500, patience=10, lr_decay_every=20,
                          lr_decay_factor=0.5, batch_size=32, test_size=0.2,
                          val_size=0.1, random_state=42, use_sklearn=False):
    """
    Train a multi-layer neural network with early stopping and learning rate scheduling.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    hidden_layer_sizes : tuple, sizes of hidden layers
    learning_rate : float, initial learning rate
    max_epochs : int, maximum number of training epochs
    patience : int, early stopping patience
    lr_decay_every : int, halve LR every this many epochs
    lr_decay_factor : float, factor to multiply LR by
    batch_size : int, mini-batch size
    test_size : float, fraction for test set
    val_size : float, fraction for validation set (from training data)
    random_state : int
    use_sklearn : bool, if True use sklearn MLPClassifier (no custom LR schedule)

    Returns
    -------
    dict with keys: model, metrics, train_losses, val_losses, stopped_epoch
    """
    print("=" * 60)
    print("NEURAL NETWORK TRAINING WITH EARLY STOPPING & LR SCHEDULING")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Split data
    # ------------------------------------------------------------------ #
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction,
        random_state=random_state, stratify=y_trainval
    )

    print(f"\nData splits:")
    print(f"  Train:      {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test:       {X_test.shape[0]} samples")

    # ------------------------------------------------------------------ #
    # 2. Standardize features
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ------------------------------------------------------------------ #
    # 3. Train model
    # ------------------------------------------------------------------ #
    if use_sklearn:
        print("\n[Mode] Using sklearn MLPClassifier with built-in early stopping")
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate,
            max_iter=max_epochs,
            early_stopping=True,
            validation_fraction=val_fraction,
            n_iter_no_change=patience,
            random_state=random_state,
            verbose=False
        )
        model.fit(X_train, y_train)
        train_losses = model.loss_curve_
        val_losses = model.validation_scores_ if hasattr(model, 'validation_scores_') else []
        stopped_epoch = model.n_iter_
        print(f"  Training stopped at iteration: {stopped_epoch}")
    else:
        print("\n[Mode] Using custom NumPy MLP with LR scheduling & early stopping")
        print(f"\nHyperparameters:")
        print(f"  Hidden layers:    {hidden_layer_sizes}")
        print(f"  Learning rate:    {learning_rate}")
        print(f"  Max epochs:       {max_epochs}")
        print(f"  Patience:         {patience}")
        print(f"  LR decay every:   {lr_decay_every} epochs")
        print(f"  LR decay factor:  {lr_decay_factor}")
        print(f"  Batch size:       {batch_size}")
        print()

        model = SimpleMLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            lr_decay_every=lr_decay_every,
            lr_decay_factor=lr_decay_factor,
            batch_size=batch_size,
            random_state=random_state
        )
        model.fit(X_train, y_train, X_val, y_val)
        train_losses = model.train_losses
        val_losses = model.val_losses
        stopped_epoch = model.stopped_epoch

    # ------------------------------------------------------------------ #
    # 4. Evaluate on test set
    # ------------------------------------------------------------------ #
    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    n_classes = len(np.unique(y))
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    if y_proba is not None:
        if n_classes == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    else:
        auc = float('nan')

    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'stopped_epoch': stopped_epoch
    }

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  F1 Score:      {f1:.4f}")
    print(f"  AUC:           {auc:.4f}")
    print(f"  Stopped Epoch: {stopped_epoch}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 5. Plot training/validation curves
    # ------------------------------------------------------------------ #
    _plot_curves(train_losses, val_losses, stopped_epoch,
                 model.learning_rates if not use_sklearn else None)

    return {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'stopped_epoch': stopped_epoch
    }


def _plot_curves(train_losses, val_losses, stopped_epoch, learning_rates=None):
    """Plot training/validation loss curves and optionally the LR schedule."""
    n_plots = 2 if learning_rates is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    ax = axes[0]
    ax.plot(epochs, train_losses, label='Training Loss', color='steelblue', linewidth=2)
    if len(val_losses) > 0:
        ax.plot(epochs, val_losses, label='Validation Loss', color='darkorange',
                linewidth=2, linestyle='--')
    ax.axvline(x=stopped_epoch, color='red', linestyle=':', linewidth=1.5,
               label=f'Early Stop (epoch {stopped_epoch})')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Learning rate schedule
    if learning_rates is not None:
        ax2 = axes[1]
        ax2.plot(epochs, learning_rates[:len(epochs)], color='green', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  [Plot] Training curves saved to 'training_curves.png'")


# ------------------------------------------------------------------ #
# Demo / self-test
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    from sklearn.datasets import make_classification

    print("Generating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    print(f"Dataset shape: X={X.shape}, y={y.shape}, classes={np.unique(y)}")

    results = train_neural_network(
        X, y,
        hidden_layer_sizes=(128, 64, 32),
        learning_rate=0.05,
        max_epochs=300,
        patience=10,
        lr_decay_every=20,
        lr_decay_factor=0.5,
        batch_size=64,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        use_sklearn=False
    )

    print("\nReturned keys:", list(results.keys()))
    print("Final metrics:", results['metrics'])