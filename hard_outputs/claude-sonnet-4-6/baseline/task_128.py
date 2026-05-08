```python
"""
Federated Learning Simulation Module with FedAvg
Simulates federated learning across multiple data partitions with non-IID data distribution.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Data Utilities
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax function."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    n = len(y)
    oh = np.zeros((n, num_classes))
    oh[np.arange(n), y.astype(int)] = 1
    return oh


# ---------------------------------------------------------------------------
# Logistic Regression Model (supports multi-class via softmax)
# ---------------------------------------------------------------------------

class LogisticRegressionModel:
    """
    Multi-class logistic regression model.
    Parameters stored as a flat numpy array for easy averaging.
    """

    def __init__(self, n_features: int, n_classes: int):
        self.n_features = n_features
        self.n_classes = n_classes
        # Weight matrix: (n_features, n_classes), bias: (n_classes,)
        scale = np.sqrt(2.0 / n_features)
        self.W = np.random.randn(n_features, n_classes) * scale
        self.b = np.zeros(n_classes)

    # ------------------------------------------------------------------
    # Parameter serialisation helpers
    # ------------------------------------------------------------------

    def get_params(self) -> np.ndarray:
        """Return a flat copy of all parameters."""
        return np.concatenate([self.W.ravel(), self.b.ravel()])

    def set_params(self, params: np.ndarray) -> None:
        """Set parameters from a flat array."""
        w_size = self.n_features * self.n_classes
        self.W = params[:w_size].reshape(self.n_features, self.n_classes)
        self.b = params[w_size:].copy()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b          # (n, n_classes)
        if self.n_classes == 2:
            # Binary: use sigmoid on the second column
            p1 = sigmoid(logits[:, 1] - logits[:, 0])
            return np.column_stack([1 - p1, p1])
        return softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------------
    # Loss & gradient
    # ------------------------------------------------------------------

    def cross_entropy_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        proba = self.predict_proba(X)
        n = len(y)
        eps = 1e-12
        log_p = np.log(proba[np.arange(n), y.astype(int)] + eps)
        return -np.mean(log_p)

    def compute_gradients(self, X: np.ndarray, y: np.ndarray,
                          l2_lambda: float = 1e-4
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Return gradients (dW, db) with optional L2 regularisation."""
        n = len(y)
        proba = self.predict_proba(X)          # (n, n_classes)
        Y_oh = one_hot(y, self.n_classes)      # (n, n_classes)
        delta = (proba - Y_oh) / n             # (n, n_classes)
        dW = X.T @ delta + l2_lambda * self.W  # (n_features, n_classes)
        db = delta.sum(axis=0)                 # (n_classes,)
        return dW, db

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, X: np.ndarray, y: np.ndarray,
                   lr: float = 0.01, l2_lambda: float = 1e-4) -> float:
        dW, db = self.compute_gradients(X, y, l2_lambda)
        self.W -= lr * dW
        self.b -= lr * db
        return self.cross_entropy_loss(X, y)


# ---------------------------------------------------------------------------
# Non-IID Data Partitioning
# ---------------------------------------------------------------------------

def partition_non_iid(X: np.ndarray, y: np.ndarray,
                      n_clients: int,
                      n_classes_per_client: int = 2,
                      seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data across clients in a non-IID fashion.

    Each client receives data predominantly from a subset of classes
    (Dirichlet-style shard approach).

    Parameters
    ----------
    X : feature matrix
    y : integer labels
    n_clients : number of clients
    n_classes_per_client : how many distinct classes each client primarily holds
    seed : random seed

    Returns
    -------
    List of (X_client, y_client) tuples, one per client.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    n_classes = len(classes)

    # Sort samples by label
    sorted_indices = np.argsort(y)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]

    # Split into shards (2 shards per client)
    n_shards = n_clients * n_classes_per_client
    shard_size = len(X) // n_shards

    shards_X = [X_sorted[i * shard_size:(i + 1) * shard_size]
                for i in range(n_shards)]
    shards_y = [y_sorted[i * shard_size:(i + 1) * shard_size]
                for i in range(n_shards)]

    # Shuffle shard indices and assign to clients
    shard_indices = list(range(n_shards))
    rng.shuffle(shard_indices)

    partitions = []
    for c in range(n_clients):
        assigned = shard_indices[c * n_classes_per_client:
                                 (c + 1) * n_classes_per_client]
        Xc = np.concatenate([shards_X[s] for s in assigned], axis=0)
        yc = np.concatenate([shards_y[s] for s in assigned], axis=0)
        partitions.append((Xc, yc))

    return partitions


# ---------------------------------------------------------------------------
# Local Training
# ---------------------------------------------------------------------------

def local_train(model: LogisticRegressionModel,
                X: np.ndarray,
                y: np.ndarray,
                local_epochs: int = 5,
                batch_size: int = 32,
                lr: float = 0.01,
                l2_lambda: float = 1e-4,
                seed: int = 0) -> Tuple[np.ndarray, float]:
    """
    Train a local copy of the model on client data for E epochs.

    Returns
    -------
    updated_params : flat parameter array after local training
    final_loss     : cross-entropy loss on local data after training
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    for epoch in range(local_epochs):
        perm = rng.permutation(n)
        X_shuf = X[perm]
        y_shuf = y[perm]

        for start in range(0, n, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            model.train_step(Xb, yb, lr=lr, l2_lambda=l2_lambda)

    final_loss = model.cross_entropy_loss(X, y)
    return model.get_params(), final_loss


# ---------------------------------------------------------------------------
# FedAvg Server
# ---------------------------------------------------------------------------

def fedavg_aggregate(client_params: List[np.ndarray],
                     client_weights: Optional[List[float]] = None
                     ) -> np.ndarray:
    """
    Weighted average of client parameters (FedAvg).

    Parameters
    ----------
    client_params  : list of flat parameter arrays from each client
    client_weights : relative weight for each client (e.g. dataset size).
                     If None, uniform averaging is used.

    Returns
    -------
    Aggregated parameter array.
    """
    if client_weights is None:
        client_weights = [1.0] * len(client_params)

    total_weight = sum(client_weights)
    aggregated = np.zeros_like(client_params[0])
    for params, w in zip(client_params, client_weights):
        aggregated += (w / total_weight) * params
    return aggregated


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def evaluate_model(model: LogisticRegressionModel,
                   X: np.ndarray,
                   y: np.ndarray) -> Tuple[float, float]:
    """Return (loss, accuracy) on a dataset."""
    loss = model.cross_entropy_loss(X, y)
    preds = model.predict(X)
    acc = np.mean(preds == y)
    return loss, acc


# ---------------------------------------------------------------------------
# Centralized Baseline
# ---------------------------------------------------------------------------

def train_centralized(X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      n_classes: int,
                      n_epochs: int = 50,
                      batch_size: int = 64,
                      lr: float = 0.01,
                      l2_lambda: float = 1e-4,
                      seed: int = 42) -> Dict[str, Any]:
    """
    Train a logistic regression model on all data combined (centralized baseline).

    Returns
    -------
    dict with keys: model, train_loss_history, test_loss_history,
                    test_acc_history, final_test_loss, final_test_acc
    """
    rng = np.random.default_rng(seed)
    n_features = X_train.shape[1]
    model = LogisticRegressionModel(n_features, n_classes)

    train_loss_hist, test_loss_hist, test_acc_hist = [], [], []
    n = len(y_train)

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        for start in range(0, n, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            model.train_step(Xb, yb, lr=lr, l2_lambda=l2_lambda)

        tr_loss = model.cross_entropy_loss(X_train, y_train)
        te_loss, te_acc = evaluate_model(model, X_test, y_test)
        train_loss_hist.append(tr_loss)
        test_loss_hist.append(te_loss)
        test_acc_hist.append(te_acc)

    return {
        "model": model,
        "train_loss_history": train_loss_hist,
        "test_loss_history": test_loss_hist,
        "test_acc_history": test_acc_hist,
        "final_test_loss": test_loss_hist[-1],
        "final_test_acc": test_acc_hist[-1],
    }


# ---------------------------------------------------------------------------
# Main Federated Learning Runner
# ---------------------------------------------------------------------------

def run_federated_learning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_clients: int = 10,
    n_rounds: int = 20,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.01,
    l2_lambda: float = 1e-4,
    n_classes_per_client: int = 2,
    client_fraction: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run federated learning simulation.

    Parameters
    ----------
    X_train, y_train : training data (all clients combined before partitioning)
    X_test, y_test   : held-out test set for global evaluation
    n_clients        : number of simulated clients
    n_rounds         : number of federated rounds (T)
    local_epochs     : local training epochs per round (E)
    batch_size       : mini-batch size for local SGD
    lr               : learning rate
    l2_lambda        : L2 regularisation coefficient
    n_classes_per_client : non-IID degree (shards per client)
    client_fraction  : fraction of clients sampled each round (1.0 = all)
    seed             : global random seed
    verbose          : print progress

    Returns
    -------
    dict with keys:
        global_model          : final LogisticRegressionModel
        convergence_history   : dict with per-round metrics
        centralized_baseline  : dict with centralized training results
        comparison_metrics    : dict comparing federated vs centralized
        client_partitions     : list of (X, y) per client
    """
    rng = np.random.default_rng(seed)
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # ------------------------------------------------------------------ #
    # 1. Partition data across clients (non-IID)
    # ------------------------------------------------------------------ #
    if verbose:
        print("=" * 60)
        print("Partitioning data across clients (non-IID)...")
    client_partitions = partition_non_iid(
        X_train, y_train,
        n_clients=n_clients,
        n_classes_per_client=n_classes_per_client,
        seed=seed,
    )
    client_sizes = [len(p[1]) for p in client_partitions]

    if verbose:
        print(f"  Clients: {n_clients}, Classes: {n_classes}, "
              f"Features: {n_features}")
        for i, (Xc, yc) in enumerate(client_partitions):
            unique, counts = np.unique(yc, return_counts=True)
            dist = dict(zip(unique.tolist(), counts.tolist()))
            print(f"  Client {i:2d}: {len(yc):5d} samples | label dist: {dist}")

    # ------------------------------------------------------------------ #
    # 2. Initialise global model
    # ------------------------------------------------------------------ #
    global_model = LogisticRegressionModel(n_features, n_classes)
    global_params = global_model.get_params()

    # ------------------------------------------------------------------ #
    # 3. Convergence tracking
    # ------------------------------------------------------------------ #
    history: Dict[str, Any] = {
        "round": [],
        "global_loss": [],
        "global_accuracy": [],
        "per_client_loss": defaultdict(list),   # client_id -> [loss per round]
        "n_clients_sampled": [],
    }

    # ------------------------------------------------------------------ #
    # 4. Federated rounds
    # ------------------------------------------------------------------ #
    if verbose:
        print("\nStarting Federated Training...")
        print("-" * 60)

    for t in range(1, n_rounds + 1):
        # Sample a subset of clients
        n_sampled = max(1, int(client_fraction * n_clients))
        sampled_ids = rng.choice(n_clients, size=n_sampled, replace=False)

        client_updated_params = []
        sampled_weights = []
        round_client_losses = {}

        for cid in sampled_ids:
            Xc, yc = client_partitions[cid]

            # Create a local copy of the global model
            local_model = LogisticRegressionModel(n_features, n_classes)
            local_model.set_params(global_params.copy())

            # Local training
            updated_params, local_loss = local_train(
                local_model, Xc, yc,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                l2_lambda=l2_lambda,
                seed=int(rng.integers(0, 1_000_000)),
            )
            client_updated_params.append(updated_params)
            sampled_weights.append(client_sizes[cid])
            round_client_losses[cid] = local_loss

        # FedAvg aggregation
        global_params = fedavg_aggregate(client_updated_params, sampled_weights)
        global_model.set_params(global_params)

        # Evaluate global model on test set
        g_loss, g_acc = evaluate_model(global_model, X_test, y_test)

        # Record history
        history["round"].append(t)
        history["global_loss"].append(g_loss)
        history["global_accuracy"].append(g_acc)
        history["n_clients_sampled"].append(n_sampled)
        for cid, loss in round_client_losses.items():
            history["per_client_loss"][int(cid)].append(loss)

        if verbose:
            avg_client_loss = np.mean(list(round_client_losses.values()))
            print(f"  Round {t:3d}/{n_rounds} | "
                  f"Global Loss: {g_loss:.4f} | "
                  f"Global Acc: {g_acc:.4f} | "
                  f"Avg Client Loss: {avg_client_loss:.4f} | "
                  f"Clients sampled: {n_sampled}/{n_clients}")

    # ------------------------------------------------------------------ #
    # 5. Centralized baseline
    # ------------------------------------------------------------------ #
    if verbose:
        print("\nTraining Centralized Baseline...")
        print("-" * 60)

    # Total epochs ≈ rounds × local_epochs for fair comparison
    central_epochs = n_rounds * local_epochs
    centralized = train_centralized(
        X_train, y_train, X_test, y_test,
        n_classes=n_classes,
        n_epochs=central_epochs,
        batch_size=batch_size,
        lr=lr,
        l2_lambda=l2_lambda,
        seed=seed,
    )

    if verbose:
        print(f"  Centralized Final Test Loss: "
              f"{centralized['final_test_loss']:.4f}")
        print(f"  Centralized Final Test Acc : "
              f"{centralized['final_test_acc']:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Comparison metrics
    # ------------------------------------------------------------------ #
    fed_final_loss = history["global_loss"][-1]
    fed_final_acc = history["global_accuracy"][-1]
    cen_final_loss = centralized["final_test_loss"]
    cen_final_acc = centralized["final_test_acc"]

    comparison_metrics = {
        "federated_final_test_loss": fed_final_loss,
        "federated_final_test_acc": fed_final_acc,
        "centralized_final_test_loss": cen_final_loss,
        "centralized_final_test_acc":