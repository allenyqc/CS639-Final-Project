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


# ─────────────────────────────────────────────
# Logistic Regression Implementation (from scratch)
# ─────────────────────────────────────────────

class LogisticRegressionModel:
    """Multinomial logistic regression with softmax."""

    def __init__(self, n_features: int, n_classes: int, learning_rate: float = 0.01,
                 l2_reg: float = 1e-4):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.l2_reg = l2_reg
        # Parameters: W shape (n_features, n_classes), b shape (n_classes,)
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 0.01, (n_features, n_classes))
        self.b = np.zeros(n_classes)

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_params(self, params: Dict[str, np.ndarray]):
        self.W = params["W"].copy()
        self.b = params["b"].copy()

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)  # numerical stability
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.softmax(X @ self.W + self.b)

    def cross_entropy_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.forward(X)
        n = len(y)
        log_probs = np.log(probs[np.arange(n), y] + 1e-12)
        loss = -log_probs.mean()
        # L2 regularization
        loss += 0.5 * self.l2_reg * np.sum(self.W ** 2)
        return float(loss)

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(y)
        probs = self.forward(X)
        # One-hot encode y
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), y] = 1.0
        delta = (probs - one_hot) / n
        dW = X.T @ delta + self.l2_reg * self.W
        db = delta.sum(axis=0)
        return dW, db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32):
        """One epoch of mini-batch SGD."""
        n = len(y)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            X_batch, y_batch = X[idx], y[idx]
            dW, db = self.compute_gradients(X_batch, y_batch)
            self.W -= self.lr * dW
            self.b -= self.lr * db


# ─────────────────────────────────────────────
# Data Partitioning (non-IID by label)
# ─────────────────────────────────────────────

def partition_data_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    n_shards_per_client: int = 2,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Non-IID partitioning: sort data by label, divide into shards,
    assign n_shards_per_client shards to each client.

    Returns list of (X_client, y_client) tuples.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(y))
    total_shards = n_clients * n_shards_per_client

    # Sort by label
    sorted_idx = np.argsort(y)
    X_sorted, y_sorted = X[sorted_idx], y[sorted_idx]

    # Split into shards
    shard_size = len(y) // total_shards
    shards_X = [X_sorted[i * shard_size:(i + 1) * shard_size] for i in range(total_shards)]
    shards_y = [y_sorted[i * shard_size:(i + 1) * shard_size] for i in range(total_shards)]

    # Randomly assign shards to clients
    shard_indices = rng.permutation(total_shards)
    client_data = []
    for c in range(n_clients):
        assigned = shard_indices[c * n_shards_per_client:(c + 1) * n_shards_per_client]
        X_c = np.concatenate([shards_X[s] for s in assigned], axis=0)
        y_c = np.concatenate([shards_y[s] for s in assigned], axis=0)
        client_data.append((X_c, y_c))

    return client_data


# ─────────────────────────────────────────────
# Local Training
# ─────────────────────────────────────────────

def local_train(
    model: LogisticRegressionModel,
    X_local: np.ndarray,
    y_local: np.ndarray,
    local_epochs: int,
    batch_size: int = 32
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Train a local copy of the model for E epochs.
    Returns updated parameters and final local loss.
    """
    for _ in range(local_epochs):
        model.train_epoch(X_local, y_local, batch_size=batch_size)
    local_loss = model.cross_entropy_loss(X_local, y_local)
    return model.get_params(), local_loss


# ─────────────────────────────────────────────
# FedAvg Aggregation
# ─────────────────────────────────────────────

def fedavg_aggregate(
    client_params: List[Dict[str, np.ndarray]],
    client_sizes: List[int]
) -> Dict[str, np.ndarray]:
    """
    Weighted average of client parameters (weighted by dataset size).
    """
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    aggregated = {}
    for key in client_params[0]:
        aggregated[key] = sum(w * p[key] for w, p in zip(weights, client_params))
    return aggregated


# ─────────────────────────────────────────────
# Centralized Baseline
# ─────────────────────────────────────────────

def train_centralized(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_features: int,
    n_classes: int,
    n_epochs: int = 50,
    learning_rate: float = 0.01,
    l2_reg: float = 1e-4,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Train a centralized model on all data combined."""
    model = LogisticRegressionModel(n_features, n_classes, learning_rate, l2_reg)
    history = {"loss": [], "accuracy": []}

    for epoch in range(n_epochs):
        model.train_epoch(X_train, y_train, batch_size=batch_size)
        loss = model.cross_entropy_loss(X_test, y_test)
        acc = model.accuracy(X_test, y_test)
        history["loss"].append(loss)
        history["accuracy"].append(acc)

    final_loss = model.cross_entropy_loss(X_test, y_test)
    final_acc = model.accuracy(X_test, y_test)

    return {
        "model": model,
        "history": history,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
    }


# ─────────────────────────────────────────────
# Federated Learning Runner
# ─────────────────────────────────────────────

def run_federated_learning(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_clients: int = 10,
    n_rounds: int = 20,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    l2_reg: float = 1e-4,
    batch_size: int = 32,
    n_shards_per_client: int = 2,
    client_fraction: float = 1.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run federated learning simulation.

    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,) — integer class labels
        X_test: Test features
        y_test: Test labels
        n_clients: Number of federated clients
        n_rounds: Number of communication rounds
        local_epochs: Local training epochs per round
        learning_rate: SGD learning rate
        l2_reg: L2 regularization coefficient
        batch_size: Mini-batch size for local training
        n_shards_per_client: Number of label shards per client (controls non-IID degree)
        client_fraction: Fraction of clients participating each round (C in FedAvg)
        seed: Random seed

    Returns:
        Dictionary with global model, convergence history, and comparison metrics.
    """
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    print(f"{'='*60}")
    print(f"Federated Learning Simulation")
    print(f"  Clients: {n_clients}, Rounds: {n_rounds}, Local Epochs: {local_epochs}")
    print(f"  Features: {n_features}, Classes: {n_classes}")
    print(f"  Train size: {len(y)}, Test size: {len(y_test)}")
    print(f"  Client fraction: {client_fraction}")
    print(f"{'='*60}")

    # ── 1. Partition data across clients (non-IID) ──
    client_data = partition_data_non_iid(X, y, n_clients, n_shards_per_client, seed)
    client_sizes = [len(d[1]) for d in client_data]

    print("\nClient data distribution (label counts):")
    for i, (_, y_c) in enumerate(client_data):
        unique, counts = np.unique(y_c, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  Client {i:2d}: {len(y_c):4d} samples | labels: {dist}")

    # ── 2. Initialize global model ──
    global_model = LogisticRegressionModel(n_features, n_classes, learning_rate, l2_reg)
    global_params = global_model.get_params()

    # ── 3. Convergence tracking ──
    convergence_history = {
        "round": [],
        "global_loss": [],
        "global_accuracy": [],
        "per_client_loss": defaultdict(list),   # client_id -> [loss per round]
        "n_participating_clients": [],
    }

    print(f"\n{'─'*60}")
    print(f"{'Round':>6} | {'Global Loss':>12} | {'Global Acc':>10} | {'Clients':>8}")
    print(f"{'─'*60}")

    # ── 4. Federated Training Rounds ──
    for round_idx in range(1, n_rounds + 1):
        # Select participating clients
        n_participating = max(1, int(client_fraction * n_clients))
        participating_ids = rng.choice(n_clients, size=n_participating, replace=False).tolist()

        round_params = []
        round_sizes = []
        round_client_losses = {}

        for cid in participating_ids:
            X_c, y_c = client_data[cid]

            # Create local model copy with current global params
            local_model = LogisticRegressionModel(n_features, n_classes, learning_rate, l2_reg)
            local_model.set_params(global_params)

            # Local training
            updated_params, local_loss = local_train(
                local_model, X_c, y_c, local_epochs, batch_size
            )
            round_params.append(updated_params)
            round_sizes.append(len(y_c))
            round_client_losses[cid] = local_loss

        # FedAvg aggregation
        global_params = fedavg_aggregate(round_params, round_sizes)
        global_model.set_params(global_params)

        # Evaluate on test set
        global_loss = global_model.cross_entropy_loss(X_test, y_test)
        global_acc = global_model.accuracy(X_test, y_test)

        # Record history
        convergence_history["round"].append(round_idx)
        convergence_history["global_loss"].append(global_loss)
        convergence_history["global_accuracy"].append(global_acc)
        convergence_history["n_participating_clients"].append(n_participating)

        # Record per-client losses (for all clients, not just participating)
        for cid in range(n_clients):
            if cid in round_client_losses:
                convergence_history["per_client_loss"][cid].append(round_client_losses[cid])
            else:
                # Non-participating client: record NaN
                convergence_history["per_client_loss"][cid].append(float("nan"))

        print(f"{round_idx:>6} | {global_loss:>12.4f} | {global_acc:>10.4f} | {n_participating:>8}")

    print(f"{'─'*60}")

    # ── 5. Centralized Baseline ──
    print(f"\n{'='*60}")
    print("Training Centralized Baseline...")
    # Use same number of total gradient steps for fair comparison
    centralized_epochs = n_rounds * local_epochs
    centralized_result = train_centralized(
        X, y, X_test, y_test,
        n_features, n_classes,
        n_epochs=centralized_epochs,
        learning_rate=learning_rate,
        l2_reg=l2_reg,
        batch_size=batch_size
    )
    print(f"Centralized Final Loss:     {centralized_result['final_loss']:.4f}")
    print(f"Centralized Final Accuracy: {centralized_result['final_accuracy']:.4f}")

    # ── 6. Comparison Metrics ──
    fed_final_loss = convergence_history["global_loss"][-1]
    fed_final_acc = convergence_history["global_accuracy"][-1]

    comparison = {
        "federated": {
            "final_loss": fed_final_loss,
            "final_accuracy": fed_final_acc,
            "best_accuracy": max(convergence_history["global_accuracy"]),
            "best_round": int(np.argmax(convergence_history["global_accuracy"])) + 1,
        },
        "centralized": {
            "final_loss": centralized_result["final_loss"],
            "final_accuracy": centralized_result["final_accuracy"],
            "best_accuracy": max(centralized_result["history"]["accuracy"]),
        },
        "accuracy_gap": centralized_result["final_accuracy"] - fed_final_acc,
        "loss_gap": fed_final_loss - centralized_result["final_loss"],
    }

    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'─'*60}")
    print(f"  Federated  — Loss: {fed_final_loss:.4f} | Accuracy: {fed_final_acc:.4f}")
    print(f"  Centralized— Loss: {centralized_result['final_loss']:.4f} | "
          f"Accuracy: {centralized_result['final_accuracy']:.4f}")
    print(f"  Accuracy Gap (Centralized - Federated): {comparison['accuracy_gap']:+.4f}")
    print(f"{'='*60}\n")

    return {
        "global_model": global_model,
        "convergence_history": dict(convergence_history),
        "comparison": comparison,
        "centralized_model": centralized_result["model"],
        "centralized_history": centralized_result["history"],
        "client_data_sizes": client_sizes,
        "n_clients": n_clients,
        "n_rounds": n_rounds,
        "local_epochs": local_epochs,
    }


# ─────────────────────────────────────────────
# Utility: Print Convergence Summary
# ─────────────────────────────────────────────

def print_convergence_summary(results: Dict[str, Any]):
    """Print a formatted convergence summary."""
    history = results["convergence_history"]
    print("\nConvergence History (every 5 rounds):")
    print(f"{'Round':>6} | {'Global Loss':>12} | {'Global Acc':>10}")
    print("─" * 36)
    for i, (r, loss, acc) in enumerate(zip(
        history["round"],
        history["global_loss"],
        history["global_accuracy"]
    )):
        if i % 5 == 0 or i == len(history["round"]) - 1:
            print(f"{r:>6} | {loss:>12.4f} | {acc:>10.4f}")

    print("\nPer-Client Average Loss (participating rounds only):")
    for cid, losses in history["per_client_loss"].items():
        valid = [l for l in losses if not np.isnan(l)]
        if valid:
            print(f"  Client {cid:2d}: mean={np.mean(valid):.4f}, "
                  f"std={np.std(valid):.4f}, rounds={len(valid)}")


# ─────────────────────────────────────────────
# Demo / Main
# ─────────────────────────────────────────────

def generate_synthetic_dataset(
    n_samples: int = 2000,
    n_features: int = 20,
    n_classes: int = 5,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    # Create class-specific means to make it learnable
    class_means = rng.standard_normal((n_classes, n_features)) * 2
    y = rng.integers(0, n_classes, n_samples)
    X += class_means[y]

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis