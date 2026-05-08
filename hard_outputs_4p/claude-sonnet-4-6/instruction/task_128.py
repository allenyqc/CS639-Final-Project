"""
Federated Learning Simulation Module

Simulates federated learning across multiple data partitions using
non-IID label-based partitioning, FedAvg aggregation, and convergence tracking.
"""

import numpy as np
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import copy

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ClientData:
    """Holds a single client's local dataset (already scaled)."""
    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    n_samples: int = field(init=False)

    def __post_init__(self):
        self.n_samples = len(self.y_train)


@dataclass
class GlobalModelParams:
    """Encapsulates logistic-regression parameters."""
    coef: np.ndarray          # shape (n_classes, n_features) or (1, n_features)
    intercept: np.ndarray     # shape (n_classes,) or (1,)


@dataclass
class ConvergenceHistory:
    """Tracks metrics across federated rounds."""
    rounds: list = field(default_factory=list)
    global_loss: list = field(default_factory=list)
    global_accuracy: list = field(default_factory=list)
    per_client_loss: dict = field(default_factory=lambda: defaultdict(list))


@dataclass
class ComparisonMetrics:
    """Comparison between federated and centralised training."""
    federated_final_loss: float
    federated_final_accuracy: float
    centralised_loss: float
    centralised_accuracy: float
    loss_gap: float           # centralised - federated  (negative = federated worse)
    accuracy_gap: float       # federated - centralised


# ---------------------------------------------------------------------------
# Non-IID Partitioning
# ---------------------------------------------------------------------------

def partition_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    n_classes_per_client: int = 2,
    random_state: int = 42,
) -> list[ClientData]:
    """
    Partition data across N clients in a non-IID fashion.

    Each client receives data predominantly from a subset of classes
    (Dirichlet-style label skew).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (already scaled by caller).
    y : np.ndarray
        Label vector.
    n_clients : int
        Number of simulated clients.
    n_classes_per_client : int
        How many distinct classes each client primarily holds.
    random_state : int
        RNG seed.

    Returns
    -------
    list[ClientData]
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    n_classes = len(classes)

    # Build per-class index lists
    class_indices: dict[int, list] = {c: [] for c in classes}
    for idx, label in enumerate(y):
        class_indices[int(label)].append(idx)
    for c in classes:
        rng.shuffle(class_indices[c])

    # Assign primary classes to each client (round-robin with overlap)
    client_primary_classes: list[list] = []
    for cid in range(n_clients):
        primary = [
            classes[(cid * n_classes_per_client + k) % n_classes]
            for k in range(n_classes_per_client)
        ]
        client_primary_classes.append(primary)

    # Distribute samples: 80 % from primary classes, 20 % from others
    client_indices: list[list] = [[] for _ in range(n_clients)]

    for c in classes:
        idxs = class_indices[c].copy()
        # Clients that have this as a primary class
        primary_clients = [
            cid for cid, pc in enumerate(client_primary_classes) if c in pc
        ]
        other_clients = [
            cid for cid in range(n_clients) if cid not in primary_clients
        ]

        n_primary = len(primary_clients)
        n_other = len(other_clients)

        if n_primary == 0:
            # Distribute evenly
            splits = np.array_split(idxs, n_clients)
            for cid, split in enumerate(splits):
                client_indices[cid].extend(split.tolist())
            continue

        # 80 % to primary clients, 20 % to others
        n_for_primary = int(0.8 * len(idxs))
        primary_idxs = idxs[:n_for_primary]
        other_idxs = idxs[n_for_primary:]

        primary_splits = np.array_split(primary_idxs, n_primary)
        for cid, split in zip(primary_clients, primary_splits):
            client_indices[cid].extend(split.tolist())

        if n_other > 0 and len(other_idxs) > 0:
            other_splits = np.array_split(other_idxs, n_other)
            for cid, split in zip(other_clients, other_splits):
                client_indices[cid].extend(split.tolist())

    # Build ClientData objects
    clients = []
    for cid in range(n_clients):
        idxs = np.array(client_indices[cid])
        if len(idxs) == 0:
            # Fallback: give a random slice
            idxs = rng.choice(len(y), max(1, len(y) // n_clients), replace=False)
        clients.append(
            ClientData(
                client_id=cid,
                X_train=X[idxs],
                y_train=y[idxs],
            )
        )

    return clients


# ---------------------------------------------------------------------------
# Local Training
# ---------------------------------------------------------------------------

def local_train(
    client: ClientData,
    global_params: Optional[GlobalModelParams],
    n_classes: int,
    local_epochs: int = 5,
    learning_rate: float = 0.01,
    random_state: int = 42,
) -> tuple[GlobalModelParams, float]:
    """
    Train a logistic regression model locally for E epochs using SGD-style
    warm-starting from the global model parameters.

    Parameters
    ----------
    client : ClientData
    global_params : GlobalModelParams or None
        Current global model parameters to initialise from.
    n_classes : int
        Total number of classes in the problem.
    local_epochs : int
        Number of local training epochs (passes over local data).
    learning_rate : float
        Step size for gradient updates.
    random_state : int

    Returns
    -------
    (GlobalModelParams, local_loss)
    """
    X, y = client.X_train, client.y_train
    n_features = X.shape[1]

    # Initialise model
    model = LogisticRegression(
        max_iter=local_epochs * 100,
        solver="saga",
        C=1.0,
        random_state=random_state,
        multi_class="auto",
        warm_start=True,
    )

    # Inject global parameters as warm start
    if global_params is not None:
        model.fit(X, y)  # fit once to initialise internal structure
        model.coef_ = global_params.coef.copy()
        model.intercept_ = global_params.intercept.copy()
        # Re-fit with warm start
        model.max_iter = local_epochs * 20
        model.fit(X, y)
    else:
        model.fit(X, y)

    # Compute local loss
    y_pred_proba = model.predict_proba(X)
    local_loss = log_loss(y, y_pred_proba)

    params = GlobalModelParams(
        coef=model.coef_.copy(),
        intercept=model.intercept_.copy(),
    )
    return params, local_loss


# ---------------------------------------------------------------------------
# FedAvg Aggregation
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    client_params: list[GlobalModelParams],
    client_weights: list[float],
) -> GlobalModelParams:
    """
    Weighted average of client model parameters (FedAvg).

    Parameters
    ----------
    client_params : list[GlobalModelParams]
    client_weights : list[float]
        Proportion of total data each client holds (sums to 1).

    Returns
    -------
    GlobalModelParams
    """
    total_weight = sum(client_weights)
    normalised = [w / total_weight for w in client_weights]

    agg_coef = sum(
        w * p.coef for w, p in zip(normalised, client_params)
    )
    agg_intercept = sum(
        w * p.intercept for w, p in zip(normalised, client_params)
    )

    return GlobalModelParams(coef=agg_coef, intercept=agg_intercept)


# ---------------------------------------------------------------------------
# Global Model Evaluation
# ---------------------------------------------------------------------------

def evaluate_global_model(
    global_params: GlobalModelParams,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, float]:
    """
    Evaluate the global model on the test set.

    Parameters
    ----------
    global_params : GlobalModelParams
    X_test : np.ndarray
    y_test : np.ndarray
    classes : np.ndarray

    Returns
    -------
    (loss, accuracy)
    """
    # Build a sklearn model shell to use predict_proba
    model = LogisticRegression()
    model.coef_ = global_params.coef
    model.intercept_ = global_params.intercept
    model.classes_ = classes

    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    loss = log_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    return loss, acc


# ---------------------------------------------------------------------------
# Centralised Baseline
# ---------------------------------------------------------------------------

def train_centralised_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Train logistic regression on all training data combined and evaluate
    on the held-out test set.

    Parameters
    ----------
    X_train, y_train : training data (already scaled)
    X_test, y_test   : test data (already scaled with train scaler)
    random_state : int

    Returns
    -------
    (loss, accuracy)
    """
    model = LogisticRegression(
        max_iter=1000,
        solver="saga",
        C=1.0,
        random_state=random_state,
        multi_class="auto",
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    loss = log_loss(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    return loss, acc


# ---------------------------------------------------------------------------
# Main Federated Learning Runner
# ---------------------------------------------------------------------------

def run_federated_learning(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = 5,
    n_rounds: int = 20,
    local_epochs: int = 5,
    n_classes_per_client: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[GlobalModelParams, ConvergenceHistory, ComparisonMetrics]:
    """
    Run federated learning simulation.

    Parameters
    ----------
    X : np.ndarray
        Raw feature matrix (unscaled).
    y : np.ndarray
        Label vector.
    n_clients : int
        Number of simulated clients.
    n_rounds : int
        Number of federated communication rounds (T).
    local_epochs : int
        Local training epochs per round (E).
    n_classes_per_client : int
        Classes per client for non-IID partitioning.
    test_size : float
        Fraction of data held out for global evaluation.
    random_state : int
        RNG seed for reproducibility.
    verbose : bool
        Print progress.

    Returns
    -------
    (global_model_params, convergence_history, comparison_metrics)
    """
    # ------------------------------------------------------------------
    # 1. Train/test split BEFORE any preprocessing
    # ------------------------------------------------------------------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ------------------------------------------------------------------
    # 2. Fit scaler ONLY on training data
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)   # transform only — no fit

    classes = np.unique(y)
    n_classes = len(classes)

    # ------------------------------------------------------------------
    # 3. Partition training data across clients (non-IID)
    # ------------------------------------------------------------------
    clients = partition_non_iid(
        X_train, y_train,
        n_clients=n_clients,
        n_classes_per_client=n_classes_per_client,
        random_state=random_state,
    )

    total_samples = sum(c.n_samples for c in clients)
    client_weights = [c.n_samples / total_samples for c in clients]

    if verbose:
        print("=" * 60)
        print("Federated Learning Simulation")
        print("=" * 60)
        print(f"  Clients          : {n_clients}")
        print(f"  Rounds           : {n_rounds}")
        print(f"  Local epochs     : {local_epochs}")
        print(f"  Classes          : {n_classes}")
        print(f"  Train samples    : {len(y_train)}")
        print(f"  Test samples     : {len(y_test)}")
        print()
        for c in clients:
            label_dist = {int(lbl): int(cnt)
                          for lbl, cnt in zip(*np.unique(c.y_train, return_counts=True))}
            print(f"  Client {c.client_id}: {c.n_samples} samples | labels: {label_dist}")
        print()

    # ------------------------------------------------------------------
    # 4. Federated training loop
    # ------------------------------------------------------------------
    history = ConvergenceHistory()
    global_params: Optional[GlobalModelParams] = None

    for round_idx in range(1, n_rounds + 1):
        # --- Local training on each client ---
        round_client_params = []
        round_client_losses = []

        for client in clients:
            params, local_loss = local_train(
                client=client,
                global_params=global_params,
                n_classes=n_classes,
                local_epochs=local_epochs,
                random_state=random_state,
            )
            round_client_params.append(params)
            round_client_losses.append(local_loss)
            history.per_client_loss[client.client_id].append(local_loss)

        # --- FedAvg aggregation ---
        global_params = fedavg_aggregate(round_client_params, client_weights)

        # --- Global evaluation on held-out test set ---
        g_loss, g_acc = evaluate_global_model(
            global_params, X_test, y_test, classes
        )

        history.rounds.append(round_idx)
        history.global_loss.append(g_loss)
        history.global_accuracy.append(g_acc)

        if verbose:
            avg_client_loss = np.mean(round_client_losses)
            print(
                f"  Round {round_idx:3d}/{n_rounds} | "
                f"Global Loss: {g_loss:.4f} | "
                f"Global Acc: {g_acc:.4f} | "
                f"Avg Client Loss: {avg_client_loss:.4f}"
            )

    # ------------------------------------------------------------------
    # 5. Centralised baseline (trained on same training data, evaluated
    #    on same test set — no data leakage)
    # ------------------------------------------------------------------
    c_loss, c_acc = train_centralised_baseline(
        X_train, y_train, X_test, y_test, random_state=random_state
    )

    fed_final_loss = history.global_loss[-1]
    fed_final_acc = history.global_accuracy[-1]

    comparison = ComparisonMetrics(
        federated_final_loss=fed_final_loss,
        federated_final_accuracy=fed_final_acc,
        centralised_loss=c_loss,
        centralised_accuracy=c_acc,
        loss_gap=c_loss - fed_final_loss,
        accuracy_gap=fed_final_acc - c_acc,
    )

    if verbose:
        print()
        print("=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"  Federated  — Loss: {fed_final_loss:.4f} | Accuracy: {fed_final_acc:.4f}")
        print(f"  Centralised— Loss: {c_loss:.4f}        | Accuracy: {c_acc:.4f}")
        print(f"  Loss gap (centralised - federated): {comparison.loss_gap:+.4f}")
        print(f"  Accuracy gap (federated - centralised): {comparison.accuracy_gap:+.4f}")
        print("=" * 60)

    return global_params, history, comparison


# ---------------------------------------------------------------------------
# Convenience: pretty-print convergence history
# ---------------------------------------------------------------------------

def print_convergence_history(history: ConvergenceHistory) -> None:
    """Print a formatted convergence table."""
    print(f"\n{'Round':>6} | {'Global Loss':>12} | {'Global Acc':>11}")
    print("-" * 36)
    for r, loss, acc in zip(history.rounds, history.global_loss, history.global_accuracy):
        print(f"{r:>6} | {loss:>12.4f} | {acc:>11.4f}")


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use breast-cancer dataset as a quick demo
    data = load_breast_cancer()
    X_demo, y_demo = data.data, data.target

    global_model, conv_history, comparison = run_federated_learning(
        X=X_demo,
        y=y_demo,
        n_clients=5,
        n_rounds=15,
        local_epochs=10,
        n_classes_per_client=1,   # strong non-IID
        test_size=0.2,
        random_state=42,
        verbose=True,
    )

    print_convergence_history(conv_history)

    # Demonstrate that the returned objects are usable
    print("\nFinal global model coef shape:", global_model.coef.shape)
    print("Convergence rounds recorded  :", len(conv_history.rounds))
    print("Per-client loss keys         :", list(conv_history.per_client_loss.keys()))