```python
"""
Federated Learning Simulation Module

Simulates federated learning across multiple data partitions with non-IID
label-based partitioning, FedAvg aggregation, and comparison against a
centralized baseline.
"""

from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClientData:
    """Holds a single client's training partition."""
    client_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    scaler: StandardScaler  # fitted on this client's data only


@dataclass
class RoundMetrics:
    """Metrics recorded after each federated round."""
    round_num: int
    global_loss: float
    global_accuracy: float
    global_f1: float
    global_auc: float
    per_client_loss: dict[int, float] = field(default_factory=dict)
    per_client_accuracy: dict[int, float] = field(default_factory=dict)


@dataclass
class FederatedResult:
    """Final output of the federated training run."""
    global_model: LogisticRegression
    global_scaler: StandardScaler
    convergence_history: list[RoundMetrics]
    centralized_metrics: dict[str, float]
    federated_final_metrics: dict[str, float]
    comparison: dict[str, Any]


# ---------------------------------------------------------------------------
# Non-IID partitioning
# ---------------------------------------------------------------------------

def partition_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    alpha: float = 0.5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Partition data across clients using a Dirichlet distribution to create
    non-IID label distributions (lower alpha → more heterogeneous).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (training data only).
    y : np.ndarray
        Labels (training data only).
    n_clients : int
        Number of simulated clients.
    alpha : float
        Dirichlet concentration parameter.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    List of (X_client, y_client) tuples, one per client.
    """
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    client_indices: list[list[int]] = [[] for _ in range(n_clients)]

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        # Sample proportions from Dirichlet distribution
        proportions = rng.dirichlet(alpha=np.full(n_clients, alpha))
        # Convert proportions to split points
        splits = (np.cumsum(proportions) * len(cls_indices)).astype(int)
        splits = np.clip(splits, 0, len(cls_indices))
        splits[-1] = len(cls_indices)  # ensure all samples are assigned
        prev = 0
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(cls_indices[prev:split].tolist())
            prev = split

    partitions = []
    for client_id in range(n_clients):
        idx = np.array(client_indices[client_id])
        if len(idx) == 0:
            logger.warning("Client %d received no samples.", client_id)
            partitions.append((np.empty((0, X.shape[1])), np.empty(0)))
        else:
            partitions.append((X[idx], y[idx]))

    return partitions


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def _init_logistic_model(
    n_classes: int,
    n_features: int,
    random_state: int = 42,
) -> LogisticRegression:
    """Create a LogisticRegression model with warm_start enabled."""
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto",
        warm_start=True,
        random_state=random_state,
        C=1.0,
    )


def _model_to_params(model: LogisticRegression) -> dict[str, np.ndarray]:
    """Extract coef_ and intercept_ from a fitted model."""
    return {
        "coef": model.coef_.copy(),
        "intercept": model.intercept_.copy(),
    }


def _params_to_model(
    params: dict[str, np.ndarray],
    template_model: LogisticRegression,
) -> LogisticRegression:
    """
    Create a new model with given parameters, copying structure from template.
    """
    new_model = copy.deepcopy(template_model)
    new_model.coef_ = params["coef"].copy()
    new_model.intercept_ = params["intercept"].copy()
    return new_model


def local_train(
    client: ClientData,
    global_params: dict[str, np.ndarray] | None,
    n_classes: int,
    local_epochs: int,
    random_state: int = 42,
) -> tuple[dict[str, np.ndarray], float]:
    """
    Train a logistic regression model on a single client's data.

    Parameters
    ----------
    client : ClientData
        Client's local dataset (already scaled).
    global_params : dict or None
        Global model parameters to initialise from (None for first round).
    n_classes : int
        Number of target classes.
    local_epochs : int
        Number of local training epochs (max_iter multiplier).
    random_state : int
        Random seed.

    Returns
    -------
    (updated_params, local_loss) tuple.
    """
    if len(client.X_train) == 0:
        logger.warning("Client %d has no data; skipping.", client.client_id)
        return global_params or {}, float("inf")

    model = _init_logistic_model(n_classes, client.X_train.shape[1], random_state)
    model.max_iter = local_epochs * 100  # approximate epoch simulation

    # Initialise from global parameters if available
    if global_params is not None:
        model.coef_ = global_params["coef"].copy()
        model.intercept_ = global_params["intercept"].copy()
        model.classes_ = np.arange(n_classes)

    try:
        model.fit(client.X_train, client.y_train)
    except ValueError as exc:
        logger.error("Client %d training failed: %s", client.client_id, exc)
        return global_params or {}, float("inf")

    # Compute local training loss
    try:
        proba = model.predict_proba(client.X_train)
        local_loss = log_loss(client.y_train, proba)
    except ValueError as exc:
        logger.warning("Could not compute local loss for client %d: %s",
                       client.client_id, exc)
        local_loss = float("inf")

    return _model_to_params(model), local_loss


# ---------------------------------------------------------------------------
# FedAvg aggregation
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    client_params: list[dict[str, np.ndarray]],
    client_sizes: list[int],
) -> dict[str, np.ndarray]:
    """
    Weighted average of model parameters (FedAvg).

    Parameters
    ----------
    client_params : list of parameter dicts
        Each dict has 'coef' and 'intercept' arrays.
    client_sizes : list of int
        Number of training samples per client (used as weights).

    Returns
    -------
    Aggregated parameter dict.
    """
    total = sum(client_sizes)
    if total == 0:
        raise ValueError("Total client data size is zero; cannot aggregate.")

    weights = [size / total for size in client_sizes]

    agg_coef = sum(w * p["coef"] for w, p in zip(weights, client_params))
    agg_intercept = sum(w * p["intercept"] for w, p in zip(weights, client_params))

    return {"coef": agg_coef, "intercept": agg_intercept}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str = "model",
) -> dict[str, float]:
    """
    Evaluate a model on the test set.

    Metrics: log-loss, accuracy, macro-F1, AUC (binary) or macro-OvR AUC.
    """
    X_scaled = scaler.transform(X_test)
    try:
        proba = model.predict_proba(X_scaled)
        loss = log_loss(y_test, proba)
    except ValueError as exc:
        logger.warning("%s: log_loss failed: %s", label, exc)
        loss = float("inf")
        proba = None

    preds = model.predict(X_scaled)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    auc = float("nan")
    if proba is not None:
        try:
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = roc_auc_score(
                    y_test, proba, multi_class="ovr", average="macro"
                )
        except ValueError as exc:
            logger.warning("%s: AUC computation failed: %s", label, exc)

    return {
        "loss": loss,
        "accuracy": acc,
        "f1_macro": f1,
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# Centralized baseline
# ---------------------------------------------------------------------------

def train_centralized_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Train a logistic regression on all training data combined and evaluate
    on the held-out test set.

    The scaler is fitted ONLY on X_train.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto",
        random_state=random_state,
        C=1.0,
    )
    model.fit(X_train_scaled, y_train)

    metrics = evaluate_model(model, scaler, X_test, y_test, label="centralized")
    logger.info("Centralized baseline — %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Main federated learning runner
# ---------------------------------------------------------------------------

def run_federated_learning(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int = 5,
    n_rounds: int = 20,
    local_epochs: int = 5,
    alpha: float = 0.5,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> FederatedResult:
    """
    Run a full federated learning simulation.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix (raw, unscaled).
    y : np.ndarray
        Target labels.
    n_clients : int
        Number of simulated federated clients.
    n_rounds : int
        Number of federated communication rounds (T).
    local_epochs : int
        Local training epochs per round (E).
    alpha : float
        Dirichlet non-IID concentration parameter.
    test_size : float
        Fraction of data reserved for the final test set.
    val_size : float
        Fraction of training data reserved for validation (not used for
        model selection here, but kept separate per best practices).
    random_state : int
        Global random seed.

    Returns
    -------
    FederatedResult dataclass.
    """
    # ------------------------------------------------------------------
    # 1. Split BEFORE any preprocessing (best practice #1)
    # ------------------------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Separate validation split (best practice #4)
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_trainval,
    )

    logger.info(
        "Data split — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    n_classes = len(np.unique(y))

    # ------------------------------------------------------------------
    # 2. Centralized baseline (scaler fitted on X_train only)
    # ------------------------------------------------------------------
    centralized_metrics = train_centralized_baseline(
        X_train, y_train, X_test, y_test, random_state=random_state
    )

    # ------------------------------------------------------------------
    # 3. Partition training data across clients (non-IID)
    # ------------------------------------------------------------------
    partitions = partition_non_iid(
        X_train, y_train,
        n_clients=n_clients,
        alpha=alpha,
        random_state=random_state,
    )

    # Build ClientData objects; each client fits its OWN scaler on its
    # local data only (best practice #1 — no leakage from other clients)
    clients: list[ClientData] = []
    for cid, (X_c, y_c) in enumerate(partitions):
        if len(X_c) == 0:
            logger.warning("Client %d has empty partition.", cid)
            scaler_c = StandardScaler()
            scaler_c.fit(X_train)  # fallback: fit on global train
            clients.append(ClientData(cid, X_c, y_c, scaler_c))
            continue
        scaler_c = StandardScaler()
        X_c_scaled = scaler_c.fit_transform(X_c)
        clients.append(ClientData(cid, X_c_scaled, y_c, scaler_c))
        logger.info(
            "Client %d — samples: %d, label dist: %s",
            cid, len(y_c),
            dict(zip(*np.unique(y_c, return_counts=True))),
        )

    # ------------------------------------------------------------------
    # 4. Global scaler for evaluation (fitted on X_train, best practice #1)
    # ------------------------------------------------------------------
    global_scaler = StandardScaler()
    global_scaler.fit(X_train)

    # ------------------------------------------------------------------
    # 5. Initialise global model parameters
    # ------------------------------------------------------------------
    # Fit a dummy model to get the right shapes
    _init_model = _init_logistic_model(n_classes, X_train.shape[1], random_state)
    _init_model.fit(global_scaler.transform(X_train), y_train)
    global_params = _model_to_params(_init_model)
    global_model = _init_model  # will be updated each round

    convergence_history: list[RoundMetrics] = []

    # ------------------------------------------------------------------
    # 6. Federated training rounds
    # ------------------------------------------------------------------
    for round_num in range(1, n_rounds + 1):
        logger.info("=== Round %d / %d ===", round_num, n_rounds)

        # --- Local training on each client ---
        all_params: list[dict[str, np.ndarray]] = []
        client_sizes: list[int] = []
        per_client_loss: dict[int, float] = {}
        per_client_acc: dict[int, float] = {}

        for client in clients:
            if len(client.X_train) == 0:
                continue

            updated_params, local_loss = local_train(
                client,
                global_params=global_params,
                n_classes=n_classes,
                local_epochs=local_epochs,
                random_state=random_state,
            )
            all_params.append(updated_params)
            client_sizes.append(len(client.X_train))
            per_client_loss[client.client_id] = local_loss

            # Per-client accuracy on local data
            tmp_model = _params_to_model(updated_params, global_model)
            tmp_model.classes_ = np.arange(n_classes)
            preds = tmp_model.predict(client.X_train)
            per_client_acc[client.client_id] = accuracy_score(
                client.y_train, preds
            )

        if not all_params:
            logger.error("No client updates received in round %d.", round_num)
            continue

        # --- FedAvg aggregation ---
        global_params = fedavg_aggregate(all_params, client_sizes)

        # --- Distribute global model ---
        global_model = _params_to_model(global_params, global_model)
        global_model.classes_ = np.arange(n_classes)

        # --- Evaluate on validation set (NOT test set) ---
        val_metrics = evaluate_model(
            global_model, global_scaler, X_val, y_val,
            label=f"round-{round_num}-val",
        )

        round_metrics = RoundMetrics(
            round_num=round_num,
            global_loss=val_metrics["loss"],
            global_accuracy=val_metrics["accuracy"],
            global_f1=val_metrics["f1_macro"],
            global_auc=val_metrics["auc"],
            per_client_loss=per_client_loss,
            per_client_accuracy=per_client_acc,
        )
        convergence_history.append(round_metrics)

        logger.info(
            "Round %d — val_loss: %.4f, val_acc: %.4f, val_f1: %.4f, val_auc: %.4f",
            round_num,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1_macro"],
            val_metrics["auc"],
        )

    # ------------------------------------------------------------------
    # 7. Final evaluation on held-out TEST set (best practice #3)
    # ------------------------------------------------------------------
    federated_final_metrics = evaluate_model(
        global_model, global_scaler, X_test, y_test, label="federated-final"
    )
    logger.info("Federated final test metrics — %s", federated_final_metrics)

    # ------------------------------------------------------------------
    # 8. Comparison
    # ------------------------------------------------------------------
    comparison: dict[str, Any] = {
        "centralized": centralized_metrics,
        "federated": federated_final_metrics,
        "delta_accuracy": (
            federated_final_metrics["accuracy"] - centralized_metrics["accuracy"]
        ),
        "delta_f1_macro": (
            federated_final_metrics["f1_macro"] - centralized_metrics["f1_macro"]
        ),
        "delta_auc": (
            federated_final_metrics["auc"] - centralized_metrics["auc"]