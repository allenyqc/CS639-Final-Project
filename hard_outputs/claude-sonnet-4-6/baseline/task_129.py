```python
"""
ML Experiment Tracking System
A comprehensive module for tracking, comparing, and managing ML experiments.
"""

import json
import os
import uuid
import time
import fcntl
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
import threading
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentLock:
    """File-based lock for concurrent access control."""

    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self._lock_file = None

    def __enter__(self):
        self._lock_file = open(self.lock_path, "w")
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lock_file:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            try:
                os.remove(self.lock_path)
            except OSError:
                pass


class ExperimentTracker:
    """
    A comprehensive ML experiment tracker that supports:
    - Hyperparameter logging
    - Metric tracking per epoch/step
    - Artifact management
    - Experiment comparison
    - Tagging and filtering
    - Auto-save to JSON
    - Resume from previous experiments
    - Context manager interface
    - Concurrent access handling
    """

    METADATA_FILE = "metadata.json"
    METRICS_FILE = "metrics.json"
    HYPERPARAMS_FILE = "hyperparams.json"
    ARTIFACTS_FILE = "artifacts.json"
    LOCK_FILE = ".lock"

    def __init__(
        self,
        experiment_name: str,
        storage_dir: str,
        experiment_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
    ):
        """
        Initialize the ExperimentTracker.

        Args:
            experiment_name: Human-readable name for the experiment.
            storage_dir: Directory where experiment data will be stored.
            experiment_id: Optional ID to resume an existing experiment.
            tags: Optional list of tags for categorization.
            description: Optional description of the experiment.
        """
        self.experiment_name = experiment_name
        self.storage_dir = Path(storage_dir)
        self.description = description
        self.tags = tags or []
        self._thread_lock = threading.Lock()
        self._active = False

        # Create storage directory
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        if experiment_id:
            self._resume_experiment(experiment_id)
        else:
            self.experiment_id = str(uuid.uuid4())[:8]
            self._initialize_new_experiment()

        self.experiment_dir = self.storage_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = str(self.experiment_dir / self.LOCK_FILE)

    def _initialize_new_experiment(self):
        """Initialize data structures for a new experiment."""
        self._metadata = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": self.tags,
            "status": "initialized",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "started_at": None,
            "ended_at": None,
        }
        self._hyperparams: Dict[str, Any] = {}
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._artifacts: List[Dict[str, Any]] = []

    def _resume_experiment(self, experiment_id: str):
        """
        Resume tracking from a previous experiment.

        Args:
            experiment_id: ID of the experiment to resume.
        """
        exp_dir = self.storage_dir / experiment_id
        if not exp_dir.exists():
            raise FileNotFoundError(
                f"Experiment {experiment_id} not found in {self.storage_dir}"
            )

        self.experiment_id = experiment_id
        self.experiment_dir = exp_dir

        # Load existing data
        self._metadata = self._load_json(exp_dir / self.METADATA_FILE)
        self._hyperparams = self._load_json(exp_dir / self.HYPERPARAMS_FILE)
        self._metrics = self._load_json(exp_dir / self.METRICS_FILE)
        self._artifacts = self._load_json(exp_dir / self.ARTIFACTS_FILE)

        # Update metadata for resume
        self._metadata["status"] = "resumed"
        self._metadata["updated_at"] = datetime.now().isoformat()
        self._metadata["experiment_name"] = self.experiment_name
        if self.tags:
            self._metadata["tags"] = list(
                set(self._metadata.get("tags", []) + self.tags)
            )

        logger.info(f"Resumed experiment {experiment_id}")

    def _load_json(self, path: Path) -> Any:
        """Load JSON data from a file, returning empty dict/list if not found."""
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        # Determine default based on filename
        if path.name in (self.ARTIFACTS_FILE,):
            return []
        return {}

    def _save_json(self, data: Any, path: Path):
        """Save data to a JSON file with atomic write."""
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, path)

    def _auto_save(self):
        """Save all tracked data to JSON files."""
        with ExperimentLock(self.lock_path):
            self._metadata["updated_at"] = datetime.now().isoformat()
            self._save_json(self._metadata, self.experiment_dir / self.METADATA_FILE)
            self._save_json(
                self._hyperparams, self.experiment_dir / self.HYPERPARAMS_FILE
            )
            self._save_json(self._metrics, self.experiment_dir / self.METRICS_FILE)
            self._save_json(self._artifacts, self.experiment_dir / self.ARTIFACTS_FILE)

    def start(self):
        """Start the experiment tracking session."""
        with self._thread_lock:
            if self._active:
                logger.warning("Experiment is already active.")
                return self

            self._active = True
            self._metadata["status"] = "running"
            self._metadata["started_at"] = datetime.now().isoformat()
            self._auto_save()
            logger.info(
                f"Started experiment '{self.experiment_name}' (ID: {self.experiment_id})"
            )
        return self

    def stop(self, status: str = "completed"):
        """
        Stop the experiment tracking session.

        Args:
            status: Final status ('completed', 'failed', 'stopped').
        """
        with self._thread_lock:
            if not self._active:
                logger.warning("Experiment is not active.")
                return

            self._active = False
            self._metadata["status"] = status
            self._metadata["ended_at"] = datetime.now().isoformat()
            self._auto_save()
            logger.info(
                f"Stopped experiment '{self.experiment_name}' with status: {status}"
            )

    def log_hyperparams(self, params: Dict[str, Any]):
        """
        Log hyperparameters for the experiment.

        Args:
            params: Dictionary of hyperparameter name-value pairs.
        """
        with self._thread_lock:
            self._hyperparams.update(params)
            self._auto_save()
            logger.debug(f"Logged hyperparams: {list(params.keys())}")

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        timestamp: Optional[str] = None,
    ):
        """
        Log a single metric value.

        Args:
            name: Metric name (e.g., 'loss', 'accuracy').
            value: Numeric value of the metric.
            step: Optional training step number.
            epoch: Optional epoch number.
            timestamp: Optional ISO timestamp string.
        """
        with self._thread_lock:
            if name not in self._metrics:
                self._metrics[name] = []

            entry = {
                "value": value,
                "step": step,
                "epoch": epoch,
                "timestamp": timestamp or datetime.now().isoformat(),
            }
            self._metrics[name].append(entry)
            self._auto_save()
            logger.debug(f"Logged metric '{name}': {value}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name-value pairs.
            step: Optional training step number.
            epoch: Optional epoch number.
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step=step, epoch=epoch)

    def log_artifact(
        self,
        file_path: str,
        artifact_type: str = "file",
        description: str = "",
        copy_to_storage: bool = False,
    ):
        """
        Log an artifact (model file, plot, etc.).

        Args:
            file_path: Path to the artifact file.
            artifact_type: Type of artifact ('model', 'plot', 'data', 'file').
            description: Optional description of the artifact.
            copy_to_storage: If True, copy the file to the experiment directory.
        """
        with self._thread_lock:
            file_path = Path(file_path)
            stored_path = str(file_path)

            if copy_to_storage and file_path.exists():
                artifacts_dir = self.experiment_dir / "artifacts"
                artifacts_dir.mkdir(exist_ok=True)
                dest = artifacts_dir / file_path.name
                shutil.copy2(file_path, dest)
                stored_path = str(dest)

            artifact_entry = {
                "original_path": str(file_path),
                "stored_path": stored_path,
                "artifact_type": artifact_type,
                "description": description,
                "filename": file_path.name,
                "exists": file_path.exists(),
                "logged_at": datetime.now().isoformat(),
            }
            self._artifacts.append(artifact_entry)
            self._auto_save()
            logger.debug(f"Logged artifact: {file_path.name}")

    def add_tags(self, tags: List[str]):
        """
        Add tags to the experiment.

        Args:
            tags: List of tag strings.
        """
        with self._thread_lock:
            existing = set(self._metadata.get("tags", []))
            existing.update(tags)
            self._metadata["tags"] = list(existing)
            self.tags = self._metadata["tags"]
            self._auto_save()

    def remove_tags(self, tags: List[str]):
        """
        Remove tags from the experiment.

        Args:
            tags: List of tag strings to remove.
        """
        with self._thread_lock:
            existing = set(self._metadata.get("tags", []))
            existing -= set(tags)
            self._metadata["tags"] = list(existing)
            self.tags = self._metadata["tags"]
            self._auto_save()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current experiment.

        Returns:
            Dictionary containing experiment summary.
        """
        final_metrics = {}
        for metric_name, values in self._metrics.items():
            if values:
                final_metrics[metric_name] = values[-1]["value"]

        best_metrics = {}
        for metric_name, values in self._metrics.items():
            if values:
                metric_values = [v["value"] for v in values]
                best_metrics[f"{metric_name}_best"] = max(metric_values)
                best_metrics[f"{metric_name}_min"] = min(metric_values)

        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "status": self._metadata.get("status"),
            "tags": self._metadata.get("tags", []),
            "created_at": self._metadata.get("created_at"),
            "started_at": self._metadata.get("started_at"),
            "ended_at": self._metadata.get("ended_at"),
            "hyperparams": copy.deepcopy(self._hyperparams),
            "final_metrics": final_metrics,
            "best_metrics": best_metrics,
            "num_artifacts": len(self._artifacts),
            "metric_history_lengths": {k: len(v) for k, v in self._metrics.items()},
        }

    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """
        Get the full history of a metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            List of metric entries with value, step, epoch, timestamp.
        """
        return copy.deepcopy(self._metrics.get(metric_name, []))

    # -------------------------------------------------------------------------
    # Context Manager Interface
    # -------------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.stop(status="failed")
            logger.error(f"Experiment failed with error: {exc_val}")
        else:
            self.stop(status="completed")
        return False  # Don't suppress exceptions

    # -------------------------------------------------------------------------
    # Class Methods for Loading and Comparing Experiments
    # -------------------------------------------------------------------------

    @classmethod
    def load_experiment(
        cls, experiment_id: str, storage_dir: str
    ) -> "ExperimentTracker":
        """
        Load an existing experiment by ID.

        Args:
            experiment_id: ID of the experiment to load.
            storage_dir: Directory where experiments are stored.

        Returns:
            ExperimentTracker instance with loaded data.
        """
        storage_path = Path(storage_dir)
        exp_dir = storage_path / experiment_id

        if not exp_dir.exists():
            raise FileNotFoundError(
                f"Experiment {experiment_id} not found in {storage_dir}"
            )

        metadata_path = exp_dir / cls.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found for experiment {experiment_id}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        tracker = cls.__new__(cls)
        tracker.experiment_id = experiment_id
        tracker.experiment_name = metadata.get("experiment_name", "unknown")
        tracker.storage_dir = storage_path
        tracker.experiment_dir = exp_dir
        tracker.description = metadata.get("description", "")
        tracker.tags = metadata.get("tags", [])
        tracker._active = False
        tracker._thread_lock = threading.Lock()
        tracker.lock_path = str(exp_dir / cls.LOCK_FILE)

        tracker._metadata = metadata
        tracker._hyperparams = tracker._load_json(exp_dir / cls.HYPERPARAMS_FILE)
        tracker._metrics = tracker._load_json(exp_dir / cls.METRICS_FILE)
        tracker._artifacts = tracker._load_json(exp_dir / cls.ARTIFACTS_FILE)

        return tracker

    @classmethod
    def list_experiments(
        cls,
        storage_dir: str,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        name_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all experiments in the storage directory with optional filtering.

        Args:
            storage_dir: Directory where experiments are stored.
            tags: Filter by tags (experiments must have ALL specified tags).
            status: Filter by status ('running', 'completed', 'failed', etc.).
            name_filter: Filter by experiment name (substring match).

        Returns:
            List of experiment summary dictionaries.
        """
        storage_path = Path(storage_dir)
        if not storage_path.exists():
            return []

        experiments = []
        for exp_dir in storage_path.iterdir():
            if not exp_dir.is_dir():
                continue
            metadata_path = exp_dir / cls.METADATA_FILE
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Apply filters
                if tags:
                    exp_tags = set(metadata.get("tags", []))
                    if not set(tags).issubset(exp_tags):
                        continue

                if status and metadata.get("status") != status:
                    continue

                if name_filter and name_filter.lower() not in metadata.get(
                    "experiment_name", ""
                ).lower():
                    continue

                # Load final metrics
                metrics_path = exp_dir / cls.METRICS_FILE
                final_metrics = {}
                if metrics_path.exists():
                    with open(metrics_path, "r") as f:
                        metrics_data = json.load(f)
                    for metric_name, values in metrics_data.items():
                        if values:
                            final_metrics[metric_name] = values[-1]["value"]

                hyperparams_path = exp_dir / cls.HYPERPARAMS_FILE
                hyperparams = {}
                if hyperparams_path.exists():
                    with open(hyperparams_path, "r") as f:
                        hyperparams = json.load(f)

                experiments.append(
                    {
                        "experiment_id": metadata.get("experiment_id"),
                        "experiment_name": metadata.get("experiment_name"),
                        "status": metadata.get("status"),
                        "tags": metadata.get("tags", []),
                        "created_at": metadata.get("created_at"),
                        "started_at": metadata.get("started_at"),
                        "ended_at": metadata.get("ended_at"),
                        "hyperparams": hyperparams,
                        "final_metrics": final_metrics,
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load experiment from {exp_dir}: {e}")
                continue

        return experiments

    @classmethod
    def compare_experiments(
        cls,
        storage_dir: str,
        sort_by: Optional[str] = None,
        ascending: bool = True,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        name_filter: Optional[str] = None,
        include_hyperparams: Optional[List[str]] = None,
        include_metrics: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compare experiments and return a sorted comparison table.

        Args:
            storage_dir: Directory where experiments are stored.
            sort_by: Metric name to sort by.
            ascending: Sort in ascending order if True.
            tags: Filter by tags.
            status: Filter by status.
            name_filter: Filter by experiment name.
            include_hyperparams: List of hyperparameter names to include.
            include_metrics: List of metric names to include.

        Returns:
            List of comparison dictionaries sorted by the chosen metric.
        """
        experiments = cls.list_experiments(
            storage_dir, tags=tags, status=status, name_filter=name_filter
        )

        comparison_table = []
        for exp in experiments:
            row = {
                "experiment_id": exp["experiment_id"],
                "experiment_name": exp["experiment_name"],
                "status": exp["status"],
                "tags": exp["tags"],
                "created_at": exp["created_at"],
            }

            #