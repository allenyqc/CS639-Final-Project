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


class ExperimentData:
    """Container for all experiment data."""

    def __init__(
        self,
        experiment_id: str,
        name: str,
        created_at: str,
        status: str = "initialized",
        hyperparameters: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        artifacts: Optional[List] = None,
        tags: Optional[List] = None,
        notes: str = "",
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ):
        self.experiment_id = experiment_id
        self.name = name
        self.created_at = created_at
        self.status = status
        self.hyperparameters = hyperparameters or {}
        self.metrics = metrics or {}  # {metric_name: [{step, value, timestamp}]}
        self.artifacts = artifacts or []
        self.tags = tags or []
        self.notes = notes
        self.started_at = started_at
        self.ended_at = ended_at
        self.duration_seconds = duration_seconds

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "created_at": self.created_at,
            "status": self.status,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "tags": self.tags,
            "notes": self.notes,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExperimentData":
        return cls(**data)

    def get_final_metrics(self) -> Dict[str, float]:
        """Return the last recorded value for each metric."""
        final = {}
        for metric_name, records in self.metrics.items():
            if records:
                final[metric_name] = records[-1]["value"]
        return final

    def get_best_metrics(self, higher_is_better: Optional[Dict[str, bool]] = None) -> Dict[str, float]:
        """Return the best recorded value for each metric."""
        higher_is_better = higher_is_better or {}
        best = {}
        for metric_name, records in self.metrics.items():
            if records:
                values = [r["value"] for r in records]
                if higher_is_better.get(metric_name, True):
                    best[metric_name] = max(values)
                else:
                    best[metric_name] = min(values)
        return best


class ExperimentTracker:
    """
    A comprehensive ML experiment tracking system.

    Supports logging hyperparameters, metrics, artifacts, tags,
    experiment comparison, concurrent access, and context manager interface.
    """

    def __init__(
        self,
        experiment_name: str,
        storage_dir: str = "./experiments",
        experiment_id: Optional[str] = None,
        auto_save_interval: int = 10,
    ):
        """
        Initialize the ExperimentTracker.

        Args:
            experiment_name: Human-readable name for the experiment.
            storage_dir: Directory to store experiment data.
            experiment_id: Optional existing experiment ID to resume.
            auto_save_interval: Number of metric logs before auto-saving.
        """
        self.experiment_name = experiment_name
        self.storage_dir = Path(storage_dir)
        self.auto_save_interval = auto_save_interval
        self._metric_log_count = 0
        self._thread_lock = threading.Lock()
        self._active = False

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        if experiment_id:
            self._load_existing(experiment_id)
        else:
            self._create_new()

    def _create_new(self):
        """Create a new experiment."""
        self.experiment_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        self.data = ExperimentData(
            experiment_id=self.experiment_id,
            name=self.experiment_name,
            created_at=now,
            status="initialized",
        )
        self.experiment_dir = self.storage_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self._save()
        logger.info(f"Created new experiment '{self.experiment_name}' with ID: {self.experiment_id}")

    def _load_existing(self, experiment_id: str):
        """Load an existing experiment by ID."""
        self.experiment_id = experiment_id
        self.experiment_dir = self.storage_dir / experiment_id
        data_path = self.experiment_dir / "experiment.json"

        if not data_path.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment_id}' not found in '{self.storage_dir}'"
            )

        with self._file_lock():
            with open(data_path, "r") as f:
                raw = json.load(f)
        self.data = ExperimentData.from_dict(raw)
        logger.info(f"Loaded experiment '{self.data.name}' (ID: {experiment_id})")

    def _file_lock(self) -> ExperimentLock:
        """Return a file lock for the current experiment."""
        lock_path = str(self.experiment_dir / ".lock")
        return ExperimentLock(lock_path)

    def _save(self):
        """Persist experiment data to disk with locking."""
        data_path = self.experiment_dir / "experiment.json"
        tmp_path = self.experiment_dir / "experiment.json.tmp"

        with self._thread_lock:
            with self._file_lock():
                with open(tmp_path, "w") as f:
                    json.dump(self.data.to_dict(), f, indent=2)
                os.replace(tmp_path, data_path)

    def start(self):
        """Mark the experiment as running."""
        if self._active:
            logger.warning("Experiment is already running.")
            return self
        self._active = True
        self.data.status = "running"
        self.data.started_at = datetime.utcnow().isoformat()
        self._save()
        logger.info(f"Experiment '{self.experiment_name}' started.")
        return self

    def stop(self, status: str = "completed"):
        """Mark the experiment as finished."""
        if not self._active:
            logger.warning("Experiment is not running.")
            return
        self._active = False
        self.data.status = status
        self.data.ended_at = datetime.utcnow().isoformat()
        if self.data.started_at:
            start = datetime.fromisoformat(self.data.started_at)
            end = datetime.fromisoformat(self.data.ended_at)
            self.data.duration_seconds = (end - start).total_seconds()
        self._save()
        logger.info(f"Experiment '{self.experiment_name}' stopped with status: {status}")

    def __enter__(self):
        """Context manager entry: start the experiment."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: stop the experiment."""
        if exc_type is not None:
            self.stop(status="failed")
            logger.error(f"Experiment failed with exception: {exc_val}")
        else:
            self.stop(status="completed")
        return False  # Do not suppress exceptions

    # -------------------------------------------------------------------------
    # Logging Methods
    # -------------------------------------------------------------------------

    def log_hyperparameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters for the experiment.

        Args:
            params: Dictionary of hyperparameter name-value pairs.
        """
        with self._thread_lock:
            self.data.hyperparameters.update(params)
        self._save()
        logger.debug(f"Logged hyperparameters: {list(params.keys())}")

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        """
        Log a single metric value.

        Args:
            name: Metric name (e.g., 'loss', 'accuracy').
            value: Numeric value of the metric.
            step: Optional training step number.
            epoch: Optional epoch number.
        """
        record = {
            "value": float(value),
            "timestamp": datetime.utcnow().isoformat(),
        }
        if step is not None:
            record["step"] = step
        if epoch is not None:
            record["epoch"] = epoch

        with self._thread_lock:
            if name not in self.data.metrics:
                self.data.metrics[name] = []
            self.data.metrics[name].append(record)
            self._metric_log_count += 1
            should_save = self._metric_log_count % self.auto_save_interval == 0

        if should_save:
            self._save()

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
        self._save()

    def log_artifact(
        self,
        file_path: str,
        artifact_type: str = "file",
        description: str = "",
        copy_to_experiment: bool = False,
    ):
        """
        Log an artifact (model checkpoint, plot, etc.).

        Args:
            file_path: Path to the artifact file.
            artifact_type: Type of artifact ('model', 'plot', 'data', 'file').
            description: Human-readable description.
            copy_to_experiment: If True, copy the file into the experiment directory.
        """
        src_path = Path(file_path)
        if not src_path.exists():
            logger.warning(f"Artifact file not found: {file_path}")

        stored_path = str(src_path.resolve())
        if copy_to_experiment and src_path.exists():
            artifacts_dir = self.experiment_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)
            dest = artifacts_dir / src_path.name
            shutil.copy2(src_path, dest)
            stored_path = str(dest)

        record = {
            "path": stored_path,
            "type": artifact_type,
            "description": description,
            "logged_at": datetime.utcnow().isoformat(),
        }
        with self._thread_lock:
            self.data.artifacts.append(record)
        self._save()
        logger.debug(f"Logged artifact: {stored_path}")

    def add_tag(self, *tags: str):
        """Add one or more tags to the experiment."""
        with self._thread_lock:
            for tag in tags:
                if tag not in self.data.tags:
                    self.data.tags.append(tag)
        self._save()

    def remove_tag(self, tag: str):
        """Remove a tag from the experiment."""
        with self._thread_lock:
            if tag in self.data.tags:
                self.data.tags.remove(tag)
        self._save()

    def add_note(self, note: str):
        """Append a note to the experiment."""
        with self._thread_lock:
            timestamp = datetime.utcnow().isoformat()
            self.data.notes += f"\n[{timestamp}] {note}"
        self._save()

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------

    @classmethod
    def resume(
        cls,
        experiment_id: str,
        storage_dir: str = "./experiments",
    ) -> "ExperimentTracker":
        """
        Resume tracking from a previous experiment.

        Args:
            experiment_id: The ID of the experiment to resume.
            storage_dir: Directory where experiments are stored.

        Returns:
            An ExperimentTracker instance loaded with the existing data.
        """
        tracker = cls.__new__(cls)
        tracker.storage_dir = Path(storage_dir)
        tracker.auto_save_interval = 10
        tracker._metric_log_count = 0
        tracker._thread_lock = threading.Lock()
        tracker._active = False
        tracker._load_existing(experiment_id)
        tracker.experiment_name = tracker.data.name
        logger.info(f"Resumed experiment '{tracker.data.name}' (ID: {experiment_id})")
        return tracker

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        Return a summary of the current experiment.

        Returns:
            Dictionary with experiment metadata, hyperparameters, and final metrics.
        """
        self._save()  # Ensure latest data is persisted
        final_metrics = self.data.get_final_metrics()
        return {
            "experiment_id": self.data.experiment_id,
            "name": self.data.name,
            "status": self.data.status,
            "created_at": self.data.created_at,
            "started_at": self.data.started_at,
            "ended_at": self.data.ended_at,
            "duration_seconds": self.data.duration_seconds,
            "tags": self.data.tags,
            "hyperparameters": copy.deepcopy(self.data.hyperparameters),
            "final_metrics": final_metrics,
            "num_artifacts": len(self.data.artifacts),
            "notes": self.data.notes.strip(),
        }

    def print_summary(self):
        """Print a formatted summary of the experiment."""
        s = self.summary()
        print("\n" + "=" * 60)
        print(f"  Experiment: {s['name']}  (ID: {s['experiment_id']})")
        print("=" * 60)
        print(f"  Status      : {s['status']}")
        print(f"  Created     : {s['created_at']}")
        print(f"  Started     : {s['started_at']}")
        print(f"  Ended       : {s['ended_at']}")
        if s["duration_seconds"] is not None:
            print(f"  Duration    : {s['duration_seconds']:.2f}s")
        print(f"  Tags        : {', '.join(s['tags']) or 'None'}")
        print("\n  Hyperparameters:")
        for k, v in s["hyperparameters"].items():
            print(f"    {k}: {v}")
        print("\n  Final Metrics:")
        for k, v in s["final_metrics"].items():
            print(f"    {k}: {v:.6g}")
        print(f"\n  Artifacts   : {s['num_artifacts']}")
        if s["notes"]:
            print(f"\n  Notes:\n{s['notes']}")
        print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # Comparison & Loading
    # -------------------------------------------------------------------------

    @staticmethod
    def load_all_experiments(storage_dir: str = "./experiments") -> List[ExperimentData]:
        """
        Load all experiments from the storage directory.

        Args:
            storage_dir: Directory where experiments are stored.

        Returns:
            List of ExperimentData objects.
        """
        storage_path = Path(storage_dir)
        experiments = []
        if not storage_path.exists():
            return experiments

        for exp_dir in sorted(storage_path.iterdir()):
            if not exp_dir.is_dir():
                continue
            data_path = exp_dir / "experiment.json"
            if not data_path.exists():
                continue
            try:
                with open(data_path, "r") as f:
                    raw = json.load(f)
                experiments.append(ExperimentData.from_dict(raw))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Could not load experiment from {exp_dir}: {e}")

        return experiments

    @staticmethod
    def filter_experiments(
        experiments: List[ExperimentData],
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        name_contains: Optional[str] = None,
        min_metric: Optional[Dict[str, float]] = None,
        max_metric: Optional[Dict[str, float]] = None,
    ) -> List[ExperimentData]:
        """
        Filter a list of experiments by various criteria.

        Args:
            experiments: List of ExperimentData to filter.
            tags: Keep only experiments that have ALL specified tags.
            status: Keep only experiments with this status.
            name_contains: Keep only experiments whose name contains this string.
            min_metric: {metric_name: min_value} — keep experiments where final metric >= min_value.
            max_metric: {metric_name: max_value} — keep experiments where final metric <= max_value.

        Returns:
            Filtered list of ExperimentData.
        """
        result = experiments

        if tags:
            result = [e for e in result if all(t in e.tags for t in tags)]

        if status:
            result = [e for e in result if e.status == status]

        if name_contains:
            result = [e for e in result if name_contains.lower() in e.name.lower()]

        if min_metric:
            filtered = []
            for e in result:
                fm = e.get_final_metrics()
                if all(fm.get(k, float("-inf")) >= v for k, v in min_metric.items()):
                    filtered.append(e)
            result = filtered

        if max_metric:
            filtered = []
            for e in result:
                fm = e.get_final_metrics()
                if all(fm.get(k, float("inf")) <= v for k, v in max_metric.items()):
                    filtered.append(e)
            result = filtered

        return result

    @staticmethod
    def compare_experiments(
        storage_dir: str = "./experiments",
        sort_by: Optional[str] = None,
        ascending: bool = True,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
        name_contains: Optional[str] = None,
        include_hyperparams: Optional[List[str]] = None,
        include_metrics: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load all experiments and produce a comparison table.

        Args:
            storage_dir: Directory where experiments are stored.
            sort_by: Metric name to sort by (or 'name', 'created_at', 'duration_seconds').
            ascending: Sort order.
            tags: Filter by tags.
            status: Filter by status.
            name