```python
"""
ML Experiment Tracking System

A comprehensive module for tracking, comparing, and managing ML experiments.
"""

import json
import os
import uuid
import hashlib
import shutil
import logging
import threading
import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import copy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_load(path: Path) -> Dict:
    """Load JSON with basic integrity check (SHA-256 sidecar file)."""
    sidecar = path.with_suffix(".sha256")
    if sidecar.exists():
        expected = sidecar.read_text().strip()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Integrity check failed for {path}. "
                "File may have been tampered with or corrupted."
            )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_json_dump(data: Dict, path: Path) -> None:
    """Write JSON and update SHA-256 sidecar for integrity verification."""
    raw = json.dumps(data, indent=2, default=str).encode("utf-8")
    path.write_bytes(raw)
    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(hashlib.sha256(raw).hexdigest())


@contextmanager
def _file_lock(lock_path: Path):
    """Cross-process file lock using fcntl (POSIX) with a fallback."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

class ExperimentData:
    """In-memory representation of a single experiment."""

    def __init__(
        self,
        experiment_id: str,
        name: str,
        tags: Optional[List[str]] = None,
    ):
        self.experiment_id: str = experiment_id
        self.name: str = name
        self.tags: List[str] = tags if tags is not None else []
        self.hyperparameters: Dict[str, Any] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}  # name -> [{step, value, ts}]
        self.artifacts: List[Dict[str, str]] = []
        self.status: str = "running"  # running | completed | failed
        self.created_at: str = _utc_now_iso()
        self.updated_at: str = _utc_now_iso()
        self.notes: str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "tags": self.tags,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ExperimentData":
        obj = cls(
            experiment_id=d["experiment_id"],
            name=d["name"],
            tags=d.get("tags", []),
        )
        obj.hyperparameters = d.get("hyperparameters", {})
        obj.metrics = d.get("metrics", {})
        obj.artifacts = d.get("artifacts", [])
        obj.status = d.get("status", "completed")
        obj.created_at = d.get("created_at", _utc_now_iso())
        obj.updated_at = d.get("updated_at", _utc_now_iso())
        obj.notes = d.get("notes", "")
        return obj

    def final_metrics(self) -> Dict[str, float]:
        """Return the last recorded value for each metric."""
        result: Dict[str, float] = {}
        for metric_name, records in self.metrics.items():
            if records:
                result[metric_name] = records[-1]["value"]
        return result


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """
    Track hyperparameters, metrics, and artifacts for ML experiments.

    Usage (context manager):
        with ExperimentTracker("my_exp", "./experiments") as tracker:
            tracker.log_hyperparameters({"lr": 0.01, "epochs": 50})
            for epoch in range(50):
                tracker.log_metric("val_loss", loss, step=epoch)
            tracker.log_artifact("/path/to/model.pt")

    Usage (manual):
        tracker = ExperimentTracker("my_exp", "./experiments")
        tracker.start()
        ...
        tracker.stop()
    """

    def __init__(
        self,
        experiment_name: str,
        storage_dir: Union[str, Path],
        tags: Optional[List[str]] = None,
        experiment_id: Optional[str] = None,
    ):
        if not experiment_name or not isinstance(experiment_name, str):
            raise ValueError("experiment_name must be a non-empty string.")

        self._name = experiment_name
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._experiment_id: str = experiment_id or str(uuid.uuid4())
        self._data: Optional[ExperimentData] = None
        self._lock = threading.Lock()  # in-process thread safety
        self._started = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    @property
    def experiment_dir(self) -> Path:
        return self._storage_dir / self._experiment_id

    @property
    def _json_path(self) -> Path:
        return self.experiment_dir / "experiment.json"

    @property
    def _lock_path(self) -> Path:
        return self.experiment_dir / ".lock"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, notes: str = "") -> "ExperimentTracker":
        """Initialise a new experiment and persist it."""
        with self._lock:
            if self._started:
                raise RuntimeError("Tracker already started.")
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            self._data = ExperimentData(
                experiment_id=self._experiment_id,
                name=self._name,
                tags=[],
            )
            self._data.notes = notes
            self._persist()
            self._started = True
            logger.info("Experiment %s started (id=%s).", self._name, self._experiment_id)
        return self

    def stop(self, status: str = "completed") -> None:
        """Finalise the experiment."""
        if status not in {"completed", "failed"}:
            raise ValueError("status must be 'completed' or 'failed'.")
        with self._lock:
            if not self._started or self._data is None:
                raise RuntimeError("Tracker has not been started.")
            self._data.status = status
            self._data.updated_at = _utc_now_iso()
            self._persist()
            self._started = False
            logger.info(
                "Experiment %s stopped with status=%s.", self._name, status
            )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        status = "failed" if exc_type is not None else "completed"
        try:
            self.stop(status=status)
        except RuntimeError as exc:
            logger.warning("Error stopping tracker: %s", exc)
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters."""
        if not isinstance(params, dict):
            raise TypeError("params must be a dict.")
        self._assert_running()
        with self._lock:
            self._data.hyperparameters.update(copy.deepcopy(params))
            self._data.updated_at = _utc_now_iso()
            self._persist()

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """Log a scalar metric at a given step/epoch."""
        if not isinstance(name, str) or not name:
            raise ValueError("Metric name must be a non-empty string.")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Metric value must be numeric: {exc}") from exc

        self._assert_running()
        with self._lock:
            if name not in self._data.metrics:
                self._data.metrics[name] = []
            record = {
                "step": step if step is not None else len(self._data.metrics[name]),
                "value": value,
                "timestamp": _utc_now_iso(),
            }
            self._data.metrics[name].append(record)
            self._data.updated_at = _utc_now_iso()
            self._persist()

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log multiple metrics at once."""
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict.")
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def log_artifact(
        self,
        source_path: Union[str, Path],
        artifact_name: Optional[str] = None,
        copy_file: bool = True,
    ) -> str:
        """
        Register an artifact (model checkpoint, plot, etc.).

        Parameters
        ----------
        source_path : path to the artifact file.
        artifact_name : optional display name.
        copy_file : if True, copy the file into the experiment directory.

        Returns
        -------
        str : stored path of the artifact.
        """
        self._assert_running()
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Artifact not found: {source}")

        artifacts_dir = self.experiment_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if copy_file:
            dest = artifacts_dir / source.name
            shutil.copy2(source, dest)
            stored_path = str(dest)
        else:
            stored_path = str(source.resolve())

        record = {
            "name": artifact_name or source.name,
            "path": stored_path,
            "logged_at": _utc_now_iso(),
        }
        with self._lock:
            self._data.artifacts.append(record)
            self._data.updated_at = _utc_now_iso()
            self._persist()

        return stored_path

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def add_tags(self, tags: List[str]) -> None:
        """Add tags to the experiment."""
        if not isinstance(tags, list):
            raise TypeError("tags must be a list of strings.")
        self._assert_running()
        with self._lock:
            for tag in tags:
                if tag not in self._data.tags:
                    self._data.tags.append(tag)
            self._data.updated_at = _utc_now_iso()
            self._persist()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the experiment."""
        self._assert_running()
        with self._lock:
            if tag in self._data.tags:
                self._data.tags.remove(tag)
            self._data.updated_at = _utc_now_iso()
            self._persist()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a snapshot of the current experiment."""
        if self._data is None:
            raise RuntimeError("Tracker has not been started.")
        with self._lock:
            snap = copy.deepcopy(self._data.to_dict())
        snap["final_metrics"] = ExperimentData.from_dict(snap).final_metrics()
        return snap

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    @classmethod
    def resume(
        cls,
        experiment_id: str,
        storage_dir: Union[str, Path],
    ) -> "ExperimentTracker":
        """
        Resume tracking from a previously saved experiment.

        The experiment status is reset to 'running' and tracking continues.
        """
        storage = Path(storage_dir)
        exp_dir = storage / experiment_id
        json_path = exp_dir / "experiment.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"No experiment found at {json_path}. "
                "Check the experiment_id and storage_dir."
            )

        data_dict = _safe_json_load(json_path)
        tracker = cls(
            experiment_name=data_dict["name"],
            storage_dir=storage_dir,
            experiment_id=experiment_id,
        )
        tracker._data = ExperimentData.from_dict(data_dict)
        tracker._data.status = "running"
        tracker._data.updated_at = _utc_now_iso()
        tracker._started = True
        tracker._persist()
        logger.info("Resumed experiment %s (id=%s).", tracker._name, experiment_id)
        return tracker

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_running(self) -> None:
        if not self._started or self._data is None:
            raise RuntimeError(
                "ExperimentTracker is not running. Call start() or use as context manager."
            )

    def _persist(self) -> None:
        """Write experiment data to disk with file-level locking."""
        with _file_lock(self._lock_path):
            _safe_json_dump(self._data.to_dict(), self._json_path)


# ---------------------------------------------------------------------------
# ExperimentRegistry — load, compare, filter experiments
# ---------------------------------------------------------------------------

class ExperimentRegistry:
    """
    Load and compare experiments stored in a directory.

    Example
    -------
    registry = ExperimentRegistry("./experiments")
    table = registry.comparison_table(sort_by="val_loss", ascending=True)
    """

    def __init__(self, storage_dir: Union[str, Path]):
        self._storage_dir = Path(storage_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> List[ExperimentData]:
        """Load all experiments from the storage directory."""
        experiments: List[ExperimentData] = []
        if not self._storage_dir.exists():
            return experiments

        for exp_dir in sorted(self._storage_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            json_path = exp_dir / "experiment.json"
            if not json_path.exists():
                continue
            try:
                data_dict = _safe_json_load(json_path)
                experiments.append(ExperimentData.from_dict(data_dict))
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                logger.warning("Skipping %s due to error: %s", exp_dir, exc)

        return experiments

    def load_by_id(self, experiment_id: str) -> ExperimentData:
        """Load a single experiment by its ID."""
        json_path = self._storage_dir / experiment_id / "experiment.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found.")
        return ExperimentData.from_dict(_safe_json_load(json_path))

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_tags(
        self,
        tags: List[str],
        match_all: bool = True,
    ) -> List[ExperimentData]:
        """
        Filter experiments by tags.

        Parameters
        ----------
        tags : list of tag strings to match.
        match_all : if True, experiment must have ALL tags; otherwise ANY.
        """
        experiments = self.load_all()
        result: List[ExperimentData] = []
        for exp in experiments:
            exp_tags = set(exp.tags)
            query_tags = set(tags)
            if match_all:
                if query_tags.issubset(exp_tags):
                    result.append(exp)
            else:
                if query_tags & exp_tags:
                    result.append(exp)
        return result

    def filter_by_status(self, status: str) -> List[ExperimentData]:
        """Filter experiments by status ('running', 'completed', 'failed')."""
        return [e for e in self.load_all() if e.status == status]

    def filter_by_name(self, name: str) -> List[ExperimentData]:
        """Filter experiments by exact name match."""
        return [e for e in self.load_all() if e.name == name]

    def filter_by_hyperparameter(
        self,
        key: str,
        value: Any,
    ) -> List[ExperimentData]:
        """Filter experiments where hyperparameter key == value."""
        return [
            e for e in self.load_all()
            if e.hyperparameters.get(key) == value
        ]

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def comparison_table(
        self,
        sort_by: Optional[str] = None,
        ascending: bool = True,
        experiments: Optional[List[ExperimentData]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build a comparison table of hyperparameters vs final metrics.

        Parameters
        ----------
        sort_by : metric name to sort by (None = no sorting).
        ascending : sort direction.
        experiments : pre-filtered list; if None, loads all experiments.

        Returns
        -------
        List of dicts, each representing one experiment row.
        """
        exps = experiments if experiments is not None else self.load_all()
        rows: List[Dict[str, Any]] = []

        for exp in exps:
            row: Dict[str, Any] = {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "status": exp.status,
                "tags": exp.tags,
                "created_at": exp.created_at,
            }
            # Flatten hyperparameters with prefix
            for k, v in exp.hyperparameters.items():
                row[f"hp_{k}"] = v
            # Flatten final metrics with prefix
            for k, v in exp.final_metrics().items():
                row[f"metric_{k}"] = v
            rows.append(row)

        if sort_by is not None:
            sort_key = f"metric_{sort_by}"
            rows.sort(
                key=lambda r: (r.get(sort_key) is None, r.get(sort_key, 0)),
                reverse=not ascending,
            )

        return rows

    def best_experiment(
        self,
        metric: str,
        ascending: bool = True,
        experiments: Optional[List[ExperimentData]] = None,
    ) -> Optional[ExperimentData]:
        """
        Return the experiment with the best value for a given metric.

        Parameters
        ----------
        metric : metric name (without 'metric_' prefix).
        ascending : True = lower is better (e.g. loss); False = higher is better.
        """
        exps = experiments if experiments is not None else self.load_all()
        candidates = [