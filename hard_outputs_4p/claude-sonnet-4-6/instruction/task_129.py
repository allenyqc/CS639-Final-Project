```python
"""
ML Experiment Tracking System
A comprehensive module for tracking, comparing, and managing ML experiments.
"""

import json
import os
import shutil
import time
import uuid
import fcntl
import logging
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_load(path: Path) -> Optional[Dict]:
    """Load JSON with a shared (read) lock."""
    try:
        with open(path, "r") as fh:
            fcntl.flock(fh, fcntl.LOCK_SH)
            try:
                return json.load(fh)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


def _safe_json_dump(data: Dict, path: Path) -> None:
    """Write JSON atomically with an exclusive lock."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            json.dump(data, fh, indent=2, default=str)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    # Atomic rename so readers never see a partial file
    os.replace(tmp_path, path)


def _file_checksum(path: Union[str, Path]) -> str:
    """SHA-256 checksum of a file (for artifact integrity)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

class ExperimentRecord:
    """In-memory representation of a single experiment."""

    def __init__(
        self,
        experiment_id: str,
        name: str,
        tags: Optional[List[str]] = None,
        resumed_from: Optional[str] = None,
    ):
        self.experiment_id = experiment_id
        self.name = name
        self.tags: List[str] = tags or []
        self.resumed_from: Optional[str] = resumed_from
        self.status: str = "running"  # running | finished | failed
        self.created_at: str = _utc_now_iso()
        self.finished_at: Optional[str] = None
        self.hyperparameters: Dict[str, Any] = {}
        # metrics[metric_name] = list of {"step": int, "value": float, "timestamp": str}
        self.metrics: Dict[str, List[Dict]] = {}
        # artifacts[name] = {"source_path": str, "stored_path": str, "checksum": str, "logged_at": str}
        self.artifacts: Dict[str, Dict] = {}
        self.notes: str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "tags": self.tags,
            "resumed_from": self.resumed_from,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ExperimentRecord":
        rec = cls(
            experiment_id=d["experiment_id"],
            name=d["name"],
            tags=d.get("tags", []),
            resumed_from=d.get("resumed_from"),
        )
        rec.status = d.get("status", "finished")
        rec.created_at = d.get("created_at", _utc_now_iso())
        rec.finished_at = d.get("finished_at")
        rec.hyperparameters = d.get("hyperparameters", {})
        rec.metrics = d.get("metrics", {})
        rec.artifacts = d.get("artifacts", {})
        rec.notes = d.get("notes", "")
        return rec

    # ------------------------------------------------------------------
    def final_metric(self, metric_name: str) -> Optional[float]:
        """Return the last logged value for a metric, or None."""
        series = self.metrics.get(metric_name)
        if not series:
            return None
        return series[-1]["value"]

    def best_metric(self, metric_name: str, mode: str = "max") -> Optional[float]:
        """Return the best (max or min) logged value for a metric."""
        series = self.metrics.get(metric_name)
        if not series:
            return None
        values = [entry["value"] for entry in series]
        return max(values) if mode == "max" else min(values)


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """
    Track hyperparameters, metrics, and artifacts for ML experiments.

    Usage (context manager — recommended):
    ----------------------------------------
    with ExperimentTracker("my_exp", storage_dir="./experiments") as tracker:
        tracker.log_hyperparameters({"lr": 0.01, "epochs": 50})
        for epoch in range(50):
            tracker.log_metric("train_loss", loss, step=epoch)
        tracker.log_artifact("model", "checkpoints/model.pt")

    Usage (manual):
    ---------------
    tracker = ExperimentTracker("my_exp", storage_dir="./experiments")
    tracker.start()
    ...
    tracker.finish()
    """

    def __init__(
        self,
        name: str,
        storage_dir: Union[str, Path] = "./experiments",
        tags: Optional[List[str]] = None,
        notes: str = "",
        experiment_id: Optional[str] = None,
        copy_artifacts: bool = True,
    ):
        """
        Parameters
        ----------
        name          : Human-readable experiment name.
        storage_dir   : Root directory where all experiments are persisted.
        tags          : Optional list of string tags for filtering.
        notes         : Free-text notes attached to the experiment.
        experiment_id : If provided, resume an existing experiment.
        copy_artifacts: If True, copy artifact files into the experiment dir.
        """
        self.storage_dir = Path(storage_dir)
        self.copy_artifacts = copy_artifacts
        self._active = False

        if experiment_id is not None:
            # Resume mode
            self._record = self._load_record(experiment_id)
            if self._record is None:
                raise ValueError(f"Experiment '{experiment_id}' not found in {storage_dir}.")
            logger.info("Resuming experiment %s (%s)", self._record.name, experiment_id)
        else:
            new_id = str(uuid.uuid4())
            self._record = ExperimentRecord(
                experiment_id=new_id,
                name=name,
                tags=tags or [],
            )
            self._record.notes = notes

        self._exp_dir = self.storage_dir / self._record.experiment_id
        self._exp_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir = self._exp_dir / "artifacts"
        self._artifacts_dir.mkdir(exist_ok=True)
        self._record_path = self._exp_dir / "record.json"

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self._record.status = "failed"
            logger.error("Experiment failed: %s", exc_val)
        self.finish()
        return False  # Do not suppress exceptions

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark experiment as running and persist initial state."""
        if self._active:
            logger.warning("Tracker already started.")
            return
        self._active = True
        self._record.status = "running"
        self._save()
        logger.info("Started experiment '%s' [%s]", self._record.name, self._record.experiment_id)

    def finish(self) -> None:
        """Mark experiment as finished and persist final state."""
        if not self._active:
            logger.warning("Tracker was not started or already finished.")
        self._active = False
        if self._record.status == "running":
            self._record.status = "finished"
        self._record.finished_at = _utc_now_iso()
        self._save()
        logger.info(
            "Finished experiment '%s' [%s] — status: %s",
            self._record.name,
            self._record.experiment_id,
            self._record.status,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Log a dictionary of hyperparameters.
        Merges with any previously logged hyperparameters.
        """
        self._assert_active()
        self._record.hyperparameters.update(deepcopy(params))
        self._save()
        logger.debug("Logged hyperparameters: %s", list(params.keys()))

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a scalar metric at a given step/epoch.

        Parameters
        ----------
        name  : Metric name (e.g. 'train_loss', 'val_accuracy').
        value : Scalar float value.
        step  : Training step or epoch index. Auto-incremented if None.
        """
        self._assert_active()
        if name not in self._record.metrics:
            self._record.metrics[name] = []
        series = self._record.metrics[name]
        if step is None:
            step = len(series)
        series.append({"step": step, "value": float(value), "timestamp": _utc_now_iso()})
        self._save()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step=step)

    def log_artifact(self, artifact_name: str, source_path: Union[str, Path]) -> None:
        """
        Register a file artifact (model checkpoint, plot, etc.).

        If copy_artifacts=True the file is copied into the experiment directory.
        A SHA-256 checksum is computed for integrity verification.
        """
        self._assert_active()
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact source not found: {source_path}")

        checksum = _file_checksum(source_path)
        stored_path = str(source_path)

        if self.copy_artifacts:
            dest = self._artifacts_dir / source_path.name
            shutil.copy2(source_path, dest)
            stored_path = str(dest)

        self._record.artifacts[artifact_name] = {
            "source_path": str(source_path),
            "stored_path": stored_path,
            "checksum": checksum,
            "logged_at": _utc_now_iso(),
        }
        self._save()
        logger.info("Logged artifact '%s' from %s", artifact_name, source_path)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment."""
        if tag not in self._record.tags:
            self._record.tags.append(tag)
            self._save()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the experiment."""
        if tag in self._record.tags:
            self._record.tags.remove(tag)
            self._save()

    def set_notes(self, notes: str) -> None:
        """Update free-text notes."""
        self._record.notes = notes
        self._save()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return a dict summarising the current experiment."""
        rec = self._record
        final_metrics = {k: rec.final_metric(k) for k in rec.metrics}
        return {
            "experiment_id": rec.experiment_id,
            "name": rec.name,
            "status": rec.status,
            "tags": rec.tags,
            "created_at": rec.created_at,
            "finished_at": rec.finished_at,
            "hyperparameters": rec.hyperparameters,
            "final_metrics": final_metrics,
            "artifacts": list(rec.artifacts.keys()),
            "notes": rec.notes,
        }

    @property
    def experiment_id(self) -> str:
        return self._record.experiment_id

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        _safe_json_dump(self._record.to_dict(), self._record_path)

    def _load_record(self, experiment_id: str) -> Optional[ExperimentRecord]:
        path = self.storage_dir / experiment_id / "record.json"
        data = _safe_json_load(path)
        if data is None:
            return None
        return ExperimentRecord.from_dict(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                "ExperimentTracker is not active. Call start() or use as a context manager."
            )


# ---------------------------------------------------------------------------
# ExperimentStore — multi-experiment management
# ---------------------------------------------------------------------------

class ExperimentStore:
    """
    Manages a collection of experiments stored in a directory.

    Provides comparison tables, filtering, and bulk operations.
    """

    def __init__(self, storage_dir: Union[str, Path] = "./experiments"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self) -> List[ExperimentRecord]:
        """Load all experiment records from the storage directory."""
        records = []
        for exp_dir in sorted(self.storage_dir.iterdir()):
            record_path = exp_dir / "record.json"
            if record_path.exists():
                data = _safe_json_load(record_path)
                if data:
                    records.append(ExperimentRecord.from_dict(data))
        return records

    def load_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Load a single experiment by ID."""
        path = self.storage_dir / experiment_id / "record.json"
        data = _safe_json_load(path)
        return ExperimentRecord.from_dict(data) if data else None

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_tags(
        self,
        tags: List[str],
        match_all: bool = True,
        records: Optional[List[ExperimentRecord]] = None,
    ) -> List[ExperimentRecord]:
        """
        Filter experiments by tags.

        Parameters
        ----------
        tags      : Tags to filter by.
        match_all : If True, experiment must have ALL tags; else ANY tag.
        records   : Pre-loaded records to filter (loads all if None).
        """
        records = records or self.load_all()
        tag_set = set(tags)
        if match_all:
            return [r for r in records if tag_set.issubset(set(r.tags))]
        return [r for r in records if tag_set.intersection(set(r.tags))]

    def filter_by_status(
        self,
        status: str,
        records: Optional[List[ExperimentRecord]] = None,
    ) -> List[ExperimentRecord]:
        """Filter experiments by status: 'running', 'finished', or 'failed'."""
        records = records or self.load_all()
        return [r for r in records if r.status == status]

    def filter_by_name(
        self,
        name: str,
        exact: bool = False,
        records: Optional[List[ExperimentRecord]] = None,
    ) -> List[ExperimentRecord]:
        """Filter experiments by name (substring match by default)."""
        records = records or self.load_all()
        if exact:
            return [r for r in records if r.name == name]
        return [r for r in records if name in r.name]

    def filter_by_hyperparameter(
        self,
        key: str,
        value: Any,
        records: Optional[List[ExperimentRecord]] = None,
    ) -> List[ExperimentRecord]:
        """Filter experiments where hyperparameter key == value."""
        records = records or self.load_all()
        return [r for r in records if r.hyperparameters.get(key) == value]

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def comparison_table(
        self,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        metric_mode: str = "final",  # "final" | "best_max" | "best_min"
        records: Optional[List[ExperimentRecord]] = None,
        include_tags: bool = True,
    ) -> pd.DataFrame:
        """
        Build a DataFrame comparing hyperparameters vs metrics across experiments.

        Parameters
        ----------
        sort_by     : Metric name to sort by (or None for creation order).
        ascending   : Sort direction.
        metric_mode : How to aggregate metrics per experiment.
                      'final'    — last logged value
                      'best_max' — maximum logged value
                      'best_min' — minimum logged value
        records     : Pre-loaded records (loads all if None).
        include_tags: Whether to include a 'tags' column.

        Returns
        -------
        pd.DataFrame with one row per experiment.
        """
        records = records or self.load_all()
        if not records:
            return pd.DataFrame()

        rows = []
        for rec in records:
            row: Dict[str, Any] = {
                "experiment_id": rec.experiment_id,
                "name": rec.name,
                "status": rec.status,
                "created_at": rec.created_at,
                "finished_at": rec.finished_at,
            }
            if include_tags:
                row["tags"] = ", ".join(rec.tags)

            # Hyperparameters
            for k, v in rec.hyperparameters.items():
                row[f"hp_{k}"] = v

            # Metrics
            for metric_name in rec.metrics:
                if metric_mode == "final":
                    row[f"metric_{metric_name}"] = rec.final_metric(metric_name)
                elif metric_mode == "best_max":
                    row[f"metric_{metric_name}"] = rec.best_metric(metric_name, mode="max")
                elif metric_mode == "best_min":
                    row[f"metric_{metric_name}"] = rec.best_metric(metric_name, mode="min")
                else:
                    raise ValueError(f"Unknown metric_mode: {metric_mode}")

            rows.append(row)

        df = pd.DataFrame(rows)

        if sort_by is not None:
            col = f"metric_{sort_by}" if not sort_by.startswith("metric_") else sort_by
            if col in df.columns:
                df = df.sort_values(col, ascending=ascending, na_position="last")
            else:
                logger.warning("