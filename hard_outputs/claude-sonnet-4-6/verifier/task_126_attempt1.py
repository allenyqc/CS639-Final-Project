```python
"""
Production-ready model serving pipeline for ML inference.
Supports pickle, joblib, and ONNX model formats with comprehensive
validation, preprocessing, logging, and error handling.
"""

import os
import json
import time
import logging
import pickle
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

import numpy as np

# Optional imports with graceful fallbacks
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def _configure_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Configure a structured logger for the serving pipeline."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


_pipeline_logger = _configure_logger("model_serving_pipeline")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureSchema:
    """Schema definition for input feature validation."""
    expected_shape: Tuple[int, ...]          # e.g. (-1, 10) where -1 = any batch size
    dtype: np.dtype = field(default_factory=lambda: np.dtype("float32"))
    value_min: Optional[float] = None        # inclusive lower bound (post-cast)
    value_max: Optional[float] = None        # inclusive upper bound (post-cast)
    feature_names: Optional[List[str]] = None  # optional column labels


@dataclass
class PreprocessingParams:
    """Saved preprocessing parameters for feature scaling."""
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    clip_min: Optional[float] = None         # clip before scaling
    clip_max: Optional[float] = None

    def __post_init__(self):
        if self.mean is not None:
            self.mean = np.asarray(self.mean, dtype=np.float64)
        if self.std is not None:
            self.std = np.asarray(self.std, dtype=np.float64)


@dataclass
class InferenceRequest:
    """A single or batch inference request."""
    features: np.ndarray
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result returned by the inference pipeline."""
    request_id: str
    predictions: np.ndarray
    confidence_scores: Optional[np.ndarray]
    latency_ms: float
    timestamp: str
    batch_size: int
    error: Optional[str] = None


@dataclass
class RequestLogEntry:
    """Structured log entry persisted for auditing / monitoring."""
    request_id: str
    timestamp: str
    batch_size: int
    input_shape: Tuple[int, ...]
    input_hash: str                          # SHA-256 of raw bytes (privacy-safe)
    predictions: List[Any]
    confidence_scores: Optional[List[Any]]
    latency_ms: float
    error: Optional[str]
    metadata: Dict[str, Any]

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ModelServingError(Exception):
    """Base exception for the serving pipeline."""

class ModelLoadError(ModelServingError):
    """Raised when the model file cannot be loaded."""

class ValidationError(ModelServingError):
    """Raised when input validation fails."""

class PreprocessingError(ModelServingError):
    """Raised when preprocessing fails."""

class InferenceError(ModelServingError):
    """Raised when model inference fails."""

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_model(model_path: Union[str, Path]) -> Any:
    """
    Load a serialized model from disk.

    Supported formats:
        - .pkl / .pickle  → pickle
        - .joblib         → joblib
        - .onnx           → ONNX Runtime

    Returns the loaded model object (or ONNX InferenceSession).
    """
    path = Path(model_path).resolve()
    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")

    suffix = path.suffix.lower()
    _pipeline_logger.info("Loading model from %s (format: %s)", path, suffix)

    try:
        if suffix in (".pkl", ".pickle"):
            with open(path, "rb") as fh:
                model = pickle.load(fh)  # noqa: S301 – intentional, caller controls path
            _pipeline_logger.info("Pickle model loaded successfully.")
            return model

        elif suffix == ".joblib":
            if not JOBLIB_AVAILABLE:
                raise ModelLoadError("joblib is not installed. Install it with: pip install joblib")
            model = joblib.load(path)
            _pipeline_logger.info("Joblib model loaded successfully.")
            return model

        elif suffix == ".onnx":
            if not ONNX_AVAILABLE:
                raise ModelLoadError(
                    "onnxruntime is not installed. Install it with: pip install onnxruntime"
                )
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3  # suppress verbose ONNX logs
            session = ort.InferenceSession(str(path), sess_options=sess_options)
            _pipeline_logger.info("ONNX model loaded successfully.")
            return session

        else:
            raise ModelLoadError(
                f"Unsupported model format '{suffix}'. "
                "Supported: .pkl, .pickle, .joblib, .onnx"
            )

    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Failed to load model from {path}: {exc}") from exc

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_input(
    features: np.ndarray,
    schema: FeatureSchema,
) -> np.ndarray:
    """
    Validate and coerce input features against the feature schema.

    Returns the validated (and dtype-cast) array.
    Raises ValidationError with a descriptive message on failure.
    """
    if not isinstance(features, np.ndarray):
        try:
            features = np.asarray(features)
        except Exception as exc:
            raise ValidationError(
                f"Cannot convert input to numpy array: {exc}"
            ) from exc

    # --- shape check ---
    if features.ndim != len(schema.expected_shape):
        raise ValidationError(
            f"Input has {features.ndim} dimensions, expected {len(schema.expected_shape)}. "
            f"Expected shape pattern: {schema.expected_shape}"
        )

    for axis, (actual, expected) in enumerate(zip(features.shape, schema.expected_shape)):
        if expected != -1 and actual != expected:
            raise ValidationError(
                f"Shape mismatch on axis {axis}: got {actual}, expected {expected}. "
                f"Full shape: {features.shape}, expected pattern: {schema.expected_shape}"
            )

    # --- dtype coercion ---
    if features.dtype != schema.dtype:
        _pipeline_logger.debug(
            "Casting input dtype from %s to %s.", features.dtype, schema.dtype
        )
        try:
            features = features.astype(schema.dtype)
        except Exception as exc:
            raise ValidationError(
                f"Cannot cast input dtype {features.dtype} to {schema.dtype}: {exc}"
            ) from exc

    # --- value range check ---
    if schema.value_min is not None and np.any(features < schema.value_min):
        bad = float(np.min(features))
        raise ValidationError(
            f"Input contains values below minimum allowed ({schema.value_min}). "
            f"Observed minimum: {bad}"
        )

    if schema.value_max is not None and np.any(features > schema.value_max):
        bad = float(np.max(features))
        raise ValidationError(
            f"Input contains values above maximum allowed ({schema.value_max}). "
            f"Observed maximum: {bad}"
        )

    # --- NaN / Inf check ---
    if not np.all(np.isfinite(features)):
        n_nan = int(np.sum(np.isnan(features)))
        n_inf = int(np.sum(np.isinf(features)))
        raise ValidationError(
            f"Input contains non-finite values: {n_nan} NaN(s), {n_inf} Inf(s)."
        )

    return features

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(
    features: np.ndarray,
    params: Optional[PreprocessingParams],
) -> np.ndarray:
    """
    Apply standard scaling (z-score) using saved mean/std.
    Optionally clips values before scaling.
    """
    if params is None:
        return features

    try:
        X = features.astype(np.float64)

        if params.clip_min is not None or params.clip_max is not None:
            X = np.clip(X, params.clip_min, params.clip_max)

        if params.mean is not None:
            X = X - params.mean

        if params.std is not None:
            # Avoid division by zero
            safe_std = np.where(params.std == 0, 1.0, params.std)
            X = X / safe_std

        return X.astype(np.float32)

    except Exception as exc:
        raise PreprocessingError(f"Preprocessing failed: {exc}") from exc

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _run_sklearn_inference(
    model: Any,
    X: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run inference on a scikit-learn compatible model."""
    predictions = model.predict(X)
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # confidence = max probability across classes
        confidence = np.max(proba, axis=1)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Sigmoid-transform decision scores as a proxy confidence
        confidence = 1.0 / (1.0 + np.exp(-scores))
        if confidence.ndim > 1:
            confidence = np.max(confidence, axis=1)
    return predictions, confidence


def _run_onnx_inference(
    session: Any,  # ort.InferenceSession
    X: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run inference on an ONNX Runtime session."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    results = session.run(output_names, {input_name: X})
    predictions = np.asarray(results[0])

    confidence = None
    # ONNX classifiers often return a second output: list of dicts or array
    if len(results) > 1:
        raw_conf = results[1]
        if isinstance(raw_conf, list) and len(raw_conf) > 0:
            if isinstance(raw_conf[0], dict):
                confidence = np.array(
                    [max(d.values()) for d in raw_conf], dtype=np.float32
                )
            else:
                arr = np.asarray(raw_conf)
                confidence = np.max(arr, axis=1) if arr.ndim > 1 else arr
        elif isinstance(raw_conf, np.ndarray):
            confidence = np.max(raw_conf, axis=1) if raw_conf.ndim > 1 else raw_conf

    return predictions, confidence


def _run_inference(
    model: Any,
    X: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Dispatch inference to the correct backend."""
    try:
        if ONNX_AVAILABLE and isinstance(model, ort.InferenceSession):
            return _run_onnx_inference(model, X)
        else:
            return _run_sklearn_inference(model, X)
    except Exception as exc:
        raise InferenceError(f"Model inference failed: {exc}") from exc

# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------

class RequestLogger:
    """
    Thread-safe request logger that writes structured JSON log entries
    to a rotating in-memory buffer and optionally to a JSONL file.
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        buffer_size: int = 1000,
    ):
        self._lock = threading.Lock()
        self._buffer: List[RequestLogEntry] = []
        self._buffer_size = buffer_size
        self._log_file = log_file
        self._logger = _configure_logger("request_logger", log_file=None)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: RequestLogEntry) -> None:
        """Record a request log entry."""
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)  # drop oldest

        # Structured log to Python logger
        self._logger.info(
            "request_id=%s | batch=%d | latency_ms=%.2f | error=%s",
            entry.request_id,
            entry.batch_size,
            entry.latency_ms,
            entry.error or "none",
        )

        # Append to JSONL file if configured
        if self._log_file:
            self._write_jsonl(entry)

    def _write_jsonl(self, entry: RequestLogEntry) -> None:
        """Append a log entry as a JSON line."""
        try:
            record = asdict(entry)
            # Convert numpy types for JSON serialisation
            record = _json_safe(record)
            with self._lock:
                with open(self._log_file, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            self._logger.warning("Failed to write log entry to file: %s", exc)

    def get_recent_logs(self, n: int = 100) -> List[RequestLogEntry]:
        """Return the most recent n log entries from the in-memory buffer."""
        with self._lock:
            return list(self._buffer[-n:])

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics over buffered requests."""
        with self._lock:
            entries = list(self._buffer)

        if not entries:
            return {"total_requests": 0}

        latencies = [e.latency_ms for e in entries]
        errors = [e for e in entries if e.error]
        total_samples = sum(e.batch_size for e in entries)

        return {
            "total_requests": len(entries),
            "total_samples": total_samples,
            "error_count": len(errors),
            "error_rate": len(errors) / len(entries),
            "latency_ms": {
                "mean": float(np.mean(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99)),
                "max": float(np.max(latencies)),
            },
        }


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

# ---------------------------------------------------------------------------
# Request ID generation
# ---------------------------------------------------------------------------

_request_counter = 0
_counter_lock = threading.Lock()


def _generate_request_id() -> str:
    """Generate a monotonically increasing, timestamp-prefixed request ID."""
    global _request_counter
    with _counter_lock:
        _request_counter += 1
        count = _request_counter
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"req-{ts}-{count:06d}"


def _hash_input(features: np.ndarray) -> str:
    """Return a SHA-256 hex digest of the raw feature bytes (privacy-safe fingerprint)."""
    return hashlib.sha256(features.tobytes()).hexdigest()

# ---------------------------------------------------------------------------
# Core pipeline class
# ---------------------------------------------------------------------------

class ModelServingPipeline:
    """
    Production-ready model serving pipeline.

    Parameters
    ----------
    model_path : str or Path
        Path to the serialized model file (.pkl, .joblib, .onnx).
    schema : FeatureSchema
        Validation schema for incoming feature arrays.
    preprocessing_params : PreprocessingParams, optional
        Saved mean/std for feature scaling. Pass None to skip scaling.
    request_log_file : str, optional
        Path to a JSONL file for persistent request logging.
    log_buffer_size : int
        Maximum number of entries kept in the in-memory log buffer.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        schema: FeatureSchema,
        preprocessing_params: Optional[PreprocessingParams] = None,
        request_log_file: Optional[str] = None,
        log_buffer_size: int = 1000,
    ):
        self._schema = schema
        self._preprocessing_params = preprocessing_params
        self._model = _load_model(model_path)
        self._request_logger = RequestLogger(
            log_file=request_log_file,
            buffer_size=log_buffer_size,
        )
        _pipeline_logger.info(
            "ModelServingPipeline initialised. Schema: shape=%s dtype=%s",
            schema.expected_shape,
            schema.dtype,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, request: InferenceRequest) -> InferenceResult:
        """
        Run inference for a single or batch request.

        Parameters
        ----------
        request : InferenceRequest
            Contains the raw feature array and optional metadata.

        Returns
        -------
        InferenceResult
            Predictions, confidence scores, latency, and request metadata.
        """
        request_id = request.request_id or _generate_request_id()
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        start_time = time.perf_counter()

        predictions: Optional[np.ndarray] = None
        confidence: Optional[np.ndarray] = None
        error_msg: Optional[str] = None

        try:
            # 1. Validate
            features = _validate_input(request.features, self._schema)

            # 2. Preprocess
            X = _preprocess(features, self._preprocessing_params)

            # 3. Infer
            predictions, confidence = _run_inference(self._model, X)

        except (ValidationError, PreprocessingError, InferenceError) as exc:
            error_msg = str(exc)
            _pipeline