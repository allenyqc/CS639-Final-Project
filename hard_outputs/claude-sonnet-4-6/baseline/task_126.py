```python
"""
Production-ready model serving pipeline for ML inference.
Supports pickle, joblib, and ONNX model formats.
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
from functools import wraps

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

def _configure_logger(name: str = "model_serving") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

_logger = _configure_logger()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureSchema:
    """Defines expected properties of the input feature matrix."""
    n_features: int
    feature_names: Optional[List[str]] = None
    dtype: str = "float32"
    value_ranges: Optional[Dict[str, Tuple[float, float]]] = None  # {feature_name: (min, max)}
    allow_nan: bool = False

    def __post_init__(self):
        if self.feature_names and len(self.feature_names) != self.n_features:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) "
                f"must match n_features ({self.n_features})."
            )


@dataclass
class PreprocessingParams:
    """Saved preprocessing parameters for feature scaling."""
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    scale_features: bool = True

    def __post_init__(self):
        if self.scale_features:
            if self.mean is None or self.std is None:
                raise ValueError("mean and std must be provided when scale_features=True.")
            self.mean = np.asarray(self.mean, dtype=np.float64)
            self.std = np.asarray(self.std, dtype=np.float64)
            if np.any(self.std == 0):
                raise ValueError("std contains zero values; division by zero would occur.")


@dataclass
class InferenceRecord:
    """Audit log record for a single inference call."""
    request_id: str
    timestamp_utc: str
    input_shape: Tuple[int, ...]
    input_hash: str                      # SHA-256 of raw input bytes (privacy-safe)
    predictions: List[Any]
    confidence_scores: Optional[List[float]]
    latency_ms: float
    model_format: str
    batch_size: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Request log store (thread-safe)
# ---------------------------------------------------------------------------

class RequestLogger:
    """Thread-safe in-memory request log with optional file persistence."""

    def __init__(self, log_file: Optional[str] = None, max_records: int = 10_000):
        self._records: List[InferenceRecord] = []
        self._lock = threading.Lock()
        self._max_records = max_records
        self._log_file = log_file
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: InferenceRecord) -> None:
        with self._lock:
            if len(self._records) >= self._max_records:
                self._records.pop(0)  # evict oldest
            self._records.append(record)
            if self._log_file:
                self._persist(record)

    def _persist(self, record: InferenceRecord) -> None:
        try:
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), default=str) + "\n")
        except OSError as exc:
            _logger.warning("Failed to persist log record: %s", exc)

    def get_records(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            records = self._records[-last_n:] if last_n else list(self._records)
        return [r.to_dict() for r in records]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._records:
                return {"total_requests": 0}
            latencies = [r.latency_ms for r in self._records if r.error is None]
            errors = sum(1 for r in self._records if r.error is not None)
            return {
                "total_requests": len(self._records),
                "successful_requests": len(self._records) - errors,
                "failed_requests": errors,
                "avg_latency_ms": float(np.mean(latencies)) if latencies else None,
                "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
                "max_latency_ms": float(np.max(latencies)) if latencies else None,
            }


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

class ModelLoader:
    """Loads ML models from pickle, joblib, or ONNX files."""

    SUPPORTED_EXTENSIONS = {".pkl", ".pickle", ".joblib", ".onnx"}

    @staticmethod
    def load(model_path: str) -> Tuple[Any, str]:
        """
        Load a model from disk.

        Returns
        -------
        (model_object, format_string)
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {model_path}")

        suffix = path.suffix.lower()
        if suffix not in ModelLoader.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported model format '{suffix}'. "
                f"Supported: {ModelLoader.SUPPORTED_EXTENSIONS}"
            )

        if suffix in {".pkl", ".pickle"}:
            return ModelLoader._load_pickle(path), "pickle"
        if suffix == ".joblib":
            return ModelLoader._load_joblib(path), "joblib"
        if suffix == ".onnx":
            return ModelLoader._load_onnx(path), "onnx"

        raise RuntimeError("Unreachable code in ModelLoader.load")  # pragma: no cover

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        _logger.info("Loading pickle model from %s", path)
        with open(path, "rb") as fh:
            model = pickle.load(fh)  # noqa: S301 – caller is responsible for trusted files
        _logger.info("Pickle model loaded successfully.")
        return model

    @staticmethod
    def _load_joblib(path: Path) -> Any:
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is not installed. Run: pip install joblib")
        _logger.info("Loading joblib model from %s", path)
        model = joblib.load(path)
        _logger.info("Joblib model loaded successfully.")
        return model

    @staticmethod
    def _load_onnx(path: Path) -> Any:
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. Run: pip install onnxruntime"
            )
        _logger.info("Loading ONNX model from %s", path)
        session = ort.InferenceSession(str(path))
        _logger.info("ONNX model loaded successfully.")
        return session


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class InputValidator:
    """Validates raw input arrays against a FeatureSchema."""

    def __init__(self, schema: FeatureSchema):
        self.schema = schema

    def validate(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and coerce input array.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        Validated numpy array.
        """
        # Convert to numpy
        try:
            X = np.asarray(X)
        except Exception as exc:
            raise TypeError(f"Cannot convert input to numpy array: {exc}") from exc

        # Shape check
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(
                f"Input must be 2-D (n_samples, n_features), got shape {X.shape}."
            )
        if X.shape[1] != self.schema.n_features:
            raise ValueError(
                f"Expected {self.schema.n_features} features, got {X.shape[1]}."
            )

        # NaN check
        if not self.schema.allow_nan and np.isnan(X).any():
            raise ValueError(
                "Input contains NaN values. Set allow_nan=True in FeatureSchema to permit them."
            )

        # Infinity check
        if not np.isfinite(X[~np.isnan(X)]).all():
            raise ValueError("Input contains infinite values.")

        # Value range check
        if self.schema.value_ranges and self.schema.feature_names:
            for idx, name in enumerate(self.schema.feature_names):
                if name in self.schema.value_ranges:
                    lo, hi = self.schema.value_ranges[name]
                    col = X[:, idx]
                    if np.any(col < lo) or np.any(col > hi):
                        raise ValueError(
                            f"Feature '{name}' has values outside expected range [{lo}, {hi}]."
                        )

        # Dtype coercion
        try:
            X = X.astype(self.schema.dtype)
        except Exception as exc:
            raise TypeError(
                f"Cannot cast input to dtype '{self.schema.dtype}': {exc}"
            ) from exc

        return X


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor:
    """Applies saved preprocessing transformations (z-score scaling)."""

    def __init__(self, params: PreprocessingParams):
        self.params = params

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.params.scale_features:
            return X
        return (X - self.params.mean) / self.params.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.params.scale_features:
            return X
        return X * self.params.std + self.params.mean


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """Runs predictions using the loaded model."""

    def __init__(self, model: Any, model_format: str):
        self.model = model
        self.model_format = model_format

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run inference.

        Returns
        -------
        (predictions, confidence_scores)
        confidence_scores is None when the model does not support probabilities.
        """
        if self.model_format == "onnx":
            return self._predict_onnx(X)
        return self._predict_sklearn_like(X)

    def _predict_sklearn_like(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        predictions = self.model.predict(X)
        confidence = None
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X)
                confidence = np.max(proba, axis=1)
            except Exception:  # noqa: BLE001
                pass
        elif hasattr(self.model, "decision_function"):
            try:
                scores = self.model.decision_function(X)
                # Normalise to [0, 1] via sigmoid for binary case
                if scores.ndim == 1:
                    confidence = 1.0 / (1.0 + np.exp(-scores))
                else:
                    confidence = np.max(scores, axis=1)
            except Exception:  # noqa: BLE001
                pass
        return predictions, confidence

    def _predict_onnx(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        session: "ort.InferenceSession" = self.model
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        X_f32 = X.astype(np.float32)
        outputs = session.run(output_names, {input_name: X_f32})

        predictions = np.asarray(outputs[0])
        confidence = None
        # ONNX classifiers often expose a second output with probabilities
        if len(outputs) > 1:
            proba_raw = outputs[1]
            if isinstance(proba_raw, list):
                # List of dicts (ONNX ZipMap output)
                try:
                    proba_arr = np.array(
                        [[v for v in d.values()] for d in proba_raw]
                    )
                    confidence = np.max(proba_arr, axis=1)
                except Exception:  # noqa: BLE001
                    pass
            else:
                proba_arr = np.asarray(proba_raw)
                if proba_arr.ndim == 2:
                    confidence = np.max(proba_arr, axis=1)
                elif proba_arr.ndim == 1:
                    confidence = proba_arr

        return predictions, confidence


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _generate_request_id() -> str:
    return hashlib.sha256(
        f"{time.time_ns()}-{threading.get_ident()}".encode()
    ).hexdigest()[:16]


def _hash_input(X: np.ndarray) -> str:
    return hashlib.sha256(X.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class ModelServingPipeline:
    """
    End-to-end production inference pipeline.

    Parameters
    ----------
    model_path : str
        Path to the serialized model file (.pkl, .pickle, .joblib, .onnx).
    schema : FeatureSchema
        Expected input feature schema.
    preprocessing_params : PreprocessingParams
        Saved preprocessing parameters.
    log_file : str, optional
        Path to JSONL file for persisting inference logs.
    max_log_records : int
        Maximum in-memory log records (default 10 000).
    """

    def __init__(
        self,
        model_path: str,
        schema: FeatureSchema,
        preprocessing_params: PreprocessingParams,
        log_file: Optional[str] = None,
        max_log_records: int = 10_000,
    ):
        self.schema = schema
        self.preprocessing_params = preprocessing_params

        # Load model
        self.model, self.model_format = ModelLoader.load(model_path)

        # Build sub-components
        self._validator = InputValidator(schema)
        self._preprocessor = Preprocessor(preprocessing_params)
        self._engine = InferenceEngine(self.model, self.model_format)
        self._request_logger = RequestLogger(log_file=log_file, max_records=max_log_records)

        _logger.info(
            "ModelServingPipeline ready | format=%s | features=%d",
            self.model_format,
            schema.n_features,
        )

    # ------------------------------------------------------------------
    # Public inference interface
    # ------------------------------------------------------------------

    def predict(
        self,
        X: Union[np.ndarray, List],
        return_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference on a single sample or a batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features,)
        return_confidence : bool
            Whether to include confidence scores in the response.

        Returns
        -------
        dict with keys:
            - request_id
            - predictions
            - confidence_scores (if available and requested)
            - latency_ms
            - timestamp_utc
            - batch_size
        """
        request_id = _generate_request_id()
        t_start = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        error_msg: Optional[str] = None
        predictions_out: List[Any] = []
        confidence_out: Optional[List[float]] = None
        input_shape: Tuple[int, ...] = ()
        input_hash = ""

        try:
            # 1. Validate
            X_valid = self._validator.validate(X)
            input_shape = X_valid.shape
            input_hash = _hash_input(X_valid)

            # 2. Preprocess
            X_scaled = self._preprocessor.transform(X_valid)

            # 3. Infer
            preds, confidence = self._engine.predict(X_scaled)

            predictions_out = preds.tolist()
            if return_confidence and confidence is not None:
                confidence_out = confidence.tolist()

        except (ValueError, TypeError) as exc:
            error_msg = f"Validation/preprocessing error: {exc}"
            _logger.error("[%s] %s", request_id, error_msg)
        except Exception as exc:  # noqa: BLE001
            error_msg = f"Inference error: {exc}"
            _logger.exception("[%s] Unexpected error during inference.", request_id)

        latency_ms = (time.perf_counter() - t_start) * 1_000

        # 4. Log
        record = InferenceRecord(
            request_id=request_id,
            timestamp_utc=timestamp,
            input_shape=input_shape,
            input_hash=input_hash,
            predictions=predictions_out,
            confidence_scores=confidence_out,
            latency_ms=round(latency_ms, 3),
            model_format=self.model_format,
            batch_size=input_shape[0] if input_shape else 0,
            error=error_msg,
        )
        self._request_logger.log(record)

        # 5. Build response
        if error_msg:
            return {
                "request_id": request_id,
                "error": error_msg,
                "latency_ms": round(latency_ms, 3),
                "timestamp_utc": timestamp,
            }

        response: Dict[str, Any] = {
            "request_id": request_id,
            "predictions": predictions_out,
            "latency_ms": round(latency_ms, 3),
            "timestamp_utc": timestamp,
            "batch_size": input_shape[0],
        }
        if return_confidence and confidence_out is not None:
            response["confidence_scores"] = confidence_out

        return response

    def predict_batch(
        self,
        batch: List[Union[np.ndarray, List]],
        return_confidence: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a list of independent samples/sub-batches.

        Each element in `batch` is treated as a separate inference request
        (useful when samples have different metadata or come from different
        sources). For homogeneous batches, pass a 2-D array to `predict`.

        Parameters
        ----------
        batch : list of array-like