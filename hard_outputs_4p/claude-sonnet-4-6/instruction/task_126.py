```python
"""
Production Model Serving Pipeline
==================================
A secure, production-ready module for serving pre-trained ML models with
validation, preprocessing, logging, and batch prediction support.
"""

import os
import time
import json
import logging
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy dependencies — imported lazily to keep the module lightweight
# ---------------------------------------------------------------------------
try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

try:
    import pickle  # stdlib – always available
    _PICKLE_AVAILABLE = True
except ImportError:
    _PICKLE_AVAILABLE = False

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("model_serving")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded from disk."""


class ValidationError(ValueError):
    """Raised when an inference request fails validation."""


class PreprocessingError(RuntimeError):
    """Raised when preprocessing fails."""


class InferenceError(RuntimeError):
    """Raised when model inference fails."""


# ---------------------------------------------------------------------------
# Request / Response data classes (plain dicts kept for zero-dependency use)
# ---------------------------------------------------------------------------
def _make_request_id(features: np.ndarray) -> str:
    """Deterministic but opaque request identifier."""
    raw = features.tobytes() + str(time.time_ns()).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
class StandardScaler:
    """
    Minimal, serialisation-friendly standard scaler.
    Fit ONLY on training data; reuse saved parameters at inference time.
    """

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean_ = mean
        self.std_ = std

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Fit on training data ONLY."""
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise PreprocessingError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.std_

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "std": self.std_.tolist() if self.std_ is not None else [],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, List[float]]) -> "StandardScaler":
        mean = np.array(d["mean"], dtype=np.float64) if d.get("mean") else None
        std = np.array(d["std"], dtype=np.float64) if d.get("std") else None
        return cls(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Request logger
# ---------------------------------------------------------------------------
class RequestLogger:
    """
    Thread-safe request logger that records inference metadata.
    Writes JSONL to a file and/or the Python logging system.
    Credentials / paths are loaded from environment variables.
    """

    def __init__(self, log_file: Optional[str] = None):
        # Prefer env-var override; fall back to constructor argument
        self._log_file = os.environ.get("SERVING_LOG_FILE", log_file)
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []

        if self._log_file:
            Path(self._log_file).parent.mkdir(parents=True, exist_ok=True)
            logger.info("Request log file: %s", self._log_file)

    def log(
        self,
        request_id: str,
        input_features: np.ndarray,
        predictions: np.ndarray,
        confidence: Optional[np.ndarray],
        latency_ms: float,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        record = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_shape": list(input_features.shape),
            # Store a hash of raw features instead of raw values to protect PII
            "input_hash": hashlib.sha256(input_features.tobytes()).hexdigest(),
            "predictions": predictions.tolist() if predictions is not None else None,
            "confidence": confidence.tolist() if confidence is not None else None,
            "latency_ms": round(latency_ms, 3),
            "error": error,
        }

        with self._lock:
            self._records.append(record)
            if self._log_file:
                try:
                    with open(self._log_file, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                except OSError as exc:
                    logger.warning("Could not write to log file: %s", exc)

        level = logging.ERROR if error else logging.INFO
        logger.log(
            level,
            "request_id=%s latency_ms=%.3f predictions=%s error=%s",
            request_id,
            latency_ms,
            record["predictions"],
            error,
        )
        return record

    def get_records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(model_path: str) -> Any:
    """
    Load a serialised model from *model_path*.
    Supported formats: .pkl / .pickle (pickle), .joblib (joblib), .onnx (ONNX).

    The path may also be supplied via the ``MODEL_PATH`` environment variable,
    which takes precedence over the argument.
    """
    path = Path(os.environ.get("MODEL_PATH", model_path))

    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")

    suffix = path.suffix.lower()

    try:
        if suffix in {".pkl", ".pickle"}:
            if not _PICKLE_AVAILABLE:
                raise ModelLoadError("pickle is not available in this environment.")
            with open(path, "rb") as fh:
                model = pickle.load(fh)  # noqa: S301 — intentional, path is validated
            logger.info("Loaded pickle model from %s", path)
            return model

        if suffix == ".joblib":
            if not _JOBLIB_AVAILABLE:
                raise ModelLoadError("joblib is not installed. Run: pip install joblib")
            model = joblib.load(path)
            logger.info("Loaded joblib model from %s", path)
            return model

        if suffix == ".onnx":
            if not _ONNX_AVAILABLE:
                raise ModelLoadError(
                    "onnxruntime is not installed. Run: pip install onnxruntime"
                )
            session = ort.InferenceSession(str(path))
            logger.info("Loaded ONNX model from %s", path)
            return session

        raise ModelLoadError(
            f"Unsupported model format '{suffix}'. "
            "Expected one of: .pkl, .pickle, .joblib, .onnx"
        )
    except (OSError, pickle.UnpicklingError, Exception) as exc:
        raise ModelLoadError(f"Failed to load model from {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Input validator
# ---------------------------------------------------------------------------
class InputValidator:
    """
    Validates incoming feature arrays against expected schema.

    Parameters
    ----------
    expected_features : int
        Number of features expected per sample.
    expected_dtype : np.dtype or str
        Expected NumPy dtype (e.g. ``np.float32``).
    feature_ranges : dict, optional
        Mapping ``{feature_index: (min_val, max_val)}``.
        ``None`` bounds are treated as unbounded.
    """

    def __init__(
        self,
        expected_features: int,
        expected_dtype: Union[np.dtype, str] = np.float32,
        feature_ranges: Optional[Dict[int, Tuple[Optional[float], Optional[float]]]] = None,
    ):
        self.expected_features = expected_features
        self.expected_dtype = np.dtype(expected_dtype)
        self.feature_ranges = feature_ranges or {}

    def validate(self, X: np.ndarray) -> np.ndarray:
        """
        Validate *X* and return a sanitised copy cast to the expected dtype.
        Raises ``ValidationError`` on any violation.
        """
        if not isinstance(X, np.ndarray):
            try:
                X = np.array(X)
            except Exception as exc:
                raise ValidationError(
                    f"Input cannot be converted to numpy array: {exc}"
                ) from exc

        # Shape check
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValidationError(
                f"Expected 2-D input (n_samples, n_features), got shape {X.shape}."
            )
        if X.shape[1] != self.expected_features:
            raise ValidationError(
                f"Expected {self.expected_features} features, got {X.shape[1]}."
            )

        # NaN / Inf check
        if not np.isfinite(X).all():
            raise ValidationError("Input contains NaN or infinite values.")

        # Cast dtype
        try:
            X = X.astype(self.expected_dtype)
        except (ValueError, TypeError) as exc:
            raise ValidationError(f"Cannot cast input to {self.expected_dtype}: {exc}") from exc

        # Value range checks
        for feat_idx, (lo, hi) in self.feature_ranges.items():
            col = X[:, feat_idx]
            if lo is not None and (col < lo).any():
                raise ValidationError(
                    f"Feature {feat_idx} has values below minimum {lo}."
                )
            if hi is not None and (col > hi).any():
                raise ValidationError(
                    f"Feature {feat_idx} has values above maximum {hi}."
                )

        return X


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _run_sklearn_inference(
    model: Any, X: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run inference for a scikit-learn–compatible model."""
    predictions = model.predict(X)
    confidence: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidence = proba.max(axis=1)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Normalise decision scores to [0, 1] via sigmoid for a rough confidence
        confidence = 1.0 / (1.0 + np.exp(-scores))
        if confidence.ndim > 1:
            confidence = confidence.max(axis=1)
    return predictions, confidence


def _run_onnx_inference(
    session: Any, X: np.ndarray
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run inference for an ONNX Runtime session."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    results = session.run(output_names, {input_name: X.astype(np.float32)})

    predictions = np.array(results[0]).flatten()
    confidence: Optional[np.ndarray] = None
    if len(results) > 1:
        proba = np.array(results[1])
        if proba.ndim == 2:
            confidence = proba.max(axis=1)
        else:
            confidence = proba.flatten()
    return predictions, confidence


# ---------------------------------------------------------------------------
# Core serving pipeline
# ---------------------------------------------------------------------------
class ModelServingPipeline:
    """
    End-to-end serving pipeline that wires together loading, validation,
    preprocessing, inference, and logging.

    Parameters
    ----------
    model_path : str
        Path to the serialised model file.
    scaler_params : dict, optional
        Dict with keys ``"mean"`` and ``"std"`` (lists of floats).
        If omitted, no scaling is applied.
    expected_features : int
        Number of input features.
    expected_dtype : str or np.dtype
        Expected input dtype (default ``float32``).
    feature_ranges : dict, optional
        Per-feature value ranges for validation.
    log_file : str, optional
        Path for JSONL request log.  Can also be set via ``SERVING_LOG_FILE``.
    """

    def __init__(
        self,
        model_path: str,
        scaler_params: Optional[Dict[str, List[float]]] = None,
        expected_features: int = 10,
        expected_dtype: Union[str, np.dtype] = "float32",
        feature_ranges: Optional[Dict[int, Tuple[Optional[float], Optional[float]]]] = None,
        log_file: Optional[str] = None,
    ):
        self.model = load_model(model_path)
        self._is_onnx = _ONNX_AVAILABLE and isinstance(self.model, ort.InferenceSession)

        self.scaler: Optional[StandardScaler] = None
        if scaler_params:
            self.scaler = StandardScaler.from_dict(scaler_params)
            logger.info("Scaler loaded with %d features.", len(scaler_params.get("mean", [])))

        self.validator = InputValidator(
            expected_features=expected_features,
            expected_dtype=expected_dtype,
            feature_ranges=feature_ranges or {},
        )
        self.request_logger = RequestLogger(log_file=log_file)
        logger.info("ModelServingPipeline initialised (ONNX=%s).", self._is_onnx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as exc:
                raise PreprocessingError(f"Scaling failed: {exc}") from exc
        return X

    def _infer(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        try:
            if self._is_onnx:
                return _run_onnx_inference(self.model, X)
            return _run_sklearn_inference(self.model, X)
        except Exception as exc:
            raise InferenceError(f"Model inference failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self, X: Union[np.ndarray, List], return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single or batch inference request.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_features,)
        return_confidence : bool
            Whether to include confidence scores in the response.

        Returns
        -------
        dict with keys: request_id, predictions, confidence, latency_ms, timestamp
        """
        start_ns = time.perf_counter_ns()
        request_id = _make_request_id(np.asarray(X, dtype=object))
        predictions = confidence = None
        error_msg: Optional[str] = None

        try:
            X_arr = self.validator.validate(X)
            X_scaled = self._preprocess(X_arr)
            predictions, confidence = self._infer(X_scaled)

            if not return_confidence:
                confidence = None

            response: Dict[str, Any] = {
                "request_id": request_id,
                "predictions": predictions.tolist(),
                "confidence": confidence.tolist() if confidence is not None else None,
                "n_samples": X_arr.shape[0],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except (ValidationError, PreprocessingError, InferenceError) as exc:
            error_msg = str(exc)
            logger.error("Inference error [%s]: %s", request_id, error_msg)
            response = {
                "request_id": request_id,
                "error": error_msg,
                "predictions": None,
                "confidence": None,
                "n_samples": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:  # noqa: BLE001
            error_msg = f"Unexpected error: {exc}"
            logger.exception("Unexpected inference error [%s]", request_id)
            response = {
                "request_id": request_id,
                "error": error_msg,
                "predictions": None,
                "confidence": None,
                "n_samples": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            latency_ms = (time.perf_counter_ns() - start_ns) / 1e6
            response["latency_ms"] = round(latency_ms, 3)

            # Always log — even on error
            _features_for_log = (
                np.asarray(X, dtype=np.float32)
                if not isinstance(X, np.ndarray)
                else X
            )
            self.request_logger.log(
                request_id=request_id,
                input_features=_features_for_log,
                predictions=np.array(predictions) if predictions is not None else np.array([]),
                confidence=np.array(confidence) if confidence is not None else None,
                latency_ms=latency_ms,
                error=error_msg,
            )

        return response

    def predict_batch(
        self,
        batch: List[Union[np.ndarray, List]],
        return_confidence: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a list of individual samples (each a 1-D array-like).
        Returns a list of response dicts in the same order.
        """
        if not isinstance(batch, (list, tuple)):
            raise ValidationError("batch must be a list of samples.")

        # Stack into a single array for efficiency, then split results
        try:
            X_batch = np.vstack([np.asarray(s, dtype=np.float32).reshape(1, -1) for s in batch])
        except Exception as exc:
            raise ValidationError(f"Could not stack batch samples: {exc}") from exc

        response = self.predict(X_batch, return_confidence=return_confidence)

        if response.get("error"):
            # Propagate the same error to every item
            return [dict(response, n_samples=1) for _ in batch]

        preds = response["predictions"]
        confs = response["confidence"]
        results = []
        for i in range(len(batch)):
            results.append(
                {
                    "request_id": response["request_id"] + f"_{i}",
                    "predictions": [preds[i]],
                    "confidence": [confs[i]] if confs is not None else None,
                    "n_samples": 1,
                    "latency_ms": response["latency_ms"],
                    "timestamp": response["timestamp"],
                }
            )
        return results

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------
    def __call__(self,