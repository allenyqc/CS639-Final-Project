```python
"""
Production-ready model serving pipeline for ML inference.

This module provides a secure, robust inference pipeline that:
- Loads pre-trained models with integrity verification
- Validates and preprocesses inputs
- Runs inference with confidence scores
- Logs requests with latency and timestamps
- Handles errors gracefully
- Supports batch prediction
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy dependencies — imported lazily so the module can be imported
# even when some backends are not installed.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """Specification for expected input features."""

    n_features: int
    feature_names: Optional[List[str]] = None
    dtype: str = "float32"
    value_ranges: Optional[Dict[str, Tuple[float, float]]] = None  # {name: (min, max)}


@dataclass
class PreprocessingParams:
    """Saved preprocessing parameters (fitted on training data only)."""

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreprocessingParams":
        return cls(
            mean=np.array(d["mean"], dtype=np.float32) if d.get("mean") is not None else None,
            std=np.array(d["std"], dtype=np.float32) if d.get("std") is not None else None,
            feature_names=d.get("feature_names"),
        )


@dataclass
class InferenceRequest:
    """A single or batch inference request."""

    features: np.ndarray  # shape (n_samples, n_features) or (n_features,)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResult:
    """Result of an inference request."""

    request_id: str
    predictions: np.ndarray
    confidence_scores: Optional[np.ndarray]
    latency_ms: float
    timestamp: str
    model_version: str
    error: Optional[str] = None


@dataclass
class LogRecord:
    """Structured log record for a single inference call."""

    request_id: str
    timestamp: str
    latency_ms: float
    n_samples: int
    input_shape: Tuple[int, ...]
    predictions: List[Any]
    confidence_scores: Optional[List[Any]]
    model_version: str
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


def _compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute the hex digest of a file using the specified algorithm."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_integrity(path: Path, expected_hash: Optional[str], algorithm: str = "sha256") -> None:
    """
    Verify file integrity against an expected hash.

    If *expected_hash* is None the check is skipped (with a warning).
    Raises ValueError on mismatch.
    """
    if expected_hash is None:
        logger.warning(
            "No expected hash provided for %s — skipping integrity check. "
            "Set MODEL_HASH env var for production use.",
            path,
        )
        return
    actual = _compute_file_hash(path, algorithm)
    if actual != expected_hash.lower():
        raise ValueError(
            f"Integrity check failed for {path}: "
            f"expected {expected_hash.lower()!r}, got {actual!r}"
        )
    logger.info("Integrity check passed for %s (%s: %s)", path, algorithm, actual)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_joblib(path: Path) -> Any:
    try:
        import joblib  # type: ignore
    except ImportError as exc:
        raise ImportError("joblib is required to load .joblib models") from exc
    return joblib.load(path)


def _load_pickle(path: Path) -> Any:
    import pickle  # noqa: S403 — guarded by integrity check above

    with open(path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301


def _load_onnx(path: Path) -> Any:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:
        raise ImportError("onnxruntime is required to load .onnx models") from exc
    return ort.InferenceSession(str(path))


_LOADERS: Dict[str, Callable[[Path], Any]] = {
    ".joblib": _load_joblib,
    ".pkl": _load_pickle,
    ".pickle": _load_pickle,
    ".onnx": _load_onnx,
}


def load_model(
    model_path: Union[str, Path],
    expected_hash: Optional[str] = None,
    hash_algorithm: str = "sha256",
) -> Any:
    """
    Load a serialized model after verifying its integrity.

    Parameters
    ----------
    model_path:
        Path to the serialized model file (.joblib, .pkl, .pickle, .onnx).
    expected_hash:
        Expected hex digest of the file.  If None, a warning is emitted.
        In production, always supply this value (e.g. from an env var).
    hash_algorithm:
        Hash algorithm to use (default: sha256).

    Returns
    -------
    Loaded model object.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    _verify_integrity(path, expected_hash, hash_algorithm)

    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(
            f"Unsupported model format '{suffix}'. "
            f"Supported: {list(_LOADERS.keys())}"
        )

    model = loader(path)
    logger.info("Model loaded from %s (format: %s)", path, suffix)
    return model


# ---------------------------------------------------------------------------
# Preprocessing parameters — load from JSON (never from pickle)
# ---------------------------------------------------------------------------


def load_preprocessing_params(params_path: Union[str, Path]) -> PreprocessingParams:
    """
    Load preprocessing parameters from a JSON file.

    The file must have been created by saving the parameters fitted
    **exclusively on the training partition**.

    Parameters
    ----------
    params_path:
        Path to a JSON file containing ``mean``, ``std``, and optionally
        ``feature_names``.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing params file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    params = PreprocessingParams.from_dict(data)
    logger.info("Preprocessing parameters loaded from %s", path)
    return params


def save_preprocessing_params(params: PreprocessingParams, params_path: Union[str, Path]) -> None:
    """Persist preprocessing parameters to a JSON file."""
    path = Path(params_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(params.to_dict(), fh, indent=2)
    logger.info("Preprocessing parameters saved to %s", path)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class ValidationError(ValueError):
    """Raised when an inference request fails validation."""


def validate_input(
    features: np.ndarray,
    spec: FeatureSpec,
) -> np.ndarray:
    """
    Validate input features against a FeatureSpec.

    Parameters
    ----------
    features:
        Raw input array, shape ``(n_features,)`` or ``(n_samples, n_features)``.
    spec:
        Expected feature specification.

    Returns
    -------
    features reshaped to ``(n_samples, n_features)``.

    Raises
    ------
    ValidationError
        On shape, dtype, or value-range violations.
    """
    if not isinstance(features, np.ndarray):
        try:
            features = np.array(features, dtype=spec.dtype)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Cannot convert input to numpy array: {exc}"
            ) from exc

    # Reshape 1-D input to (1, n_features)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.ndim != 2:
        raise ValidationError(
            f"Expected 2-D input (n_samples, n_features), got shape {features.shape}"
        )

    n_samples, n_features = features.shape
    if n_features != spec.n_features:
        raise ValidationError(
            f"Expected {spec.n_features} features, got {n_features}"
        )

    # Cast dtype
    try:
        features = features.astype(spec.dtype)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Cannot cast features to {spec.dtype}: {exc}") from exc

    # Check for NaN / Inf
    if not np.all(np.isfinite(features)):
        raise ValidationError("Input contains NaN or infinite values")

    # Per-feature value range checks
    if spec.value_ranges and spec.feature_names:
        for idx, name in enumerate(spec.feature_names):
            if name in spec.value_ranges:
                lo, hi = spec.value_ranges[name]
                col = features[:, idx]
                if np.any(col < lo) or np.any(col > hi):
                    raise ValidationError(
                        f"Feature '{name}' has values outside [{lo}, {hi}]: "
                        f"min={col.min():.4f}, max={col.max():.4f}"
                    )

    return features


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess(
    features: np.ndarray,
    params: PreprocessingParams,
) -> np.ndarray:
    """
    Apply standard scaling using pre-fitted parameters.

    Parameters
    ----------
    features:
        Validated input array, shape ``(n_samples, n_features)``.
    params:
        Preprocessing parameters fitted on the training set.

    Returns
    -------
    Scaled feature array.
    """
    if params.mean is None or params.std is None:
        logger.debug("No scaling parameters — returning raw features")
        return features

    mean = params.mean.reshape(1, -1)
    std = params.std.reshape(1, -1)

    # Guard against zero std (avoid division by zero)
    safe_std = np.where(std == 0, 1.0, std)
    scaled = (features - mean) / safe_std
    return scaled.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _predict_sklearn(model: Any, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run inference with a scikit-learn compatible model."""
    predictions = model.predict(X)
    confidence: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # For binary: max probability; for multi-class: max probability per sample
        confidence = proba.max(axis=1)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Normalise to [0, 1] via sigmoid for a rough confidence proxy
        confidence = 1.0 / (1.0 + np.exp(-scores))
        if confidence.ndim > 1:
            confidence = confidence.max(axis=1)
    return predictions, confidence


def _predict_onnx(model: Any, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run inference with an ONNX Runtime session."""
    input_name = model.get_inputs()[0].name
    output_names = [o.name for o in model.get_outputs()]
    outputs = model.run(output_names, {input_name: X})
    predictions = np.array(outputs[0]).flatten()
    confidence: Optional[np.ndarray] = None
    if len(outputs) > 1:
        proba = np.array(outputs[1])
        if proba.ndim == 2:
            confidence = proba.max(axis=1)
        else:
            confidence = proba
    return predictions, confidence


def _run_inference(model: Any, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Dispatch inference to the appropriate backend."""
    try:
        import onnxruntime as ort  # type: ignore

        if isinstance(model, ort.InferenceSession):
            return _predict_onnx(model, X)
    except ImportError:
        pass

    # Assume scikit-learn compatible API
    return _predict_sklearn(model, X)


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------


class RequestLogger:
    """
    Structured request logger.

    Writes JSON log records to a rotating file and/or the standard logger.
    Credentials and paths are read from environment variables — never
    hardcoded.
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        log_to_stdout: bool = True,
    ) -> None:
        self._log_to_stdout = log_to_stdout
        self._records: List[LogRecord] = []

        # Resolve log directory from env var or argument
        _dir = log_dir or os.environ.get("INFERENCE_LOG_DIR")
        self._log_path: Optional[Path] = None
        if _dir:
            self._log_path = Path(_dir)
            self._log_path.mkdir(parents=True, exist_ok=True)

    def log(self, record: LogRecord) -> None:
        """Persist a log record."""
        self._records.append(record)
        record_dict = asdict(record)

        if self._log_to_stdout:
            logger.info("INFERENCE | %s", json.dumps(record_dict, default=str))

        if self._log_path is not None:
            log_file = self._log_path / "inference_requests.jsonl"
            try:
                with open(log_file, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record_dict, default=str) + "\n")
            except OSError as exc:
                logger.error("Failed to write log record to %s: %s", log_file, exc)

    def get_records(self) -> List[LogRecord]:
        """Return all in-memory log records."""
        return list(self._records)

    def clear(self) -> None:
        """Clear in-memory records."""
        self._records.clear()


# ---------------------------------------------------------------------------
# Core inference pipeline
# ---------------------------------------------------------------------------


class InferencePipeline:
    """
    Production inference pipeline.

    Parameters
    ----------
    model:
        Loaded model object (sklearn, ONNX, etc.).
    feature_spec:
        Expected input feature specification.
    preprocessing_params:
        Scaling parameters fitted on the training set.
    model_version:
        Semantic version string for the model (e.g. "1.2.3").
    request_logger:
        Optional RequestLogger instance.
    """

    def __init__(
        self,
        model: Any,
        feature_spec: FeatureSpec,
        preprocessing_params: PreprocessingParams,
        model_version: str = "unknown",
        request_logger: Optional[RequestLogger] = None,
    ) -> None:
        self._model = model
        self._spec = feature_spec
        self._params = preprocessing_params
        self._version = model_version
        self._logger = request_logger or RequestLogger()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, request: InferenceRequest) -> InferenceResult:
        """
        Run inference for a single or batch request.

        Parameters
        ----------
        request:
            InferenceRequest containing raw feature array.

        Returns
        -------
        InferenceResult with predictions, confidence scores, and metadata.
        """
        start = time.perf_counter()
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        error_msg: Optional[str] = None
        predictions: np.ndarray = np.array([])
        confidence: Optional[np.ndarray] = None

        try:
            # 1. Validate
            validated = validate_input(request.features, self._spec)

            # 2. Preprocess
            scaled = preprocess(validated, self._params)

            # 3. Inference
            predictions, confidence = _run_inference(self._model, scaled)

        except ValidationError as exc:
            error_msg = f"ValidationError: {exc}"
            logger.warning("Request %s failed validation: %s", request.request_id, exc)
        except (RuntimeError, ValueError, TypeError) as exc:
            error_msg = f"InferenceError: {exc}"
            logger.error("Request %s inference error: %s", request.request_id, exc)

        latency_ms = (time.perf_counter() - start) * 1000.0

        # 4. Log
        log_record = LogRecord(
            request_id=request.request_id,
            timestamp=timestamp,
            latency_ms=round(latency_ms, 3),
            n_samples=int(request.features.shape[0]) if request.features.ndim > 1 else 1,
            input_shape=tuple(request.features.shape),
            predictions=predictions.tolist() if predictions.size > 0 else [],
            confidence_scores=confidence.tolist() if confidence is not None else None,
            model_version=self._version,
            error=error_msg,
            metadata=request.metadata,
        )
        self._logger.log(log_record)

        return InferenceResult(
            request_id=request.request_id,
            predictions=predictions,
            confidence_scores=confidence,
            latency_ms=latency_ms,
            timestamp=timestamp,
            model_version=self._version,
            error=error_msg,
        )

    def predict_batch(
        self,
        feature_list: List[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[InferenceResult]:
        """
        Run inference on a list of feature arrays.

        Each element of *feature_list* may be a 1-D array (single sample)
        or a 2-D array (mini-batch).  Results are returned in the same order.

        Parameters
        ----------
        feature_list:
            List of raw feature arrays.
        metadata:
            Optional metadata attached to every request.

        Returns
        -------
        List of InferenceResult objects.
        """
        results: List[InferenceResult] = []
        for features in feature_list:
            req = InferenceRequest(
                features=np.array(features),
                metadata=metadata,
            )
            results.append(self.predict(req))
        return results

    def predict_array(
        self,
        features: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """
        Convenience wrapper