```python
"""
Image Classification Pipeline with Data Augmentation
=====================================================
Builds a complete ML pipeline: augmentation → normalization → MLP classification.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import rotate
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NormalizationParams:
    """Per-channel mean and std computed ONLY from training data."""
    mean: np.ndarray          # shape (C,) or scalar for grayscale
    std: np.ndarray           # shape (C,) or scalar for grayscale


@dataclass
class AugmentationStats:
    """Statistics about the augmentation process."""
    original_train_size: int
    augmented_train_size: int
    augmentation_factor: float
    operations_applied: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """All outputs returned by the pipeline."""
    model: MLPClassifier
    label_encoder: LabelEncoder
    norm_params: NormalizationParams
    augmentation_stats: AugmentationStats
    metrics: dict[str, Any]   # accuracy, macro_f1, confusion_matrix
    X_test_flat: np.ndarray   # normalised, flattened test features
    y_test: np.ndarray        # original (encoded) test labels


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _random_horizontal_flip(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip image horizontally with 50 % probability."""
    if rng.random() < 0.5:
        return np.fliplr(image)
    return image


def _random_rotation(
    image: np.ndarray,
    rng: np.random.Generator,
    max_angle: float = 15.0,
) -> np.ndarray:
    """Rotate image by a random angle in [-max_angle, +max_angle] degrees."""
    angle = rng.uniform(-max_angle, max_angle)
    axes = (0, 1)  # rotate in the H×W plane; works for both HW and HWC
    return rotate(image, angle=angle, axes=axes, reshape=False, mode="nearest")


def _random_brightness_contrast(
    image: np.ndarray,
    rng: np.random.Generator,
    brightness_delta: float = 0.2,
    contrast_range: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """Adjust brightness (additive) and contrast (multiplicative)."""
    img = image.astype(np.float64)
    # Contrast: scale around mean
    contrast_factor = rng.uniform(*contrast_range)
    mean_val = img.mean()
    img = mean_val + contrast_factor * (img - mean_val)
    # Brightness: additive shift relative to value range
    value_range = img.max() - img.min() + 1e-8
    brightness_shift = rng.uniform(-brightness_delta, brightness_delta) * value_range
    img = img + brightness_shift
    # Clip to original dtype range
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        img = np.clip(img, info.min, info.max)
    else:
        img = np.clip(img, 0.0, 1.0)
    return img.astype(image.dtype)


def _add_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    noise_std_fraction: float = 0.02,
) -> np.ndarray:
    """Add Gaussian noise scaled to the image value range."""
    img = image.astype(np.float64)
    value_range = img.max() - img.min() + 1e-8
    noise = rng.normal(0.0, noise_std_fraction * value_range, size=img.shape)
    img = img + noise
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        img = np.clip(img, info.min, info.max)
    else:
        img = np.clip(img, 0.0, 1.0)
    return img.astype(image.dtype)


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply all augmentation operations to a single image."""
    img = _random_horizontal_flip(image, rng)
    img = _random_rotation(img, rng)
    img = _random_brightness_contrast(img, rng)
    img = _add_gaussian_noise(img, rng)
    return img


def augment_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    target_multiplier: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create `target_multiplier` augmented copies of the dataset and concatenate
    with the originals, roughly tripling the dataset size (original + 2 copies).

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W) or (N, H, W, C)
    labels : np.ndarray, shape (N,)
    target_multiplier : int
        Number of *additional* augmented copies (default 2 → 3× total).
    seed : int

    Returns
    -------
    augmented_images, augmented_labels
    """
    rng = np.random.default_rng(seed)
    all_images = [images]
    all_labels = [labels]

    for copy_idx in range(target_multiplier):
        logger.info("Generating augmented copy %d / %d …", copy_idx + 1, target_multiplier)
        aug_images = np.stack(
            [augment_image(img, rng) for img in images], axis=0
        )
        all_images.append(aug_images)
        all_labels.append(labels)

    return np.concatenate(all_images, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def compute_normalization_params(images: np.ndarray) -> NormalizationParams:
    """
    Compute per-channel mean and std from a set of images.
    Images can be (N, H, W) [grayscale] or (N, H, W, C) [colour].
    """
    if images.ndim == 3:
        # Grayscale: treat as single channel
        flat = images.reshape(len(images), -1).astype(np.float64)
        mean = np.array([flat.mean()])
        std = np.array([flat.std() + 1e-8])
    elif images.ndim == 4:
        n, h, w, c = images.shape
        flat = images.reshape(n, h * w, c).astype(np.float64)
        mean = flat.mean(axis=(0, 1))          # shape (C,)
        std = flat.std(axis=(0, 1)) + 1e-8     # shape (C,)
    else:
        raise ValueError(
            f"Expected images with 3 or 4 dimensions, got {images.ndim}."
        )
    return NormalizationParams(mean=mean, std=std)


def normalize_images(images: np.ndarray, params: NormalizationParams) -> np.ndarray:
    """Apply per-channel z-score normalisation using pre-computed params."""
    imgs = images.astype(np.float64)
    if imgs.ndim == 3:
        imgs = (imgs - params.mean[0]) / params.std[0]
    else:
        imgs = (imgs - params.mean) / params.std
    return imgs


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten spatial (and channel) dimensions: (N, ...) → (N, D)."""
    return images.reshape(len(images), -1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_image_classification_pipeline(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    augmentation_copies: int = 2,
    mlp_hidden_layer_sizes: tuple[int, ...] = (256, 128),
    mlp_max_iter: int = 300,
    mlp_random_state: int = 0,
    random_state: int = 42,
) -> PipelineResult:
    """
    End-to-end image classification pipeline.

    Parameters
    ----------
    images : np.ndarray
        Array of images, shape (N, H, W) or (N, H, W, C).
    labels : np.ndarray
        1-D array of class labels (int or str), shape (N,).
    test_size : float
        Fraction of data held out for final evaluation.
    val_size : float
        Fraction of *training* data used as a validation set for early stopping.
    augmentation_copies : int
        Number of augmented copies added to training data (default 2 → 3× size).
    mlp_hidden_layer_sizes : tuple[int, ...]
        Hidden layer sizes for MLPClassifier.
    mlp_max_iter : int
        Maximum training iterations.
    mlp_random_state : int
        Random state for MLPClassifier.
    random_state : int
        Global random state for splits and augmentation.

    Returns
    -------
    PipelineResult
    """
    # ------------------------------------------------------------------
    # 0. Input validation
    # ------------------------------------------------------------------
    if not isinstance(images, np.ndarray):
        raise TypeError("images must be a numpy ndarray.")
    if not isinstance(labels, np.ndarray):
        raise TypeError("labels must be a numpy ndarray.")
    if images.ndim not in (3, 4):
        raise ValueError("images must be 3-D (N,H,W) or 4-D (N,H,W,C).")
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length.")
    if len(images) < 10:
        raise ValueError("Dataset too small (< 10 samples).")

    logger.info("Dataset: %d images, shape per image: %s", len(images), images.shape[1:])

    # ------------------------------------------------------------------
    # 1. Encode labels
    # ------------------------------------------------------------------
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # ------------------------------------------------------------------
    # 2. Train / test split BEFORE any preprocessing or augmentation
    # ------------------------------------------------------------------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        images,
        encoded_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_labels,
    )
    logger.info(
        "Split → train: %d, test: %d", len(X_train_raw), len(X_test_raw)
    )

    # ------------------------------------------------------------------
    # 3. Augment ONLY the training set
    # ------------------------------------------------------------------
    original_train_size = len(X_train_raw)
    X_train_aug, y_train_aug = augment_dataset(
        X_train_raw,
        y_train,
        target_multiplier=augmentation_copies,
        seed=random_state,
    )
    augmented_train_size = len(X_train_aug)
    aug_factor = augmented_train_size / original_train_size
    logger.info(
        "Augmentation: %d → %d samples (×%.2f)",
        original_train_size,
        augmented_train_size,
        aug_factor,
    )

    aug_stats = AugmentationStats(
        original_train_size=original_train_size,
        augmented_train_size=augmented_train_size,
        augmentation_factor=aug_factor,
        operations_applied=[
            "random_horizontal_flip",
            "random_rotation_±15°",
            "random_brightness_contrast",
            "gaussian_noise",
        ],
    )

    # ------------------------------------------------------------------
    # 4. Compute normalisation params from augmented training data ONLY
    # ------------------------------------------------------------------
    norm_params = compute_normalization_params(X_train_aug)
    logger.info("Normalisation params — mean: %s, std: %s", norm_params.mean, norm_params.std)

    # ------------------------------------------------------------------
    # 5. Normalise and flatten
    # ------------------------------------------------------------------
    X_train_norm = normalize_images(X_train_aug, norm_params)
    X_test_norm = normalize_images(X_test_raw, norm_params)

    X_train_flat = flatten_images(X_train_norm)
    X_test_flat = flatten_images(X_test_norm)

    # ------------------------------------------------------------------
    # 6. Validation split (from augmented training data) for early stopping
    #    The test set is NOT touched here.
    # ------------------------------------------------------------------
    val_abs = max(1, int(len(X_train_flat) * val_size))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_flat,
        y_train_aug,
        test_size=val_abs,
        random_state=random_state,
        stratify=y_train_aug,
    )
    logger.info(
        "Inner split → fit: %d, validation: %d", len(X_tr), len(X_val)
    )

    # ------------------------------------------------------------------
    # 7. Train MLP with early stopping on the validation set
    # ------------------------------------------------------------------
    mlp = MLPClassifier(
        hidden_layer_sizes=mlp_hidden_layer_sizes,
        max_iter=mlp_max_iter,
        early_stopping=True,
        validation_fraction=0.0,   # we supply our own val set via warm-start trick
        n_iter_no_change=15,
        random_state=mlp_random_state,
        verbose=False,
    )

    # sklearn's MLPClassifier early_stopping uses an internal split when
    # validation_fraction > 0.  To use our own val set we pass it as the
    # training data and rely on n_iter_no_change with the internal split
    # set to a tiny fraction.  Alternatively, we do manual early stopping:
    best_val_f1 = -1.0
    best_weights: list[np.ndarray] | None = None
    best_intercepts: list[np.ndarray] | None = None
    patience = 15
    no_improve = 0

    # Use partial_fit for manual early stopping
    classes = np.unique(y_tr)
    mlp_manual = MLPClassifier(
        hidden_layer_sizes=mlp_hidden_layer_sizes,
        max_iter=1,
        warm_start=True,
        random_state=mlp_random_state,
        verbose=False,
    )

    for epoch in range(mlp_max_iter):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp_manual.fit(X_tr, y_tr)   # warm_start=True continues training

        val_preds = mlp_manual.predict(X_val)
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1 = val_f1
            best_weights = [w.copy() for w in mlp_manual.coefs_]
            best_intercepts = [b.copy() for b in mlp_manual.intercepts_]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (best val macro-F1=%.4f)", epoch, best_val_f1)
            break

    # Restore best weights
    if best_weights is not None:
        mlp_manual.coefs_ = best_weights
        mlp_manual.intercepts_ = best_intercepts

    logger.info("Training complete. Best validation macro-F1: %.4f", best_val_f1)

    # ------------------------------------------------------------------
    # 8. Final evaluation on the held-out test set (only once)
    # ------------------------------------------------------------------
    y_pred = mlp_manual.predict(X_test_flat)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics: dict[str, Any] = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
        "class_names": le.classes_.tolist(),
        "best_val_macro_f1": float(best_val_f1),
    }

    logger.info("Test accuracy: %.4f | Test macro-F1: %.4f", acc, macro_f1)
    logger.info("Confusion matrix:\n%s", cm)

    return PipelineResult(
        model=mlp_manual,
        label_encoder=le,
        norm_params=norm_params,
        augmentation_stats=aug_stats,
        metrics=metrics,
        X_test_flat=X_test_flat,
        y_test=y_test,
    )


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict(
    model: MLPClassifier,
    label_encoder: LabelEncoder,
    norm_params: NormalizationParams,
    images: np.ndarray,
) -> np.ndarray:
    """
    Run inference on new images using a trained pipeline.

    Parameters
    ----------
    model : MLPClassifier
    label_encoder : LabelEncoder
    norm_params : NormalizationParams
    images : np.ndarray, shape (N, H, W) or (N, H, W, C)

    Returns
    -------
    np.ndarray of decoded class labels, shape (N,)
    """
    if not isinstance(images, np.ndarray):
        raise TypeError("images must be a numpy ndarray.")
    norm = normalize_images(images, norm_params)
    flat = flatten_images(norm)
    encoded_preds = model.predict(flat)
    return label_encoder.inverse_transform(encoded_preds)


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Synthetic dataset: 200 grayscale 32×32 images, 5 classes
    N, H, W, num_classes = 200, 32, 32, 5
    synthetic_images = rng.integers(0, 256, size=(N, H, W), dtype=np.uint8)
    synthetic_labels = rng.integers(0, num_classes, size=N)

    result = build_image_classification_pipeline(
        images=synthetic_images,
        labels=synthetic_labels,
        test_size=0.2,
        val_size=0.1,
        augmentation_copies=2,
        mlp_hidden_layer_sizes=(128, 64),
        mlp_max_iter=50,
    )

    print("\n=== Pipeline Results ===")
    print(f"Accuracy      : {result.metrics['accuracy']:.4f}")
    print(f"Macro F1      : {result.metrics['macro_f1']:.4f}")
    print(f"Best Val F1   : {result.metrics['best_val_macro_f1']:.4f}")
    print(f"Augmentation  : {result.augmentation_stats.original_train_size} → "
          f"{result.augmentation_stats.augmented_train_size} "
          f"(×{result.augmentation_stats.augmentation_factor:.2f})")
    print(f"Norm mean     : {