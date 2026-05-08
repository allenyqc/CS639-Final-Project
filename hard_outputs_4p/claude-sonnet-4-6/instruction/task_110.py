```python
"""
image_classification_pipeline.py

A machine learning pipeline for image classification with data augmentation.
Follows best practices: train/test split before preprocessing, fit transformers
only on training data, and evaluate only on held-out test set.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data classes for structured returns
# ---------------------------------------------------------------------------

@dataclass
class NormalizationParams:
    """Per-channel mean and std computed ONLY from training data."""
    mean: np.ndarray          # shape (C,) or scalar for grayscale
    std: np.ndarray           # shape (C,) or scalar for grayscale
    n_channels: int


@dataclass
class AugmentationStats:
    """Statistics about the augmentation process."""
    original_train_size: int
    augmented_train_size: int
    augmentation_factor: float
    operations_applied: list = field(default_factory=list)
    noise_std: float = 0.02
    rotation_range_deg: float = 15.0
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)


@dataclass
class PipelineResult:
    """Complete result of the classification pipeline."""
    model: MLPClassifier
    scaler: StandardScaler
    metrics: Dict[str, Any]
    normalization_params: NormalizationParams
    augmentation_stats: AugmentationStats
    train_size: int
    test_size: int


# ---------------------------------------------------------------------------
# Augmentation helpers (operate on single images, values in [0, 1])
# ---------------------------------------------------------------------------

def _to_float(image: np.ndarray) -> np.ndarray:
    """Convert image to float32 in [0, 1]."""
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def _clip(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)


def random_horizontal_flip(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip image horizontally with probability 0.5."""
    if rng.random() < 0.5:
        return np.flip(image, axis=1)
    return image.copy()


def random_rotation(
    image: np.ndarray,
    rng: np.random.Generator,
    max_angle_deg: float = 15.0,
) -> np.ndarray:
    """
    Rotate image by a random angle in [-max_angle_deg, +max_angle_deg].
    Uses a simple affine rotation via index mapping (no scipy dependency).
    """
    angle_deg = rng.uniform(-max_angle_deg, max_angle_deg)
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    h, w = image.shape[:2]
    cy, cx = h / 2.0, w / 2.0

    # Build output coordinate grid
    ys, xs = np.mgrid[0:h, 0:w]
    # Translate to center
    ys_c = ys - cy
    xs_c = xs - cx
    # Inverse rotation
    src_y = cos_a * ys_c + sin_a * xs_c + cy
    src_x = -sin_a * ys_c + cos_a * xs_c + cx

    # Nearest-neighbor sampling
    src_y = np.round(src_y).astype(int)
    src_x = np.round(src_x).astype(int)

    valid = (src_y >= 0) & (src_y < h) & (src_x >= 0) & (src_x < w)

    if image.ndim == 3:
        out = np.zeros_like(image)
        out[valid] = image[src_y[valid], src_x[valid]]
    else:
        out = np.zeros_like(image)
        out[valid] = image[src_y[valid], src_x[valid]]

    return out


def random_brightness_contrast(
    image: np.ndarray,
    rng: np.random.Generator,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """
    Randomly adjust brightness (additive scale) and contrast (multiplicative
    around the mean).
    """
    img = image.copy()
    # Brightness: multiply by random factor
    b_factor = rng.uniform(*brightness_range)
    img = img * b_factor

    # Contrast: scale around channel mean
    c_factor = rng.uniform(*contrast_range)
    if img.ndim == 3:
        mean = img.mean(axis=(0, 1), keepdims=True)
    else:
        mean = img.mean()
    img = mean + c_factor * (img - mean)

    return _clip(img)


def add_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Add zero-mean Gaussian noise."""
    noise = rng.normal(0.0, noise_std, size=image.shape).astype(np.float32)
    return _clip(image + noise)


def augment_image(
    image: np.ndarray,
    rng: np.random.Generator,
    max_angle_deg: float = 15.0,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
    noise_std: float = 0.02,
) -> np.ndarray:
    """Apply the full augmentation pipeline to a single image."""
    img = _to_float(image)
    img = random_horizontal_flip(img, rng)
    img = random_rotation(img, rng, max_angle_deg)
    img = random_brightness_contrast(img, rng, brightness_range, contrast_range)
    img = add_gaussian_noise(img, rng, noise_std)
    return img


# ---------------------------------------------------------------------------
# Dataset augmentation (roughly triples the dataset)
# ---------------------------------------------------------------------------

def augment_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    augmentation_factor: int = 2,
    seed: int = 42,
    max_angle_deg: float = 15.0,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
    noise_std: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, AugmentationStats]:
    """
    Augment the dataset by creating `augmentation_factor` additional copies
    of each image with random transformations.  The original images are kept,
    so the total size is (1 + augmentation_factor) × original_size ≈ 3×.

    Parameters
    ----------
    images : np.ndarray  shape (N, H, W) or (N, H, W, C)
    labels : np.ndarray  shape (N,)
    augmentation_factor : int  number of augmented copies per original image
    seed : int  random seed for reproducibility

    Returns
    -------
    augmented_images, augmented_labels, AugmentationStats
    """
    rng = np.random.default_rng(seed)
    original_size = len(images)

    aug_images = [_to_float(img) for img in images]  # keep originals as float
    aug_labels = list(labels)

    for _ in range(augmentation_factor):
        for img, lbl in zip(images, labels):
            aug_img = augment_image(
                img, rng,
                max_angle_deg=max_angle_deg,
                brightness_range=brightness_range,
                contrast_range=contrast_range,
                noise_std=noise_std,
            )
            aug_images.append(aug_img)
            aug_labels.append(lbl)

    aug_images_arr = np.array(aug_images, dtype=np.float32)
    aug_labels_arr = np.array(aug_labels)

    stats = AugmentationStats(
        original_train_size=original_size,
        augmented_train_size=len(aug_images_arr),
        augmentation_factor=(1 + augmentation_factor),
        operations_applied=[
            "random_horizontal_flip",
            "random_rotation",
            "random_brightness_contrast",
            "gaussian_noise",
        ],
        noise_std=noise_std,
        rotation_range_deg=max_angle_deg,
        brightness_range=brightness_range,
        contrast_range=contrast_range,
    )
    return aug_images_arr, aug_labels_arr, stats


# ---------------------------------------------------------------------------
# Normalization (fit ONLY on training data)
# ---------------------------------------------------------------------------

def compute_normalization_params(images: np.ndarray) -> NormalizationParams:
    """
    Compute per-channel mean and std from a set of images.
    images: float32 array of shape (N, H, W) or (N, H, W, C).
    """
    if images.ndim == 3:
        # Grayscale: (N, H, W)
        mean = images.mean(axis=(0, 1, 2), keepdims=False)
        std = images.std(axis=(0, 1, 2), keepdims=False)
        std = np.where(std == 0, 1.0, std)
        return NormalizationParams(
            mean=np.array([mean], dtype=np.float32),
            std=np.array([std], dtype=np.float32),
            n_channels=1,
        )
    elif images.ndim == 4:
        # Color: (N, H, W, C)
        n_channels = images.shape[-1]
        mean = images.mean(axis=(0, 1, 2))   # shape (C,)
        std = images.std(axis=(0, 1, 2))     # shape (C,)
        std = np.where(std == 0, 1.0, std)
        return NormalizationParams(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            n_channels=n_channels,
        )
    else:
        raise ValueError(f"Expected 3-D or 4-D image array, got shape {images.shape}")


def normalize_images(images: np.ndarray, params: NormalizationParams) -> np.ndarray:
    """Apply per-channel normalization using pre-computed params."""
    imgs = images.astype(np.float32)
    if imgs.ndim == 3:
        imgs = (imgs - params.mean[0]) / params.std[0]
    else:
        imgs = (imgs - params.mean[np.newaxis, np.newaxis, np.newaxis, :]) / \
               params.std[np.newaxis, np.newaxis, np.newaxis, :]
    return imgs


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten spatial (and channel) dimensions: (N, H, W[, C]) → (N, H*W*C)."""
    return images.reshape(len(images), -1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_image_classification_pipeline(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    augmentation_factor: int = 2,
    max_angle_deg: float = 15.0,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
    noise_std: float = 0.02,
    mlp_hidden_layer_sizes: Tuple[int, ...] = (256, 128),
    mlp_max_iter: int = 300,
    mlp_learning_rate_init: float = 1e-3,
    mlp_early_stopping: bool = True,
    mlp_validation_fraction: float = 0.1,
    verbose: bool = False,
) -> PipelineResult:
    """
    Build an image classification pipeline with data augmentation.

    Parameters
    ----------
    images : np.ndarray
        Array of images with shape (N, H, W) for grayscale or (N, H, W, C)
        for colour.  Pixel values may be uint8 [0, 255] or float [0, 1].
    labels : np.ndarray
        Integer class labels of shape (N,).
    test_size : float
        Fraction of data reserved for the held-out test set.
    random_state : int
        Master random seed.
    augmentation_factor : int
        Number of augmented copies per original training image.
        Total training size ≈ (1 + augmentation_factor) × original_train_size.
    max_angle_deg : float
        Maximum rotation angle in degrees (±).
    brightness_range : tuple
        (min, max) multiplicative brightness factor.
    contrast_range : tuple
        (min, max) multiplicative contrast factor.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    mlp_hidden_layer_sizes : tuple
        Hidden layer sizes for MLPClassifier.
    mlp_max_iter : int
        Maximum training iterations.
    mlp_learning_rate_init : float
        Initial learning rate.
    mlp_early_stopping : bool
        Whether to use early stopping.
    mlp_validation_fraction : float
        Fraction of training data used for early-stopping validation.
    verbose : bool
        Print progress information.

    Returns
    -------
    PipelineResult
        Contains the trained model, scaler, metrics, normalization params,
        and augmentation statistics.
    """
    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------
    if not isinstance(images, np.ndarray):
        images = np.array(images)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    if images.ndim not in (3, 4):
        raise ValueError(
            f"images must be 3-D (N, H, W) or 4-D (N, H, W, C), got {images.ndim}-D"
        )
    if len(images) != len(labels):
        raise ValueError(
            f"images and labels must have the same length: "
            f"{len(images)} vs {len(labels)}"
        )
    if len(images) < 10:
        raise ValueError("Need at least 10 samples to build a meaningful pipeline.")

    if verbose:
        print(f"[Pipeline] Dataset: {len(images)} samples, shape {images.shape[1:]}")

    # ------------------------------------------------------------------
    # 1. Train / test split BEFORE any preprocessing or augmentation
    # ------------------------------------------------------------------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        images, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )

    if verbose:
        print(f"[Pipeline] Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

    # ------------------------------------------------------------------
    # 2. Augment ONLY the training set
    # ------------------------------------------------------------------
    X_train_aug, y_train_aug, aug_stats = augment_dataset(
        X_train_raw, y_train,
        augmentation_factor=augmentation_factor,
        seed=random_state,
        max_angle_deg=max_angle_deg,
        brightness_range=brightness_range,
        contrast_range=contrast_range,
        noise_std=noise_std,
    )

    if verbose:
        print(
            f"[Pipeline] After augmentation: {len(X_train_aug)} training samples "
            f"(×{aug_stats.augmentation_factor:.1f})"
        )

    # ------------------------------------------------------------------
    # 3. Compute normalization parameters from TRAINING data only
    # ------------------------------------------------------------------
    norm_params = compute_normalization_params(X_train_aug)

    if verbose:
        print(f"[Pipeline] Normalization — mean: {norm_params.mean}, std: {norm_params.std}")

    # ------------------------------------------------------------------
    # 4. Normalize both sets using training statistics
    # ------------------------------------------------------------------
    X_train_norm = normalize_images(X_train_aug, norm_params)
    X_test_norm = normalize_images(
        _to_float_array(X_test_raw), norm_params
    )

    # ------------------------------------------------------------------
    # 5. Flatten images
    # ------------------------------------------------------------------
    X_train_flat = flatten_images(X_train_norm)
    X_test_flat = flatten_images(X_test_norm)

    if verbose:
        print(f"[Pipeline] Feature dimension after flattening: {X_train_flat.shape[1]}")

    # ------------------------------------------------------------------
    # 6. StandardScaler — fit ONLY on training data
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)   # transform only, never fit

    # ------------------------------------------------------------------
    # 7. Train MLP classifier
    # ------------------------------------------------------------------
    mlp = MLPClassifier(
        hidden_layer_sizes=mlp_hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=mlp_learning_rate_init,
        max_iter=mlp_max_iter,
        early_stopping=mlp_early_stopping,
        validation_fraction=mlp_validation_fraction,
        random_state=random_state,
        verbose=False,
    )

    if verbose:
        print("[Pipeline] Training MLP …")

    mlp.fit(X_train_scaled, y_train_aug)

    if verbose:
        print(f"[Pipeline] Training complete. Iterations: {mlp.n_iter_}")

    # ------------------------------------------------------------------
    # 8. Evaluate on the held-out TEST set ONLY
    # ------------------------------------------------------------------
    y_pred = mlp.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": conf_matrix,
        "n_test_samples": len(y_test),
        "n_train_samples": len(y_train_aug),
        "classes": np.unique(labels).tolist(),
    }

    if verbose:
        print(f"[Pipeline] Test accuracy : {accuracy:.4f}")
        print(f"[Pipeline] Test macro-F1 : {macro_f1:.4f}")
        print(f"[Pipeline] Confusion matrix:\n{conf_matrix}")

    return PipelineResult(
        model=mlp,
        scaler=scaler,
        metrics=metrics,
        normalization_params=norm_params,
        augmentation_stats=aug_stats,
        train_size=len(X_train_aug),
        test_size=len(X_test_raw),
    )


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict(
    result: PipelineResult,
    images: np.ndarray,
) -> np.ndarray:
    """
    Run inference on new images using a