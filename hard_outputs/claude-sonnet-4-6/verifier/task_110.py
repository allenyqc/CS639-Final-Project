"""
Image Classification Pipeline with Data Augmentation
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip image horizontally with probability p."""
    if np.random.random() < p:
        return image[:, ::-1, :].copy() if image.ndim == 3 else image[:, ::-1].copy()
    return image.copy()


def _random_rotation(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Rotate image by a random angle in [-max_angle, +max_angle] degrees."""
    angle = np.random.uniform(-max_angle, max_angle)
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    out = np.zeros_like(image)

    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xs_c = xs - cx
    ys_c = ys - cy

    src_x = cos_a * xs_c + sin_a * ys_c + cx
    src_y = -sin_a * xs_c + cos_a * ys_c + cy

    src_xi = np.round(src_x).astype(int)
    src_yi = np.round(src_y).astype(int)

    valid = (src_xi >= 0) & (src_xi < w) & (src_yi >= 0) & (src_yi < h)
    if image.ndim == 3:
        out[ys[valid], xs[valid]] = image[src_yi[valid], src_xi[valid]]
    else:
        out[ys[valid], xs[valid]] = image[src_yi[valid], src_xi[valid]]

    return out


def _random_brightness_contrast(
    image: np.ndarray,
    brightness_delta: float = 0.2,
    contrast_delta: float = 0.2,
) -> np.ndarray:
    """Randomly adjust brightness and contrast."""
    img = image.astype(np.float32)
    b = np.random.uniform(-brightness_delta, brightness_delta) * 255.0
    img = img + b
    c = np.random.uniform(1.0 - contrast_delta, 1.0 + contrast_delta)
    mean_val = img.mean()
    img = (img - mean_val) * c + mean_val
    return np.clip(img, 0, 255).astype(image.dtype)


def _add_gaussian_noise(
    image: np.ndarray, mean: float = 0.0, std: float = 10.0
) -> np.ndarray:
    """Add Gaussian noise to the image."""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(image.dtype)


def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply a full chain of augmentations to a single image."""
    img = _random_horizontal_flip(image)
    img = _random_rotation(img, max_angle=15.0)
    img = _random_brightness_contrast(img, brightness_delta=0.2, contrast_delta=0.2)
    img = _add_gaussian_noise(img, mean=0.0, std=10.0)
    return img


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def compute_normalization_params(
    images: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std from a batch of images.

    Parameters
    ----------
    images : np.ndarray
        Shape (N, H, W, C) for colour or (N, H, W) for grayscale.

    Returns
    -------
    mean : np.ndarray  shape (C,) or (1,) for grayscale
    std  : np.ndarray  shape (C,) or (1,) for grayscale
    """
    if images.ndim == 4:
        axes = (0, 1, 2)
        mean = images.mean(axis=axes)
        std = images.std(axis=axes)
    else:
        mean = np.array([images.mean()])
        std = np.array([images.std()])
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_images(
    images: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Normalize images using pre-computed per-channel mean and std."""
    imgs = images.astype(np.float32)
    if imgs.ndim == 4:
        imgs = (imgs - mean[None, None, None, :]) / std[None, None, None, :]
    else:
        imgs = (imgs - mean[0]) / std[0]
    return imgs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_image_classification_pipeline(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    augmentation_factor: int = 2,
    mlp_hidden_layer_sizes: Tuple[int, ...] = (256, 128),
    mlp_max_iter: int = 300,
    mlp_learning_rate_init: float = 1e-3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Build an image classification pipeline with data augmentation.

    The pipeline strictly follows the correct ML protocol:
      1. Split raw data into train / test.
      2. Fit all statistics (label encoder, normalisation) on the training
         split only.
      3. Augment the training split only.
      4. Normalise both splits using training-set statistics.
      5. Train the MLP on the augmented, normalised training set.
      6. Evaluate on the held-out test set.

    Parameters
    ----------
    images : np.ndarray
        Array of images with shape (N, H, W, C) or (N, H, W).
        Pixel values should be in [0, 255].
    labels : np.ndarray
        1-D array of integer or string labels, length N.
    test_size : float
        Fraction of data reserved for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.
    augmentation_factor : int
        Number of *additional* augmented copies per original training image.
        Setting this to 2 triples the training set size (1 original + 2 copies).
    mlp_hidden_layer_sizes : tuple
        Hidden layer sizes for MLPClassifier.
    mlp_max_iter : int
        Maximum iterations for MLPClassifier.
    mlp_learning_rate_init : float
        Initial learning rate for MLPClassifier.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    result : dict with keys
        - "model"             : trained MLPClassifier
        - "label_encoder"     : LabelEncoder fitted on training labels only
        - "metrics"           : dict with accuracy, macro_f1, confusion_matrix,
                                class_names
        - "norm_params"       : dict with "mean" and "std" (per-channel,
                                computed from training set only)
        - "augmentation_stats": dict describing augmentation details
    """
    np.random.seed(random_state)

    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    images = np.array(images)
    labels = np.array(labels)

    if images.ndim not in (3, 4):
        raise ValueError(
            f"images must be 3-D (N, H, W) or 4-D (N, H, W, C), "
            f"got shape {images.shape}"
        )
    if len(images) != len(labels):
        raise ValueError(
            f"images and labels must have the same length, "
            f"got {len(images)} and {len(labels)}"
        )

    n_original = len(images)

    if verbose:
        print(f"[Pipeline] Dataset: {n_original} images, shape {images.shape[1:]}")

    # ------------------------------------------------------------------
    # 2. Train / test split on raw data BEFORE any fitting or augmentation
    # ------------------------------------------------------------------
    X_train_orig, X_test, y_train_raw, y_test_raw = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    if verbose:
        print(
            f"[Pipeline] Split → train: {len(X_train_orig)}, "
            f"test: {len(X_test)} (before augmentation)"
        )

    # ------------------------------------------------------------------
    # 3. Fit LabelEncoder on training labels ONLY
    # ------------------------------------------------------------------
    le = LabelEncoder()
    y_train_orig = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    if verbose:
        print(f"[Pipeline] Classes (from training set): {list(le.classes_)}")

    # ------------------------------------------------------------------
    # 4. Compute normalisation parameters from training set ONLY
    # ------------------------------------------------------------------
    norm_mean, norm_std = compute_normalization_params(X_train_orig)

    if verbose:
        print(f"[Pipeline] Per-channel mean (train): {norm_mean}")
        print(f"[Pipeline] Per-channel std  (train): {norm_std}")

    # ------------------------------------------------------------------
    # 5. Augment training set ONLY
    # ------------------------------------------------------------------
    aug_image_batches = [X_train_orig]
    aug_label_batches = [y_train_orig]

    aug_counts = {
        "horizontal_flips": 0,
        "rotations": 0,
        "brightness_contrast": 0,
        "gaussian_noise": 0,
    }

    for copy_idx in range(augmentation_factor):
        augmented_batch = []
        for img in X_train_orig:
            aug_img = augment_image(img)
            augmented_batch.append(aug_img)
            # All four transforms are always applied inside augment_image
            aug_counts["horizontal_flips"] += 1
            aug_counts["rotations"] += 1
            aug_counts["brightness_contrast"] += 1
            aug_counts["gaussian_noise"] += 1

        aug_image_batches.append(np.array(augmented_batch))
        aug_label_batches.append(y_train_orig)

        if verbose:
            print(
                f"[Pipeline] Augmentation copy {copy_idx + 1}/{augmentation_factor} "
                f"complete ({len(augmented_batch)} images added)"
            )

    X_train_aug = np.concatenate(aug_image_batches, axis=0)
    y_train_aug = np.concatenate(aug_label_batches, axis=0)

    augmentation_stats = {
        "original_train_size": len(X_train_orig),
        "augmented_train_size": len(X_train_aug),
        "augmentation_factor": augmentation_factor + 1,   # original + copies
        "augmentation_counts_per_copy": aug_counts,
        "total_augmented_images": len(X_train_aug) - len(X_train_orig),
    }

    if verbose:
        print(
            f"[Pipeline] Training set after augmentation: {len(X_train_aug)} images "
            f"(~{len(X_train_aug) / len(X_train_orig):.1f}x original)"
        )

    # ------------------------------------------------------------------
    # 6. Normalise using training-set statistics ONLY
    #    (test set is normalised with the same training statistics)
    # ------------------------------------------------------------------
    X_train_norm = normalize_images(X_train_aug, norm_mean, norm_std)
    X_test_norm = normalize_images(X_test, norm_mean, norm_std)

    # ------------------------------------------------------------------
    # 7. Flatten for MLP input
    # ------------------------------------------------------------------
    X_train_flat = X_train_norm.reshape(len(X_train_norm), -1)
    X_test_flat = X_test_norm.reshape(len(X_test_norm), -1)

    if verbose:
        print(f"[Pipeline] Flattened feature size: {X_train_flat.shape[1]}")

    # ------------------------------------------------------------------
    # 8. Train MLP on augmented training data ONLY
    # ------------------------------------------------------------------
    if verbose:
        print(
            f"[Pipeline] Training MLPClassifier "
            f"hidden_layers={mlp_hidden_layer_sizes}, max_iter={mlp_max_iter} ..."
        )

    mlp = MLPClassifier(
        hidden_layer_sizes=mlp_hidden_layer_sizes,
        activation="relu",
        solver="adam",
        learning_rate_init=mlp_learning_rate_init,
        max_iter=mlp_max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False,
    )
    mlp.fit(X_train_flat, y_train_aug)

    if verbose:
        print(f"[Pipeline] Training complete. Iterations: {mlp.n_iter_}")

    # ------------------------------------------------------------------
    # 9. Evaluate on held-out test set
    # ------------------------------------------------------------------
    y_pred = mlp.predict(X_test_flat)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
        "class_names": list(le.classes_),
    }

    if verbose:
        print("\n[Pipeline] === Evaluation Results ===")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Macro F1 : {macro_f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

    # ------------------------------------------------------------------
    # 10. Return all artefacts
    # ------------------------------------------------------------------
    return {
        "model": mlp,
        "label_encoder": le,
        "metrics": metrics,
        "norm_params": {
            "mean": norm_mean,
            "std": norm_std,
        },
        "augmentation_stats": augmentation_stats,
    }


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict(
    model: MLPClassifier,
    label_encoder: LabelEncoder,
    norm_params: Dict[str, np.ndarray],
    images: np.ndarray,
) -> np.ndarray:
    """
    Run inference on new images using a trained pipeline.

    Parameters
    ----------
    model : MLPClassifier
        Trained classifier returned by build_image_classification_pipeline.
    label_encoder : LabelEncoder
        Fitted label encoder returned by build_image_classification_pipeline.
    norm_params : dict
        Dict with "mean" and "std" keys returned by
        build_image_classification_pipeline.
    images : np.ndarray
        New images with the same spatial shape as the training images.

    Returns
    -------
    predictions : np.ndarray
        Decoded class labels.
    """
    images = np.array(images)
    imgs_norm = normalize_images(images, norm_params["mean"], norm_params["std"])
    imgs_flat = imgs_norm.reshape(len(imgs_norm), -1)
    preds_enc = model.predict(imgs_flat)
    return label_encoder.inverse_transform(preds_enc)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

def _demo():
    """Generate synthetic data and run the full pipeline as a smoke test."""
    print("=" * 60)
    print("Running smoke test with synthetic data ...")
    print("=" * 60)

    rng = np.random.default_rng(0)
    n_samples = 200
    H, W, C = 16, 16, 3

    # Two easily separable classes: bright vs dark images
    images_class0 = rng.integers(150, 255, size=(n_samples // 2, H, W, C), dtype=np.uint8)
    images_class1 = rng.integers(0, 100, size=(n_samples // 2, H, W, C), dtype=np.uint8)
    images = np.concatenate([images_class0, images_class1], axis=0)
    labels = np.array(["bright"] * (n_samples // 2) + ["dark"] * (n_samples // 2))

    result = build_image_classification_pipeline(
        images=images,
        labels=labels,
        test_size=0.2,
        random_state=42,
        augmentation_factor=2,
        mlp_hidden_layer_sizes=(64, 32),
        mlp_max_iter=200,
        verbose=True,
    )

    print("\n[Demo] Augmentation stats:")
    for k, v in result["augmentation_stats"].items():
        print(f"  {k}: {v}")

    # Test inference helper
    sample_images = images[:5]
    preds = predict(
        result["model"],
        result["label_encoder"],
        result["norm_params"],
        sample_images,
    )
    print(f"\n[Demo] Sample predictions : {preds}")
    print(f"[Demo] True labels        : {labels[:5]}")

    return result


if __name__ == "__main__":
    _demo()