"""
Image Classification Pipeline with Data Augmentation
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip image horizontally with probability p."""
    if np.random.random() < p:
        return image[:, ::-1, :] if image.ndim == 3 else image[:, ::-1]
    return image.copy()


def random_rotation(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Rotate image by a random angle in [-max_angle, +max_angle] degrees."""
    from scipy.ndimage import rotate as scipy_rotate
    angle = np.random.uniform(-max_angle, max_angle)
    if image.ndim == 3:
        rotated = scipy_rotate(image, angle, axes=(0, 1), reshape=False, mode='reflect')
    else:
        rotated = scipy_rotate(image, angle, reshape=False, mode='reflect')
    return rotated.astype(image.dtype)


def random_brightness_contrast(
    image: np.ndarray,
    brightness_range: tuple = (-0.2, 0.2),
    contrast_range: tuple = (0.8, 1.2),
) -> np.ndarray:
    """
    Adjust brightness (additive) and contrast (multiplicative).
    Assumes pixel values in [0, 255] or [0, 1]; clips to original range.
    """
    img = image.astype(np.float64)
    orig_max = 255.0 if image.max() > 1.0 else 1.0

    # Contrast: scale around mean
    alpha = np.random.uniform(*contrast_range)
    mean_val = img.mean()
    img = mean_val + alpha * (img - mean_val)

    # Brightness: additive shift
    beta = np.random.uniform(*brightness_range) * orig_max
    img = img + beta

    img = np.clip(img, 0, orig_max)
    return img.astype(image.dtype)


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    std_fraction: float = 0.05,
) -> np.ndarray:
    """Add Gaussian noise scaled by std_fraction * pixel_range."""
    img = image.astype(np.float64)
    pixel_range = 255.0 if image.max() > 1.0 else 1.0
    noise = np.random.normal(mean, std_fraction * pixel_range, image.shape)
    img = np.clip(img + noise, 0, pixel_range)
    return img.astype(image.dtype)


def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply the full augmentation pipeline to a single image."""
    img = random_horizontal_flip(image)
    img = random_rotation(img, max_angle=15.0)
    img = random_brightness_contrast(img)
    img = add_gaussian_noise(img)
    return img


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_image_classification_pipeline(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    augmentation_factor: int = 2,          # extra copies → total ≈ 3×
    mlp_hidden_layer_sizes: tuple = (256, 128),
    mlp_max_iter: int = 300,
    mlp_learning_rate_init: float = 1e-3,
) -> dict:
    """
    Build an image classification pipeline with data augmentation.

    Parameters
    ----------
    images : np.ndarray, shape (N, H, W) or (N, H, W, C)
        Raw pixel images (uint8 or float).
    labels : np.ndarray, shape (N,)
        Class labels (int or str).
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Seed for reproducibility.
    augmentation_factor : int
        Number of *additional* augmented copies per original training image.
        With factor=2 the training set is roughly tripled.
    mlp_hidden_layer_sizes : tuple
        Hidden layer sizes for MLPClassifier.
    mlp_max_iter : int
        Maximum training iterations.
    mlp_learning_rate_init : float
        Initial learning rate.

    Returns
    -------
    dict with keys:
        model               – trained MLPClassifier
        label_encoder       – fitted LabelEncoder
        normalization       – {'mean': ..., 'std': ...} per-channel stats
        metrics             – {'accuracy', 'f1_macro', 'confusion_matrix'}
        augmentation_stats  – counts and factor info
    """
    np.random.seed(random_state)

    images = np.array(images)
    labels = np.array(labels)

    if images.ndim == 3:                    # (N, H, W) → (N, H, W, 1)
        images = images[..., np.newaxis]

    N, H, W, C = images.shape
    print(f"[Pipeline] Dataset: {N} images, shape {H}×{W}×{C}, "
          f"{len(np.unique(labels))} classes.")

    # ------------------------------------------------------------------
    # 1. Train / test split (on original data, before augmentation)
    # ------------------------------------------------------------------
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        images, labels_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_enc,
    )
    print(f"[Pipeline] Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

    # ------------------------------------------------------------------
    # 2. Data augmentation (training set only)
    # ------------------------------------------------------------------
    aug_images_list = [X_train_raw]
    aug_labels_list = [y_train]

    for copy_idx in range(augmentation_factor):
        aug_batch = np.stack([augment_image(img) for img in X_train_raw])
        aug_images_list.append(aug_batch)
        aug_labels_list.append(y_train)
        print(f"[Augmentation] Generated copy {copy_idx + 1}/{augmentation_factor} "
              f"({len(aug_batch)} images).")

    X_train_aug = np.concatenate(aug_images_list, axis=0)
    y_train_aug = np.concatenate(aug_labels_list, axis=0)

    # Shuffle augmented training set
    shuffle_idx = np.random.permutation(len(X_train_aug))
    X_train_aug = X_train_aug[shuffle_idx]
    y_train_aug = y_train_aug[shuffle_idx]

    aug_stats = {
        "original_train_size": len(X_train_raw),
        "augmented_train_size": len(X_train_aug),
        "augmentation_factor_approx": len(X_train_aug) / len(X_train_raw),
        "test_size": len(X_test_raw),
        "augmentation_copies": augmentation_factor,
        "techniques": [
            "random_horizontal_flip",
            "random_rotation_±15°",
            "random_brightness_contrast",
            "gaussian_noise",
        ],
    }
    print(f"[Augmentation] Training set size: "
          f"{aug_stats['original_train_size']} → {aug_stats['augmented_train_size']} "
          f"(×{aug_stats['augmentation_factor_approx']:.2f})")

    # ------------------------------------------------------------------
    # 3. Per-channel normalization statistics (from augmented training set)
    # ------------------------------------------------------------------
    # Shape: (N_aug, H, W, C) → compute mean/std per channel
    train_float = X_train_aug.astype(np.float64)
    ch_mean = train_float.mean(axis=(0, 1, 2))   # shape (C,)
    ch_std  = train_float.std(axis=(0, 1, 2))    # shape (C,)
    ch_std  = np.where(ch_std == 0, 1.0, ch_std) # avoid division by zero

    norm_params = {"mean": ch_mean, "std": ch_std}
    print(f"[Normalization] Per-channel mean: {ch_mean}, std: {ch_std}")

    def normalize(arr: np.ndarray) -> np.ndarray:
        """Normalize images using training statistics."""
        arr = arr.astype(np.float64)
        return (arr - ch_mean) / ch_std   # broadcasting over (N, H, W, C)

    # ------------------------------------------------------------------
    # 4. Normalize and flatten
    # ------------------------------------------------------------------
    X_train_norm = normalize(X_train_aug)
    X_test_norm  = normalize(X_test_raw)

    X_train_flat = X_train_norm.reshape(len(X_train_norm), -1)
    X_test_flat  = X_test_norm.reshape(len(X_test_norm), -1)

    print(f"[Pipeline] Feature vector size: {X_train_flat.shape[1]}")

    # ------------------------------------------------------------------
    # 5. Train MLP classifier
    # ------------------------------------------------------------------
    clf = MLPClassifier(
        hidden_layer_sizes=mlp_hidden_layer_sizes,
        activation='relu',
        solver='adam',
        learning_rate_init=mlp_learning_rate_init,
        max_iter=mlp_max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False,
    )
    print("[Training] Fitting MLPClassifier …")
    clf.fit(X_train_flat, y_train_aug)
    print(f"[Training] Converged after {clf.n_iter_} iterations.")

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test_flat)

    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm    = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy":         acc,
        "f1_macro":         f1,
        "confusion_matrix": cm,
    }
    print(f"[Evaluation] Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    print(f"[Evaluation] Confusion matrix:\n{cm}")

    # ------------------------------------------------------------------
    # 7. Return everything
    # ------------------------------------------------------------------
    return {
        "model":              clf,
        "label_encoder":      le,
        "normalization":      norm_params,
        "metrics":            metrics,
        "augmentation_stats": aug_stats,
    }


# ---------------------------------------------------------------------------
# Convenience: predict on new images
# ---------------------------------------------------------------------------

def predict(
    model: MLPClassifier,
    label_encoder: LabelEncoder,
    norm_params: dict,
    images: np.ndarray,
) -> np.ndarray:
    """
    Predict class labels for new images using a trained pipeline.

    Parameters
    ----------
    model         : trained MLPClassifier returned by build_image_classification_pipeline
    label_encoder : LabelEncoder returned by the pipeline
    norm_params   : normalization dict {'mean': ..., 'std': ...}
    images        : np.ndarray of shape (N, H, W) or (N, H, W, C)

    Returns
    -------
    np.ndarray of decoded class labels, shape (N,)
    """
    images = np.array(images)
    if images.ndim == 3:
        images = images[..., np.newaxis]

    imgs_float = images.astype(np.float64)
    imgs_norm  = (imgs_float - norm_params["mean"]) / norm_params["std"]
    imgs_flat  = imgs_norm.reshape(len(imgs_norm), -1)

    preds_enc = model.predict(imgs_flat)
    return label_encoder.inverse_transform(preds_enc)


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate synthetic data: 200 images, 16×16 RGB, 4 classes
    rng = np.random.default_rng(0)
    N_demo, H_demo, W_demo, C_demo = 200, 16, 16, 3
    n_classes = 4

    demo_images = rng.integers(0, 256, (N_demo, H_demo, W_demo, C_demo), dtype=np.uint8)
    demo_labels = rng.integers(0, n_classes, N_demo)

    result = build_image_classification_pipeline(
        images=demo_images,
        labels=demo_labels,
        test_size=0.2,
        random_state=42,
        augmentation_factor=2,
        mlp_hidden_layer_sizes=(128, 64),
        mlp_max_iter=200,
    )

    print("\n=== Pipeline Results ===")
    print(f"Accuracy : {result['metrics']['accuracy']:.4f}")
    print(f"Macro-F1 : {result['metrics']['f1_macro']:.4f}")
    print(f"Augmentation stats: {result['augmentation_stats']}")
    print(f"Norm mean: {result['normalization']['mean']}")
    print(f"Norm std : {result['normalization']['std']}")

    # Demo prediction
    sample = demo_images[:5]
    preds = predict(
        result["model"],
        result["label_encoder"],
        result["normalization"],
        sample,
    )
    print(f"\nSample predictions: {preds}")