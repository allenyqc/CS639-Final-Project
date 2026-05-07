```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler


def train_and_optimize_threshold(
    X=None,
    y=None,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    logistic_regression_kwargs=None,
):
    """
    Trains a logistic regression classifier, finds the classification threshold
    that maximizes F1-score on a validation set, and reports the final test F1
    at that threshold.

    Parameters
    ----------
    X : array-like, optional
        Feature matrix. If None, a synthetic dataset is generated.
    y : array-like, optional
        Target vector. If None, a synthetic dataset is generated.
    test_size : float
        Proportion of data to use as the test set.
    val_size : float
        Proportion of training data to use as the validation set.
    random_state : int
        Random seed for reproducibility.
    logistic_regression_kwargs : dict, optional
        Additional keyword arguments passed to LogisticRegression.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'model': trained LogisticRegression model
        - 'scaler': fitted StandardScaler
        - 'best_threshold': threshold that maximizes validation F1
        - 'val_f1_at_best_threshold': validation F1 at best threshold
        - 'test_f1_default_threshold': test F1 at default threshold (0.5)
        - 'test_f1_optimized_threshold': test F1 at optimized threshold
        - 'all_thresholds': array of evaluated thresholds
        - 'all_val_f1_scores': array of validation F1 scores per threshold
    """
    if logistic_regression_kwargs is None:
        logistic_regression_kwargs = {"max_iter": 1000, "random_state": random_state}

    # Generate synthetic data if none provided
    if X is None or y is None:
        print("No data provided. Generating synthetic classification dataset...")
        X, y = make_classification(
            n_samples=5000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            weights=[0.7, 0.3],  # Imbalanced classes to make threshold tuning meaningful
            random_state=random_state,
        )

    X = np.array(X)
    y = np.array(y)

    # Split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    print(f"Dataset splits:")
    print(f"  Train size : {len(X_train)} samples")
    print(f"  Val size   : {len(X_val)} samples")
    print(f"  Test size  : {len(X_test)} samples")
    print(f"  Class distribution (train): {np.bincount(y_train)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(**logistic_regression_kwargs)
    model.fit(X_train_scaled, y_train)
    print("\nLogistic Regression model trained successfully.")

    # Get predicted probabilities on validation set
    val_probs = model.predict_proba(X_val_scaled)[:, 1]

    # Use precision_recall_curve to get candidate thresholds efficiently
    precisions, recalls, pr_thresholds = precision_recall_curve(y_val, val_probs)

    # Compute F1 for each threshold (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precisions[:-1] + recalls[:-1]) > 0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
            0.0,
        )

    # Also evaluate a fine-grained grid of thresholds for completeness
    grid_thresholds = np.linspace(0.01, 0.99, 199)
    grid_f1_scores = np.array(
        [
            f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
            for t in grid_thresholds
        ]
    )

    # Combine PR-curve thresholds and grid thresholds
    all_thresholds = np.concatenate([pr_thresholds, grid_thresholds])
    all_f1_scores = np.concatenate([f1_scores, grid_f1_scores])

    # Find best threshold
    best_idx = np.argmax(all_f1_scores)
    best_threshold = all_thresholds[best_idx]
    best_val_f1 = all_f1_scores[best_idx]

    print(f"\nThreshold Optimization (on validation set):")
    print(f"  Evaluated {len(all_thresholds)} thresholds")
    print(f"  Best threshold : {best_threshold:.4f}")
    print(f"  Best val F1    : {best_val_f1:.4f}")

    # Evaluate on test set
    test_probs = model.predict_proba(X_test_scaled)[:, 1]

    # Default threshold (0.5)
    test_preds_default = (test_probs >= 0.5).astype(int)
    test_f1_default = f1_score(y_test, test_preds_default, zero_division=0)

    # Optimized threshold
    test_preds_optimized = (test_probs >= best_threshold).astype(int)
    test_f1_optimized = f1_score(y_test, test_preds_optimized, zero_division=0)

    print(f"\nTest Set Results:")
    print(f"  F1 at default threshold (0.50)          : {test_f1_default:.4f}")
    print(f"  F1 at optimized threshold ({best_threshold:.4f})  : {test_f1_optimized:.4f}")
    print(
        f"  Improvement                              : {test_f1_optimized - test_f1_default:+.4f}"
    )

    results = {
        "model": model,
        "scaler": scaler,
        "best_threshold": best_threshold,
        "val_f1_at_best_threshold": best_val_f1,
        "test_f1_default_threshold": test_f1_default,
        "test_f1_optimized_threshold": test_f1_optimized,
        "all_thresholds": all_thresholds,
        "all_val_f1_scores": all_f1_scores,
    }

    return results


def plot_threshold_vs_f1(results):
    """
    Plots validation F1 score as a function of classification threshold.

    Parameters
    ----------
    results : dict
        Output dictionary from train_and_optimize_threshold().
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plot.")
        return

    thresholds = results["all_thresholds"]
    f1_scores = results["all_val_f1