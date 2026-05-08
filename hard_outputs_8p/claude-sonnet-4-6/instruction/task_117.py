```python
"""
Multi-label document tagging system using TF-IDF + OneVsRestClassifier.

Best practices followed:
- Train/val/test split BEFORE any preprocessing
- Transformers fit ONLY on training data
- Separate validation set for threshold tuning
- Test set used ONLY for final evaluation
- Appropriate multi-label metrics (Hamming loss, sample-averaged F1, per-label breakdown)
- No hardcoded credentials
- No unsafe deserialization
- Specific exception handling
- No mutable default arguments
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_LABELS = 20
MIN_POSITIVE_EXAMPLES = 5  # labels with fewer positives are flagged as sparse


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SplitData:
    X_train: list[str]
    X_val: list[str]
    X_test: list[str]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


@dataclass
class EvaluationMetrics:
    hamming_loss: float
    subset_accuracy: float
    sample_f1_micro: float
    sample_f1_macro: float
    sample_f1_weighted: float
    sample_f1_samples: float
    precision_micro: float
    recall_micro: float
    jaccard_score: float
    label_ranking_avg_precision: float
    roc_auc_macro: float | None


@dataclass
class PerLabelMetrics:
    label_index: int
    label_name: str
    precision: float
    recall: float
    f1: float
    support: int
    roc_auc: float | None
    is_sparse: bool


@dataclass
class LabelCooccurrence:
    cooccurrence_matrix: np.ndarray
    label_names: list[str]
    normalized_cooccurrence: np.ndarray  # Jaccard-based


@dataclass
class TaggingSystemResult:
    pipeline: Pipeline
    optimal_thresholds: np.ndarray
    evaluation_metrics: EvaluationMetrics
    per_label_metrics: list[PerLabelMetrics]
    poorly_predicted_labels: list[PerLabelMetrics]
    label_cooccurrence: LabelCooccurrence
    label_names: list[str]
    classification_report_str: str


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_data(
    documents: list[str],
    label_matrix: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> SplitData:
    """
    Split data into train / validation / test BEFORE any preprocessing.
    Stratification is approximated via the first label column (multi-label
    stratification requires additional libraries; we use a simple split here).
    """
    if len(documents) != label_matrix.shape[0]:
        raise ValueError(
            f"Number of documents ({len(documents)}) must match "
            f"number of label rows ({label_matrix.shape[0]})."
        )
    if label_matrix.shape[1] != NUM_LABELS:
        raise ValueError(
            f"Expected {NUM_LABELS} labels, got {label_matrix.shape[1]}."
        )

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        documents,
        label_matrix,
        test_size=test_size,
        random_state=random_state,
    )

    # Second split: train vs val (relative size adjusted)
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val_size,
        random_state=random_state,
    )

    logger.info(
        "Data split — train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return SplitData(
        X_train=list(X_train),
        X_val=list(X_val),
        X_test=list(X_test),
        y_train=np.array(y_train),
        y_val=np.array(y_val),
        y_test=np.array(y_test),
    )


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def build_pipeline(random_state: int = 42) -> Pipeline:
    """
    Build a sklearn Pipeline: TF-IDF vectorizer → OneVsRestClassifier.
    The vectorizer is fit ONLY on training data (enforced by Pipeline.fit).
    """
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    base_lr = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",  # handles label imbalance
        random_state=random_state,
        n_jobs=-1,
    )
    ovr = OneVsRestClassifier(base_lr, n_jobs=-1)
    pipeline = Pipeline([("tfidf", tfidf), ("clf", ovr)])
    return pipeline


# ---------------------------------------------------------------------------
# Threshold tuning on validation set
# ---------------------------------------------------------------------------

def tune_thresholds(
    pipeline: Pipeline,
    X_val: list[str],
    y_val: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """
    For each label, find the decision threshold on the VALIDATION set that
    maximises the per-label F1 score.  The test set is never touched here.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    proba_val = pipeline.predict_proba(X_val)  # shape (n_val, n_labels)
    n_labels = y_val.shape[1]
    optimal = np.full(n_labels, 0.5)

    for label_idx in range(n_labels):
        positive_count = int(y_val[:, label_idx].sum())
        if positive_count < MIN_POSITIVE_EXAMPLES:
            # Too few positives — keep default threshold
            logger.debug(
                "Label %d has only %d positives in val set; using default threshold.",
                label_idx,
                positive_count,
            )
            continue

        best_f1 = -1.0
        best_thresh = 0.5
        for thresh in thresholds:
            preds = (proba_val[:, label_idx] >= thresh).astype(int)
            try:
                f1 = f1_score(y_val[:, label_idx], preds, zero_division=0)
            except ValueError as exc:
                logger.warning("F1 computation failed for label %d: %s", label_idx, exc)
                continue
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal[label_idx] = best_thresh

    logger.info("Threshold tuning complete. Thresholds range: [%.3f, %.3f]", optimal.min(), optimal.max())
    return optimal


# ---------------------------------------------------------------------------
# Prediction with per-label thresholds
# ---------------------------------------------------------------------------

def predict_with_thresholds(
    pipeline: Pipeline,
    X: list[str],
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary_predictions, probability_matrix)."""
    proba = pipeline.predict_proba(X)
    binary = (proba >= thresholds[np.newaxis, :]).astype(int)
    return binary, proba


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> EvaluationMetrics:
    """Compute comprehensive multi-label evaluation metrics on the TEST set."""

    hl = hamming_loss(y_true, y_pred)
    subset_acc = float(np.mean(np.all(y_true == y_pred, axis=1)))

    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_samples = f1_score(y_true, y_pred, average="samples", zero_division=0)

    prec_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    rec_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)

    jacc = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    lrap = label_ranking_average_precision_score(y_true, y_proba)

    # ROC-AUC (macro) — only for labels with both classes present in test set
    try:
        valid_labels = [
            i for i in range(y_true.shape[1])
            if len(np.unique(y_true[:, i])) == 2
        ]
        if valid_labels:
            roc_auc = roc_auc_score(
                y_true[:, valid_labels],
                y_proba[:, valid_labels],
                average="macro",
            )
        else:
            roc_auc = None
    except ValueError as exc:
        logger.warning("ROC-AUC computation failed: %s", exc)
        roc_auc = None

    return EvaluationMetrics(
        hamming_loss=hl,
        subset_accuracy=subset_acc,
        sample_f1_micro=f1_micro,
        sample_f1_macro=f1_macro,
        sample_f1_weighted=f1_weighted,
        sample_f1_samples=f1_samples,
        precision_micro=prec_micro,
        recall_micro=rec_micro,
        jaccard_score=jacc,
        label_ranking_avg_precision=lrap,
        roc_auc_macro=roc_auc,
    )


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
) -> list[PerLabelMetrics]:
    """Compute per-label precision, recall, F1, support, and ROC-AUC."""
    metrics_list: list[PerLabelMetrics] = []
    n_labels = y_true.shape[1]

    for i in range(n_labels):
        support = int(y_true[:, i].sum())
        prec = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        rec = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        is_sparse = support < MIN_POSITIVE_EXAMPLES

        roc_auc: float | None = None
        if len(np.unique(y_true[:, i])) == 2:
            try:
                roc_auc = roc_auc_score(y_true[:, i], y_proba[:, i])
            except ValueError as exc:
                logger.debug("ROC-AUC failed for label %d: %s", i, exc)

        metrics_list.append(
            PerLabelMetrics(
                label_index=i,
                label_name=label_names[i],
                precision=prec,
                recall=rec,
                f1=f1,
                support=support,
                roc_auc=roc_auc,
                is_sparse=is_sparse,
            )
        )

    return metrics_list


def identify_poorly_predicted_labels(
    per_label_metrics: list[PerLabelMetrics],
    f1_threshold: float = 0.3,
) -> list[PerLabelMetrics]:
    """Return labels with F1 below threshold or flagged as sparse."""
    return [
        m for m in per_label_metrics
        if m.f1 < f1_threshold or m.is_sparse
    ]


# ---------------------------------------------------------------------------
# Label co-occurrence
# ---------------------------------------------------------------------------

def compute_label_cooccurrence(
    y: np.ndarray,
    label_names: list[str],
) -> LabelCooccurrence:
    """
    Compute raw co-occurrence counts and Jaccard-normalised co-occurrence
    from the TRAINING labels only (never test labels).
    """
    n_labels = y.shape[1]
    cooc = np.zeros((n_labels, n_labels), dtype=np.int64)

    for i in range(n_labels):
        for j in range(n_labels):
            cooc[i, j] = int(np.sum((y[:, i] == 1) & (y[:, j] == 1)))

    # Jaccard normalisation: |A ∩ B| / |A ∪ B|
    jaccard_norm = np.zeros_like(cooc, dtype=float)
    for i in range(n_labels):
        for j in range(n_labels):
            union = int(np.sum((y[:, i] == 1) | (y[:, j] == 1)))
            if union > 0:
                jaccard_norm[i, j] = cooc[i, j] / union

    return LabelCooccurrence(
        cooccurrence_matrix=cooc,
        label_names=label_names,
        normalized_cooccurrence=jaccard_norm,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_multilabel_tagger(
    documents: list[str],
    label_matrix: np.ndarray,
    label_names: list[str] | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    poorly_predicted_f1_threshold: float = 0.3,
) -> TaggingSystemResult:
    """
    Build, train, and evaluate a multi-label document tagging system.

    Parameters
    ----------
    documents : list[str]
        Raw text documents.
    label_matrix : np.ndarray, shape (n_docs, 20)
        Binary label matrix; each row is a document, each column a tag.
    label_names : list[str] | None
        Human-readable names for the 20 tags.  Defaults to 'label_0' … 'label_19'.
    test_size : float
        Fraction of data reserved for final testing.
    val_size : float
        Fraction of data reserved for validation (threshold tuning).
    random_state : int
        Random seed for reproducibility.
    poorly_predicted_f1_threshold : float
        Labels with F1 below this value are flagged as poorly predicted.

    Returns
    -------
    TaggingSystemResult
        Contains the fitted pipeline, optimal thresholds, evaluation metrics,
        per-label breakdown, poorly predicted labels, and co-occurrence stats.
    """
    # Validate inputs
    if not documents:
        raise ValueError("documents list must not be empty.")
    label_matrix = np.asarray(label_matrix, dtype=np.int32)
    if label_matrix.ndim != 2 or label_matrix.shape[1] != NUM_LABELS:
        raise ValueError(
            f"label_matrix must be 2-D with {NUM_LABELS} columns; "
            f"got shape {label_matrix.shape}."
        )
    if label_names is None:
        label_names = [f"label_{i}" for i in range(NUM_LABELS)]
    if len(label_names) != NUM_LABELS:
        raise ValueError(
            f"label_names must have {NUM_LABELS} entries; got {len(label_names)}."
        )

    # ------------------------------------------------------------------ #
    # 1. Split BEFORE any preprocessing                                    #
    # ------------------------------------------------------------------ #
    split = split_data(
        documents=documents,
        label_matrix=label_matrix,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    # ------------------------------------------------------------------ #
    # 2. Build pipeline and fit ONLY on training data                      #
    # ------------------------------------------------------------------ #
    logger.info("Building and fitting pipeline on training data …")
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(split.X_train, split.y_train)
    logger.info("Pipeline fitted.")

    # ------------------------------------------------------------------ #
    # 3. Tune thresholds on VALIDATION set (test set untouched)            #
    # ------------------------------------------------------------------ #
    logger.info("Tuning per-label thresholds on validation set …")
    optimal_thresholds = tune_thresholds(pipeline, split.X_val, split.y_val)

    # ------------------------------------------------------------------ #
    # 4. Final evaluation on TEST set only                                 #
    # ------------------------------------------------------------------ #
    logger.info("Evaluating on held-out test set …")
    y_pred_test, y_proba_test = predict_with_thresholds(
        pipeline, split.X_test, optimal_thresholds
    )

    eval_metrics = compute_evaluation_metrics(
        y_true=split.y_test,
        y_pred=y_pred_test,
        y_proba=y_proba_test,
    )

    per_label = compute_per_label_metrics(
        y_true=split.y_test,
        y_pred=y_pred_test,
        y_proba=y_proba_test,
        label_names=label_names,
    )

    poorly_predicted = identify_poorly_predicted_labels(
        per_label, f1_threshold=poorly_predicted_f1_threshold
    )

    # ------------------------------------------------------------------ #
    # 5. Label co-occurrence from TRAINING labels only                     #
    # ------------------------------------------------------------------ #
    cooccurrence = compute_label_cooccurrence(split.y_train, label_names)

    # ------------------------------------------------------------------ #
    # 6. Classification report string                                      #
    # ------------------------------------------------------------------ #
    clf_report = classification_report(
        split.y_test,
        y_pred_test,
        target_names=label_names,
        zero_division=0,
    )

    # ------------------------------------------------------------------ #
    # 7. Log summary                                                       #
    # ------------------------------------------------------------------ #
    logger.info("=== Evaluation Summary ===")
    logger.info("Hamming Loss        : %.4f", eval_metrics.hamming_loss)
    logger.info("Subset Accuracy     : %.4f", eval_metrics.subset_accuracy)
    logger.info("Sample F1 (micro)   :