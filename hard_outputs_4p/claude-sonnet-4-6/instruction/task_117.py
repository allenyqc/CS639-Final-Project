```python
"""
Multi-label document tagging system.

Follows strict train/test separation: all transformers are fit only on the
training partition; the test set is used exclusively for final evaluation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

NUM_LABELS = 20


@dataclass
class PerLabelMetrics:
    label_index: int
    label_name: str
    precision: float
    recall: float
    f1: float
    support: int          # positive examples in test set
    train_support: int    # positive examples in training set
    roc_auc: float | None
    is_poorly_predicted: bool


@dataclass
class TaggingResult:
    model: Pipeline
    vectorizer: TfidfVectorizer          # fitted on train only
    label_names: list[str]
    # ---- global metrics ----
    hamming_loss: float
    subset_accuracy: float
    micro_f1: float
    macro_f1: float
    weighted_f1: float
    micro_precision: float
    micro_recall: float
    micro_roc_auc: float | None
    jaccard_micro: float
    # ---- per-label breakdown ----
    per_label_metrics: list[PerLabelMetrics]
    poorly_predicted_labels: list[str]
    # ---- label co-occurrence (train set) ----
    label_cooccurrence: pd.DataFrame
    # ---- raw predictions ----
    y_test: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray | None
    # ---- split info ----
    train_size: int
    test_size: int
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    documents: list[str],
    label_matrix: np.ndarray,
    label_names: list[str] | None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Basic shape / type validation."""
    if len(documents) != label_matrix.shape[0]:
        raise ValueError(
            f"Number of documents ({len(documents)}) must match "
            f"number of rows in label_matrix ({label_matrix.shape[0]})."
        )
    if label_matrix.shape[1] != NUM_LABELS:
        raise ValueError(
            f"label_matrix must have exactly {NUM_LABELS} columns; "
            f"got {label_matrix.shape[1]}."
        )
    if label_matrix.dtype not in (np.int32, np.int64, np.float32, np.float64, bool):
        label_matrix = label_matrix.astype(int)
    if label_names is None:
        label_names = [f"tag_{i:02d}" for i in range(NUM_LABELS)]
    if len(label_names) != NUM_LABELS:
        raise ValueError(
            f"label_names must have {NUM_LABELS} entries; got {len(label_names)}."
        )
    return documents, label_matrix, label_names


def _compute_label_cooccurrence(
    y_train: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    """Compute label co-occurrence matrix from the training set only."""
    y = y_train.astype(float)
    cooc = y.T @ y                          # shape (L, L)
    df = pd.DataFrame(cooc, index=label_names, columns=label_names)
    return df


def _per_label_breakdown(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    y_train: np.ndarray,
    label_names: list[str],
    poorly_predicted_threshold: float = 0.4,
) -> tuple[list[PerLabelMetrics], list[str]]:
    """Compute per-label precision, recall, F1, support, and ROC-AUC."""
    metrics_list: list[PerLabelMetrics] = []
    poorly_predicted: list[str] = []

    for idx, name in enumerate(label_names):
        yt = y_test[:, idx]
        yp = y_pred[:, idx]
        train_support = int(y_train[:, idx].sum())
        test_support = int(yt.sum())

        if test_support == 0:
            # No positive examples in test — metrics are undefined
            metrics_list.append(
                PerLabelMetrics(
                    label_index=idx,
                    label_name=name,
                    precision=float("nan"),
                    recall=float("nan"),
                    f1=float("nan"),
                    support=0,
                    train_support=train_support,
                    roc_auc=None,
                    is_poorly_predicted=True,
                )
            )
            poorly_predicted.append(name)
            continue

        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)

        roc_auc: float | None = None
        if y_prob is not None and len(np.unique(yt)) > 1:
            try:
                roc_auc = roc_auc_score(yt, y_prob[:, idx])
            except Exception:
                roc_auc = None

        is_poor = f1 < poorly_predicted_threshold
        if is_poor:
            poorly_predicted.append(name)

        metrics_list.append(
            PerLabelMetrics(
                label_index=idx,
                label_name=name,
                precision=prec,
                recall=rec,
                f1=f1,
                support=test_support,
                train_support=train_support,
                roc_auc=roc_auc,
                is_poorly_predicted=is_poor,
            )
        )

    return metrics_list, poorly_predicted


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def build_multilabel_tagger(
    documents: list[str],
    label_matrix: np.ndarray,
    label_names: list[str] | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    # TF-IDF hyper-parameters (tuned on training data only)
    tfidf_max_features: int = 30_000,
    tfidf_ngram_range: tuple[int, int] = (1, 2),
    tfidf_min_df: int = 2,
    tfidf_sublinear_tf: bool = True,
    # Logistic Regression hyper-parameters
    lr_C: float = 1.0,
    lr_max_iter: int = 1_000,
    lr_solver: str = "lbfgs",
    lr_class_weight: str | None = "balanced",
    # Poorly-predicted threshold
    poorly_predicted_f1_threshold: float = 0.4,
) -> TaggingResult:
    """
    Build, train, and evaluate a multi-label document tagging system.

    Parameters
    ----------
    documents:
        Raw text documents.
    label_matrix:
        Binary matrix of shape (n_docs, 20).
    label_names:
        Optional list of 20 tag names.
    test_size:
        Fraction of data held out for final evaluation.
    random_state:
        Reproducibility seed.
    tfidf_*:
        TF-IDF vectorizer settings.
    lr_*:
        LogisticRegression settings.
    poorly_predicted_f1_threshold:
        Labels with F1 below this value are flagged as poorly predicted.

    Returns
    -------
    TaggingResult dataclass with model, metrics, per-label breakdown, and
    label co-occurrence statistics.
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    documents, label_matrix, label_names = _validate_inputs(
        documents, label_matrix, label_names
    )
    label_matrix = label_matrix.astype(int)

    # ------------------------------------------------------------------
    # 2. Train / test split — BEFORE any preprocessing
    # ------------------------------------------------------------------
    (
        docs_train, docs_test,
        y_train, y_test,
    ) = train_test_split(
        documents,
        label_matrix,
        test_size=test_size,
        random_state=random_state,
        # stratify on the number of labels per document (coarse stratification)
        stratify=None,   # multi-label stratification requires extra deps
    )

    train_size = len(docs_train)
    test_size_actual = len(docs_test)

    # ------------------------------------------------------------------
    # 3. Fit TF-IDF vectorizer on TRAINING data only
    # ------------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=tfidf_ngram_range,
        min_df=tfidf_min_df,
        sublinear_tf=tfidf_sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(docs_train)   # fit ONLY on train
    X_test = vectorizer.transform(docs_test)          # transform test

    # ------------------------------------------------------------------
    # 4. Handle labels with very few positive training examples
    #    Labels with 0 training positives cannot be learned; we track them
    #    but still include them in the OvR classifier (it will predict 0).
    # ------------------------------------------------------------------
    train_label_counts = y_train.sum(axis=0)
    zero_support_labels = [
        label_names[i] for i, c in enumerate(train_label_counts) if c == 0
    ]
    if zero_support_labels:
        warnings.warn(
            f"Labels with zero training examples (will always predict 0): "
            f"{zero_support_labels}",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 5. Build and train the classifier
    # ------------------------------------------------------------------
    base_lr = LogisticRegression(
        C=lr_C,
        max_iter=lr_max_iter,
        solver=lr_solver,
        class_weight=lr_class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    ovr = OneVsRestClassifier(base_lr, n_jobs=-1)
    ovr.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Predict on TEST set (never used for fitting)
    # ------------------------------------------------------------------
    y_pred = ovr.predict(X_test)

    y_prob: np.ndarray | None = None
    try:
        y_prob = ovr.predict_proba(X_test)
    except AttributeError:
        pass

    # ------------------------------------------------------------------
    # 7. Global evaluation metrics
    # ------------------------------------------------------------------
    hl = hamming_loss(y_test, y_pred)
    subset_acc = accuracy_score(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    micro_prec = precision_score(y_test, y_pred, average="micro", zero_division=0)
    micro_rec = recall_score(y_test, y_pred, average="micro", zero_division=0)
    jac_micro = jaccard_score(y_test, y_pred, average="micro", zero_division=0)

    micro_roc_auc: float | None = None
    if y_prob is not None:
        try:
            # Only compute if at least one label has both classes in test
            valid_cols = [
                i for i in range(NUM_LABELS)
                if len(np.unique(y_test[:, i])) > 1
            ]
            if valid_cols:
                micro_roc_auc = roc_auc_score(
                    y_test[:, valid_cols],
                    y_prob[:, valid_cols],
                    average="micro",
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 8. Per-label breakdown
    # ------------------------------------------------------------------
    per_label_metrics, poorly_predicted_labels = _per_label_breakdown(
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        y_train=y_train,
        label_names=label_names,
        poorly_predicted_threshold=poorly_predicted_f1_threshold,
    )

    # ------------------------------------------------------------------
    # 9. Label co-occurrence (training set only)
    # ------------------------------------------------------------------
    cooc_df = _compute_label_cooccurrence(y_train, label_names)

    # ------------------------------------------------------------------
    # 10. Wrap everything in a Pipeline for convenient inference
    # ------------------------------------------------------------------
    # We store vectorizer separately (already fitted) and expose a
    # lightweight Pipeline-like wrapper for predict convenience.
    # The Pipeline below is NOT re-fitted; it is assembled post-hoc.
    inference_pipeline = Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("clf", ovr),
        ]
    )
    # Mark the pipeline as already fitted by setting _is_fitted attribute
    # (sklearn >= 1.0 checks this via __sklearn_is_fitted__)
    inference_pipeline._is_fitted = True  # type: ignore[attr-defined]

    return TaggingResult(
        model=inference_pipeline,
        vectorizer=vectorizer,
        label_names=label_names,
        hamming_loss=hl,
        subset_accuracy=subset_acc,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        micro_precision=micro_prec,
        micro_recall=micro_rec,
        micro_roc_auc=micro_roc_auc,
        jaccard_micro=jac_micro,
        per_label_metrics=per_label_metrics,
        poorly_predicted_labels=poorly_predicted_labels,
        label_cooccurrence=cooc_df,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        train_size=train_size,
        test_size=test_size_actual,
        extra={
            "zero_support_train_labels": zero_support_labels,
            "train_label_counts": dict(zip(label_names, train_label_counts.tolist())),
        },
    )


# ---------------------------------------------------------------------------
# Convenience: predict tags for new documents
# ---------------------------------------------------------------------------

def predict_tags(
    result: TaggingResult,
    new_documents: list[str],
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Predict tags for new (unseen) documents using a fitted TaggingResult.

    Parameters
    ----------
    result:
        A TaggingResult returned by build_multilabel_tagger.
    new_documents:
        List of raw text strings.
    threshold:
        If provided, apply this probability threshold instead of the
        classifier's default decision boundary.  Only used when
        predict_proba is available.

    Returns
    -------
    dict with keys:
        'binary_predictions'  – (n_docs, 20) int array
        'probabilities'       – (n_docs, 20) float array or None
        'tag_lists'           – list of lists of tag name strings
    """
    pipeline = result.model
    binary_preds: np.ndarray
    probs: np.ndarray | None = None

    try:
        probs = pipeline.predict_proba(new_documents)
        if threshold is not None:
            binary_preds = (probs >= threshold).astype(int)
        else:
            binary_preds = pipeline.predict(new_documents)
    except AttributeError:
        binary_preds = pipeline.predict(new_documents)

    tag_lists = [
        [result.label_names[j] for j in range(NUM_LABELS) if binary_preds[i, j] == 1]
        for i in range(len(new_documents))
    ]

    return {
        "binary_predictions": binary_preds,
        "probabilities": probs,
        "tag_lists": tag_lists,
    }


# ---------------------------------------------------------------------------
# Convenience: pretty-print a summary
# ---------------------------------------------------------------------------

def print_summary(result: TaggingResult) -> None:
    """Print a human-readable summary of the TaggingResult."""
    sep = "=" * 60
    print(sep)
    print("MULTI-LABEL DOCUMENT TAGGING — EVALUATION SUMMARY")
    print(sep)
    print(f"  Train size        : {result.train_size}")
    print(f"  Test  size        : {result.test_size}")
    print(f"  Number of labels  : {NUM_LABELS}")
    print()
    print("  GLOBAL METRICS (test set)")
    print(f"    Hamming loss      : {result.hamming_loss:.4f}")
    print(f"    Subset accuracy   : {result.subset_accuracy:.4f}")
    print(f"    Micro F1          : {result.micro_f1:.4f}")
    print(f"    Macro F1          : {result.macro_f1:.4f}")
    print(f"    Weighted F1       : {result.weighted_f1:.4f}")
    print(f"    Micro Precision   : {result.micro_precision:.4f}")
    print(f"    Micro Recall      : {result.micro_recall:.4f}")
    print(f"    Micro Jaccard     : {result.jaccard_micro:.4f}")
    if result.micro_roc_auc is not None:
        print(f"    Micro ROC-AUC     : {result.micro_roc_auc:.4f}")
    print()
    print("  PER-LABEL BREAKDOWN")
    header = f"  {'Label':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} "
    header += f"{'AUC':>6} {'Test+':>6} {'Train+':>7} {'Poor?':>6}"
    print(header)
    print("  " + "-" * 70)
    for m in result.per_label_metrics:
        prec_s = f"{m.precision:.3f}" if not np.isnan(m.precision) else "  N/A"
        rec_s = f"{m.recall:.3f}" if not np.isnan(m.recall) else "  N/A"
        f1_s = f"{m.f1:.3f}" if not np.isnan(m.f1) else "  N/A"
        auc_s = f"{m.roc_auc:.3f}" if m.roc_auc is not None else "  N/A"
        poor_s = "  YES" if m.is_poorly_predicted else "   no"
        print(
            f"  {m.label_name:<20} {prec_s:>6} {rec_s:>6} {f1_s:>6} "
            f"{auc_s:>6} {m.support:>6} {m.train_support:>7