```python
"""
Multi-label document tagging system using TF-IDF and OneVsRestClassifier.
"""

import numpy as np
import warnings
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    hamming_loss,
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    coverage_error,
    label_ranking_average_precision_score,
    label_ranking_loss,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
import scipy.sparse as sp

warnings.filterwarnings("ignore")

NUM_TAGS = 20


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_metric(func, y_true, y_pred, **kwargs):
    """Call a metric function and return NaN on failure."""
    try:
        return float(func(y_true, y_pred, **kwargs))
    except Exception:
        return float("nan")


def _per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: list) -> dict:
    """
    Compute per-label precision, recall, F1, support, and flag poorly predicted labels.

    A label is considered 'poorly predicted' if:
      - F1 < 0.30, OR
      - fewer than 5 positive training examples exist (rare label).
    """
    n_labels = y_true.shape[1]
    per_label = {}

    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        support = int(yt.sum())

        if support == 0:
            per_label[name] = {
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "support": 0,
                "poorly_predicted": True,
                "reason": "no positive examples in test set",
            }
            continue

        p = _safe_metric(precision_score, yt, yp, zero_division=0)
        r = _safe_metric(recall_score, yt, yp, zero_division=0)
        f = _safe_metric(f1_score, yt, yp, zero_division=0)

        poorly = f < 0.30 or support < 5
        reason = []
        if f < 0.30:
            reason.append(f"low F1 ({f:.3f})")
        if support < 5:
            reason.append(f"rare label (support={support})")

        per_label[name] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": support,
            "poorly_predicted": poorly,
            "reason": "; ".join(reason) if reason else "ok",
        }

    return per_label


def _label_cooccurrence(y: np.ndarray, label_names: list) -> dict:
    """
    Compute label co-occurrence statistics.

    Returns:
        co_matrix  : raw co-occurrence count matrix (n_labels × n_labels)
        co_prob    : conditional probability P(label_j | label_i)
        top_pairs  : top-10 most co-occurring label pairs (excluding diagonal)
    """
    if sp.issparse(y):
        y = y.toarray()

    co_matrix = (y.T @ y).astype(int)  # shape (n_labels, n_labels)

    # Conditional probability: P(j | i) = co(i,j) / count(i)
    counts = co_matrix.diagonal().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        co_prob = np.where(counts[:, None] > 0, co_matrix / counts[:, None], 0.0)

    # Top co-occurring pairs (upper triangle, excluding diagonal)
    n = co_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((co_matrix[i, j], label_names[i], label_names[j]))
    pairs.sort(reverse=True)
    top_pairs = [
        {"label_a": a, "label_b": b, "count": int(c)} for c, a, b in pairs[:10]
    ]

    return {
        "co_matrix": co_matrix,
        "co_prob": co_prob,
        "label_names": label_names,
        "top_pairs": top_pairs,
    }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_multilabel_tagger(
    documents: list,
    label_matrix: np.ndarray,
    label_names: Optional[list] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    tfidf_max_features: int = 20_000,
    tfidf_ngram_range: tuple = (1, 2),
    lr_C: float = 1.0,
    lr_max_iter: int = 1_000,
    min_label_freq: int = 2,
) -> dict:
    """
    Build, train, and evaluate a multi-label document tagging system.

    Parameters
    ----------
    documents       : list of raw text strings (corpus).
    label_matrix    : binary ndarray of shape (n_docs, n_labels).
                      Each row is a document; each column is a tag.
    label_names     : optional list of tag names (length == n_labels).
                      Defaults to ["tag_0", "tag_1", ...].
    test_size       : fraction of data reserved for evaluation.
    random_state    : random seed for reproducibility.
    tfidf_max_features : vocabulary size cap for TF-IDF.
    tfidf_ngram_range  : n-gram range for TF-IDF.
    lr_C            : regularisation strength for LogisticRegression.
    lr_max_iter     : max iterations for LogisticRegression solver.
    min_label_freq  : labels with fewer than this many positives in the
                      *training* set are handled with class_weight='balanced'.

    Returns
    -------
    dict with keys:
        model            : fitted sklearn Pipeline
        vectorizer       : fitted TfidfVectorizer (also inside pipeline)
        evaluation       : overall evaluation metrics dict
        per_label        : per-label performance breakdown dict
        poorly_predicted : list of label names flagged as poorly predicted
        cooccurrence     : label co-occurrence statistics dict
        label_names      : list of tag names used
        splits           : dict with train/test indices and label matrices
    """
    # ------------------------------------------------------------------
    # 0. Validate inputs
    # ------------------------------------------------------------------
    documents = list(documents)
    label_matrix = np.asarray(label_matrix, dtype=int)

    n_docs, n_labels = label_matrix.shape
    if n_docs != len(documents):
        raise ValueError(
            f"Number of documents ({len(documents)}) must match "
            f"number of rows in label_matrix ({n_docs})."
        )
    if n_labels != NUM_TAGS:
        raise ValueError(
            f"label_matrix must have exactly {NUM_TAGS} columns; got {n_labels}."
        )

    if label_names is None:
        label_names = [f"tag_{i}" for i in range(n_labels)]
    if len(label_names) != n_labels:
        raise ValueError("len(label_names) must equal number of label columns.")

    # ------------------------------------------------------------------
    # 1. Train / test split  (stratify on most-common label for balance)
    # ------------------------------------------------------------------
    indices = np.arange(n_docs)
    # Use the most frequent label as stratification proxy
    most_common_label = int(label_matrix.sum(axis=0).argmax())
    stratify_col = label_matrix[:, most_common_label]

    # If stratification is impossible (all same class), skip it
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
        )

    X_train_raw = [documents[i] for i in train_idx]
    X_test_raw = [documents[i] for i in test_idx]
    y_train = label_matrix[train_idx]
    y_test = label_matrix[test_idx]

    # ------------------------------------------------------------------
    # 2. TF-IDF vectorisation
    # ------------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=tfidf_ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]*\b",
        min_df=2,
    )

    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # ------------------------------------------------------------------
    # 3. Build OneVsRestClassifier with LogisticRegression
    #    Use class_weight='balanced' for rare labels
    # ------------------------------------------------------------------
    train_label_counts = y_train.sum(axis=0)  # shape (n_labels,)
    rare_labels = set(
        np.where(train_label_counts < min_label_freq)[0].tolist()
    )

    # We use a single OvR classifier; to handle rare labels we use
    # class_weight='balanced' globally (it helps rare classes most).
    base_lr = LogisticRegression(
        C=lr_C,
        max_iter=lr_max_iter,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
        multi_class="ovr",
    )
    ovr = OneVsRestClassifier(base_lr, n_jobs=-1)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    ovr.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Predict
    # ------------------------------------------------------------------
    y_pred = ovr.predict(X_test)
    # Decision scores for ranking-based metrics
    try:
        y_scores = ovr.predict_proba(X_test)
    except AttributeError:
        y_scores = ovr.decision_function(X_test)

    # ------------------------------------------------------------------
    # 6. Overall evaluation metrics
    # ------------------------------------------------------------------
    evaluation = {}

    # Hamming loss (fraction of wrong labels)
    evaluation["hamming_loss"] = _safe_metric(hamming_loss, y_test, y_pred)

    # Jaccard similarity (averaged over samples)
    evaluation["jaccard_score_samples"] = _safe_metric(
        jaccard_score, y_test, y_pred, average="samples", zero_division=0
    )
    evaluation["jaccard_score_macro"] = _safe_metric(
        jaccard_score, y_test, y_pred, average="macro", zero_division=0
    )

    # F1 variants
    for avg in ("micro", "macro", "weighted", "samples"):
        evaluation[f"f1_{avg}"] = _safe_metric(
            f1_score, y_test, y_pred, average=avg, zero_division=0
        )

    # Precision / Recall
    for avg in ("micro", "macro", "weighted"):
        evaluation[f"precision_{avg}"] = _safe_metric(
            precision_score, y_test, y_pred, average=avg, zero_division=0
        )
        evaluation[f"recall_{avg}"] = _safe_metric(
            recall_score, y_test, y_pred, average=avg, zero_division=0
        )

    # Ranking-based metrics (require probability scores)
    evaluation["coverage_error"] = _safe_metric(coverage_error, y_test, y_scores)
    evaluation["label_ranking_avg_precision"] = _safe_metric(
        label_ranking_average_precision_score, y_test, y_scores
    )
    evaluation["label_ranking_loss"] = _safe_metric(
        label_ranking_loss, y_test, y_scores
    )

    # Exact match ratio (subset accuracy)
    evaluation["exact_match_ratio"] = float(
        np.mean(np.all(y_pred == y_test, axis=1))
    )

    # Average number of predicted labels per document
    evaluation["avg_predicted_labels"] = float(y_pred.sum(axis=1).mean())
    evaluation["avg_true_labels"] = float(y_test.sum(axis=1).mean())

    # ------------------------------------------------------------------
    # 7. Per-label performance
    # ------------------------------------------------------------------
    per_label = _per_label_metrics(y_test, y_pred, label_names)
    poorly_predicted = [
        name for name, stats in per_label.items() if stats["poorly_predicted"]
    ]

    # ------------------------------------------------------------------
    # 8. Label co-occurrence (computed on full dataset)
    # ------------------------------------------------------------------
    cooccurrence = _label_cooccurrence(label_matrix, label_names)

    # ------------------------------------------------------------------
    # 9. Wrap vectorizer + classifier in a Pipeline for easy reuse
    # ------------------------------------------------------------------
    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", ovr),
        ]
    )
    # The vectorizer is already fitted; mark pipeline as fitted by
    # calling fit on empty data is not needed — we just store it.

    # ------------------------------------------------------------------
    # 10. Return everything
    # ------------------------------------------------------------------
    return {
        "model": pipeline,
        "vectorizer": vectorizer,
        "classifier": ovr,
        "evaluation": evaluation,
        "per_label": per_label,
        "poorly_predicted": poorly_predicted,
        "cooccurrence": cooccurrence,
        "label_names": label_names,
        "splits": {
            "train_indices": train_idx,
            "test_indices": test_idx,
            "y_train": y_train,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_scores": y_scores,
        },
        "rare_labels": [label_names[i] for i in sorted(rare_labels)],
    }


# ---------------------------------------------------------------------------
# Convenience: predict on new documents using a fitted result dict
# ---------------------------------------------------------------------------

def predict_tags(
    result: dict,
    new_documents: list,
    threshold: Optional[float] = None,
) -> dict:
    """
    Predict tags for new documents using a fitted result from build_multilabel_tagger.

    Parameters
    ----------
    result        : dict returned by build_multilabel_tagger.
    new_documents : list of raw text strings.
    threshold     : if provided, use this probability threshold instead of
                    the default 0.5 decision boundary.

    Returns
    -------
    dict with keys:
        binary_predictions : ndarray (n_docs, n_labels)
        probabilities      : ndarray (n_docs, n_labels)
        tag_lists          : list of lists of predicted tag names per document
    """
    pipeline = result["model"]
    label_names = result["label_names"]

    vectorizer = result["vectorizer"]
    clf = result["classifier"]

    X = vectorizer.transform(new_documents)

    try:
        probs = clf.predict_proba(X)
    except AttributeError:
        probs = clf.decision_function(X)

    if threshold is not None:
        binary = (probs >= threshold).astype(int)
    else:
        binary = clf.predict(X)

    tag_lists = [
        [label_names[j] for j in range(len(label_names)) if binary[i, j] == 1]
        for i in range(len(new_documents))
    ]

    return {
        "binary_predictions": binary,
        "probabilities": probs,
        "tag_lists": tag_lists,
    }


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _generate_synthetic_corpus(n_docs: int = 500, n_labels: int = 20, seed: int = 0):
    """Generate a tiny synthetic corpus for testing."""
    rng = np.random.default_rng(seed)

    # Vocabulary per tag
    tag_vocab = {
        i: [f"word_{i}_{k}" for k in range(20)] for i in range(n_labels)
    }
    common_words = [f"common_{k}" for k in range(30)]

    documents = []
    for _ in range(n_docs):
        words = list(rng.choice(common_words, size=10, replace=True))
        # Add tag-specific words
        active_tags = rng.integers(0, n_labels, size=rng.integers(1, 5))
        for t in active_tags:
            words += list(rng.choice(tag_vocab[t], size=5, replace=True))
        rng.shuffle(words)
        documents.append(" ".join(words))

    # Binary label matrix with ~20% density, some rare labels
    label_matrix = rng.binomial(1, 0.15, size=(n_docs, n_labels))
    # Make labels 18 and 19 very rare
    label_matrix[:, 18] = rng.binomial(1, 0.02, size=n_docs)
    label_matrix[:, 19] = rng.binomial(1, 0.01, size=n_docs)

    return documents, label_matrix


if __name__ == "__main__":
    print("Generating synthetic corpus …")
    docs, labels = _generate_synthetic_corpus(n_docs=600)

    tag_names = [f"tag_{i:02d}" for i in range(NUM_TAGS)]

    print(f"Corpus size : {len(docs)} documents")
    print(f"Label shape : {labels.shape}")
    print(f"Label density: {labels.mean():.3f}")
    print()

    result = build_multilabel_tagger(
        documents=docs,
        label_matrix=labels,
        label_names=tag_names,
        test_size=0.20,
        random_state=42,
    )

    print("=" * 60)
    print("OVERALL EVALUATION METRICS")
    print("=" * 60)
    for k, v in result["evaluation"].items():
        print(f"  {k:<40s}: {v:.4f}")

    print()
    print("=" * 60)
    print("PER-LABEL PERFORMANCE")
    print("=" * 60)
    print(f"  {'Label':<12} {'P':>6} {'R':>6} {'F1':>6} {'Sup':>5}  Status")
    print("  " + "-" * 55)
    for name, stats in result["per_label"].items():
        p = f"{stats['precision']:.3f}" if not np.isnan(stats["precision"]) else "  nan"
        r = f"{stats['recall']:.3f}" if not np.isnan(stats["recall"]) else "  nan"
        f = f"{stats['f1']:.3f}" if not np.isnan(stats["f1"]) else "  nan"
        flag = "⚠ " + stats["reason"] if stats["poorly_predicted"] else "✓"
        print(f"  {name:<12} {p:>6} {r:>6} {f:>6} {stats['support']:>5}  {flag}")

    print()
    print("=" * 60)
    print(f"POORLY PREDICTED LABELS ({len(result['poorly_predicted'])})")
    print("=" * 60)
    for name in result["poorly_predicted"]:
        print(f"  • {name}: {result['per_label'][name]['reason']}")

    print()
    print("=" * 60)
    print("RARE LABELS (< 2 positive training examples)")
    print("=" * 60)
    print(" ", result["rare_labels"] or "none")

    print()
    print("=" * 60)
    print("TOP-10 LABEL CO-OCCURRENCES")
    print("=" * 60)
    for pair in result["cooccurrence