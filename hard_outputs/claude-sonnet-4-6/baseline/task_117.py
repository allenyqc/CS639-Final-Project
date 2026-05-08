```python
"""
Multi-label document tagging system using TF-IDF and OneVsRestClassifier.
"""

import numpy as np
import warnings
from typing import Any

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


def _per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict:
    """Compute per-label precision, recall, F1, support, and flag sparse labels."""
    n_labels = y_true.shape[1]
    per_label: dict[str, dict] = {}

    for i, name in enumerate(label_names):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        support = int(col_true.sum())

        if support == 0:
            per_label[name] = {
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "support": 0,
                "predicted_positive": int(col_pred.sum()),
                "sparse": True,
                "poorly_predicted": True,
            }
            continue

        p = _safe_metric(precision_score, col_true, col_pred, zero_division=0)
        r = _safe_metric(recall_score, col_true, col_pred, zero_division=0)
        f = _safe_metric(f1_score, col_true, col_pred, zero_division=0)

        per_label[name] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": support,
            "predicted_positive": int(col_pred.sum()),
            "sparse": support < 5,
            "poorly_predicted": f < 0.3,
        }

    return per_label


def _label_cooccurrence(y: np.ndarray, label_names: list[str]) -> dict:
    """Compute label co-occurrence matrix and statistics."""
    n_labels = y.shape[1]
    cooc_matrix = (y.T @ y).astype(int)  # shape (n_labels, n_labels)

    # Build a readable dict
    cooc_dict: dict = {
        "matrix": cooc_matrix.tolist(),
        "label_names": label_names,
        "top_pairs": [],
    }

    # Find top co-occurring pairs (off-diagonal)
    pairs = []
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            count = int(cooc_matrix[i, j])
            if count > 0:
                pairs.append((label_names[i], label_names[j], count))

    pairs.sort(key=lambda x: -x[2])
    cooc_dict["top_pairs"] = pairs[:20]  # top-20 pairs

    return cooc_dict


# ---------------------------------------------------------------------------
# Core system
# ---------------------------------------------------------------------------

def build_multilabel_tagger(
    documents: list[str],
    label_matrix: np.ndarray,
    label_names: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    tfidf_kwargs: dict | None = None,
    lr_kwargs: dict | None = None,
) -> dict[str, Any]:
    """
    Build, train, and evaluate a multi-label document tagging system.

    Parameters
    ----------
    documents : list of str
        Raw text documents.
    label_matrix : np.ndarray, shape (n_docs, 20)
        Binary matrix; entry [i, j] == 1 means document i has tag j.
    label_names : list of str, optional
        Human-readable names for the 20 tags.
    test_size : float
        Fraction of data used for evaluation.
    random_state : int
        Seed for reproducibility.
    tfidf_kwargs : dict, optional
        Extra keyword arguments forwarded to TfidfVectorizer.
    lr_kwargs : dict, optional
        Extra keyword arguments forwarded to LogisticRegression.

    Returns
    -------
    dict with keys:
        model            – trained Pipeline (TF-IDF + OneVsRestClassifier)
        evaluation       – overall evaluation metrics
        per_label        – per-label breakdown dict
        cooccurrence     – label co-occurrence statistics
        label_names      – list of tag names used
        X_test_docs      – held-out documents
        y_test           – true labels for held-out set
        y_pred           – predicted labels for held-out set
    """
    # ------------------------------------------------------------------ #
    # 0. Validate inputs
    # ------------------------------------------------------------------ #
    documents = list(documents)
    label_matrix = np.asarray(label_matrix, dtype=int)

    if label_matrix.ndim != 2 or label_matrix.shape[1] != NUM_TAGS:
        raise ValueError(
            f"label_matrix must have shape (n_docs, {NUM_TAGS}), "
            f"got {label_matrix.shape}"
        )
    if len(documents) != label_matrix.shape[0]:
        raise ValueError(
            "Number of documents must equal number of rows in label_matrix."
        )

    if label_names is None:
        label_names = [f"tag_{i:02d}" for i in range(NUM_TAGS)]
    if len(label_names) != NUM_TAGS:
        raise ValueError(f"label_names must have exactly {NUM_TAGS} entries.")

    # ------------------------------------------------------------------ #
    # 1. Train / test split  (stratify on first label as proxy)
    # ------------------------------------------------------------------ #
    indices = np.arange(len(documents))
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=label_matrix[:, 0],
        )
    except ValueError:
        # Fallback when stratification is impossible (too few positives)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
        )

    docs_train = [documents[i] for i in train_idx]
    docs_test  = [documents[i] for i in test_idx]
    y_train    = label_matrix[train_idx]
    y_test     = label_matrix[test_idx]

    # ------------------------------------------------------------------ #
    # 2. Build pipeline
    # ------------------------------------------------------------------ #
    _tfidf_kwargs = dict(
        max_features=50_000,
        sublinear_tf=True,
        min_df=1,
        ngram_range=(1, 2),
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]*\b",
    )
    if tfidf_kwargs:
        _tfidf_kwargs.update(tfidf_kwargs)

    _lr_kwargs = dict(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )
    if lr_kwargs:
        _lr_kwargs.update(lr_kwargs)

    base_lr  = LogisticRegression(**_lr_kwargs)
    ovr      = OneVsRestClassifier(base_lr, n_jobs=-1)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**_tfidf_kwargs)),
        ("clf",   ovr),
    ])

    # ------------------------------------------------------------------ #
    # 3. Handle labels with zero positive examples in training set
    # ------------------------------------------------------------------ #
    # Identify labels that have at least one positive example in training
    active_labels = np.where(y_train.sum(axis=0) > 0)[0]
    sparse_labels = np.where(y_train.sum(axis=0) == 0)[0]

    # Train on all labels; sklearn handles zero-support labels gracefully
    # when class_weight='balanced' is used (it will just predict 0 always).
    pipeline.fit(docs_train, y_train)

    # ------------------------------------------------------------------ #
    # 4. Predict
    # ------------------------------------------------------------------ #
    y_pred       = pipeline.predict(docs_test)
    y_pred_proba = None
    try:
        y_pred_proba = pipeline.predict_proba(docs_test)
    except AttributeError:
        pass

    # ------------------------------------------------------------------ #
    # 5. Overall evaluation metrics
    # ------------------------------------------------------------------ #
    evaluation: dict[str, Any] = {}

    evaluation["hamming_loss"]   = _safe_metric(hamming_loss, y_test, y_pred)
    evaluation["jaccard_macro"]  = _safe_metric(
        jaccard_score, y_test, y_pred, average="macro", zero_division=0
    )
    evaluation["jaccard_micro"]  = _safe_metric(
        jaccard_score, y_test, y_pred, average="micro", zero_division=0
    )
    evaluation["f1_macro"]       = _safe_metric(
        f1_score, y_test, y_pred, average="macro", zero_division=0
    )
    evaluation["f1_micro"]       = _safe_metric(
        f1_score, y_test, y_pred, average="micro", zero_division=0
    )
    evaluation["f1_samples"]     = _safe_metric(
        f1_score, y_test, y_pred, average="samples", zero_division=0
    )
    evaluation["precision_macro"] = _safe_metric(
        precision_score, y_test, y_pred, average="macro", zero_division=0
    )
    evaluation["precision_micro"] = _safe_metric(
        precision_score, y_test, y_pred, average="micro", zero_division=0
    )
    evaluation["recall_macro"]   = _safe_metric(
        recall_score, y_test, y_pred, average="macro", zero_division=0
    )
    evaluation["recall_micro"]   = _safe_metric(
        recall_score, y_test, y_pred, average="micro", zero_division=0
    )

    # Ranking-based metrics (require probability estimates)
    if y_pred_proba is not None:
        evaluation["coverage_error"] = _safe_metric(
            coverage_error, y_test, y_pred_proba
        )
        evaluation["label_ranking_avg_precision"] = _safe_metric(
            label_ranking_average_precision_score, y_test, y_pred_proba
        )
        evaluation["label_ranking_loss"] = _safe_metric(
            label_ranking_loss, y_test, y_pred_proba
        )

    # Subset accuracy (exact match)
    evaluation["subset_accuracy"] = float(
        np.mean(np.all(y_test == y_pred, axis=1))
    )

    # Average number of tags per document
    evaluation["avg_true_tags_per_doc"]  = float(y_test.sum(axis=1).mean())
    evaluation["avg_pred_tags_per_doc"]  = float(y_pred.sum(axis=1).mean())

    # Classification report as string
    try:
        evaluation["classification_report"] = classification_report(
            y_test, y_pred,
            target_names=label_names,
            zero_division=0,
        )
    except Exception:
        evaluation["classification_report"] = "N/A"

    # ------------------------------------------------------------------ #
    # 6. Per-label breakdown
    # ------------------------------------------------------------------ #
    per_label = _per_label_metrics(y_test, y_pred, label_names)

    # Identify poorly predicted labels
    poorly_predicted = [
        name for name, stats in per_label.items() if stats["poorly_predicted"]
    ]
    evaluation["poorly_predicted_labels"] = poorly_predicted
    evaluation["n_poorly_predicted"]      = len(poorly_predicted)
    evaluation["sparse_labels_in_train"]  = [label_names[i] for i in sparse_labels]

    # ------------------------------------------------------------------ #
    # 7. Label co-occurrence statistics (computed on full corpus)
    # ------------------------------------------------------------------ #
    cooccurrence = _label_cooccurrence(label_matrix, label_names)

    # ------------------------------------------------------------------ #
    # 8. Return everything
    # ------------------------------------------------------------------ #
    return {
        "model":        pipeline,
        "evaluation":   evaluation,
        "per_label":    per_label,
        "cooccurrence": cooccurrence,
        "label_names":  label_names,
        "X_test_docs":  docs_test,
        "y_test":       y_test,
        "y_pred":       y_pred,
        "y_pred_proba": y_pred_proba,
    }


# ---------------------------------------------------------------------------
# Convenience: predict on new documents
# ---------------------------------------------------------------------------

def predict_tags(
    model_pipeline: Pipeline,
    new_documents: list[str],
    label_names: list[str] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Predict tags for new documents using a trained pipeline.

    Parameters
    ----------
    model_pipeline : Pipeline
        Trained pipeline returned by build_multilabel_tagger.
    new_documents : list of str
        New raw text documents.
    label_names : list of str, optional
        Tag names; defaults to tag_00 … tag_19.
    threshold : float, optional
        If provided, apply this probability threshold instead of the
        default decision boundary (requires predict_proba support).

    Returns
    -------
    dict with 'binary_predictions', 'label_names', and optionally 'probabilities'.
    """
    if label_names is None:
        label_names = [f"tag_{i:02d}" for i in range(NUM_TAGS)]

    result: dict[str, Any] = {"label_names": label_names}

    if threshold is not None:
        try:
            proba = model_pipeline.predict_proba(new_documents)
            binary = (proba >= threshold).astype(int)
            result["probabilities"]      = proba
            result["binary_predictions"] = binary
            return result
        except AttributeError:
            pass  # fall through to standard predict

    binary = model_pipeline.predict(new_documents)
    result["binary_predictions"] = binary

    try:
        result["probabilities"] = model_pipeline.predict_proba(new_documents)
    except AttributeError:
        pass

    return result


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

def _generate_synthetic_corpus(
    n_docs: int = 500,
    n_tags: int = NUM_TAGS,
    random_state: int = 0,
) -> tuple[list[str], np.ndarray]:
    """Generate a tiny synthetic corpus for testing."""
    rng = np.random.default_rng(random_state)

    tag_keywords = [
        ["python", "code", "programming", "software", "developer"],
        ["machine", "learning", "model", "neural", "training"],
        ["data", "analysis", "statistics", "dataset", "csv"],
        ["web", "html", "css", "javascript", "browser"],
        ["cloud", "aws", "azure", "server", "deployment"],
        ["security", "encryption", "firewall", "vulnerability", "hack"],
        ["database", "sql", "query", "table", "index"],
        ["mobile", "android", "ios", "app", "smartphone"],
        ["network", "protocol", "tcp", "router", "bandwidth"],
        ["devops", "docker", "kubernetes", "pipeline", "ci"],
        ["nlp", "text", "language", "tokenize", "corpus"],
        ["vision", "image", "pixel", "detection", "camera"],
        ["finance", "stock", "market", "investment", "portfolio"],
        ["health", "medical", "patient", "diagnosis", "treatment"],
        ["education", "student", "course", "lecture", "exam"],
        ["gaming", "game", "player", "level", "score"],
        ["science", "research", "experiment", "hypothesis", "lab"],
        ["environment", "climate", "carbon", "renewable", "ecology"],
        ["politics", "election", "government", "policy", "vote"],
        ["sports", "athlete", "team", "championship", "tournament"],
    ]

    documents: list[str] = []
    labels = np.zeros((n_docs, n_tags), dtype=int)

    for i in range(n_docs):
        # Each document gets 1-4 tags
        n_active = rng.integers(1, 5)
        active_tags = rng.choice(n_tags, size=n_active, replace=False)
        labels[i, active_tags] = 1

        words = []
        for tag in active_tags:
            kws = tag_keywords[tag]
            chosen = rng.choice(kws, size=rng.integers(3, 8), replace=True)
            words.extend(chosen.tolist())

        # Add some noise words
        noise = ["the", "a", "is", "in", "of", "and", "to", "for", "with"]
        words += rng.choice(noise, size=rng.integers(5, 15), replace=True).tolist()
        rng.shuffle(words)
        documents.append(" ".join(words))

    return documents, labels


if __name__ == "__main__":
    print("Generating synthetic corpus …")
    docs, labels = _generate_synthetic_corpus(n_docs=600, random_state=42)

    tag_names = [
        "python", "ml", "data", "web", "cloud",
        "security", "database", "mobile", "network", "devops",
        "nlp", "vision", "finance", "health", "education",
        "gaming", "science", "environment", "politics", "sports",
    ]

    print("Training multi-label tagger …")
    results = build_multilabel_tagger(
        documents=docs,
        label_matrix=labels,
        label_names=tag_names,
        test_size=0.2,
        random_state=42,
    )

    ev = results["evaluation"]
    print("\n=== Overall Evaluation ===")
    print(f"  Hamming Loss          : {ev['hamming_loss']:.4f}")
    print(f"  F1 Micro              : {ev['f1_micro']:.4f}")
    print(f"  F1 Macro              : {ev['f1_macro']:.4f}")
    print(f"  Subset Accuracy       : {ev['subset_accuracy']:.4f}")
    print(f"  Avg true tags/doc     : {ev['avg_true_tags_per_doc']:.2f}")
    print(f"  Avg pred tags/doc     : {ev['avg_pred_tags_per_doc']:.2f}")
    if "label_ranking_avg_precision" in ev:
        print(f"  Label Ranking AvgPrec : {ev['label_ranking_avg_precision']:.4f}")

    print("\n=== Per-Label F1 ===")
    for name, stats in results["per_label"].items():
        flag = " ⚠ POOR" if stats["poorly_predicted"] else ""
        sparse = " (sparse)" if stats["sparse"] else ""
        f1_val = f"{stats['f1']:.3f}" if not np.isnan(stats["f1"]) else "  N/A"
        print(f"  {name:<15} F1={f1_val}  support={stats['support']}{sparse}{flag}")

    print("\n=== Poorly Predicted Labels ===")
    print(" ", ev["poorly_predicted_labels"])

    print("\n=== Top Label Co-occurrences