```python
"""
Full text classification pipeline for sentiment analysis.

Best practices followed:
- Train/test split BEFORE any preprocessing or fitting
- All transformers fit ONLY on training data
- Test set used ONLY for final evaluation
- Separate validation split available for threshold tuning
- Appropriate metrics (accuracy + macro F1 + full report) for potentially imbalanced data
- No hardcoded credentials
- No unsafe deserialization
- No mutable default arguments (use None sentinel)
- Specific exception handling
"""

from __future__ import annotations

import re
import string
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Optional stemming support
try:
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import PorterStemmer
    import nltk

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the sentiment analysis pipeline."""

    test_size: float = 0.2
    val_size: float = 0.1  # fraction of training data used as validation
    random_state: int = 42
    # Preprocessing
    use_stemming: bool = False
    language: str = "english"
    # TF-IDF
    max_features: int = 20_000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    sublinear_tf: bool = True
    # Feature selection
    top_k_features: int = 5_000
    # Classifier
    svc_C: float = 1.0
    svc_max_iter: int = 2_000
    dual: bool = True


@dataclass
class PipelineArtifacts:
    """Container for all trained pipeline components and results."""

    vectorizer: TfidfVectorizer
    selector: SelectKBest
    classifier: LinearSVC
    selected_feature_names: List[str]
    train_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    label_classes: np.ndarray
    config: PipelineConfig


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------


def _ensure_nltk_resources(language: str = "english") -> None:
    """Download required NLTK resources if not already present."""
    if not _NLTK_AVAILABLE:
        return
    for resource in ["stopwords", "punkt"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def _get_stopwords(language: str = "english") -> frozenset:
    """Return a frozenset of stopwords for the given language."""
    if _NLTK_AVAILABLE:
        _ensure_nltk_resources(language)
        try:
            return frozenset(nltk_stopwords.words(language))
        except OSError:
            pass
    # Minimal English fallback
    _FALLBACK = frozenset(
        {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him",
            "his", "himself", "she", "her", "hers", "herself", "it", "its",
            "itself", "they", "them", "their", "theirs", "themselves", "what",
            "which", "who", "whom", "this", "that", "these", "those", "am",
            "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "having", "do", "does", "did", "doing", "a", "an", "the",
            "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below",
            "to", "from", "up", "down", "in", "out", "on", "off", "over",
            "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "both", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "s", "t",
            "can", "will", "just", "don", "should", "now", "d", "ll", "m",
            "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn",
            "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
            "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
        }
    )
    return _FALLBACK


def preprocess_text(
    text: str,
    stopwords: Optional[frozenset] = None,
    stemmer: Optional[Any] = None,
) -> str:
    """
    Preprocess a single text document.

    Steps:
        1. Lowercase
        2. Remove punctuation
        3. Tokenise by whitespace
        4. Remove stopwords
        5. Optionally stem tokens

    Parameters
    ----------
    text:
        Raw input string.
    stopwords:
        Frozenset of stopwords to remove.  Pass ``None`` to skip removal.
    stemmer:
        An object with a ``stem(word) -> str`` method (e.g. PorterStemmer).
        Pass ``None`` to skip stemming.

    Returns
    -------
    str
        Cleaned, space-joined token string.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation (replace with space to avoid word merging)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # 3. Remove digits and extra whitespace
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Tokenise
    tokens = text.split()

    # 5. Remove stopwords
    if stopwords is not None:
        tokens = [t for t in tokens if t not in stopwords]

    # 6. Stem
    if stemmer is not None:
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def preprocess_corpus(
    documents: List[str],
    config: PipelineConfig,
) -> List[str]:
    """
    Apply :func:`preprocess_text` to every document in *documents*.

    Parameters
    ----------
    documents:
        List of raw text strings.
    config:
        Pipeline configuration.

    Returns
    -------
    List[str]
        Preprocessed documents.
    """
    sw = _get_stopwords(config.language)
    stemmer: Optional[Any] = None
    if config.use_stemming:
        if _NLTK_AVAILABLE:
            stemmer = PorterStemmer()
        else:
            warnings.warn(
                "NLTK is not installed; stemming will be skipped.",
                RuntimeWarning,
                stacklevel=2,
            )

    return [preprocess_text(doc, stopwords=sw, stemmer=stemmer) for doc in documents]


# ---------------------------------------------------------------------------
# 2. Vectorisation
# ---------------------------------------------------------------------------


def build_vectorizer(config: PipelineConfig) -> TfidfVectorizer:
    """
    Construct (but do NOT fit) a :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.

    Parameters
    ----------
    config:
        Pipeline configuration.

    Returns
    -------
    TfidfVectorizer
        Unfitted vectorizer.
    """
    return TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        sublinear_tf=config.sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-z][a-z]+\b",
    )


def fit_vectorizer(
    vectorizer: TfidfVectorizer,
    train_docs: List[str],
) -> TfidfVectorizer:
    """
    Fit *vectorizer* on *train_docs* only.

    Parameters
    ----------
    vectorizer:
        Unfitted TfidfVectorizer.
    train_docs:
        Preprocessed training documents.

    Returns
    -------
    TfidfVectorizer
        Fitted vectorizer.
    """
    vectorizer.fit(train_docs)
    return vectorizer


# ---------------------------------------------------------------------------
# 3. Feature selection
# ---------------------------------------------------------------------------


def build_selector(config: PipelineConfig) -> SelectKBest:
    """
    Construct (but do NOT fit) a chi-squared :class:`~sklearn.feature_selection.SelectKBest`.

    Parameters
    ----------
    config:
        Pipeline configuration.

    Returns
    -------
    SelectKBest
        Unfitted selector.
    """
    k = config.top_k_features
    return SelectKBest(score_func=chi2, k=k)


def fit_selector(
    selector: SelectKBest,
    X_train,
    y_train: np.ndarray,
) -> SelectKBest:
    """
    Fit *selector* on training data only.

    Parameters
    ----------
    selector:
        Unfitted SelectKBest.
    X_train:
        Sparse TF-IDF matrix for training documents.
    y_train:
        Training labels.

    Returns
    -------
    SelectKBest
        Fitted selector.
    """
    selector.fit(X_train, y_train)
    return selector


def get_selected_feature_names(
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
) -> List[str]:
    """
    Return the feature names that survived chi-squared selection.

    Parameters
    ----------
    vectorizer:
        Fitted TfidfVectorizer.
    selector:
        Fitted SelectKBest.

    Returns
    -------
    List[str]
        Selected feature names.
    """
    all_names = np.array(vectorizer.get_feature_names_out())
    support_mask = selector.get_support()
    return all_names[support_mask].tolist()


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------


def train_classifier(
    X_train,
    y_train: np.ndarray,
    config: PipelineConfig,
) -> LinearSVC:
    """
    Train a :class:`~sklearn.svm.LinearSVC` on the (already transformed) training data.

    Parameters
    ----------
    X_train:
        Feature matrix for training documents (after vectorisation + selection).
    y_train:
        Training labels.
    config:
        Pipeline configuration.

    Returns
    -------
    LinearSVC
        Fitted classifier.
    """
    clf = LinearSVC(
        C=config.svc_C,
        max_iter=config.svc_max_iter,
        dual=config.dual,
        random_state=config.random_state,
    )
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    classifier: LinearSVC,
    X: Any,
    y_true: np.ndarray,
    split_name: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate *classifier* on a feature matrix *X* and true labels *y_true*.

    Parameters
    ----------
    classifier:
        Fitted LinearSVC.
    X:
        Feature matrix.
    y_true:
        Ground-truth labels.
    split_name:
        Human-readable name for the split (used in the report header).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys ``accuracy``, ``macro_f1``, and ``report``.
    """
    y_pred = classifier.predict(X)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)

    metrics: Dict[str, Any] = {
        "split": split_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report": report,
    }
    return metrics


# ---------------------------------------------------------------------------
# 6. Full pipeline orchestration
# ---------------------------------------------------------------------------


def build_sentiment_pipeline(
    documents: List[str],
    labels: List[Any],
    config: Optional[PipelineConfig] = None,
    verbose: bool = True,
) -> PipelineArtifacts:
    """
    Build, train, and evaluate a full sentiment analysis pipeline.

    Data splitting strategy
    -----------------------
    1. ``documents`` / ``labels`` are split into **train+val** and **test**
       BEFORE any preprocessing or fitting.
    2. The train+val portion is further split into **train** and **val**.
    3. All transformers (vectorizer, selector) are fit ONLY on the **train**
       partition.
    4. The **val** set is available for threshold tuning / early stopping
       (not used here, but metrics are reported).
    5. The **test** set is used ONLY for final metric reporting.

    Parameters
    ----------
    documents:
        Raw text documents.
    labels:
        Corresponding class labels (strings or integers).
    config:
        :class:`PipelineConfig` instance.  Defaults to ``PipelineConfig()``.
    verbose:
        If ``True``, print progress and metrics to stdout.

    Returns
    -------
    PipelineArtifacts
        All trained components, selected feature names, and metrics.

    Raises
    ------
    ValueError
        If ``documents`` and ``labels`` have different lengths or are empty.
    """
    if config is None:
        config = PipelineConfig()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if len(documents) != len(labels):
        raise ValueError(
            f"documents and labels must have the same length "
            f"(got {len(documents)} vs {len(labels)})."
        )
    if len(documents) == 0:
        raise ValueError("documents must not be empty.")

    docs_arr = np.array(documents)
    labels_arr = np.array(labels)

    # ------------------------------------------------------------------
    # Step 1: Train+val / test split  (BEFORE any preprocessing)
    # ------------------------------------------------------------------
    if verbose:
        print(f"[1/7] Splitting data — test_size={config.test_size}, "
              f"val_size={config.val_size} (of train+val)")

    docs_trainval, docs_test, y_trainval, y_test = train_test_split(
        docs_arr,
        labels_arr,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels_arr,
    )

    # Validation split from the train+val portion
    # val_size is expressed as a fraction of the ORIGINAL dataset;
    # we convert it to a fraction of train+val.
    val_fraction_of_trainval = config.val_size / (1.0 - config.test_size)
    val_fraction_of_trainval = min(max(val_fraction_of_trainval, 0.01), 0.5)

    docs_train, docs_val, y_train, y_val = train_test_split(
        docs_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=config.random_state,
        stratify=y_trainval,
    )

    if verbose:
        print(
            f"    Train: {len(docs_train)}, Val: {len(docs_val)}, "
            f"Test: {len(docs_test)}"
        )

    # ------------------------------------------------------------------
    # Step 2: Preprocessing  (fit nothing here — pure text transformation)
    # ------------------------------------------------------------------
    if verbose:
        print("[2/7] Preprocessing text …")

    train_clean = preprocess_corpus(docs_train.tolist(), config)
    val_clean = preprocess_corpus(docs_val.tolist(), config)
    test_clean = preprocess_corpus(docs_test.tolist(), config)

    # ------------------------------------------------------------------
    # Step 3: Vectorisation — fit ONLY on train
    # ------------------------------------------------------------------
    if verbose:
        print("[3/7] Fitting TF-IDF vectorizer on training data …")

    vectorizer = build_vectorizer(config)
    fit_vectorizer(vectorizer, train_clean)

    X_train_tfidf = vectorizer.transform(train_clean)
    X_val_tfidf = vectorizer.transform(val_clean)
    X_test_tfidf = vectorizer.transform(test_clean)

    # ------------------------------------------------------------------
    # Step 4: Chi-squared feature selection — fit ONLY on train
    # ------------------------------------------------------------------
    actual_k = min(config.top_k_features, X_train_tfidf.shape[1])
    if actual_k != config.top_k_features:
        warnings.warn(
            f"top_k_features ({config.top_k_features}) exceeds vocabulary size "
            f"({X_train_tfidf.shape[1]}); using k={actual_k}.",
            RuntimeWarning,
            stacklevel=2,
        )
        config = PipelineConfig(
            **{**config.__dict__, "top_k_features": actual_k}
        )

    if verbose:
        print(f"[4/7] Chi-squared feature selection (k={config.top_k_features}) …")

    selector = build_selector(config)
    fit_selector(selector, X_train_tfidf, y_train)

    X_train_sel = selector.transform(X_train_tfidf)
    X_val_sel = selector.transform(X_val_tfidf)
    X_test_sel = selector.transform(X_test_tfidf)

    selected_features = get_selected_feature_names(vectorizer, selector)

    if verbose:
        print(f"    Selected {len(selected_features)} features.")

    # ------------------------------------------------------------------
    # Step 5: Train classifier — ONLY on train partition
    # ------------------------------------------------------------------
    if verbose:
        print("[5/7] Training LinearSVC …")

    classifier = train_classifier(X_train_sel, y_train, config)

    # ------------------------------------------------------------------
    # Step 6: Evaluate on train, val, and test
    # ------------------------------------------------------------------
    if verbose:
        print("[6/7] Evaluating …")

    train_metrics = evaluate(classifier, X_train_sel, y_train, split_name="train")
    val_metrics = evaluate(classifier, X_val_sel, y_val, split_name="val")
    test_metrics = evaluate(classifier, X_test_sel, y_test, split_name="test")

    if verbose:
        print(
            f"\n{'='*60}\n"
            f"  Train  — Accuracy: {train_metrics['accuracy']:.4f}  "
            f"Macro-F1: {train_metrics['macro_f1']:.4f}\n"
            f"  Val    — Accuracy: {val_metrics['accuracy']:.4f}  "
            f"Macro-F1: {val_metrics['macro_f1']:.4f}\n"
            f"  Test   — Accuracy: {test_metrics['accuracy']:.4f}  "
            f"Macro-F1: {test_metrics['macro_f1']:.4f}\n"