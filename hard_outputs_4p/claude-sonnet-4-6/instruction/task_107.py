"""
Full text classification pipeline for sentiment analysis.

Best practices enforced:
- Train/test split happens BEFORE any preprocessing or fitting.
- All transformers are fit ONLY on training data.
- Test set is used exclusively for final evaluation.
- No credentials or secrets are hardcoded.
"""

from __future__ import annotations

import re
import string
from typing import Any

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Ensure NLTK resources are available
# ---------------------------------------------------------------------------
import nltk

for _resource in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"tokenizers/{_resource}" if _resource == "punkt" else f"corpora/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

_STOP_WORDS: set[str] = set(stopwords.words("english"))
_STEMMER = PorterStemmer()


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    apply_stemming: bool = False,
) -> str:
    """
    Lowercase, remove punctuation, optionally remove stopwords and stem.

    Parameters
    ----------
    text:
        Raw input string.
    remove_stopwords:
        Whether to remove English stopwords.
    apply_stemming:
        Whether to apply Porter stemming.

    Returns
    -------
    str
        Cleaned text.
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation and digits
    text = re.sub(r"[%s\d]" % re.escape(string.punctuation), " ", text)

    # Tokenise on whitespace
    tokens: list[str] = text.split()

    # Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]

    # Stemming
    if apply_stemming:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


def preprocess_corpus(
    documents: list[str],
    *,
    remove_stopwords: bool = True,
    apply_stemming: bool = False,
) -> list[str]:
    """
    Apply :func:`preprocess_text` to every document in *documents*.

    Parameters
    ----------
    documents:
        List of raw text strings.
    remove_stopwords:
        Passed through to :func:`preprocess_text`.
    apply_stemming:
        Passed through to :func:`preprocess_text`.

    Returns
    -------
    list[str]
        Preprocessed documents.
    """
    return [
        preprocess_text(doc, remove_stopwords=remove_stopwords, apply_stemming=apply_stemming)
        for doc in documents
    ]


# ---------------------------------------------------------------------------
# 2. Vectorisation
# ---------------------------------------------------------------------------

def build_tfidf_vectorizer(
    *,
    max_features: int | None = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Create (but do NOT fit) a :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.

    Parameters
    ----------
    max_features:
        Maximum vocabulary size.
    ngram_range:
        The lower and upper boundary of the range of n-values for n-grams.
    sublinear_tf:
        Apply sublinear TF scaling (``1 + log(tf)``).

    Returns
    -------
    TfidfVectorizer
        Unfitted vectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-z]{2,}\b",
    )


def fit_vectorizer(
    vectorizer: TfidfVectorizer,
    train_documents: list[str],
) -> spmatrix:
    """
    Fit *vectorizer* on *train_documents* and transform them.

    .. warning::
        Call this function **only** with training data.

    Parameters
    ----------
    vectorizer:
        An unfitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    train_documents:
        Preprocessed training documents.

    Returns
    -------
    spmatrix
        Sparse TF-IDF matrix for the training set.
    """
    return vectorizer.fit_transform(train_documents)


# ---------------------------------------------------------------------------
# 3. Feature selection
# ---------------------------------------------------------------------------

def build_feature_selector(*, k: int = 5_000) -> SelectKBest:
    """
    Create (but do NOT fit) a chi-squared :class:`~sklearn.feature_selection.SelectKBest`.

    Parameters
    ----------
    k:
        Number of top features to retain.

    Returns
    -------
    SelectKBest
        Unfitted selector instance.
    """
    return SelectKBest(score_func=chi2, k=k)


def fit_feature_selector(
    selector: SelectKBest,
    X_train: spmatrix,
    y_train: np.ndarray,
) -> spmatrix:
    """
    Fit *selector* on training data and transform it.

    .. warning::
        Call this function **only** with training data.

    Parameters
    ----------
    selector:
        An unfitted :class:`~sklearn.feature_selection.SelectKBest`.
    X_train:
        TF-IDF matrix for the training set.
    y_train:
        Training labels.

    Returns
    -------
    spmatrix
        Reduced sparse matrix for the training set.
    """
    return selector.fit_transform(X_train, y_train)


def get_selected_feature_names(
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
) -> list[str]:
    """
    Return the vocabulary terms that survived feature selection.

    Parameters
    ----------
    vectorizer:
        A **fitted** :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    selector:
        A **fitted** :class:`~sklearn.feature_selection.SelectKBest`.

    Returns
    -------
    list[str]
        Selected feature names.
    """
    all_features: np.ndarray = np.array(vectorizer.get_feature_names_out())
    support_mask: np.ndarray = selector.get_support()
    return all_features[support_mask].tolist()


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def train_classifier(
    X_train: spmatrix,
    y_train: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 2_000,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
) -> LinearSVC:
    """
    Fit a :class:`~sklearn.svm.LinearSVC` on the training data.

    Parameters
    ----------
    X_train:
        Feature matrix (training).
    y_train:
        Labels (training).
    C:
        Regularisation parameter.
    max_iter:
        Maximum number of iterations.
    class_weight:
        Weighting strategy for imbalanced classes.
    random_state:
        Reproducibility seed.

    Returns
    -------
    LinearSVC
        Fitted classifier.
    """
    clf = LinearSVC(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    clf: LinearSVC,
    X_test: spmatrix,
    y_test: np.ndarray,
    *,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate *clf* on the held-out test set.

    Parameters
    ----------
    clf:
        A **fitted** :class:`~sklearn.svm.LinearSVC`.
    X_test:
        Feature matrix (test).
    y_test:
        True labels (test).
    target_names:
        Human-readable class names for the classification report.

    Returns
    -------
    dict
        Dictionary with keys ``accuracy``, ``macro_f1``, and
        ``classification_report``.
    """
    y_pred: np.ndarray = clf.predict(X_test)

    accuracy: float = accuracy_score(y_test, y_pred)
    macro_f1: float = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report: str = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# 6. End-to-end pipeline
# ---------------------------------------------------------------------------

def build_sentiment_pipeline(
    documents: list[str],
    labels: list[Any],
    *,
    # Preprocessing
    remove_stopwords: bool = True,
    apply_stemming: bool = False,
    # Vectorisation
    max_features: int | None = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    # Feature selection
    k: int = 5_000,
    # Classifier
    C: float = 1.0,
    max_iter: int = 2_000,
    class_weight: str | dict | None = "balanced",
    # Split
    test_size: float = 0.2,
    random_state: int = 42,
    # Reporting
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build, train, and evaluate a full sentiment-analysis pipeline.

    Data-leakage safeguards
    -----------------------
    * The train/test split is performed **before** any preprocessing.
    * The TF-IDF vectorizer and chi-squared selector are fit **only** on the
      training partition.
    * The test set is used **exclusively** for final evaluation.

    Parameters
    ----------
    documents:
        Raw text documents.
    labels:
        Corresponding class labels.
    remove_stopwords:
        Remove English stopwords during preprocessing.
    apply_stemming:
        Apply Porter stemming during preprocessing.
    max_features:
        Maximum TF-IDF vocabulary size.
    ngram_range:
        N-gram range for the TF-IDF vectorizer.
    sublinear_tf:
        Apply sublinear TF scaling.
    k:
        Number of features to keep after chi-squared selection.
    C:
        LinearSVC regularisation parameter.
    max_iter:
        Maximum LinearSVC iterations.
    class_weight:
        Class-weight strategy for LinearSVC.
    test_size:
        Fraction of data reserved for testing.
    random_state:
        Global reproducibility seed.
    target_names:
        Human-readable class names for the classification report.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``vectorizer``
            Fitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
        ``selector``
            Fitted :class:`~sklearn.feature_selection.SelectKBest`.
        ``classifier``
            Fitted :class:`~sklearn.svm.LinearSVC`.
        ``selected_features``
            List of selected feature names.
        ``metrics``
            Dictionary with ``accuracy``, ``macro_f1``, and
            ``classification_report``.
        ``splits``
            Dictionary with ``X_train``, ``X_test``, ``y_train``, ``y_test``
            (after all transformations).
    """
    if len(documents) != len(labels):
        raise ValueError(
            f"documents and labels must have the same length, "
            f"got {len(documents)} and {len(labels)}."
        )

    labels_array: np.ndarray = np.asarray(labels)

    # ------------------------------------------------------------------
    # Step 1 — Split BEFORE any fitting or preprocessing
    # ------------------------------------------------------------------
    docs_train_raw, docs_test_raw, y_train, y_test = train_test_split(
        documents,
        labels_array,
        test_size=test_size,
        random_state=random_state,
        stratify=labels_array,
    )

    # ------------------------------------------------------------------
    # Step 2 — Preprocess (no fitting involved; safe to apply to both)
    # ------------------------------------------------------------------
    docs_train: list[str] = preprocess_corpus(
        docs_train_raw,
        remove_stopwords=remove_stopwords,
        apply_stemming=apply_stemming,
    )
    docs_test: list[str] = preprocess_corpus(
        docs_test_raw,
        remove_stopwords=remove_stopwords,
        apply_stemming=apply_stemming,
    )

    # ------------------------------------------------------------------
    # Step 3 — Vectorise: fit on TRAIN, transform both
    # ------------------------------------------------------------------
    vectorizer: TfidfVectorizer = build_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
    )
    X_train_tfidf: spmatrix = fit_vectorizer(vectorizer, docs_train)
    X_test_tfidf: spmatrix = vectorizer.transform(docs_test)

    # ------------------------------------------------------------------
    # Step 4 — Feature selection: fit on TRAIN, transform both
    # ------------------------------------------------------------------
    # Clamp k to the actual vocabulary size to avoid errors
    actual_k: int = min(k, X_train_tfidf.shape[1])
    selector: SelectKBest = build_feature_selector(k=actual_k)
    X_train_sel: spmatrix = fit_feature_selector(selector, X_train_tfidf, y_train)
    X_test_sel: spmatrix = selector.transform(X_test_tfidf)

    selected_features: list[str] = get_selected_feature_names(vectorizer, selector)

    # ------------------------------------------------------------------
    # Step 5 — Train classifier on TRAIN only
    # ------------------------------------------------------------------
    clf: LinearSVC = train_classifier(
        X_train_sel,
        y_train,
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # Step 6 — Evaluate on TEST only
    # ------------------------------------------------------------------
    metrics: dict[str, Any] = evaluate_classifier(
        clf,
        X_test_sel,
        y_test,
        target_names=target_names,
    )

    return {
        "vectorizer": vectorizer,
        "selector": selector,
        "classifier": clf,
        "selected_features": selected_features,
        "metrics": metrics,
        "splits": {
            "X_train": X_train_sel,
            "X_test": X_test_sel,
            "y_train": y_train,
            "y_test": y_test,
        },
    }


# ---------------------------------------------------------------------------
# 7. Convenience: predict on new documents
# ---------------------------------------------------------------------------

def predict(
    new_documents: list[str],
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
    clf: LinearSVC,
    *,
    remove_stopwords: bool = True,
    apply_stemming: bool = False,
) -> np.ndarray:
    """
    Preprocess, vectorise, select features, and classify new documents.

    Parameters
    ----------
    new_documents:
        Raw text strings to classify.
    vectorizer:
        Fitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    selector:
        Fitted :class:`~sklearn.feature_selection.SelectKBest`.
    clf:
        Fitted :class:`~sklearn.svm.LinearSVC`.
    remove_stopwords:
        Must match the setting used during training.
    apply_stemming:
        Must match the setting used during training.

    Returns
    -------
    np.ndarray
        Predicted class labels.
    """
    cleaned: list[str] = preprocess_corpus(
        new_documents,
        remove_stopwords=remove_stopwords,
        apply_stemming=apply_stemming,
    )
    X_tfidf: spmatrix = vectorizer.transform(cleaned)
    X_sel: spmatrix = selector.transform(X_tfidf)
    return clf.predict(X_sel)


# ---------------------------------------------------------------------------
# 8. Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal synthetic dataset for a quick smoke test
    _sample_docs = [
        "I absolutely loved this movie, it was fantastic!",
        "Terrible film, complete waste of time.",
        "Great acting and wonderful story.",
        "Boring and predictable plot.",
        "One of the best movies I have ever seen.",
        "Awful experience, would not recommend.",
        "Brilliant cinematography and superb direction.",
        "Dull characters and poor dialogue.",
        "A masterpiece of modern cinema.",
        "Disappointing and poorly executed.",
        "Highly entertaining and well-paced.",
        "Slow and uninteresting from start to finish.",
        "The performances were outstanding.",
        "Completely forgettable and mediocre.",
        "An absolute delight to watch.",
        "Dreadful script and bad acting.",
    ] * 10  # repeat to have enough samples for stratified split

    _sample_labels = (["positive", "negative"] * 8) * 10

    result = build_sentiment_pipeline(
        _sample_docs,
        _sample_labels,
        max_features=5_000,
        ngram_range=(1, 2),
        k=500,
        apply_stemming=True,
        target_names=["negative", "positive"],
        random_state=0,
    )

    print("=== Sentiment Analysis Pipeline Results ===")
    print(f"Accuracy : {result['metrics']['accuracy']:.4f}")
    print(f"Macro F1 : {result['metrics']['macro_f1']:.4f}")
    print("\nClassification Report:")
    print(result["metrics"]["classification_report"])
    print(f"Number of selected features: {len(result['selected_features'])}")
    print("Top-10 selected features:", result["selected_features"][:10])

    # Demonstrate inference on unseen text
    new_texts = ["This was an amazing experience!", "Terrible and boring."]
    preds = predict(
        new_texts,
        result["vectorizer"],
        result["selector"],
        result["classifier"],
        apply_stemming=True,
    )
    print("\nPredictions for new texts:")
    for text, label in zip(new_texts, preds):
        print(f"  '{text}' -> {label}")