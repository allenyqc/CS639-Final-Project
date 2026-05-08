Looking at the code, it actually already correctly fixes the data leakage issue - the vectorizer is fitted only on training documents. The original violation mentioned has been addressed in this code.

However, reviewing more carefully, I can see the code is already correct. The task says "The previous solution violates ML/Python protocol" with the violation being "text vectorizer fitted on full corpus before split; fit only on training documents" - but the provided code already does this correctly.

Let me re-read... The code provided IS the fixed version, and it's already correct. But since we're asked to fix it, let me ensure everything is properly structured and there are no subtle issues remaining.

```python
"""
Full Text Classification Pipeline for Sentiment Analysis
"""

import re
import string
import warnings
from typing import Optional, Union

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

import nltk

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(
    text: str,
    use_stemming: bool = False,
    language: str = "english",
) -> str:
    """
    Preprocess a single text document.

    Steps:
        1. Lowercase
        2. Remove punctuation and digits
        3. Tokenise (whitespace split)
        4. Remove stopwords
        5. Optionally apply Porter stemming

    Parameters
    ----------
    text : str
        Raw input text.
    use_stemming : bool
        Whether to apply Porter stemming.
    language : str
        Language for stopword removal.

    Returns
    -------
    str
        Cleaned, space-joined token string.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation and digits
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)

    # 3. Tokenise
    tokens = text.split()

    # 4. Remove stopwords
    stop_words = set(stopwords.words(language))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # 5. Optional stemming
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def preprocess_corpus(
    documents: list,
    use_stemming: bool = False,
    language: str = "english",
) -> list:
    """
    Apply :func:`preprocess_text` to every document in a corpus.

    Parameters
    ----------
    documents : list[str]
        Raw text documents.
    use_stemming : bool
        Whether to apply Porter stemming.
    language : str
        Language for stopword removal.

    Returns
    -------
    list[str]
        Preprocessed documents.
    """
    return [
        preprocess_text(doc, use_stemming=use_stemming, language=language)
        for doc in documents
    ]


# ---------------------------------------------------------------------------
# 2. Vectorisation
# ---------------------------------------------------------------------------

def build_tfidf_vectorizer(
    max_features: Optional[int] = 20_000,
    ngram_range: tuple = (1, 2),
    sublinear_tf: bool = True,
    min_df: int = 2,
) -> TfidfVectorizer:
    """
    Create a configured :class:`TfidfVectorizer`.

    Parameters
    ----------
    max_features : int or None
        Maximum vocabulary size.
    ngram_range : tuple
        The lower and upper boundary of the range of n-values for n-grams.
    sublinear_tf : bool
        Apply sublinear TF scaling (1 + log(tf)).
    min_df : int
        Minimum document frequency for a term to be included.

    Returns
    -------
    TfidfVectorizer
        Configured (unfitted) vectorizer.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        analyzer="word",
        strip_accents="unicode",
    )


def vectorize(
    vectorizer: TfidfVectorizer,
    train_docs: list,
    test_docs: list,
) -> tuple:
    """
    Fit the vectorizer on training data only, then transform both splits.

    The vectorizer is fitted exclusively on ``train_docs`` to prevent
    data leakage from the test set into the vocabulary and IDF weights.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Unfitted vectorizer.
    train_docs : list[str]
        Preprocessed training documents.
    test_docs : list[str]
        Preprocessed test documents.

    Returns
    -------
    tuple
        (fitted_vectorizer, X_train_tfidf, X_test_tfidf)
    """
    # Fit ONLY on training documents to avoid data leakage.
    # Vocabulary and IDF weights are learned exclusively from training data.
    X_train = vectorizer.fit_transform(train_docs)
    # Transform test documents using the vocabulary and IDF weights
    # learned from training data only — no fitting on test data.
    X_test = vectorizer.transform(test_docs)
    return vectorizer, X_train, X_test


# ---------------------------------------------------------------------------
# 3. Feature Selection
# ---------------------------------------------------------------------------

def select_features(
    X_train,
    y_train,
    X_test,
    feature_names: np.ndarray,
    k: Union[int, str] = 5_000,
) -> tuple:
    """
    Apply chi-squared feature selection fitted on training data only.

    The selector is fitted on ``X_train`` and ``y_train`` exclusively;
    ``X_test`` is only transformed (not used to fit) to prevent leakage.

    Parameters
    ----------
    X_train : sparse matrix
        TF-IDF training matrix.
    y_train : array-like
        Training labels.
    X_test : sparse matrix
        TF-IDF test matrix.
    feature_names : np.ndarray
        All feature names from the vectorizer.
    k : int or 'all'
        Number of top features to keep.

    Returns
    -------
    tuple
        (selector, X_train_sel, X_test_sel, selected_feature_names)
    """
    selector = SelectKBest(chi2, k=k)
    # Fit and transform training data only — chi-squared statistics are
    # computed from training features and labels exclusively.
    X_train_sel = selector.fit_transform(X_train, y_train)
    # Transform test data using the support mask learned from training data.
    # The selector is never re-fitted or updated with test information.
    X_test_sel = selector.transform(X_test)

    support_mask = selector.get_support()
    selected_feature_names = feature_names[support_mask]

    return selector, X_train_sel, X_test_sel, selected_feature_names


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def train_classifier(
    X_train,
    y_train,
    C: float = 1.0,
    max_iter: int = 2_000,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
) -> LinearSVC:
    """
    Train a :class:`LinearSVC` classifier.

    Parameters
    ----------
    X_train : array-like or sparse matrix
        Feature matrix.
    y_train : array-like
        Target labels.
    C : float
        Regularisation parameter.
    max_iter : int
        Maximum number of iterations.
    class_weight : str or None
        Class weight strategy.
    random_state : int
        Random seed for reproducibility.

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

def evaluate(
    clf: LinearSVC,
    X_test,
    y_test,
    target_names: Optional[list] = None,
) -> dict:
    """
    Evaluate a trained classifier on the test set.

    Parameters
    ----------
    clf : LinearSVC
        Fitted classifier.
    X_test : array-like or sparse matrix
        Test feature matrix.
    y_test : array-like
        True test labels.
    target_names : list[str] or None
        Human-readable class names for the report.

    Returns
    -------
    dict
        Dictionary with keys: ``accuracy``, ``macro_f1``,
        ``classification_report``, ``predictions``.
    """
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=target_names)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "predictions": y_pred,
    }


# ---------------------------------------------------------------------------
# 6. Full Pipeline
# ---------------------------------------------------------------------------

def build_sentiment_pipeline(
    documents: list,
    labels: list,
    *,
    use_stemming: bool = False,
    language: str = "english",
    max_features: Optional[int] = 20_000,
    ngram_range: tuple = (1, 2),
    sublinear_tf: bool = True,
    min_df: int = 2,
    k_best: Union[int, str] = 5_000,
    C: float = 1.0,
    max_iter: int = 2_000,
    class_weight: Optional[str] = "balanced",
    test_size: float = 0.2,
    random_state: int = 42,
    target_names: Optional[list] = None,
) -> dict:
    """
    End-to-end sentiment analysis pipeline.

    Data separation protocol (strictly enforced):
      - Step 1: Raw documents and labels are split into train/test sets
                BEFORE any fitting step.  This is the critical boundary
                that prevents data leakage.
      - Step 2: Preprocessing is a deterministic, label-independent
                transformation (lowercasing, stopword removal, stemming)
                applied identically to both splits; it requires no fitting
                and introduces no leakage.
      - Step 3: The TF-IDF vectorizer is fitted ONLY on training documents.
                Vocabulary and IDF weights are learned from training data
                exclusively; test documents are only transformed.
      - Step 4: The chi-squared selector is fitted ONLY on training
                features and labels.  The support mask is then applied to
                the test matrix via transform — no test information is used
                during fitting.
      - Step 5: The LinearSVC classifier is fitted ONLY on training data.
      - Step 6: Evaluation is performed on the held-out test set using
                components that were fitted exclusively on training data.

    Parameters
    ----------
    documents : list[str]
        Raw text documents.
    labels : list
        Corresponding class labels.
    use_stemming : bool
        Apply Porter stemming during preprocessing.
    language : str
        Stopword language.
    max_features : int or None
        TF-IDF vocabulary size cap.
    ngram_range : tuple
        N-gram range for TF-IDF.
    sublinear_tf : bool
        Sublinear TF scaling.
    min_df : int
        Minimum document frequency.
    k_best : int or 'all'
        Number of chi-squared selected features.
    C : float
        SVM regularisation strength.
    max_iter : int
        SVM maximum iterations.
    class_weight : str or None
        SVM class weighting.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Global random seed.
    target_names : list[str] or None
        Class names for the evaluation report.

    Returns
    -------
    dict
        {
            "vectorizer":        fitted TfidfVectorizer (on train only),
            "selector":          fitted SelectKBest (on train only),
            "classifier":        fitted LinearSVC,
            "selected_features": np.ndarray of selected feature names,
            "metrics":           dict with accuracy, macro_f1, report,
            "train_docs":        preprocessed training documents,
            "test_docs":         preprocessed test documents,
            "y_train":           training labels,
            "y_test":            test labels,
        }
    """
    labels = np.array(labels)

    # ------------------------------------------------------------------ #
    # Step 1 — Train / test split on RAW documents FIRST                  #
    # ------------------------------------------------------------------ #
    # The split must happen before any fitting step.  Splitting on raw
    # (unprocessed, un-vectorised) documents guarantees that no estimator
    # can observe test-set information during fitting.
    print("[1/6] Splitting raw data into train and test sets …")
    docs_train_raw, docs_test_raw, y_train, y_test = train_test_split(
        documents,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    print(f"    Train size: {len(docs_train_raw):,}  |  Test size: {len(docs_test_raw):,}")

    # ------------------------------------------------------------------ #
    # Step 2 — Preprocessing (no fitting required; no leakage possible)  #
    # ------------------------------------------------------------------ #
    # Lowercasing, punctuation removal, stopword filtering, and optional
    # stemming are all deterministic, label-independent transformations.
    # They are applied identically to both splits and do not require
    # fitting, so they cannot introduce data leakage.
    print("[2/6] Preprocessing text …")
    docs_train = preprocess_corpus(
        docs_train_raw, use_stemming=use_stemming, language=language
    )
    docs_test = preprocess_corpus(
        docs_test_raw, use_stemming=use_stemming, language=language
    )

    # ------------------------------------------------------------------ #
    # Step 3 — TF-IDF vectorisation (fit on TRAINING data only)          #
    # ------------------------------------------------------------------ #
    # The vectorizer learns its vocabulary and IDF weights exclusively
    # from training documents.  Test documents are transformed using
    # those pre-learned weights — the vectorizer is never re-fitted or
    # updated with any test-set information.
    print("[3/6] Vectorising with TF-IDF (fit on training data only) …")
    vectorizer = build_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
    )
    vectorizer, X_train_tfidf, X_test_tfidf = vectorize(
        vectorizer, docs_train, docs_test
    )
    all_feature_names = np.array(vectorizer.get_feature_names_out())
    print(f"    Vocabulary size (learned from training docs only): {len(all_feature_names):,}")

    # ------------------------------------------------------------------ #
    # Step 4 — Chi-squared feature selection (fit on TRAINING data only) #
    # ------------------------------------------------------------------ #
    # Chi-squared statistics are computed from training features and
    # training labels only.  The resulting support mask is applied to
    # the test matrix via transform — the selector is never re-fitted
    # or updated with test-set information.
    print("[4/6] Selecting top features with chi-squared (fit on training data only) …")
    effective_k = k_best
    if isinstance(k_best, int):
        effective_k = min(k_best, X_train_tfidf.shape[1])
    selector, X_train_sel, X_test_sel, selected_features = select_features(
        X_train_tfidf, y_train, X_test_tfidf, all_feature_names, k=effective_k
    )
    print(f"    Features after selection: {X_train_sel.shape[1]:,}")

    # ------------------------------------------------------------------ #
    # Step 5 — Train LinearSVC on TRAINING data only                     #
    # ------------------------------------------------------------------ #
    print("[5/6] Training LinearSVC on training data only …")
    clf = train_classifier(
        X_train_sel,
        y_train,
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )

    # ------------------------------------------------------------------ #
    # Step 6 — Evaluate on held-out TEST set                             #
    # ------------------------------------------------------------------ #
    # All pipeline components (vectorizer, selector, classifier) were
    # fitted exclusively on training data.  Evaluation uses only the
    # transform/predict interface — no fitting occurs here.
    print("[6/6] Evaluating on held-out test set …")
    metrics = evaluate(clf, X_test_sel, y_test, target_names=target_names)

    print("\n=== Results ===")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    print("\n" + metrics["classification_report"])

    return {
        "vectorizer": vectorizer,
        "selector": selector,
        "classifier": clf,
        "selected_features": selected_features,
        "metrics": metrics,
        "train_docs": docs_train,
        "test_docs": docs_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# 7. Inference helper
# ---------------------------------------------------------------------------

def predict(
    text: Union[str, list],
    vectorizer: TfidfVectorizer,
    selector: SelectKBest,
    classifier: LinearSVC,
    use_stemming: bool = False,
    language: str = "english",
) -> np.ndarray:
    """
    Predict sentiment for new text(s) using fitted pipeline components.

    All components (vectorizer, selector, classifier) were fitted on
    training data only and are used here in transform/predict mode only.
    No fitting occurs at inference time, so no leakage is possible.

    Parameters
    ----------
    text : str or list[str]
        One or more raw text documents.
    vectorizer : TfidfVectorizer
        Fitted vectorizer (trained on training documents only).
    selector : SelectKBest
        Fitted feature selector (trained on training data only).
    classifier : LinearSVC
        Fitted classifier (trained on training data only).
    use_stemming : bool
        Must match the setting used during training.
    language : str
        Must match the setting used during training.

    Returns
    -------
    np.ndarray
        Predicted class labels.
    """
    if isinstance(text, str):
        text = [text]

    cleaned = preprocess_corpus(text, use_stemming=use_stemming, language=language)
    # Transform only — the vectorizer vocabulary and IDF weights were
    # learned from training data and are not updated here.
    X_tfidf = vectorizer.transform(cleaned)
    # Transform only — the support mask was learned from training data
    # and is not updated here.
    X_sel = selector.transform(X_tfidf)
    return classifier.predict(X_sel)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import fetch_20newsgroups

    categories = ["rec.sport.hockey", "sci.med"]
    data = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )

    result = build_sentiment_pipeline(
        documents=data.data,
        labels=data.target,
        use_stemming=False,