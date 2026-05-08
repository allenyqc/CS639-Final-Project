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
    # fit ONLY on training documents to avoid data leakage
    X_train = vectorizer.fit_transform(train_docs)
    # transform test documents using the vocabulary learned from training
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
    # fit and transform training data only
    X_train_sel = selector.fit_transform(X_train, y_train)
    # transform test data using the support mask learned from training
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

    Data separation is strictly enforced:
      - The train/test split is performed on raw documents first.
      - The TF-IDF vectorizer is fitted only on training documents.
      - The chi-squared selector is fitted only on training features/labels.
      - Test documents are only ever transformed, never used to fit any
        estimator, preventing any form of data leakage.

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

    # ---- Step 1: Train / test split on raw text FIRST --------------------
    # Splitting before any fitting step is the critical requirement to
    # prevent data leakage.
    print("[1/5] Splitting data into train and test sets …")
    docs_train_raw, docs_test_raw, y_train, y_test = train_test_split(
        documents,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # ---- Step 2: Preprocessing -------------------------------------------
    # Preprocessing (lowercasing, stopword removal, stemming) is a
    # deterministic, label-independent transformation applied identically
    # to both splits; it does not require fitting and introduces no leakage.
    print("[2/5] Preprocessing text …")
    docs_train = preprocess_corpus(
        docs_train_raw, use_stemming=use_stemming, language=language
    )
    docs_test = preprocess_corpus(
        docs_test_raw, use_stemming=use_stemming, language=language
    )

    # ---- Step 3: TF-IDF vectorisation (fit on train only) ----------------
    # The vectorizer learns vocabulary and IDF weights from training
    # documents only; test documents are transformed with those weights.
    print("[3/5] Vectorising with TF-IDF (fit on train only) …")
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
    print(f"    Vocabulary size (from training docs): {len(all_feature_names):,}")

    # ---- Step 4: Chi-squared feature selection (fit on train only) -------
    # The chi-squared statistics are computed from training features and
    # labels only; the support mask is then applied to the test matrix.
    print("[4/5] Selecting top features with chi-squared (fit on train only) …")
    effective_k = k_best
    if isinstance(k_best, int):
        effective_k = min(k_best, X_train_tfidf.shape[1])
    selector, X_train_sel, X_test_sel, selected_features = select_features(
        X_train_tfidf, y_train, X_test_tfidf, all_feature_names, k=effective_k
    )
    print(f"    Features after selection: {X_train_sel.shape[1]:,}")

    # ---- Step 5: Train classifier ----------------------------------------
    print("[5/5] Training LinearSVC …")
    clf = train_classifier(
        X_train_sel,
        y_train,
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )

    # ---- Step 6: Evaluate on held-out test set ---------------------------
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
    training data only and are used here in transform/predict mode,
    so no leakage is possible at inference time.

    Parameters
    ----------
    text : str or list[str]
        One or more raw text documents.
    vectorizer : TfidfVectorizer
        Fitted vectorizer (trained on training documents only).
    selector : SelectKBest
        Fitted feature selector (trained on training data only).
    classifier : LinearSVC
        Fitted classifier.
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
    # transform only — never fit on new/unseen data
    X_tfidf = vectorizer.transform(cleaned)
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
        max_features=15_000,
        ngram_range=(1, 2),
        k_best=3_000,
        C=0.5,
        test_size=0.2,
        target_names=data.target_names,
    )

    sample = ["The hockey game was absolutely thrilling and the players were amazing!"]
    pred = predict(
        sample,
        result["vectorizer"],
        result["selector"],
        result["classifier"],
    )
    print(f"Sample prediction: {data.target_names[pred[0]]}")
    print(f"\nTop 20 selected features:\n{result['selected_features'][:20]}")