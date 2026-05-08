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
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# Download NLTK resources if not already present
import nltk

for resource in ["stopwords", "punkt"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


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
        - Lowercase
        - Remove punctuation
        - Tokenize
        - Remove stopwords
        - Optionally apply Porter stemming

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

    # Lowercase
    text = text.lower()

    # Remove punctuation and digits
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)

    # Tokenize (simple whitespace split after cleaning)
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words(language))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # Optional stemming
    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


def preprocess_corpus(
    documents: list[str],
    use_stemming: bool = False,
    language: str = "english",
) -> list[str]:
    """
    Preprocess a list of text documents.

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
        List of preprocessed documents.
    """
    return [preprocess_text(doc, use_stemming=use_stemming, language=language) for doc in documents]


# ---------------------------------------------------------------------------
# 2. Vectorization
# ---------------------------------------------------------------------------

def build_tfidf_vectorizer(
    max_features: Optional[int] = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Build a TF-IDF vectorizer with configurable parameters.

    Parameters
    ----------
    max_features : int or None
        Maximum vocabulary size.
    ngram_range : tuple
        Range of n-gram sizes (min_n, max_n).
    min_df : int
        Minimum document frequency for a term to be included.
    sublinear_tf : bool
        Apply sublinear TF scaling (1 + log(tf)).

    Returns
    -------
    TfidfVectorizer
        Configured (unfitted) vectorizer.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
    )


def vectorize_corpus(
    train_docs: list[str],
    test_docs: list[str],
    vectorizer: Optional[TfidfVectorizer] = None,
    max_features: Optional[int] = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple:
    """
    Fit a TF-IDF vectorizer on training data and transform both splits.

    Parameters
    ----------
    train_docs : list[str]
        Preprocessed training documents.
    test_docs : list[str]
        Preprocessed test documents.
    vectorizer : TfidfVectorizer or None
        Pre-built vectorizer; if None, one is created.
    max_features : int or None
        Passed to build_tfidf_vectorizer if vectorizer is None.
    ngram_range : tuple
        Passed to build_tfidf_vectorizer if vectorizer is None.

    Returns
    -------
    X_train_tfidf : sparse matrix
    X_test_tfidf  : sparse matrix
    vectorizer    : fitted TfidfVectorizer
    """
    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer(
            max_features=max_features, ngram_range=ngram_range
        )

    X_train_tfidf = vectorizer.fit_transform(train_docs)
    X_test_tfidf = vectorizer.transform(test_docs)

    return X_train_tfidf, X_test_tfidf, vectorizer


# ---------------------------------------------------------------------------
# 3. Feature Selection
# ---------------------------------------------------------------------------

def select_features(
    X_train,
    y_train,
    X_test,
    feature_names: list[str],
    k: Union[int, str] = 5_000,
) -> tuple:
    """
    Apply chi-squared feature selection to keep the top-k features.

    Parameters
    ----------
    X_train : sparse matrix
        TF-IDF training matrix.
    y_train : array-like
        Training labels.
    X_test : sparse matrix
        TF-IDF test matrix.
    feature_names : list[str]
        All feature names from the vectorizer.
    k : int or 'all'
        Number of top features to keep.

    Returns
    -------
    X_train_sel : sparse matrix
    X_test_sel  : sparse matrix
    selector    : fitted SelectKBest
    selected_feature_names : list[str]
    """
    # Clamp k to the actual number of features
    n_features = X_train.shape[1]
    if isinstance(k, int):
        k = min(k, n_features)

    selector = SelectKBest(chi2, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    selected_mask = selector.get_support()
    selected_feature_names = [name for name, keep in zip(feature_names, selected_mask) if keep]

    return X_train_sel, X_test_sel, selector, selected_feature_names


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def train_classifier(
    X_train,
    y_train,
    C: float = 1.0,
    max_iter: int = 2_000,
    class_weight: Optional[str] = "balanced",
) -> LinearSVC:
    """
    Train a LinearSVC classifier.

    Parameters
    ----------
    X_train : sparse matrix
        Feature matrix for training.
    y_train : array-like
        Training labels.
    C : float
        Regularisation parameter.
    max_iter : int
        Maximum number of iterations.
    class_weight : str or None
        'balanced' adjusts weights inversely proportional to class frequencies.

    Returns
    -------
    LinearSVC
        Fitted classifier.
    """
    clf = LinearSVC(C=C, max_iter=max_iter, class_weight=class_weight, dual=True)
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    clf: LinearSVC,
    X_test,
    y_test,
    target_names: Optional[list[str]] = None,
) -> dict:
    """
    Evaluate a trained classifier on the test set.

    Parameters
    ----------
    clf : LinearSVC
        Fitted classifier.
    X_test : sparse matrix
        Test feature matrix.
    y_test : array-like
        True test labels.
    target_names : list[str] or None
        Human-readable class names for the report.

    Returns
    -------
    dict with keys:
        'accuracy'           : float
        'macro_f1'           : float
        'classification_report' : str
        'predictions'        : np.ndarray
    """
    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        ),
        "predictions": y_pred,
    }
    return metrics


# ---------------------------------------------------------------------------
# 6. Full Pipeline
# ---------------------------------------------------------------------------

def build_sentiment_pipeline(
    documents: list[str],
    labels: list,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    use_stemming: bool = False,
    language: str = "english",
    max_features: Optional[int] = 10_000,
    ngram_range: tuple[int, int] = (1, 2),
    top_k_features: Union[int, str] = 5_000,
    svc_C: float = 1.0,
    svc_max_iter: int = 2_000,
    class_weight: Optional[str] = "balanced",
    target_names: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    Build and evaluate a full text classification pipeline for sentiment analysis.

    Parameters
    ----------
    documents : list[str]
        Raw text documents.
    labels : list
        Corresponding class labels.
    test_size : float
        Fraction of data to use as the held-out test set.
    random_state : int
        Seed for reproducibility.
    use_stemming : bool
        Whether to apply Porter stemming during preprocessing.
    language : str
        Language for stopword removal.
    max_features : int or None
        Maximum TF-IDF vocabulary size.
    ngram_range : tuple
        N-gram range for TF-IDF.
    top_k_features : int or 'all'
        Number of features to keep after chi-squared selection.
    svc_C : float
        Regularisation parameter for LinearSVC.
    svc_max_iter : int
        Maximum iterations for LinearSVC.
    class_weight : str or None
        Class weighting strategy for LinearSVC.
    target_names : list[str] or None
        Human-readable class names for the evaluation report.
    verbose : bool
        Whether to print progress and results.

    Returns
    -------
    dict with keys:
        'vectorizer'             : fitted TfidfVectorizer
        'selector'               : fitted SelectKBest
        'classifier'             : fitted LinearSVC
        'selected_feature_names' : list[str]
        'metrics'                : dict (accuracy, macro_f1, classification_report, predictions)
        'splits'                 : dict (train/test indices and preprocessed docs)
    """
    labels = np.array(labels)

    # ---- Step 1: Train/test split ----------------------------------------
    if verbose:
        print(f"[1/6] Splitting data  ({len(documents)} samples, "
              f"test_size={test_size}) ...")
    (
        docs_train_raw, docs_test_raw,
        y_train, y_test,
        idx_train, idx_test,
    ) = train_test_split(
        documents, labels, np.arange(len(documents)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # ---- Step 2: Preprocessing -------------------------------------------
    if verbose:
        print(f"[2/6] Preprocessing text (stemming={use_stemming}) ...")
    docs_train = preprocess_corpus(docs_train_raw, use_stemming=use_stemming, language=language)
    docs_test = preprocess_corpus(docs_test_raw, use_stemming=use_stemming, language=language)

    # ---- Step 3: TF-IDF Vectorization ------------------------------------
    if verbose:
        print(f"[3/6] Vectorizing with TF-IDF "
              f"(max_features={max_features}, ngram_range={ngram_range}) ...")
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_corpus(
        docs_train, docs_test,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    all_feature_names = vectorizer.get_feature_names_out().tolist()
    if verbose:
        print(f"       Vocabulary size: {len(all_feature_names)}")

    # ---- Step 4: Chi-squared Feature Selection ---------------------------
    if verbose:
        print(f"[4/6] Selecting top-{top_k_features} features via chi-squared ...")
    X_train_sel, X_test_sel, selector, selected_feature_names = select_features(
        X_train_tfidf, y_train, X_test_tfidf,
        feature_names=all_feature_names,
        k=top_k_features,
    )
    if verbose:
        print(f"       Features after selection: {len(selected_feature_names)}")

    # ---- Step 5: Training ------------------------------------------------
    if verbose:
        print(f"[5/6] Training LinearSVC (C={svc_C}) ...")
    clf = train_classifier(
        X_train_sel, y_train,
        C=svc_C,
        max_iter=svc_max_iter,
        class_weight=class_weight,
    )

    # ---- Step 6: Evaluation ----------------------------------------------
    if verbose:
        print("[6/6] Evaluating on held-out test set ...")
    metrics = evaluate_classifier(clf, X_test_sel, y_test, target_names=target_names)

    if verbose:
        print("\n" + "=" * 60)
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
        print("\nClassification Report:")
        print(metrics["classification_report"])
        print("=" * 60)

    return {
        "vectorizer": vectorizer,
        "selector": selector,
        "classifier": clf,
        "selected_feature_names": selected_feature_names,
        "metrics": metrics,
        "splits": {
            "train_indices": idx_train,
            "test_indices": idx_test,
            "train_docs_preprocessed": docs_train,
            "test_docs_preprocessed": docs_test,
            "y_train": y_train,
            "y_test": y_test,
        },
    }


# ---------------------------------------------------------------------------
# 7. Inference helper
# ---------------------------------------------------------------------------

def predict(
    pipeline_components: dict,
    new_documents: list[str],
    use_stemming: bool = False,
    language: str = "english",
) -> np.ndarray:
    """
    Run inference on new documents using a trained pipeline.

    Parameters
    ----------
    pipeline_components : dict
        Output of build_sentiment_pipeline.
    new_documents : list[str]
        Raw text documents to classify.
    use_stemming : bool
        Must match the setting used during training.
    language : str
        Must match the setting used during training.

    Returns
    -------
    np.ndarray
        Predicted labels.
    """
    vectorizer: TfidfVectorizer = pipeline_components["vectorizer"]
    selector: SelectKBest = pipeline_components["selector"]
    clf: LinearSVC = pipeline_components["classifier"]

    preprocessed = preprocess_corpus(new_documents, use_stemming=use_stemming, language=language)
    X_tfidf = vectorizer.transform(preprocessed)
    X_sel = selector.transform(X_tfidf)
    return clf.predict(X_sel)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal synthetic dataset for a quick smoke test
    _docs = [
        "I absolutely love this product, it is amazing!",
        "Terrible experience, would not recommend to anyone.",
        "Great quality and fast shipping, very happy.",
        "Worst purchase ever, complete waste of money.",
        "Fantastic! Exceeded all my expectations.",
        "Horrible customer service, never buying again.",
        "Really good value for the price, satisfied.",
        "Broken on arrival, very disappointed.",
        "Outstanding performance, highly recommend.",
        "Poor quality, fell apart after one day.",
        "Superb craftsmanship and excellent support.",
        "Awful product, nothing like the description.",
        "Delightful experience from start to finish.",
        "Dreadful quality, complete rubbish.",
        "Impressive features and easy to use.",
        "Useless gadget, total disappointment.",
        "Wonderful gift, everyone loved it.",
        "Disgusting quality, avoid at all costs.",
        "Brilliant product, works perfectly.",
        "Shocking service, still waiting for refund.",
    ]
    _labels = [
        "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative",
    ]

    result = build_sentiment_pipeline(
        _docs,
        _labels,
        test_size=0.25,
        use_stemming=True,
        max_features=500,
        ngram_range=(1, 2),
        top_k_features=100,
        target_names=["negative", "positive"],
        verbose=True,
    )

    # Inference demo
    new_texts = ["This is an incredible product!", "Terrible, I hate it."]
    preds = predict(result, new_texts, use_stemming=True)
    print("\nInference on new texts:")
    for text, label in zip(new_texts, preds):
        print(f"  '{text}'  =>  {label}")