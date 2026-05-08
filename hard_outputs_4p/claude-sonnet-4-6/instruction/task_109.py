```python
"""
Document Similarity Search System using Word2Vec Embeddings.

This module builds a document similarity search system that:
1. Accepts a corpus of text documents and queries
2. Preprocesses documents (tokenize, lowercase, remove stopwords)
3. Trains a Word2Vec model on the training corpus
4. Represents documents as mean word vectors
5. Splits documents into index (train) and held-out evaluation sets
6. Finds top-5 most similar documents per query using cosine similarity
7. Evaluates retrieval quality using Precision@5 and MRR
8. Returns the trained model, evaluation metrics, and example results
"""

import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Optional: NLTK for stopwords and tokenization
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available; falling back to basic tokenization.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Holds the retrieval result for a single query."""

    query: str
    top5_doc_indices: list[int]
    top5_doc_texts: list[str]
    top5_similarities: list[float]
    relevant_doc_indices: list[int]
    precision_at_5: float
    reciprocal_rank: float


@dataclass
class EvaluationMetrics:
    """Aggregate evaluation metrics over all queries."""

    precision_at_5: float
    mrr: float
    per_query_results: list[RetrievalResult] = field(default_factory=list)


@dataclass
class SimilaritySearchSystem:
    """Container for the trained system and its outputs."""

    word2vec_model: Word2Vec
    index_doc_indices: list[int]  # original indices of index-set documents
    eval_doc_indices: list[int]  # original indices of held-out documents
    evaluation_metrics: EvaluationMetrics
    example_results: list[RetrievalResult]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_STOP_WORDS: set[str] = set()
if NLTK_AVAILABLE:
    _STOP_WORDS = set(stopwords.words("english"))
else:
    # Minimal English stop-word list as fallback
    _STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "this", "that", "these", "those", "it", "its", "i", "we", "you",
        "he", "she", "they", "me", "us", "him", "her", "them", "my",
        "our", "your", "his", "their", "what", "which", "who", "whom",
        "when", "where", "why", "how", "all", "each", "every", "more",
        "most", "other", "some", "such", "than", "then", "there",
    }


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, removing punctuation."""
    if NLTK_AVAILABLE:
        tokens = word_tokenize(text.lower())
    else:
        tokens = re.findall(r"[a-z]+", text.lower())
    return tokens


def preprocess_text(text: str, stop_words: set[str] | None = None) -> list[str]:
    """
    Preprocess a single document:
      - Lowercase
      - Tokenize
      - Remove stopwords and non-alphabetic tokens
      - Remove very short tokens (length < 2)

    Parameters
    ----------
    text : str
        Raw document text.
    stop_words : set[str] | None
        Set of stop words to remove. Defaults to English NLTK stop words.

    Returns
    -------
    list[str]
        List of cleaned tokens.
    """
    if stop_words is None:
        stop_words = _STOP_WORDS
    tokens = _tokenize(text)
    tokens = [
        t for t in tokens
        if t.isalpha() and len(t) >= 2 and t not in stop_words
    ]
    return tokens


# ---------------------------------------------------------------------------
# Document vector computation
# ---------------------------------------------------------------------------


def document_vector(
    tokens: list[str],
    model: Word2Vec,
    vector_size: int,
) -> np.ndarray:
    """
    Compute the mean Word2Vec vector for a list of tokens.

    Out-of-vocabulary (OOV) tokens are silently ignored.
    If no tokens are in-vocabulary, returns a zero vector.

    Parameters
    ----------
    tokens : list[str]
        Preprocessed tokens.
    model : Word2Vec
        Trained Word2Vec model.
    vector_size : int
        Dimensionality of word vectors.

    Returns
    -------
    np.ndarray
        Mean document vector of shape (vector_size,).
    """
    vectors = [
        model.wv[token]
        for token in tokens
        if token in model.wv
    ]
    if not vectors:
        return np.zeros(vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_top_k(
    query_vec: np.ndarray,
    index_vecs: np.ndarray,
    k: int = 5,
) -> tuple[list[int], list[float]]:
    """
    Retrieve the top-k most similar documents by cosine similarity.

    Parameters
    ----------
    query_vec : np.ndarray
        Query vector of shape (vector_size,).
    index_vecs : np.ndarray
        Matrix of document vectors, shape (n_docs, vector_size).
    k : int
        Number of top results to return.

    Returns
    -------
    tuple[list[int], list[float]]
        Sorted (descending) indices into index_vecs and their similarity scores.
    """
    if np.all(query_vec == 0):
        logger.warning("Query vector is all zeros (all tokens OOV). Returning empty results.")
        return [], []

    # cosine_similarity expects 2-D arrays
    sims = cosine_similarity(query_vec.reshape(1, -1), index_vecs)[0]
    top_k_local = int(min(k, len(sims)))
    top_indices = np.argsort(sims)[::-1][:top_k_local].tolist()
    top_sims = sims[top_indices].tolist()
    return top_indices, top_sims


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: list[int], relevant: list[int], k: int = 5) -> float:
    """Compute Precision@k."""
    if not relevant or not retrieved:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for idx in retrieved_k if idx in relevant_set)
    return hits / k


def reciprocal_rank(retrieved: list[int], relevant: list[int]) -> float:
    """Compute Reciprocal Rank (RR) for a single query."""
    relevant_set = set(relevant)
    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main system builder
# ---------------------------------------------------------------------------


def build_similarity_search_system(
    documents: list[str],
    queries: list[str],
    relevant_doc_indices: list[list[int]],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    epochs: int = 10,
    top_k: int = 5,
    n_example_results: int = 3,
) -> SimilaritySearchSystem:
    """
    Build and evaluate a Word2Vec-based document similarity search system.

    Data-leakage-free pipeline:
      1. Split document indices into train (index) / test (eval) BEFORE any fitting.
      2. Preprocess ALL documents using a fixed rule-based preprocessor
         (no learned parameters, so safe to apply to both splits).
      3. Train Word2Vec ONLY on the training (index) split.
      4. Compute document vectors for both splits using the trained model.
      5. Evaluate retrieval on the held-out (eval) split.

    Parameters
    ----------
    documents : list[str]
        Corpus of raw text documents.
    queries : list[str]
        List of query strings.
    relevant_doc_indices : list[list[int]]
        For each query, a list of document indices (into `documents`) that are
        considered relevant. Used for evaluation.
    test_size : float
        Fraction of documents to hold out for evaluation.
    random_state : int
        Random seed for reproducibility.
    vector_size : int
        Dimensionality of Word2Vec embeddings.
    window : int
        Word2Vec context window size.
    min_count : int
        Minimum word frequency for Word2Vec vocabulary.
    workers : int
        Number of worker threads for Word2Vec training.
    epochs : int
        Number of training epochs for Word2Vec.
    top_k : int
        Number of top documents to retrieve per query.
    n_example_results : int
        Number of example retrieval results to include in the output.

    Returns
    -------
    SimilaritySearchSystem
        Trained model, evaluation metrics, and example results.

    Raises
    ------
    ValueError
        If inputs are inconsistent.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not documents:
        raise ValueError("`documents` must be a non-empty list.")
    if not queries:
        raise ValueError("`queries` must be a non-empty list.")
    if len(queries) != len(relevant_doc_indices):
        raise ValueError(
            "`queries` and `relevant_doc_indices` must have the same length. "
            f"Got {len(queries)} queries and {len(relevant_doc_indices)} relevance lists."
        )
    n_docs = len(documents)
    for q_idx, rel_list in enumerate(relevant_doc_indices):
        for doc_idx in rel_list:
            if not (0 <= doc_idx < n_docs):
                raise ValueError(
                    f"relevant_doc_indices[{q_idx}] contains out-of-range index {doc_idx}. "
                    f"Valid range: [0, {n_docs - 1}]."
                )

    # ------------------------------------------------------------------
    # Step 1: Split document indices BEFORE any fitting
    # ------------------------------------------------------------------
    all_indices = list(range(n_docs))
    index_indices, eval_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(
        "Split: %d index documents, %d evaluation documents.",
        len(index_indices),
        len(eval_indices),
    )

    # ------------------------------------------------------------------
    # Step 2: Preprocess documents
    # Note: The preprocessor is purely rule-based (no learned parameters),
    # so it is safe to apply to all documents without leakage.
    # ------------------------------------------------------------------
    logger.info("Preprocessing documents...")
    all_tokens: list[list[str]] = [preprocess_text(doc) for doc in documents]

    # Tokens for the index (training) split only
    index_tokens: list[list[str]] = [all_tokens[i] for i in index_indices]

    # ------------------------------------------------------------------
    # Step 3: Train Word2Vec ONLY on the index (training) split
    # ------------------------------------------------------------------
    logger.info("Training Word2Vec on %d documents...", len(index_tokens))
    w2v_model = Word2Vec(
        sentences=index_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=random_state,
    )
    logger.info(
        "Word2Vec vocabulary size: %d", len(w2v_model.wv)
    )

    # ------------------------------------------------------------------
    # Step 4: Compute document vectors
    # ------------------------------------------------------------------
    logger.info("Computing document vectors...")
    all_doc_vecs: np.ndarray = np.vstack([
        document_vector(tokens, w2v_model, vector_size)
        for tokens in all_tokens
    ])  # shape: (n_docs, vector_size)

    index_vecs: np.ndarray = all_doc_vecs[index_indices]  # (n_index, vector_size)

    # ------------------------------------------------------------------
    # Step 5: Preprocess queries (using the same rule-based preprocessor)
    # ------------------------------------------------------------------
    logger.info("Preprocessing %d queries...", len(queries))
    query_tokens: list[list[str]] = [preprocess_text(q) for q in queries]
    query_vecs: np.ndarray = np.vstack([
        document_vector(tokens, w2v_model, vector_size)
        for tokens in query_tokens
    ])  # shape: (n_queries, vector_size)

    # ------------------------------------------------------------------
    # Step 6 & 7: Retrieve top-k and evaluate
    # ------------------------------------------------------------------
    logger.info("Running retrieval and evaluation...")

    # Map from local index-set position → original document index
    local_to_original: dict[int, int] = {
        local: orig for local, orig in enumerate(index_indices)
    }

    per_query_results: list[RetrievalResult] = []
    precision_scores: list[float] = []
    rr_scores: list[float] = []

    for q_idx, (query, q_vec, rel_orig_indices) in enumerate(
        zip(queries, query_vecs, relevant_doc_indices)
    ):
        # Retrieve top-k local indices within the index set
        top_local_indices, top_sims = retrieve_top_k(q_vec, index_vecs, k=top_k)

        # Convert local indices back to original document indices
        top_orig_indices: list[int] = [
            local_to_original[local_idx] for local_idx in top_local_indices
        ]

        # Compute metrics
        p5 = precision_at_k(top_orig_indices, rel_orig_indices, k=top_k)
        rr = reciprocal_rank(top_orig_indices, rel_orig_indices)

        precision_scores.append(p5)
        rr_scores.append(rr)

        result = RetrievalResult(
            query=query,
            top5_doc_indices=top_orig_indices,
            top5_doc_texts=[documents[i] for i in top_orig_indices],
            top5_similarities=top_sims,
            relevant_doc_indices=rel_orig_indices,
            precision_at_5=p5,
            reciprocal_rank=rr,
        )
        per_query_results.append(result)

    mean_p5 = float(np.mean(precision_scores)) if precision_scores else 0.0
    mrr = float(np.mean(rr_scores)) if rr_scores else 0.0

    logger.info("Precision@5: %.4f | MRR: %.4f", mean_p5, mrr)

    metrics = EvaluationMetrics(
        precision_at_5=mean_p5,
        mrr=mrr,
        per_query_results=per_query_results,
    )

    # ------------------------------------------------------------------
    # Step 8: Select example results
    # ------------------------------------------------------------------
    example_results = per_query_results[:n_example_results]

    return SimilaritySearchSystem(
        word2vec_model=w2v_model,
        index_doc_indices=index_indices,
        eval_doc_indices=eval_indices,
        evaluation_metrics=metrics,
        example_results=example_results,
    )


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------


def print_system_summary(system: SimilaritySearchSystem, max_text_len: int = 80) -> None:
    """Print a human-readable summary of the system's outputs."""
    metrics = system.evaluation_metrics
    print("=" * 70)
    print("DOCUMENT SIMILARITY SEARCH SYSTEM — SUMMARY")
    print("=" * 70)
    print(f"  Index set size   : {len(system.index_doc_indices)} documents")
    print(f"  Eval set size    : {len(system.eval_doc_indices)} documents")
    print(f"  W2V vocab size   : {len(system.word2vec_model.wv)}")
    print(f"  Precision@5      : {metrics.precision_at_5:.4f}")
    print(f"  MRR              : {metrics.mrr:.4f}")
    print()
    print("EXAMPLE RETRIEVAL RESULTS")
    print("-" * 70)
    for res in system.example_results:
        print(f"  Query : {res.query!r}")
        print(f"  Relevant doc indices: {res.relevant_doc_indices}")
        print(f"  Top-5 retrieved (orig idx | sim | text snippet):")
        for rank, (idx, sim, text) in enumerate(
            zip(res.top5_doc_indices, res.top5_similarities, res.top5_doc_texts),
            start=1,
        ):
            snippet = text[:max_text_len].replace("\n", " ")
            hit = "✓" if idx in res.relevant_doc_indices else " "
            print(f"    {rank}. [{hit}] doc#{idx:4d} | sim={sim:.4f} | {snippet!r}")
        print(f"  P@5={res.precision_at_5:.2f}  RR={res.reciprocal_rank:.4f}")
        print()


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------


def _demo() -> None:
    """Run a small demo with synthetic documents and queries."""
    corpus = [
        # Technology cluster
        "Machine learning algorithms improve predictive accuracy in data science.",
        "Deep learning neural networks excel at image recognition tasks.",
        "Natural language processing enables computers to understand human text.",
        "Reinforcement learning agents learn optimal policies through trial and error.",
        "Convolutional neural networks are widely used in computer vision.",
        "Transformer models revolutionized natural language understanding.",
        "Support vector machines classify data using hyperplane boundaries.",
        "Random forests combine multiple decision trees for robust predictions.",
        "Gradient boosting methods achieve state-of-the-art tabular performance.",
        "Autoencoders learn compressed representations of input data.",
        # Science cluster
        "Quantum mechanics describes the behavior of subatomic particles.",
        "General relativity explains gravity as curvature of spacetime.",
        "DNA replication ensures genetic information is passed to daughter cells.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The periodic table organizes elements by atomic number and properties.",
        "Black holes are regions where gravity prevents even light from escaping.",
        "Evolution by natural selection drives biodiversity on Earth.",
        "Neurons transmit electrical signals across synapses in the brain.",
        "Climate change is driven by greenhouse gas emissions from human activity.",
        "Plate tectonics explains the movement of Earth's lithospheric plates.",
        # Arts cluster
        "Impressionist painters captured fleeting light and color in their works.",
        "Shakespeare's plays explore themes of power, love