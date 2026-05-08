```python
"""
Document Similarity Search System using Word2Vec Embeddings.

This module builds a document similarity search system that:
1. Accepts a corpus of text documents and queries
2. Preprocesses documents (tokenize, lowercase, remove stopwords)
3. Trains a Word2Vec model on the corpus (training partition only)
4. Represents documents as mean word vectors
5. Splits documents into index/evaluation sets
6. Finds top-5 similar documents per query using cosine similarity
7. Evaluates retrieval quality with Precision@5 and MRR
8. Returns the trained model, metrics, and example results
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords (lightweight built-in set; swap for nltk.corpus.stopwords if needed)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "just", "as", "if",
    "this", "that", "these", "those", "it", "its", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "his", "her", "they", "their", "them",
})


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Stores retrieval results for a single query."""
    query: str
    query_tokens: list[str]
    top5_doc_indices: list[int]
    top5_similarities: list[float]
    top5_doc_snippets: list[str]
    relevant_doc_indices: list[int]
    precision_at_5: float
    reciprocal_rank: float


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics across all queries."""
    precision_at_5: float
    mean_reciprocal_rank: float
    num_queries: int
    per_query_results: list[RetrievalResult] = field(default_factory=list)


@dataclass
class SimilaritySearchSystem:
    """Container for the trained system and its outputs."""
    word2vec_model: Word2Vec
    evaluation_metrics: EvaluationMetrics
    example_results: list[RetrievalResult]
    index_doc_indices: list[int]       # indices into original corpus
    eval_doc_indices: list[int]        # held-out indices
    doc_vectors: np.ndarray            # shape (n_index_docs, vector_size)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def remove_stopwords(tokens: list[str], stopwords: frozenset[str] = _STOPWORDS) -> list[str]:
    """Remove stopwords from a token list."""
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def preprocess(text: str, stopwords: frozenset[str] = _STOPWORDS) -> list[str]:
    """Full preprocessing pipeline: tokenize → lowercase → remove stopwords."""
    return remove_stopwords(tokenize(text), stopwords)


# ---------------------------------------------------------------------------
# Document vector computation
# ---------------------------------------------------------------------------

def document_vector(
    tokens: list[str],
    model: Word2Vec,
    vector_size: int,
) -> Optional[np.ndarray]:
    """
    Compute the mean Word2Vec vector for a list of tokens.

    Returns None if no token is in the vocabulary (OOV document).
    OOV tokens are silently skipped.
    """
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def build_document_matrix(
    tokenized_docs: list[list[str]],
    model: Word2Vec,
    vector_size: int,
) -> tuple[np.ndarray, list[int]]:
    """
    Build a matrix of document vectors.

    Returns:
        matrix: shape (n_valid_docs, vector_size)
        valid_positions: positions in tokenized_docs that produced a valid vector
    """
    vectors = []
    valid_positions = []
    for i, tokens in enumerate(tokenized_docs):
        vec = document_vector(tokens, model, vector_size)
        if vec is not None:
            vectors.append(vec)
            valid_positions.append(i)
        else:
            logger.warning("Document %d produced no vector (all OOV); skipping.", i)
    if not vectors:
        raise ValueError("No documents produced valid vectors. Check your corpus.")
    return np.vstack(vectors), valid_positions


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_top_k(
    query_vector: np.ndarray,
    doc_matrix: np.ndarray,
    k: int = 5,
) -> tuple[list[int], list[float]]:
    """
    Retrieve the top-k most similar documents by cosine similarity.

    Args:
        query_vector: shape (vector_size,)
        doc_matrix:   shape (n_docs, vector_size)
        k:            number of results to return

    Returns:
        top_k_indices:      indices into doc_matrix (sorted by descending similarity)
        top_k_similarities: corresponding cosine similarity scores
    """
    query_2d = query_vector.reshape(1, -1)
    sims = cosine_similarity(query_2d, doc_matrix)[0]  # shape (n_docs,)
    top_k = min(k, len(sims))
    top_indices = np.argsort(sims)[::-1][:top_k].tolist()
    top_sims = sims[top_indices].tolist()
    return top_indices, top_sims


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[int], relevant: list[int], k: int = 5) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for idx in retrieved_k if idx in relevant)
    return hits / k


def reciprocal_rank(retrieved: list[int], relevant: list[int]) -> float:
    """Reciprocal rank of the first relevant document in the retrieved list."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main system builder
# ---------------------------------------------------------------------------

def build_similarity_search_system(
    corpus: list[str],
    queries: list[str],
    relevant_docs: list[list[int]],
    *,
    eval_fraction: float = 0.2,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    epochs: int = 10,
    top_k: int = 5,
    random_seed: int = 42,
    stopwords: Optional[frozenset[str]] = None,
) -> SimilaritySearchSystem:
    """
    Build and evaluate a Word2Vec-based document similarity search system.

    Args:
        corpus:        List of raw text documents.
        queries:       List of query strings.
        relevant_docs: For each query, a list of corpus indices considered relevant.
                       Indices refer to the *original* corpus list.
        eval_fraction: Fraction of corpus to hold out for evaluation.
        vector_size:   Dimensionality of Word2Vec embeddings.
        window:        Context window size for Word2Vec.
        min_count:     Minimum word frequency for Word2Vec vocabulary.
        workers:       Number of training threads.
        epochs:        Number of training epochs.
        top_k:         Number of documents to retrieve per query.
        random_seed:   Random seed for reproducibility.
        stopwords:     Custom stopword set; defaults to built-in set.

    Returns:
        SimilaritySearchSystem dataclass with model, metrics, and results.

    Raises:
        ValueError: If corpus is empty, queries/relevant_docs lengths mismatch,
                    or no valid document vectors can be built.
    """
    if not corpus:
        raise ValueError("corpus must not be empty.")
    if len(queries) != len(relevant_docs):
        raise ValueError(
            f"queries ({len(queries)}) and relevant_docs ({len(relevant_docs)}) "
            "must have the same length."
        )

    _stopwords = stopwords if stopwords is not None else _STOPWORDS
    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Step 1: Split corpus BEFORE any preprocessing
    # ------------------------------------------------------------------
    n_docs = len(corpus)
    n_eval = max(1, int(n_docs * eval_fraction))
    n_train = n_docs - n_eval

    all_indices = np.arange(n_docs)
    rng.shuffle(all_indices)
    train_indices: list[int] = sorted(all_indices[:n_train].tolist())
    eval_indices: list[int] = sorted(all_indices[n_train:].tolist())

    logger.info(
        "Corpus split: %d train (index) documents, %d held-out evaluation documents.",
        len(train_indices),
        len(eval_indices),
    )

    # ------------------------------------------------------------------
    # Step 2: Preprocess — fit vocabulary ONLY on training partition
    # ------------------------------------------------------------------
    train_docs_raw = [corpus[i] for i in train_indices]
    eval_docs_raw = [corpus[i] for i in eval_indices]

    train_tokens: list[list[str]] = [preprocess(doc, _stopwords) for doc in train_docs_raw]
    eval_tokens: list[list[str]] = [preprocess(doc, _stopwords) for doc in eval_docs_raw]
    query_tokens: list[list[str]] = [preprocess(q, _stopwords) for q in queries]

    # ------------------------------------------------------------------
    # Step 3: Train Word2Vec on training partition only
    # ------------------------------------------------------------------
    logger.info("Training Word2Vec (vector_size=%d, window=%d, epochs=%d)...",
                vector_size, window, epochs)
    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=random_seed,
    )
    logger.info("Vocabulary size: %d", len(w2v_model.wv))

    # ------------------------------------------------------------------
    # Step 4: Build document vectors for the index (training) set
    # ------------------------------------------------------------------
    index_matrix, valid_local_positions = build_document_matrix(
        train_tokens, w2v_model, vector_size
    )
    # Map local positions back to original corpus indices
    index_doc_indices = [train_indices[p] for p in valid_local_positions]
    logger.info(
        "Index matrix shape: %s (%d/%d documents have valid vectors).",
        index_matrix.shape,
        len(index_doc_indices),
        len(train_indices),
    )

    # ------------------------------------------------------------------
    # Step 5 & 6: For each query, retrieve top-k from the index set
    # ------------------------------------------------------------------
    per_query_results: list[RetrievalResult] = []

    for q_idx, (query_raw, q_tokens, rel_orig_indices) in enumerate(
        zip(queries, query_tokens, relevant_docs)
    ):
        # Compute query vector (OOV tokens silently skipped)
        q_vec = document_vector(q_tokens, w2v_model, vector_size)

        if q_vec is None:
            logger.warning(
                "Query %d ('%s') produced no vector (all OOV). "
                "Assigning zero scores.",
                q_idx,
                query_raw[:60],
            )
            top_local_indices: list[int] = list(range(min(top_k, len(index_doc_indices))))
            top_sims: list[float] = [0.0] * len(top_local_indices)
        else:
            top_local_indices, top_sims = retrieve_top_k(q_vec, index_matrix, k=top_k)

        # Convert local index-matrix positions → original corpus indices
        top_orig_indices = [index_doc_indices[li] for li in top_local_indices]

        # Map relevant original indices to those present in the index set
        rel_in_index = [i for i in rel_orig_indices if i in index_doc_indices]

        p5 = precision_at_k(top_orig_indices, rel_in_index, k=top_k)
        rr = reciprocal_rank(top_orig_indices, rel_in_index)

        snippets = [corpus[oi][:120] for oi in top_orig_indices]

        per_query_results.append(
            RetrievalResult(
                query=query_raw,
                query_tokens=q_tokens,
                top5_doc_indices=top_orig_indices,
                top5_similarities=top_sims,
                top5_doc_snippets=snippets,
                relevant_doc_indices=rel_in_index,
                precision_at_5=p5,
                reciprocal_rank=rr,
            )
        )

    # ------------------------------------------------------------------
    # Step 7: Aggregate evaluation metrics
    # ------------------------------------------------------------------
    avg_p5 = float(np.mean([r.precision_at_5 for r in per_query_results]))
    avg_mrr = float(np.mean([r.reciprocal_rank for r in per_query_results]))

    metrics = EvaluationMetrics(
        precision_at_5=avg_p5,
        mean_reciprocal_rank=avg_mrr,
        num_queries=len(queries),
        per_query_results=per_query_results,
    )

    logger.info(
        "Evaluation — Precision@5: %.4f | MRR: %.4f (over %d queries)",
        avg_p5,
        avg_mrr,
        len(queries),
    )

    # ------------------------------------------------------------------
    # Step 8: Return system
    # ------------------------------------------------------------------
    example_results = per_query_results[:min(3, len(per_query_results))]

    return SimilaritySearchSystem(
        word2vec_model=w2v_model,
        evaluation_metrics=metrics,
        example_results=example_results,
        index_doc_indices=index_doc_indices,
        eval_doc_indices=eval_indices,
        doc_vectors=index_matrix,
    )


# ---------------------------------------------------------------------------
# Convenience: query a trained system at inference time
# ---------------------------------------------------------------------------

def query_system(
    system: SimilaritySearchSystem,
    corpus: list[str],
    query: str,
    top_k: int = 5,
    stopwords: Optional[frozenset[str]] = None,
) -> list[tuple[int, float, str]]:
    """
    Query a trained SimilaritySearchSystem for the most similar documents.

    Args:
        system:   A trained SimilaritySearchSystem.
        corpus:   The original corpus list (needed for text snippets).
        query:    Raw query string.
        top_k:    Number of results to return.
        stopwords: Custom stopword set.

    Returns:
        List of (original_corpus_index, cosine_similarity, text_snippet) tuples.
    """
    _stopwords = stopwords if stopwords is not None else _STOPWORDS
    q_tokens = preprocess(query, _stopwords)
    q_vec = document_vector(q_tokens, system.word2vec_model, system.word2vec_model.vector_size)

    if q_vec is None:
        logger.warning("Query produced no vector (all OOV). Returning empty results.")
        return []

    top_local, top_sims = retrieve_top_k(q_vec, system.doc_vectors, k=top_k)
    results = []
    for li, sim in zip(top_local, top_sims):
        orig_idx = system.index_doc_indices[li]
        snippet = corpus[orig_idx][:120]
        results.append((orig_idx, sim, snippet))
    return results


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _DEMO_CORPUS = [
        "Machine learning is a subset of artificial intelligence focused on building systems that learn from data.",
        "Deep learning uses neural networks with many layers to model complex patterns in data.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and understand visual information from the world.",
        "Reinforcement learning trains agents to make decisions by rewarding desired behaviors.",
        "Support vector machines are supervised learning models used for classification and regression.",
        "Random forests combine multiple decision trees to improve predictive accuracy.",
        "Gradient boosting builds an ensemble of weak learners sequentially to reduce errors.",
        "Convolutional neural networks are particularly effective for image recognition tasks.",
        "Recurrent neural networks are designed to work with sequential data such as text or time series.",
        "Transfer learning leverages pre-trained models to solve new but related tasks efficiently.",
        "Clustering algorithms group similar data points together without labeled training data.",
        "Dimensionality reduction techniques like PCA compress data while preserving structure.",
        "Bayesian inference updates beliefs about model parameters given observed data.",
        "Generative adversarial networks consist of a generator and discriminator trained together.",
        "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
        "Transformers have revolutionized NLP by enabling parallel processing of sequences.",
        "Word embeddings represent words as dense vectors capturing semantic relationships.",
        "Autoencoders learn compressed representations of data in an unsupervised manner.",
        "Hyperparameter tuning optimizes model configuration to improve generalization.",
    ]

    _DEMO_QUERIES = [
        "neural networks for image classification",
        "text understanding with language models",
        "unsupervised clustering of data points",
    ]

    # Relevant documents per query (indices into _DEMO_CORPUS)
    _DEMO_RELEVANT = [
        [1, 4, 8, 9],   # neural networks / deep learning / CNN / RNN
        [2, 15, 16, 17],  # NLP / attention / transformers / word embeddings
        [11, 12, 18],    # clustering / dimensionality reduction / autoencoders
    ]

    system = build_similarity_search_system(
        corpus=_DEMO_CORPUS,
        queries=_DEMO_QUERIES,
        relevant_docs=_DEMO_RELEVANT,
        eval_fraction=0.2,
        vector_size=50,
        window=3,
        min_count=1,
        epochs=20,
        random_seed=42,
    )

    print("\n=== Evaluation Metrics ===")
    print(f"Precision@5 : {system.evaluation_metrics.precision_at_5:.4f}")
    print(f"MRR         : {system.evaluation_metrics.mean_reciprocal_rank:.4f}")

    print("\n=== Example Retrieval Results ===")
    for result in system.example_results:
        print(f"\nQuery: '{result.query}'")
        print(f"  P@5={result.precision_at_5:.2f}  RR={result.reciprocal_rank:.2f}")
        for rank, (idx, sim, snippet) in enumerate(