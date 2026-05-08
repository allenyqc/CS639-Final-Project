"""
Document Similarity Search System using Word2Vec Embeddings
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
except ImportError:
    raise ImportError("Please install gensim: pip install gensim")

try:
    from nltk.corpus import stopwords
    import nltk
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
except ImportError:
    # Fallback stopwords if NLTK is not available
    STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "s", "t", "can", "will", "just", "don", "should",
        "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
        "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
        "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
        "won", "wouldn",
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(text: str, stop_words: Optional[set] = None) -> List[str]:
    """
    Tokenize, lowercase, remove punctuation and stopwords from a text string.

    Parameters
    ----------
    text : str
        Raw input text.
    stop_words : set, optional
        Set of stopwords to remove. Defaults to STOP_WORDS.

    Returns
    -------
    List[str]
        List of cleaned tokens.
    """
    if stop_words is None:
        stop_words = STOP_WORDS

    # Lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize using gensim's simple_preprocess (min_len=2 removes single chars)
    tokens = simple_preprocess(text, deacc=True, min_len=2)

    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    return tokens


# ---------------------------------------------------------------------------
# Document Vector Representation
# ---------------------------------------------------------------------------

def document_vector(
    tokens: List[str],
    model: Word2Vec,
    vector_size: int,
) -> np.ndarray:
    """
    Compute the mean Word2Vec vector for a list of tokens.

    Out-of-vocabulary (OOV) tokens are silently ignored.  If *all* tokens are
    OOV the function returns a zero vector.

    Parameters
    ----------
    tokens : List[str]
        Pre-processed tokens for a single document.
    model : Word2Vec
        Trained gensim Word2Vec model.
    vector_size : int
        Dimensionality of the word vectors.

    Returns
    -------
    np.ndarray
        Mean embedding vector of shape (vector_size,).
    """
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])

    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size, dtype=np.float32)


# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns 0.0 if either vector is the zero vector.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_top_k(
    query_vec: np.ndarray,
    index_vectors: np.ndarray,
    index_ids: List[Any],
    k: int = 5,
) -> List[Tuple[Any, float]]:
    """
    Retrieve the top-k most similar documents from the index.

    Parameters
    ----------
    query_vec : np.ndarray
        Query embedding vector.
    index_vectors : np.ndarray
        Matrix of shape (n_docs, vector_size) containing index document vectors.
    index_ids : List[Any]
        Document identifiers aligned with rows of *index_vectors*.
    k : int
        Number of results to return.

    Returns
    -------
    List[Tuple[Any, float]]
        Sorted list of (doc_id, similarity_score) pairs, highest first.
    """
    # Vectorised cosine similarity
    norms = np.linalg.norm(index_vectors, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0.0:
        return [(doc_id, 0.0) for doc_id in index_ids[:k]]

    # Avoid division by zero for zero-norm document vectors
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalised = index_vectors / safe_norms
    similarities = normalised @ query_vec / query_norm

    # Zero out similarities for zero-norm documents
    zero_mask = (norms.squeeze() == 0)
    similarities[zero_mask] = 0.0

    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [(index_ids[i], float(similarities[i])) for i in top_k_indices]


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: List[Any], relevant: List[Any], k: int = 5) -> float:
    """
    Compute Precision@k.

    Parameters
    ----------
    retrieved : List[Any]
        Ordered list of retrieved document IDs (top-k).
    relevant : List[Any]
        List of ground-truth relevant document IDs.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        Precision@k value in [0, 1].
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    retrieved_at_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
    return hits / k


def reciprocal_rank(retrieved: List[Any], relevant: List[Any]) -> float:
    """
    Compute the Reciprocal Rank for a single query.

    Parameters
    ----------
    retrieved : List[Any]
        Ordered list of retrieved document IDs.
    relevant : List[Any]
        List of ground-truth relevant document IDs.

    Returns
    -------
    float
        Reciprocal rank value in (0, 1], or 0.0 if no relevant doc found.
    """
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main System
# ---------------------------------------------------------------------------

def build_similarity_search_system(
    documents: List[str],
    queries: List[str],
    relevant_docs: Dict[str, List[int]],
    doc_ids: Optional[List[Any]] = None,
    eval_fraction: float = 0.2,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    epochs: int = 10,
    top_k: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Build and evaluate a Word2Vec-based document similarity search system.

    Parameters
    ----------
    documents : List[str]
        Corpus of raw text documents.
    queries : List[str]
        List of query strings to evaluate.
    relevant_docs : Dict[str, List[int]]
        Mapping from query string to list of relevant document indices
        (0-based indices into *documents*).
    doc_ids : List[Any], optional
        Custom document identifiers. Defaults to integer indices.
    eval_fraction : float
        Fraction of documents to hold out for evaluation (default 0.2).
    vector_size : int
        Dimensionality of Word2Vec embeddings.
    window : int
        Context window size for Word2Vec.
    min_count : int
        Minimum word frequency for Word2Vec vocabulary.
    workers : int
        Number of worker threads for Word2Vec training.
    epochs : int
        Number of training epochs for Word2Vec.
    top_k : int
        Number of top results to retrieve per query.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "model"           : trained gensim Word2Vec model
        - "metrics"         : {"precision@5": float, "mrr": float}
        - "retrieval_results": per-query retrieval details
        - "index_doc_ids"   : IDs of documents in the search index
        - "eval_doc_ids"    : IDs of held-out evaluation documents
    """
    np.random.seed(random_seed)

    if not documents:
        raise ValueError("The documents list must not be empty.")

    n_docs = len(documents)

    # Assign document IDs
    if doc_ids is None:
        doc_ids = list(range(n_docs))
    elif len(doc_ids) != n_docs:
        raise ValueError("Length of doc_ids must match length of documents.")

    # ------------------------------------------------------------------
    # Step 1 – Preprocess all documents
    # ------------------------------------------------------------------
    print("[1/6] Preprocessing documents …")
    tokenized_docs = [preprocess_text(doc) for doc in documents]

    # ------------------------------------------------------------------
    # Step 2 – Train Word2Vec
    # ------------------------------------------------------------------
    print("[2/6] Training Word2Vec model …")
    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=random_seed,
    )
    print(f"       Vocabulary size: {len(model.wv)}")

    # ------------------------------------------------------------------
    # Step 3 – Compute document vectors
    # ------------------------------------------------------------------
    print("[3/6] Computing document vectors …")
    doc_vectors = np.array(
        [document_vector(tokens, model, vector_size) for tokens in tokenized_docs],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------
    # Step 4 – Split into index and evaluation sets
    # ------------------------------------------------------------------
    print("[4/6] Splitting into index / evaluation sets …")
    indices = np.arange(n_docs)
    np.random.shuffle(indices)

    n_eval = max(1, int(n_docs * eval_fraction))
    eval_indices = indices[:n_eval].tolist()
    index_indices = indices[n_eval:].tolist()

    # Ensure at least one document in the index
    if not index_indices:
        index_indices = eval_indices[:1]
        eval_indices = eval_indices[1:] or eval_indices[:1]

    index_doc_ids = [doc_ids[i] for i in index_indices]
    eval_doc_ids = [doc_ids[i] for i in eval_indices]
    index_vectors = doc_vectors[index_indices]

    print(f"       Index size: {len(index_doc_ids)} | Eval size: {len(eval_doc_ids)}")

    # ------------------------------------------------------------------
    # Step 5 – Preprocess queries and retrieve results
    # ------------------------------------------------------------------
    print("[5/6] Running retrieval for each query …")
    retrieval_results: Dict[str, Any] = {}
    precision_scores: List[float] = []
    rr_scores: List[float] = []

    for query in queries:
        q_tokens = preprocess_text(query)
        q_vec = document_vector(q_tokens, model, vector_size)

        top_results = retrieve_top_k(q_vec, index_vectors, index_doc_ids, k=top_k)
        retrieved_ids = [doc_id for doc_id, _ in top_results]

        # Map relevant doc indices to doc_ids
        relevant_ids = [doc_ids[idx] for idx in relevant_docs.get(query, [])]

        p_at_k = precision_at_k(retrieved_ids, relevant_ids, k=top_k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        precision_scores.append(p_at_k)
        rr_scores.append(rr)

        retrieval_results[query] = {
            "query_tokens": q_tokens,
            "top_results": top_results,          # [(doc_id, score), …]
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "precision@5": p_at_k,
            "reciprocal_rank": rr,
        }

    # ------------------------------------------------------------------
    # Step 6 – Aggregate metrics
    # ------------------------------------------------------------------
    print("[6/6] Computing evaluation metrics …")
    mean_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    mrr = float(np.mean(rr_scores)) if rr_scores else 0.0

    metrics = {
        "precision@5": mean_precision,
        "mrr": mrr,
        "per_query_precision@5": {q: retrieval_results[q]["precision@5"] for q in queries},
        "per_query_rr": {q: retrieval_results[q]["reciprocal_rank"] for q in queries},
    }

    print("\n=== Evaluation Results ===")
    print(f"  Mean Precision@{top_k} : {mean_precision:.4f}")
    print(f"  MRR              : {mrr:.4f}")

    return {
        "model": model,
        "metrics": metrics,
        "retrieval_results": retrieval_results,
        "index_doc_ids": index_doc_ids,
        "eval_doc_ids": eval_doc_ids,
        "tokenized_docs": tokenized_docs,
        "doc_vectors": doc_vectors,
    }


# ---------------------------------------------------------------------------
# Demo / Self-test
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Run a small self-contained demonstration."""

    corpus = [
        # Technology
        "Machine learning algorithms improve with more data and better features.",
        "Deep learning neural networks have revolutionized computer vision tasks.",
        "Natural language processing enables computers to understand human text.",
        "Reinforcement learning agents learn by interacting with their environment.",
        "Transfer learning allows models trained on one task to be applied to another.",
        # Science
        "Quantum mechanics describes the behavior of particles at the atomic scale.",
        "The theory of relativity changed our understanding of space and time.",
        "DNA carries the genetic information for all living organisms on Earth.",
        "Climate change is driven by greenhouse gas emissions from human activities.",
        "The Higgs boson was discovered at the Large Hadron Collider in 2012.",
        # Sports
        "Football is the most popular sport in the world by number of fans.",
        "The Olympic Games bring together athletes from every nation on Earth.",
        "Basketball was invented by James Naismith in Springfield Massachusetts.",
        "Tennis grand slams are held in Australia France England and the United States.",
        "Swimming is an excellent full-body workout that improves cardiovascular health.",
        # Food
        "Italian cuisine is famous for pasta pizza and gelato around the world.",
        "Fermentation is used to produce bread cheese wine and many other foods.",
        "Spices have been traded across continents for thousands of years.",
        "Vegetarian and vegan diets are growing in popularity for health and ethics.",
        "Chocolate is made from cacao beans which are native to Central America.",
    ]

    queries = [
        "deep learning and neural networks for image recognition",
        "genetic information and DNA in biology",
        "popular team sports and competitions",
        "fermented foods and traditional cooking techniques",
    ]

    # Ground-truth: indices of relevant documents for each query
    relevant_docs: Dict[str, List[int]] = {
        "deep learning and neural networks for image recognition": [0, 1, 2, 3, 4],
        "genetic information and DNA in biology": [5, 6, 7, 8, 9],
        "popular team sports and competitions": [10, 11, 12, 13, 14],
        "fermented foods and traditional cooking techniques": [15, 16, 17, 18, 19],
    }

    results = build_similarity_search_system(
        documents=corpus,
        queries=queries,
        relevant_docs=relevant_docs,
        eval_fraction=0.2,
        vector_size=50,
        window=3,
        min_count=1,
        epochs=20,
        top_k=5,
        random_seed=42,
    )

    print("\n=== Example Retrieval Results ===")
    for query, info in results["retrieval_results"].items():
        print(f"\nQuery : {query!r}")
        print(f"  Tokens  : {info['query_tokens']}")
        print(f"  P@5     : {info['precision@5']:.2f}  |  RR: {info['reciprocal_rank']:.4f}")
        print("  Top-5 retrieved documents:")
        for rank, (doc_id, score) in enumerate(info["top_results"], start=1):
            snippet = corpus[doc_id][:70] + ("…" if len(corpus[doc_id]) > 70 else "")
            print(f"    {rank}. [id={doc_id}, sim={score:.4f}] {snippet}")

    print("\n=== Final Metrics ===")
    print(f"  Precision@5 : {results['metrics']['precision@5']:.4f}")
    print(f"  MRR         : {results['metrics']['mrr']:.4f}")
    print(f"\nWord2Vec vocabulary size: {len(results['model'].wv)}")


if __name__ == "__main__":
    _demo()