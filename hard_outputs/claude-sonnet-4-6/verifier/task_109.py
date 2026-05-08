"""
Document Similarity Search System using Word2Vec Embeddings
"""

import re
import numpy as np
from typing import Optional
from collections import defaultdict

import gensim
from gensim.models import Word2Vec

# ---------------------------------------------------------------------------
# Optional: NLTK stop-words (fall back to a built-in list if unavailable)
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords

    try:
        STOPWORDS = set(nltk_stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(nltk_stopwords.words("english"))
except ImportError:
    # Minimal built-in stop-word list
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "this", "that", "these", "those", "it", "its", "as", "if",
        "then", "than", "when", "where", "who", "which", "what", "how",
    }


# ===========================================================================
# 1. Text pre-processing
# ===========================================================================

def preprocess(text: str) -> list[str]:
    """
    Tokenise *text*, lowercase every token, strip non-alphabetic characters,
    and remove stop-words.  Returns a list of clean tokens.
    """
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ===========================================================================
# 2. Document vectorisation
# ===========================================================================

def document_vector(tokens: list[str], model: Word2Vec) -> Optional[np.ndarray]:
    """
    Return the mean Word2Vec vector for *tokens*.
    Tokens that are out-of-vocabulary are silently ignored.
    Returns *None* when no token has a vector (empty / all-OOV document).
    """
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


# ===========================================================================
# 3. Cosine similarity
# ===========================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ===========================================================================
# 4. Retrieval
# ===========================================================================

def retrieve_top_k(
    query_vec: np.ndarray,
    index_vecs: dict[int, np.ndarray],
    k: int = 5,
) -> list[tuple[int, float]]:
    """
    Return the *k* index document ids most similar to *query_vec*, sorted by
    descending cosine similarity.

    Parameters
    ----------
    query_vec  : query embedding
    index_vecs : mapping {doc_id: embedding} for the searchable index
    k          : number of results to return

    Returns
    -------
    List of (doc_id, similarity_score) tuples.
    """
    scores = [
        (doc_id, cosine_similarity(query_vec, vec))
        for doc_id, vec in index_vecs.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


# ===========================================================================
# 5. Evaluation metrics
# ===========================================================================

def precision_at_k(retrieved: list[int], relevant: set[int], k: int = 5) -> float:
    """Fraction of the top-*k* retrieved documents that are relevant."""
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k if k > 0 else 0.0


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    """
    Reciprocal rank of the first relevant document in *retrieved*.
    Returns 0 if no relevant document is found.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# ===========================================================================
# 6. Main pipeline
# ===========================================================================

def build_similarity_search_system(
    documents: list[str],
    queries: list[str],
    relevant_docs: dict[int, set[int]],   # query_idx -> set of relevant doc_ids
    index_ratio: float = 0.8,
    w2v_vector_size: int = 100,
    w2v_window: int = 5,
    w2v_min_count: int = 1,
    w2v_epochs: int = 10,
    top_k: int = 5,
    random_seed: int = 42,
) -> dict:
    """
    Build a Word2Vec-based document similarity search system.

    Parameters
    ----------
    documents      : list of raw document strings (the corpus)
    queries        : list of raw query strings
    relevant_docs  : ground-truth relevance judgements
                     {query_index: {relevant_doc_id, ...}}
    index_ratio    : fraction of documents used as the searchable index
                     (the rest form the held-out evaluation set)
    w2v_*          : Word2Vec hyper-parameters
    top_k          : number of results to retrieve per query
    random_seed    : reproducibility seed

    Returns
    -------
    dict with keys:
        "model"            – trained gensim Word2Vec model
        "metrics"          – {"precision@5": float, "mrr": float}
        "retrieval_results"– list of per-query result dicts
        "index_doc_ids"    – list of document ids in the index
        "held_out_doc_ids" – list of held-out document ids
    """
    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Step 1 – Pre-process documents
    # ------------------------------------------------------------------
    tokenised_docs: list[list[str]] = [preprocess(doc) for doc in documents]

    # ------------------------------------------------------------------
    # Step 2 – Train Word2Vec on the full corpus
    # ------------------------------------------------------------------
    print("[INFO] Training Word2Vec model …")
    w2v_model = Word2Vec(
        sentences=tokenised_docs,
        vector_size=w2v_vector_size,
        window=w2v_window,
        min_count=w2v_min_count,
        workers=4,
        seed=random_seed,
        epochs=w2v_epochs,
    )
    print(f"[INFO] Vocabulary size: {len(w2v_model.wv)}")

    # ------------------------------------------------------------------
    # Step 3 – Compute document embeddings
    # ------------------------------------------------------------------
    doc_vectors: dict[int, Optional[np.ndarray]] = {}
    for doc_id, tokens in enumerate(tokenised_docs):
        vec = document_vector(tokens, w2v_model)
        doc_vectors[doc_id] = vec

    # ------------------------------------------------------------------
    # Step 4 – Split into index / held-out sets
    # ------------------------------------------------------------------
    all_ids = list(range(len(documents)))
    rng.shuffle(all_ids)
    split = max(1, int(len(all_ids) * index_ratio))
    index_doc_ids: list[int] = sorted(all_ids[:split])
    held_out_doc_ids: list[int] = sorted(all_ids[split:])

    # Build the searchable index (skip docs with no vector)
    index_vecs: dict[int, np.ndarray] = {
        doc_id: doc_vectors[doc_id]
        for doc_id in index_doc_ids
        if doc_vectors[doc_id] is not None
    }
    print(
        f"[INFO] Index size: {len(index_vecs)} docs  |  "
        f"Held-out size: {len(held_out_doc_ids)} docs"
    )

    # ------------------------------------------------------------------
    # Step 5 – Pre-process queries and compute query embeddings
    # ------------------------------------------------------------------
    tokenised_queries: list[list[str]] = [preprocess(q) for q in queries]
    query_vectors: list[Optional[np.ndarray]] = [
        document_vector(tokens, w2v_model) for tokens in tokenised_queries
    ]

    # ------------------------------------------------------------------
    # Step 6 – Retrieve top-k for each query and evaluate
    # ------------------------------------------------------------------
    precision_scores: list[float] = []
    rr_scores: list[float] = []
    retrieval_results: list[dict] = []

    for q_idx, (query, q_vec) in enumerate(zip(queries, query_vectors)):
        relevant = relevant_docs.get(q_idx, set())

        if q_vec is None:
            print(f"[WARN] Query {q_idx} has no valid tokens after pre-processing.")
            retrieval_results.append(
                {
                    "query_idx": q_idx,
                    "query": query,
                    "top_k_results": [],
                    "precision@5": 0.0,
                    "reciprocal_rank": 0.0,
                    "relevant_doc_ids": relevant,
                }
            )
            precision_scores.append(0.0)
            rr_scores.append(0.0)
            continue

        top_k_results = retrieve_top_k(q_vec, index_vecs, k=top_k)
        retrieved_ids = [doc_id for doc_id, _ in top_k_results]

        p_at_k = precision_at_k(retrieved_ids, relevant, k=top_k)
        rr = reciprocal_rank(retrieved_ids, relevant)

        precision_scores.append(p_at_k)
        rr_scores.append(rr)

        retrieval_results.append(
            {
                "query_idx": q_idx,
                "query": query,
                "top_k_results": [
                    {
                        "doc_id": doc_id,
                        "score": round(score, 4),
                        "snippet": documents[doc_id][:120] + "…"
                        if len(documents[doc_id]) > 120
                        else documents[doc_id],
                        "is_relevant": doc_id in relevant,
                    }
                    for doc_id, score in top_k_results
                ],
                "precision@5": round(p_at_k, 4),
                "reciprocal_rank": round(rr, 4),
                "relevant_doc_ids": relevant,
            }
        )

    # ------------------------------------------------------------------
    # Step 7 – Aggregate metrics
    # ------------------------------------------------------------------
    mean_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
    mrr = float(np.mean(rr_scores)) if rr_scores else 0.0

    metrics = {
        "precision@5": round(mean_precision, 4),
        "mrr": round(mrr, 4),
    }

    print(f"\n[RESULTS] Precision@{top_k}: {metrics['precision@5']:.4f}")
    print(f"[RESULTS] MRR:           {metrics['mrr']:.4f}")

    return {
        "model": w2v_model,
        "metrics": metrics,
        "retrieval_results": retrieval_results,
        "index_doc_ids": index_doc_ids,
        "held_out_doc_ids": held_out_doc_ids,
    }


# ===========================================================================
# 7. Demo / self-test
# ===========================================================================

def _demo() -> None:
    """
    Small self-contained demonstration using a synthetic corpus so the module
    can be run directly without any external data.
    """
    corpus = [
        # Technology cluster
        "Machine learning algorithms learn patterns from data automatically.",
        "Deep learning uses neural networks with many hidden layers.",
        "Natural language processing enables computers to understand human text.",
        "Computer vision systems can recognise objects in images and videos.",
        "Reinforcement learning trains agents through reward and punishment signals.",
        "Transfer learning reuses pre-trained models for new tasks efficiently.",
        "Convolutional neural networks excel at image classification tasks.",
        "Recurrent neural networks process sequential data like time series.",
        "Transformer models revolutionised natural language understanding tasks.",
        "Generative adversarial networks create realistic synthetic images.",
        # Science cluster
        "Quantum mechanics describes the behaviour of particles at atomic scales.",
        "Relativity theory explains gravity as curvature of spacetime.",
        "DNA carries genetic information encoded in nucleotide sequences.",
        "Evolution drives biodiversity through natural selection over generations.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "Black holes are regions where gravity prevents even light from escaping.",
        "Neurons transmit electrical signals throughout the nervous system.",
        "Climate change alters global weather patterns and sea levels.",
        "Vaccines stimulate the immune system to fight specific pathogens.",
        "Plate tectonics explains earthquakes and volcanic activity on Earth.",
        # History cluster
        "The Renaissance was a cultural movement that began in Italy.",
        "World War Two ended with the surrender of Germany and Japan.",
        "Ancient Rome built an extensive empire across Europe and beyond.",
        "The Industrial Revolution transformed manufacturing and society.",
        "The French Revolution overthrew the monarchy and established a republic.",
        "Ancient Egypt built monumental pyramids as royal tombs.",
        "The Cold War was a geopolitical tension between the USA and USSR.",
        "The Silk Road connected trade routes across Asia and Europe.",
        "The printing press enabled mass production of books and knowledge.",
        "The American Civil War ended slavery in the United States.",
    ]

    queries = [
        "neural network deep learning",
        "genetic DNA biology",
        "ancient history empire",
        "image recognition computer",
        "quantum physics particles",
    ]

    # Ground-truth: doc ids that are genuinely relevant to each query
    relevant_docs: dict[int, set[int]] = {
        0: {0, 1, 6, 7, 8, 9},   # neural / deep learning docs
        1: {12, 13, 16},           # biology / genetics docs
        2: {20, 21, 22, 24, 25},   # history / ancient docs
        3: {2, 3, 6},              # computer vision / image docs
        4: {10, 11, 15},           # quantum / physics docs
    }

    results = build_similarity_search_system(
        documents=corpus,
        queries=queries,
        relevant_docs=relevant_docs,
        index_ratio=0.8,
        w2v_vector_size=50,
        w2v_window=3,
        w2v_min_count=1,
        w2v_epochs=20,
        top_k=5,
        random_seed=42,
    )

    print("\n" + "=" * 60)
    print("EXAMPLE RETRIEVAL RESULTS")
    print("=" * 60)
    for qr in results["retrieval_results"]:
        print(f"\nQuery [{qr['query_idx']}]: \"{qr['query']}\"")
        print(f"  Precision@5 = {qr['precision@5']}  |  RR = {qr['reciprocal_rank']}")
        for rank, hit in enumerate(qr["top_k_results"], start=1):
            marker = "✓" if hit["is_relevant"] else " "
            print(
                f"  {rank}. [{marker}] doc_id={hit['doc_id']:2d}  "
                f"score={hit['score']:.4f}  \"{hit['snippet']}\""
            )

    print("\n" + "=" * 60)
    print(f"Overall Precision@5 : {results['metrics']['precision@5']}")
    print(f"Overall MRR         : {results['metrics']['mrr']}")
    print("=" * 60)


if __name__ == "__main__":
    _demo()