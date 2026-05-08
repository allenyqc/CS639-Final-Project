```python
"""
Learning-to-Rank System for Document Retrieval

This module implements a pointwise learning-to-rank approach for document retrieval,
using multiple features to predict relevance scores and rank documents per query.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, ndcg_score
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    "bm25_score",
    "tfidf_similarity",
    "doc_length",
    "query_term_freq",
    "pagerank_score",
]

@dataclass
class QueryDocumentPair:
    """Represents a single query-document pair with features and relevance label."""
    query_id: str
    doc_id: str
    features: Dict[str, float]          # keys must match FEATURE_NAMES
    relevance: int                       # 0, 1, or 2

@dataclass
class RankedDocument:
    """A document with its predicted score and true relevance."""
    doc_id: str
    predicted_score: float
    true_relevance: int
    rank: int

@dataclass
class QueryResult:
    """Per-query ranking result."""
    query_id: str
    ranked_docs: List[RankedDocument]
    ndcg_at_5: float
    ndcg_at_10: float
    map_score: float
    baseline_ndcg_at_5: float
    baseline_ndcg_at_10: float
    baseline_map: float

@dataclass
class EvaluationMetrics:
    """Aggregate evaluation metrics across all queries."""
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float
    mean_map: float
    baseline_mean_ndcg_at_5: float
    baseline_mean_ndcg_at_10: float
    baseline_mean_map: float
    improvement_ndcg_at_5: float
    improvement_ndcg_at_10: float
    improvement_map: float
    mse: float
    num_queries: int
    num_pairs: int

@dataclass
class LTRResult:
    """Full result object returned by the LTR system."""
    model: Any
    scaler: StandardScaler
    metrics: EvaluationMetrics
    per_query: Dict[str, QueryResult]
    feature_importances: Optional[Dict[str, float]]


# ─────────────────────────────────────────────
# Metric Helpers
# ─────────────────────────────────────────────

def _dcg(relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = [rel / np.log2(i + 2) for i, rel in enumerate(relevances)]
    return float(np.sum(gains))


def _ndcg(relevances: List[int], k: int) -> float:
    """Normalised DCG at k."""
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = _dcg(ideal, k)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(relevances, k) / ideal_dcg


def _average_precision(relevances: List[int]) -> float:
    """Mean Average Precision (binary: relevant if label >= 1)."""
    num_relevant = sum(1 for r in relevances if r >= 1)
    if num_relevant == 0:
        return 0.0
    precision_sum = 0.0
    hits = 0
    for i, rel in enumerate(relevances):
        if rel >= 1:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / num_relevant


# ─────────────────────────────────────────────
# Feature Matrix Builder
# ─────────────────────────────────────────────

def _build_feature_matrix(
    pairs: List[QueryDocumentPair],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Convert list of QueryDocumentPair into numpy arrays.

    Returns
    -------
    X        : (n, 5) feature matrix
    y        : (n,)   relevance labels
    groups   : (n,)   integer group ids (one per query)
    query_ids: list of query id strings (length n)
    doc_ids  : list of doc id strings   (length n)
    """
    rows, labels, query_ids, doc_ids = [], [], [], []
    for pair in pairs:
        row = [pair.features.get(f, 0.0) for f in FEATURE_NAMES]
        rows.append(row)
        labels.append(pair.relevance)
        query_ids.append(pair.query_id)
        doc_ids.append(pair.doc_id)

    X = np.array(rows, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)

    # Encode query ids as integer groups
    unique_qids = list(dict.fromkeys(query_ids))          # preserve order
    qid_to_int = {qid: i for i, qid in enumerate(unique_qids)}
    groups = np.array([qid_to_int[qid] for qid in query_ids], dtype=np.int64)

    return X, y, groups, query_ids, doc_ids


# ─────────────────────────────────────────────
# Ranking & Evaluation per Query
# ─────────────────────────────────────────────

def _rank_and_evaluate_query(
    doc_ids: List[str],
    true_relevances: List[int],
    predicted_scores: List[float],
    baseline_scores: List[float],
) -> QueryResult:
    """Rank documents by predicted score and compute metrics."""
    n = len(doc_ids)

    # ---- LTR ranking ----
    ltr_order = np.argsort(predicted_scores)[::-1]
    ranked_docs = [
        RankedDocument(
            doc_id=doc_ids[idx],
            predicted_score=predicted_scores[idx],
            true_relevance=true_relevances[idx],
            rank=rank + 1,
        )
        for rank, idx in enumerate(ltr_order)
    ]
    ltr_relevances = [true_relevances[idx] for idx in ltr_order]

    ndcg5  = _ndcg(ltr_relevances, 5)
    ndcg10 = _ndcg(ltr_relevances, 10)
    ap     = _average_precision(ltr_relevances)

    # ---- Baseline (BM25-only) ranking ----
    base_order = np.argsort(baseline_scores)[::-1]
    base_relevances = [true_relevances[idx] for idx in base_order]

    base_ndcg5  = _ndcg(base_relevances, 5)
    base_ndcg10 = _ndcg(base_relevances, 10)
    base_ap     = _average_precision(base_relevances)

    return QueryResult(
        query_id="",          # filled by caller
        ranked_docs=ranked_docs,
        ndcg_at_5=ndcg5,
        ndcg_at_10=ndcg10,
        map_score=ap,
        baseline_ndcg_at_5=base_ndcg5,
        baseline_ndcg_at_10=base_ndcg10,
        baseline_map=base_ap,
    )


# ─────────────────────────────────────────────
# Main LTR Function
# ─────────────────────────────────────────────

def build_ltr_system(
    train_pairs: List[QueryDocumentPair],
    test_pairs: List[QueryDocumentPair],
    model_type: str = "gradient_boosting",
    cv_folds: int = 3,
    random_state: int = 42,
) -> LTRResult:
    """
    Build and evaluate a pointwise Learning-to-Rank system.

    Parameters
    ----------
    train_pairs  : Training query-document pairs with relevance labels.
    test_pairs   : Test query-document pairs for evaluation.
    model_type   : One of 'gradient_boosting', 'random_forest', 'ridge'.
    cv_folds     : Number of cross-validation folds (group-aware).
    random_state : Random seed for reproducibility.

    Returns
    -------
    LTRResult containing the trained model, metrics, and per-query breakdowns.
    """
    if not train_pairs:
        raise ValueError("train_pairs must not be empty.")
    if not test_pairs:
        raise ValueError("test_pairs must not be empty.")

    # ── 1. Build feature matrices ──────────────────────────────────────────
    X_train, y_train, groups_train, _, _ = _build_feature_matrix(train_pairs)
    X_test,  y_test,  groups_test,  test_qids, test_dids = _build_feature_matrix(test_pairs)

    # ── 2. Scale features ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 3. Select & train model ───────────────────────────────────────────
    model = _create_model(model_type, random_state)

    # Optional: group-aware cross-validation on training set
    if cv_folds > 1 and len(np.unique(groups_train)) >= cv_folds:
        _cross_validate(model, X_train_s, y_train, groups_train, cv_folds)

    model.fit(X_train_s, y_train)

    # ── 4. Predict on test set ────────────────────────────────────────────
    y_pred = model.predict(X_test_s)
    mse    = float(mean_squared_error(y_test, y_pred))

    # BM25 baseline scores (first feature column)
    bm25_idx = FEATURE_NAMES.index("bm25_score")
    baseline_scores = X_test[:, bm25_idx]

    # ── 5. Group test pairs by query ──────────────────────────────────────
    query_groups: Dict[str, Dict[str, List]] = {}
    for i, qid in enumerate(test_qids):
        if qid not in query_groups:
            query_groups[qid] = {
                "doc_ids": [], "true_rel": [],
                "pred_scores": [], "bm25_scores": [],
            }
        query_groups[qid]["doc_ids"].append(test_dids[i])
        query_groups[qid]["true_rel"].append(int(y_test[i]))
        query_groups[qid]["pred_scores"].append(float(y_pred[i]))
        query_groups[qid]["bm25_scores"].append(float(baseline_scores[i]))

    # ── 6. Per-query evaluation ───────────────────────────────────────────
    per_query: Dict[str, QueryResult] = {}
    for qid, data in query_groups.items():
        result = _rank_and_evaluate_query(
            doc_ids=data["doc_ids"],
            true_relevances=data["true_rel"],
            predicted_scores=data["pred_scores"],
            baseline_scores=data["bm25_scores"],
        )
        result.query_id = qid
        per_query[qid] = result

    # ── 7. Aggregate metrics ──────────────────────────────────────────────
    metrics = _aggregate_metrics(per_query, mse, len(test_pairs))

    # ── 8. Feature importances (if available) ─────────────────────────────
    feature_importances = _extract_importances(model)

    return LTRResult(
        model=model,
        scaler=scaler,
        metrics=metrics,
        per_query=per_query,
        feature_importances=feature_importances,
    )


# ─────────────────────────────────────────────
# Helper: Model Factory
# ─────────────────────────────────────────────

def _create_model(model_type: str, random_state: int):
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=random_state,
        )
    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "ridge":
        return Ridge(alpha=1.0)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from 'gradient_boosting', 'random_forest', 'ridge'."
        )


# ─────────────────────────────────────────────
# Helper: Cross-Validation
# ─────────────────────────────────────────────

def _cross_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> List[float]:
    """Group-aware k-fold CV; returns per-fold MSE (informational only)."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_mses = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        fold_mses.append(mean_squared_error(y[val_idx], preds))
    return fold_mses


# ─────────────────────────────────────────────
# Helper: Aggregate Metrics
# ─────────────────────────────────────────────

def _aggregate_metrics(
    per_query: Dict[str, QueryResult],
    mse: float,
    num_pairs: int,
) -> EvaluationMetrics:
    ndcg5_list  = [r.ndcg_at_5        for r in per_query.values()]
    ndcg10_list = [r.ndcg_at_10       for r in per_query.values()]
    map_list    = [r.map_score         for r in per_query.values()]
    b_ndcg5     = [r.baseline_ndcg_at_5  for r in per_query.values()]
    b_ndcg10    = [r.baseline_ndcg_at_10 for r in per_query.values()]
    b_map       = [r.baseline_map        for r in per_query.values()]

    mean_n5  = float(np.mean(ndcg5_list))
    mean_n10 = float(np.mean(ndcg10_list))
    mean_map = float(np.mean(map_list))
    b_n5     = float(np.mean(b_ndcg5))
    b_n10    = float(np.mean(b_ndcg10))
    b_m      = float(np.mean(b_map))

    return EvaluationMetrics(
        mean_ndcg_at_5=mean_n5,
        mean_ndcg_at_10=mean_n10,
        mean_map=mean_map,
        baseline_mean_ndcg_at_5=b_n5,
        baseline_mean_ndcg_at_10=b_n10,
        baseline_mean_map=b_m,
        improvement_ndcg_at_5=mean_n5  - b_n5,
        improvement_ndcg_at_10=mean_n10 - b_n10,
        improvement_map=mean_map - b_m,
        mse=mse,
        num_queries=len(per_query),
        num_pairs=num_pairs,
    )


# ─────────────────────────────────────────────
# Helper: Feature Importances
# ─────────────────────────────────────────────

def _extract_importances(model) -> Optional[Dict[str, float]]:
    if hasattr(model, "feature_importances_"):
        return dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
    if hasattr(model, "coef_"):
        coefs = np.abs(model.coef_)
        coefs = coefs / coefs.sum() if coefs.sum() > 0 else coefs
        return dict(zip(FEATURE_NAMES, coefs.tolist()))
    return None


# ─────────────────────────────────────────────
# Inference Helper
# ─────────────────────────────────────────────

def rank_documents(
    ltr_result: LTRResult,
    query_id: str,
    pairs: List[QueryDocumentPair],
) -> List[RankedDocument]:
    """
    Rank a list of documents for a new query using a trained LTR model.

    Parameters
    ----------
    ltr_result : Trained LTRResult from build_ltr_system.
    query_id   : Identifier for the query (informational).
    pairs      : List of QueryDocumentPair (relevance labels may be 0/dummy).

    Returns
    -------
    List of RankedDocument sorted by predicted score (best first).
    """
    X, _, _, _, doc_ids = _build_feature_matrix(pairs)
    X_s = ltr_result.scaler.transform(X)
    scores = ltr_result.model.predict(X_s)

    order = np.argsort(scores)[::-1]
    return [
        RankedDocument(
            doc_id=doc_ids[idx],
            predicted_score=float(scores[idx]),
            true_relevance=pairs[idx].relevance,
            rank=rank + 1,
        )
        for rank, idx in enumerate(order)
    ]


# ─────────────────────────────────────────────
# Pretty Printer
# ─────────────────────────────────────────────

def print_report(result: LTRResult) -> None:
    """Print a human-readable summary of the LTR evaluation."""
    m = result.metrics
    sep = "─" * 60

    print(sep)
    print("  Learning-to-Rank Evaluation Report")
    print(sep)
    print(f"  Queries evaluated : {m.num_queries}")
    print(f"  Total test pairs  : {m.num_pairs}")
    print(f"  Regression MSE    : {m.mse:.4f}")
    print()
    print(f"  {'Metric':<22} {'LTR':>8}  {'BM25':>8}  {'Δ':>8}")
    print(f"  {'─'*22} {'─'*8}  {'─'*8}  {'─'*8}")
    print(f"  {'NDCG@5':<22} {m.mean_ndcg_at_5:>8.4f}  "
          f"{m.baseline_mean_ndcg_at_5:>8.4f}  "
          f"{m.improvement_ndcg_at_5:>+8.4f}")
    print(f"  {'NDCG@10':<22} {m.mean_ndcg_at_10:>8.4f}  "
          f"{m.baseline_mean_ndcg_at_10:>8.4f}  "
          f"{m.improvement_ndcg_at_10:>+8.4f}")