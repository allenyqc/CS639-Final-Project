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

@dataclass
class QueryDocumentPair:
    """Represents a single query-document pair with features and relevance label."""
    query_id: str
    doc_id: str
    bm25_score: float
    tfidf_similarity: float
    doc_length: float
    query_term_frequency: float
    pagerank_score: float
    relevance_label: int  # 0=irrelevant, 1=partial, 2=highly relevant

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.bm25_score,
            self.tfidf_similarity,
            self.doc_length,
            self.query_term_frequency,
            self.pagerank_score,
        ])


@dataclass
class RankingResult:
    """Stores ranking results for a single query."""
    query_id: str
    doc_ids: List[str]
    true_labels: List[int]
    predicted_scores: List[float]
    bm25_scores: List[float]
    ranked_doc_ids: List[str] = field(default_factory=list)
    ranked_true_labels: List[int] = field(default_factory=list)
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    map_score: float = 0.0
    bm25_ndcg_at_5: float = 0.0
    bm25_ndcg_at_10: float = 0.0
    bm25_map_score: float = 0.0


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics across all queries."""
    mean_ndcg_at_5: float
    mean_ndcg_at_10: float
    mean_average_precision: float
    bm25_mean_ndcg_at_5: float
    bm25_mean_ndcg_at_10: float
    bm25_mean_average_precision: float
    improvement_ndcg_at_5: float
    improvement_ndcg_at_10: float
    improvement_map: float
    rmse: float
    num_queries: int
    num_pairs: int


# ─────────────────────────────────────────────
# Metric Helpers
# ─────────────────────────────────────────────

def _dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    gains = [rel / np.log2(i + 2) for i, rel in enumerate(relevances)]
    return sum(gains)


def _ndcg_at_k(true_labels: List[int], predicted_scores: List[float], k: int) -> float:
    """Compute Normalised DCG at k."""
    if len(true_labels) == 0:
        return 0.0
    # Sort by predicted score descending
    order = np.argsort(predicted_scores)[::-1]
    ranked_labels = [true_labels[i] for i in order]
    # Ideal ranking
    ideal_labels = sorted(true_labels, reverse=True)
    dcg = _dcg_at_k(ranked_labels, k)
    idcg = _dcg_at_k(ideal_labels, k)
    return dcg / idcg if idcg > 0 else 0.0


def _average_precision(true_labels: List[int], predicted_scores: List[float]) -> float:
    """Compute Average Precision (binary relevance: label > 0 is relevant)."""
    if not any(l > 0 for l in true_labels):
        return 0.0
    order = np.argsort(predicted_scores)[::-1]
    ranked_labels = [true_labels[i] for i in order]
    precisions, num_relevant = [], 0
    for rank, label in enumerate(ranked_labels, start=1):
        if label > 0:
            num_relevant += 1
            precisions.append(num_relevant / rank)
    total_relevant = sum(1 for l in true_labels if l > 0)
    return sum(precisions) / total_relevant if total_relevant > 0 else 0.0


# ─────────────────────────────────────────────
# Core LTR System
# ─────────────────────────────────────────────

class LearningToRankSystem:
    """
    Pointwise Learning-to-Rank system for document retrieval.

    Features used:
        - BM25 score
        - TF-IDF cosine similarity
        - Document length (normalised)
        - Query term frequency
        - PageRank-like authority score

    The model predicts a continuous relevance score; documents are then
    ranked per query by that predicted score.
    """

    FEATURE_NAMES = [
        "bm25_score",
        "tfidf_similarity",
        "doc_length",
        "query_term_frequency",
        "pagerank_score",
    ]

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        n_splits: int = 3,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        model_type : str
            One of 'gradient_boosting', 'random_forest', 'ridge'.
        n_splits : int
            Number of cross-validation folds (grouped by query).
        random_state : int
            Random seed for reproducibility.
        """
        self.model_type = model_type
        self.n_splits = n_splits
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.scaler = StandardScaler()
        self._is_fitted = False

    # ── Model factory ──────────────────────────────────────────────────────

    def _build_model(self) -> Any:
        if self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=self.random_state,
            )
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "ridge":
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    # ── Data preparation ───────────────────────────────────────────────────

    @staticmethod
    def _pairs_to_arrays(
        pairs: List[QueryDocumentPair],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """Convert list of pairs to numpy arrays."""
        X = np.array([p.to_feature_vector() for p in pairs])
        y = np.array([p.relevance_label for p in pairs], dtype=float)
        query_ids = np.array([p.query_id for p in pairs])
        doc_ids = [p.doc_id for p in pairs]
        pair_query_ids = [p.query_id for p in pairs]
        return X, y, query_ids, doc_ids, pair_query_ids

    # ── Training ───────────────────────────────────────────────────────────

    def fit(self, pairs: List[QueryDocumentPair]) -> "LearningToRankSystem":
        """
        Train the pointwise ranking model.

        Parameters
        ----------
        pairs : list of QueryDocumentPair
            Training data.

        Returns
        -------
        self
        """
        if not pairs:
            raise ValueError("Training data is empty.")

        X, y, query_ids, _, _ = self._pairs_to_arrays(pairs)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build and train model on full training set
        self.model = self._build_model()
        self.model.fit(X_scaled, y)
        self._is_fitted = True

        print(f"[LTR] Trained {self.model_type} on {len(pairs)} pairs "
              f"({len(np.unique(query_ids))} queries).")
        return self

    # ── Prediction ─────────────────────────────────────────────────────────

    def predict(self, pairs: List[QueryDocumentPair]) -> np.ndarray:
        """Predict relevance scores for query-document pairs."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X = np.array([p.to_feature_vector() for p in pairs])
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    # ── Per-query ranking ──────────────────────────────────────────────────

    def rank_documents(
        self, pairs: List[QueryDocumentPair]
    ) -> Dict[str, RankingResult]:
        """
        Rank documents per query using predicted relevance scores.

        Parameters
        ----------
        pairs : list of QueryDocumentPair

        Returns
        -------
        dict mapping query_id -> RankingResult
        """
        predicted_scores = self.predict(pairs)

        # Group by query
        query_groups: Dict[str, List[Tuple[int, QueryDocumentPair]]] = {}
        for idx, pair in enumerate(pairs):
            query_groups.setdefault(pair.query_id, []).append((idx, pair))

        results: Dict[str, RankingResult] = {}
        for qid, indexed_pairs in query_groups.items():
            indices = [i for i, _ in indexed_pairs]
            group_pairs = [p for _, p in indexed_pairs]

            true_labels = [p.relevance_label for p in group_pairs]
            pred_scores = [float(predicted_scores[i]) for i in indices]
            bm25_scores = [p.bm25_score for p in group_pairs]
            doc_ids = [p.doc_id for p in group_pairs]

            # Rank by predicted score
            rank_order = np.argsort(pred_scores)[::-1]
            ranked_doc_ids = [doc_ids[i] for i in rank_order]
            ranked_true_labels = [true_labels[i] for i in rank_order]

            result = RankingResult(
                query_id=qid,
                doc_ids=doc_ids,
                true_labels=true_labels,
                predicted_scores=pred_scores,
                bm25_scores=bm25_scores,
                ranked_doc_ids=ranked_doc_ids,
                ranked_true_labels=ranked_true_labels,
            )

            # LTR metrics
            result.ndcg_at_5 = _ndcg_at_k(true_labels, pred_scores, k=5)
            result.ndcg_at_10 = _ndcg_at_k(true_labels, pred_scores, k=10)
            result.map_score = _average_precision(true_labels, pred_scores)

            # BM25 baseline metrics
            result.bm25_ndcg_at_5 = _ndcg_at_k(true_labels, bm25_scores, k=5)
            result.bm25_ndcg_at_10 = _ndcg_at_k(true_labels, bm25_scores, k=10)
            result.bm25_map_score = _average_precision(true_labels, bm25_scores)

            results[qid] = result

        return results

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(
        self, pairs: List[QueryDocumentPair]
    ) -> Tuple[EvaluationMetrics, Dict[str, RankingResult]]:
        """
        Evaluate the ranking system on a set of query-document pairs.

        Parameters
        ----------
        pairs : list of QueryDocumentPair

        Returns
        -------
        (EvaluationMetrics, per_query_results)
        """
        per_query = self.rank_documents(pairs)

        # Aggregate
        ndcg5_list = [r.ndcg_at_5 for r in per_query.values()]
        ndcg10_list = [r.ndcg_at_10 for r in per_query.values()]
        map_list = [r.map_score for r in per_query.values()]
        bm25_ndcg5_list = [r.bm25_ndcg_at_5 for r in per_query.values()]
        bm25_ndcg10_list = [r.bm25_ndcg_at_10 for r in per_query.values()]
        bm25_map_list = [r.bm25_map_score for r in per_query.values()]

        # RMSE on relevance prediction
        predicted = self.predict(pairs)
        true = np.array([p.relevance_label for p in pairs], dtype=float)
        rmse = float(np.sqrt(mean_squared_error(true, predicted)))

        mean_ndcg5 = float(np.mean(ndcg5_list))
        mean_ndcg10 = float(np.mean(ndcg10_list))
        mean_map = float(np.mean(map_list))
        bm25_mean_ndcg5 = float(np.mean(bm25_ndcg5_list))
        bm25_mean_ndcg10 = float(np.mean(bm25_ndcg10_list))
        bm25_mean_map = float(np.mean(bm25_map_list))

        metrics = EvaluationMetrics(
            mean_ndcg_at_5=mean_ndcg5,
            mean_ndcg_at_10=mean_ndcg10,
            mean_average_precision=mean_map,
            bm25_mean_ndcg_at_5=bm25_mean_ndcg5,
            bm25_mean_ndcg_at_10=bm25_mean_ndcg10,
            bm25_mean_average_precision=bm25_mean_map,
            improvement_ndcg_at_5=mean_ndcg5 - bm25_mean_ndcg5,
            improvement_ndcg_at_10=mean_ndcg10 - bm25_mean_ndcg10,
            improvement_map=mean_map - bm25_mean_map,
            rmse=rmse,
            num_queries=len(per_query),
            num_pairs=len(pairs),
        )

        return metrics, per_query

    # ── Cross-validation ───────────────────────────────────────────────────

    def cross_validate(
        self, pairs: List[QueryDocumentPair]
    ) -> Dict[str, List[float]]:
        """
        Group k-fold cross-validation (folds are split by query).

        Returns
        -------
        dict with lists of per-fold metric values.
        """
        X, y, query_ids, doc_ids, pair_query_ids = self._pairs_to_arrays(pairs)

        # Encode query ids as integers for GroupKFold
        unique_qids = list(dict.fromkeys(pair_query_ids))
        qid_to_int = {qid: i for i, qid in enumerate(unique_qids)}
        groups = np.array([qid_to_int[qid] for qid in pair_query_ids])

        n_splits = min(self.n_splits, len(unique_qids))
        gkf = GroupKFold(n_splits=n_splits)

        cv_results: Dict[str, List[float]] = {
            "ndcg_at_5": [], "ndcg_at_10": [], "map": [],
            "bm25_ndcg_at_5": [], "bm25_ndcg_at_10": [], "bm25_map": [],
            "rmse": [],
        }

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            train_pairs = [pairs[i] for i in train_idx]
            test_pairs = [pairs[i] for i in test_idx]

            # Fit a fresh model on this fold
            fold_system = LearningToRankSystem(
                model_type=self.model_type,
                random_state=self.random_state,
            )
            fold_system.fit(train_pairs)
            fold_metrics, _ = fold_system.evaluate(test_pairs)

            cv_results["ndcg_at_5"].append(fold_metrics.mean_ndcg_at_5)
            cv_results["ndcg_at_10"].append(fold_metrics.mean_ndcg_at_10)
            cv_results["map"].append(fold_metrics.mean_average_precision)
            cv_results["bm25_ndcg_at_5"].append(fold_metrics.bm25_mean_ndcg_at_5)
            cv_results["bm25_ndcg_at_10"].append(fold_metrics.bm25_mean_ndcg_at_10)
            cv_results["bm25_map"].append(fold_metrics.bm25_mean_average_precision)
            cv_results["rmse"].append(fold_metrics.rmse)

            print(f"  Fold {fold + 1}/{n_splits}: "
                  f"nDCG@5={fold_metrics.mean_ndcg_at_5:.4f}, "
                  f"MAP={fold_metrics.mean_average_precision:.4f}, "
                  f"RMSE={fold_metrics.rmse:.4f}")

        return cv_results

    # ── Feature importance ─────────────────────────────────────────────────

    def feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importances (tree-based models only)."""
        if not self._is_fitted:
            return None
        if not hasattr(self.model, "feature_importances_"):
            return None
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.FEATURE_NAMES,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # ── Pretty reporting ───────────────────────────────────────────────────

    @staticmethod
    def print_report(
        metrics: EvaluationMetrics,
        per_query: Dict[str, RankingResult],
        top_n_queries: int = 5,
    ) -> None:
        """Print a formatted evaluation report."""
        sep = "─" * 60
        print(f"\n{sep}")
        print("  LEARNING-TO-RANK EVALUATION REPORT")
        print(sep)
        print(f"  Queries evaluated : {metrics.num_queries}")
        print(f"  Total pairs       : {metrics.num_pairs}")
        print(sep)
        print(f"  {'Metric':<25} {'LTR Model':>10} {'BM25 Only':>10} {'Δ':>8}")
        print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*8}")

        def row(name, ltr, bm25, delta):
            sign = "+"