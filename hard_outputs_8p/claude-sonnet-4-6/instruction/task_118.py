```python
"""
Learning-to-Rank module for document retrieval.

Implements a pointwise ranking approach using regression to predict relevance scores,
with proper train/validation/test splits, group-aware splitting, and NDCG/MAP evaluation.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "bm25_score",
    "tfidf_similarity",
    "doc_length",
    "query_term_freq",
    "pagerank_score",
]


@dataclass
class RankingResult:
    """Container for all outputs of the ranking system."""

    model: Pipeline
    scaler_fitted_on_train: bool
    train_ndcg: float
    val_ndcg: float
    test_ndcg: float
    train_map: float
    val_map: float
    test_map: float
    baseline_test_ndcg: float
    baseline_test_map: float
    per_query_metrics: pd.DataFrame
    feature_importances: dict[str, float]
    best_params: dict[str, Any]
    predictions_df: pd.DataFrame


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _dcg_at_k(relevances: np.ndarray, k: int | None = None) -> float:
    """Discounted Cumulative Gain at k."""
    if k is not None:
        relevances = relevances[:k]
    gains = 2.0 ** relevances - 1.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains / discounts))


def _ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int | None = None) -> float:
    """Normalised DCG at k for a single query."""
    order = np.argsort(y_score)[::-1]
    ideal_order = np.argsort(y_true)[::-1]
    dcg = _dcg_at_k(y_true[order], k)
    idcg = _dcg_at_k(y_true[ideal_order], k)
    if idcg == 0.0:
        return 1.0  # all docs irrelevant → perfect by convention
    return dcg / idcg


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision for a single query (binary relevance: label > 0)."""
    order = np.argsort(y_score)[::-1]
    y_sorted = (y_true[order] > 0).astype(float)
    n_relevant = y_sorted.sum()
    if n_relevant == 0:
        return 0.0
    precisions = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
    return float(np.sum(precisions * y_sorted) / n_relevant)


def _mean_ndcg(df: pd.DataFrame, score_col: str, k: int | None = None) -> float:
    """Mean NDCG across all queries in a DataFrame."""
    scores = []
    for _, grp in df.groupby("query_id"):
        scores.append(_ndcg_at_k(grp["relevance"].values, grp[score_col].values, k))
    return float(np.mean(scores)) if scores else 0.0


def _mean_ap(df: pd.DataFrame, score_col: str) -> float:
    """Mean Average Precision across all queries."""
    scores = []
    for _, grp in df.groupby("query_id"):
        scores.append(_average_precision(grp["relevance"].values, grp[score_col].values))
    return float(np.mean(scores)) if scores else 0.0


def _per_query_metrics(df: pd.DataFrame, score_col: str, k: int | None = None) -> pd.DataFrame:
    rows = []
    for qid, grp in df.groupby("query_id"):
        ndcg = _ndcg_at_k(grp["relevance"].values, grp[score_col].values, k)
        ap = _average_precision(grp["relevance"].values, grp[score_col].values)
        rows.append({"query_id": qid, "ndcg": ndcg, "ap": ap, "n_docs": len(grp)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data generation (synthetic, for demonstration)
# ---------------------------------------------------------------------------


def generate_synthetic_data(
    n_queries: int = 50,
    docs_per_query: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic query-document dataset.

    Each row represents one (query, document) pair with five features and a
    relevance label in {0, 1, 2}.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for qid in range(n_queries):
        for did in range(docs_per_query):
            bm25 = rng.exponential(scale=3.0)
            tfidf = rng.beta(2, 5)
            doc_len = rng.integers(50, 2000)
            qtf = rng.poisson(lam=2)
            pagerank = rng.beta(1, 9)

            # Relevance is correlated with features
            logit = (
                0.4 * bm25
                + 2.0 * tfidf
                + 0.001 * qtf
                - 0.0001 * doc_len
                + 1.5 * pagerank
                + rng.normal(0, 0.5)
            )
            prob = 1 / (1 + np.exp(-logit + 1.5))
            relevance = int(np.clip(rng.binomial(2, prob), 0, 2))

            rows.append(
                {
                    "query_id": f"q{qid:03d}",
                    "doc_id": f"d{did:04d}",
                    "bm25_score": bm25,
                    "tfidf_similarity": tfidf,
                    "doc_length": doc_len,
                    "query_term_freq": qtf,
                    "pagerank_score": pagerank,
                    "relevance": relevance,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------


def group_aware_split(
    df: pd.DataFrame,
    group_col: str = "query_id",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data so that the same query never appears in more than one partition.

    Returns (train_df, val_df, test_df).
    """
    groups = df[group_col].values

    # First split: hold out test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Second split: carve out validation from train_val
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_fraction_of_trainval, random_state=random_state + 1
    )
    train_idx2, val_idx2 = next(
        gss_val.split(train_val_df, groups=train_val_df[group_col].values)
    )

    train_df = train_val_df.iloc[train_idx2].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx2].reset_index(drop=True)

    # Sanity checks
    train_queries = set(train_df[group_col])
    val_queries = set(val_df[group_col])
    test_queries = set(test_df[group_col])
    assert train_queries.isdisjoint(test_queries), "Train/test query overlap!"
    assert train_queries.isdisjoint(val_queries), "Train/val query overlap!"
    assert val_queries.isdisjoint(test_queries), "Val/test query overlap!"

    logger.info(
        "Split sizes — train queries: %d, val queries: %d, test queries: %d",
        len(train_queries),
        len(val_queries),
        len(test_queries),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def _ndcg_scorer(estimator: Any, X: np.ndarray, y: np.ndarray) -> float:
    """Custom scorer for RandomizedSearchCV using NDCG (no query grouping in CV)."""
    preds = estimator.predict(X)
    order = np.argsort(preds)[::-1]
    return _dcg_at_k(y[order]) / max(_dcg_at_k(np.sort(y)[::-1]), 1e-9)


def build_pipeline() -> Pipeline:
    """Return an sklearn Pipeline with scaler + GBM regressor."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    random_state=42,
                ),
            ),
        ]
    )


def tune_hyperparameters(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
    cv: int = 3,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Tune hyperparameters via RandomizedSearchCV on the training set only.
    Returns the best pipeline and best params.
    """
    param_dist = {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "regressor__max_depth": [3, 4, 5, 6],
        "regressor__subsample": [0.6, 0.8, 1.0],
        "regressor__min_samples_leaf": [1, 5, 10],
    }

    scorer = make_scorer(_ndcg_scorer, greater_is_better=True)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info("Best CV NDCG (train only): %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)
    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------------------
# Main ranking system
# ---------------------------------------------------------------------------


def build_ranking_system(
    df: pd.DataFrame | None = None,
    n_queries: int = 50,
    docs_per_query: int = 20,
    test_size: float = 0.2,
    val_size: float = 0.1,
    n_iter_search: int = 20,
    random_state: int = 42,
) -> RankingResult:
    """
    Build and evaluate a learning-to-rank system.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-built dataset. If None, synthetic data is generated.
    n_queries : int
        Number of queries for synthetic data generation.
    docs_per_query : int
        Documents per query for synthetic data.
    test_size : float
        Fraction of queries held out for testing.
    val_size : float
        Fraction of queries held out for validation.
    n_iter_search : int
        Number of iterations for RandomizedSearchCV.
    random_state : int
        Global random seed.

    Returns
    -------
    RankingResult
    """
    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    if df is None:
        logger.info("Generating synthetic data (%d queries × %d docs).", n_queries, docs_per_query)
        df = generate_synthetic_data(n_queries, docs_per_query, random_state)

    required_cols = {"query_id", "doc_id", "relevance"} | set(FEATURE_COLS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    # ------------------------------------------------------------------
    # 2. Group-aware split (BEFORE any preprocessing)
    # ------------------------------------------------------------------
    train_df, val_df, test_df = group_aware_split(
        df,
        group_col="query_id",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    X_train = train_df[FEATURE_COLS].values.astype(np.float64)
    y_train = train_df["relevance"].values.astype(np.float64)

    X_val = val_df[FEATURE_COLS].values.astype(np.float64)
    y_val = val_df["relevance"].values.astype(np.float64)

    X_test = test_df[FEATURE_COLS].values.astype(np.float64)
    y_test = test_df["relevance"].values.astype(np.float64)

    # ------------------------------------------------------------------
    # 3. Hyperparameter search on TRAIN only; scaler fitted inside pipeline
    # ------------------------------------------------------------------
    pipeline = build_pipeline()
    best_pipeline, best_params = tune_hyperparameters(
        pipeline, X_train, y_train, n_iter=n_iter_search, random_state=random_state
    )

    # ------------------------------------------------------------------
    # 4. Predict on all splits
    # ------------------------------------------------------------------
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["pred_score"] = best_pipeline.predict(X_train)
    val_df["pred_score"] = best_pipeline.predict(X_val)
    test_df["pred_score"] = best_pipeline.predict(X_test)

    # BM25-only baseline (uses raw bm25_score as ranking signal)
    test_df["bm25_baseline"] = test_df["bm25_score"]

    # ------------------------------------------------------------------
    # 5. Evaluate — NDCG and MAP
    # ------------------------------------------------------------------
    train_ndcg = _mean_ndcg(train_df, "pred_score")
    val_ndcg = _mean_ndcg(val_df, "pred_score")
    test_ndcg = _mean_ndcg(test_df, "pred_score")

    train_map = _mean_ap(train_df, "pred_score")
    val_map = _mean_ap(val_df, "pred_score")
    test_map = _mean_ap(test_df, "pred_score")

    baseline_test_ndcg = _mean_ndcg(test_df, "bm25_baseline")
    baseline_test_map = _mean_ap(test_df, "bm25_baseline")

    logger.info("=== Evaluation Results ===")
    logger.info("Train  NDCG: %.4f | MAP: %.4f", train_ndcg, train_map)
    logger.info("Val    NDCG: %.4f | MAP: %.4f", val_ndcg, val_map)
    logger.info("Test   NDCG: %.4f | MAP: %.4f", test_ndcg, test_map)
    logger.info(
        "BM25 Baseline — Test NDCG: %.4f | MAP: %.4f", baseline_test_ndcg, baseline_test_map
    )

    # ------------------------------------------------------------------
    # 6. Per-query breakdown on test set
    # ------------------------------------------------------------------
    pq_model = _per_query_metrics(test_df, "pred_score")
    pq_baseline = _per_query_metrics(test_df, "bm25_baseline")
    pq_model = pq_model.rename(columns={"ndcg": "model_ndcg", "ap": "model_ap"})
    pq_baseline = pq_baseline.rename(columns={"ndcg": "baseline_ndcg", "ap": "baseline_ap"})
    per_query_df = pq_model.merge(pq_baseline[["query_id", "baseline_ndcg", "baseline_ap"]], on="query_id")
    per_query_df["ndcg_improvement"] = per_query_df["model_ndcg"] - per_query_df["baseline_ndcg"]
    per_query_df["ap_improvement"] = per_query_df["model_ap"] - per_query_df["baseline_ap"]

    # ------------------------------------------------------------------
    # 7. Feature importances (from the GBM inside the pipeline)
    # ------------------------------------------------------------------
    gbm = best_pipeline.named_steps["regressor"]
    importances = dict(zip(FEATURE_COLS, gbm.feature_importances_))
    logger.info("Feature importances: %s", importances)

    # ------------------------------------------------------------------
    # 8. Combine predictions for return
    # ------------------------------------------------------------------
    test_df["split"] = "test"
    val_df["split"] = "val"
    train_df["split"] = "train"
    predictions_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    return RankingResult(
        model=best_pipeline,
        scaler_fitted_on_train=True,
        train_ndcg=train_ndcg,
        val_ndcg=val_ndcg,
        test_ndcg=test_ndcg,
        train_map=train_map,
        val_map=val_map,
        test_map=test_map,
        baseline_test_ndcg=baseline_test_ndcg,
        baseline_test_map=baseline_test_map,
        per_query_metrics=per_query_df,
        feature_importances=importances,
        best_params=best_params,
        predictions_df=predictions_df,
    )


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


def rank_documents(
    model: Pipeline,
    query_doc_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank a set of documents for a single query using a trained pipeline.

    Parameters
    ----------
    model : fitted Pipeline
    query_doc_features : DataFrame with at least FEATURE_COLS columns.

    Returns
    -------
    DataFrame sorted by predicted relevance score (descending).
    """
    missing = set(FEATURE_COLS) - set(query_doc_features.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = query_doc_features[FEATURE_COLS].values.astype(np.float64)
    scores = model.predict(X)
    result = query_doc_features.copy()