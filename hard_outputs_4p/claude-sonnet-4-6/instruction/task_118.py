```python
"""
Learning-to-Rank module for document retrieval.

Implements a pointwise ranking approach using regression to predict relevance scores,
with proper train/test splitting, feature scaling fitted only on training data,
and evaluation using standard IR metrics (NDCG, MAP, MRR).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryDocumentPair:
    """Represents a single query-document pair with features and label."""
    query_id: str
    doc_id: str
    bm25_score: float
    tfidf_similarity: float
    doc_length: float
    query_term_frequency: float
    authority_score: float
    relevance_label: int  # 0, 1, or 2

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.bm25_score,
            self.tfidf_similarity,
            self.doc_length,
            self.query_term_frequency,
            self.authority_score,
        ], dtype=np.float64)


@dataclass
class RankingResult:
    """Holds evaluation results for the ranking system."""
    model: Any
    scaler: StandardScaler
    feature_names: List[str]
    # Aggregate metrics
    ltr_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    # Per-query breakdowns
    per_query_ltr: Dict[str, Dict[str, float]]
    per_query_baseline: Dict[str, Dict[str, float]]
    # Predictions on test set
    test_predictions: pd.DataFrame
    # Training info
    best_params: Optional[Dict[str, Any]] = None
    cv_results: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# IR Evaluation metrics
# ---------------------------------------------------------------------------

def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain at k."""
    relevances = np.asarray(relevances, dtype=float)[:k]
    if len(relevances) == 0:
        return 0.0
    gains = (2.0 ** relevances - 1.0) / np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Normalised DCG at k."""
    ideal = dcg_at_k(np.sort(relevances)[::-1], k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal


def average_precision(relevances: np.ndarray, threshold: int = 1) -> float:
    """Average Precision (binary relevance: label >= threshold is relevant)."""
    binary = (np.asarray(relevances, dtype=float) >= threshold).astype(float)
    if binary.sum() == 0:
        return 0.0
    precisions = []
    num_relevant = 0
    for i, rel in enumerate(binary):
        if rel:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    return float(np.mean(precisions)) if precisions else 0.0


def reciprocal_rank(relevances: np.ndarray, threshold: int = 1) -> float:
    """Reciprocal Rank of the first relevant document."""
    for i, rel in enumerate(relevances):
        if rel >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevances: np.ndarray, k: int, threshold: int = 1) -> float:
    """Precision at k."""
    top_k = np.asarray(relevances, dtype=float)[:k]
    return float(np.mean(top_k >= threshold))


def evaluate_ranking(
    query_groups: Dict[str, Tuple[np.ndarray, np.ndarray]],
    k_values: Tuple[int, ...] = (5, 10),
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Evaluate ranking quality across queries.

    Parameters
    ----------
    query_groups : dict mapping query_id -> (predicted_scores, true_labels)
    k_values     : cutoff values for NDCG and Precision

    Returns
    -------
    aggregate_metrics, per_query_metrics
    """
    per_query: Dict[str, Dict[str, float]] = {}

    for qid, (scores, labels) in query_groups.items():
        order = np.argsort(scores)[::-1]
        ranked_labels = labels[order]

        q_metrics: Dict[str, float] = {}
        for k in k_values:
            q_metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_labels, k)
            q_metrics[f"p@{k}"] = precision_at_k(ranked_labels, k)
        q_metrics["map"] = average_precision(ranked_labels)
        q_metrics["mrr"] = reciprocal_rank(ranked_labels)
        per_query[qid] = q_metrics

    # Aggregate (macro average over queries)
    all_keys = list(next(iter(per_query.values())).keys())
    aggregate: Dict[str, float] = {}
    for key in all_keys:
        aggregate[key] = float(np.mean([per_query[qid][key] for qid in per_query]))

    return aggregate, per_query


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def pairs_to_dataframe(pairs: List[QueryDocumentPair]) -> pd.DataFrame:
    """Convert a list of QueryDocumentPair objects to a DataFrame."""
    records = []
    for p in pairs:
        records.append({
            "query_id": p.query_id,
            "doc_id": p.doc_id,
            "bm25_score": p.bm25_score,
            "tfidf_similarity": p.tfidf_similarity,
            "doc_length": p.doc_length,
            "query_term_frequency": p.query_term_frequency,
            "authority_score": p.authority_score,
            "relevance_label": p.relevance_label,
        })
    return pd.DataFrame(records)


FEATURE_COLS = [
    "bm25_score",
    "tfidf_similarity",
    "doc_length",
    "query_term_frequency",
    "authority_score",
]


def split_by_query(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test ensuring all documents for a given query
    end up in the same partition (group-aware split).
    """
    query_ids = df["query_id"].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=query_ids))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def build_ltr_model(
    train_df: pd.DataFrame,
    n_cv_folds: int = 3,
    random_state: int = 42,
) -> Tuple[Any, StandardScaler, Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Train a pointwise LTR model (GradientBoostingRegressor) with
    group-aware cross-validation for hyperparameter search.

    The scaler is fitted ONLY on training data.

    Returns
    -------
    best_model, scaler, best_params, cv_results_df
    """
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["relevance_label"].values.astype(float)
    groups_train = train_df["query_id"].values

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    base_model = GradientBoostingRegressor(random_state=random_state)

    # Group K-Fold so that CV respects query boundaries
    cv = GroupKFold(n_splits=n_cv_folds)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    grid_search.fit(X_train_scaled, y_train, groups=groups_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    return best_model, scaler, best_params, cv_results_df


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_scores(
    model: Any,
    scaler: StandardScaler,
    df: pd.DataFrame,
) -> np.ndarray:
    """Apply the trained scaler and model to produce relevance score predictions."""
    X = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)  # transform only — never fit on test data
    return model.predict(X_scaled)


def build_query_groups(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "relevance_label",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Group (scores, labels) by query_id for evaluation."""
    groups: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for qid, sub in df.groupby("query_id"):
        groups[str(qid)] = (
            sub[score_col].values.astype(float),
            sub[label_col].values.astype(float),
        )
    return groups


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_ranking_system(
    pairs: List[QueryDocumentPair],
    test_size: float = 0.25,
    n_cv_folds: int = 3,
    k_values: Tuple[int, ...] = (5, 10),
    random_state: int = 42,
) -> RankingResult:
    """
    Build and evaluate a learning-to-rank system.

    Parameters
    ----------
    pairs        : list of QueryDocumentPair instances
    test_size    : fraction of queries to hold out for testing
    n_cv_folds   : number of group-CV folds for hyperparameter search
    k_values     : cutoff values for NDCG / Precision
    random_state : reproducibility seed

    Returns
    -------
    RankingResult with model, metrics, and per-query breakdowns
    """
    if len(pairs) == 0:
        raise ValueError("pairs list is empty.")

    # ------------------------------------------------------------------ #
    # 1. Convert to DataFrame and split BEFORE any preprocessing          #
    # ------------------------------------------------------------------ #
    df = pairs_to_dataframe(pairs)

    n_queries = df["query_id"].nunique()
    if n_queries < 2:
        raise ValueError(
            f"Need at least 2 distinct queries for a train/test split; got {n_queries}."
        )

    train_df, test_df = split_by_query(df, test_size=test_size, random_state=random_state)

    print(f"[LTR] Train queries: {train_df['query_id'].nunique()}, "
          f"docs: {len(train_df)}")
    print(f"[LTR] Test  queries: {test_df['query_id'].nunique()}, "
          f"docs: {len(test_df)}")

    # ------------------------------------------------------------------ #
    # 2. Train model (scaler fitted on train only)                        #
    # ------------------------------------------------------------------ #
    model, scaler, best_params, cv_results = build_ltr_model(
        train_df, n_cv_folds=n_cv_folds, random_state=random_state
    )
    print(f"[LTR] Best hyperparameters: {best_params}")

    # ------------------------------------------------------------------ #
    # 3. Predict on test set (scaler.transform, not fit_transform)        #
    # ------------------------------------------------------------------ #
    test_df = test_df.copy()
    test_df["ltr_score"] = predict_scores(model, scaler, test_df)
    test_df["bm25_score_baseline"] = test_df["bm25_score"]  # alias for clarity

    # Regression metrics on test set
    mse = mean_squared_error(test_df["relevance_label"], test_df["ltr_score"])
    mae = mean_absolute_error(test_df["relevance_label"], test_df["ltr_score"])
    print(f"[LTR] Test MSE: {mse:.4f}, MAE: {mae:.4f}")

    # ------------------------------------------------------------------ #
    # 4. Evaluate ranking quality                                         #
    # ------------------------------------------------------------------ #
    ltr_groups = build_query_groups(test_df, score_col="ltr_score")
    baseline_groups = build_query_groups(test_df, score_col="bm25_score_baseline")

    ltr_agg, ltr_per_query = evaluate_ranking(ltr_groups, k_values=k_values)
    baseline_agg, baseline_per_query = evaluate_ranking(baseline_groups, k_values=k_values)

    # Add regression metrics to aggregate
    ltr_agg["mse"] = mse
    ltr_agg["mae"] = mae

    # ------------------------------------------------------------------ #
    # 5. Print comparison summary                                         #
    # ------------------------------------------------------------------ #
    print("\n[LTR] ===== Evaluation Summary (test set) =====")
    print(f"{'Metric':<15} {'LTR Model':>12} {'BM25 Baseline':>15}")
    print("-" * 44)
    for key in sorted(ltr_agg.keys()):
        if key in baseline_agg:
            print(f"{key:<15} {ltr_agg[key]:>12.4f} {baseline_agg[key]:>15.4f}")
        else:
            print(f"{key:<15} {ltr_agg[key]:>12.4f} {'N/A':>15}")

    return RankingResult(
        model=model,
        scaler=scaler,
        feature_names=FEATURE_COLS,
        ltr_metrics=ltr_agg,
        baseline_metrics=baseline_agg,
        per_query_ltr=ltr_per_query,
        per_query_baseline=baseline_per_query,
        test_predictions=test_df.reset_index(drop=True),
        best_params=best_params,
        cv_results=cv_results,
    )


# ---------------------------------------------------------------------------
# Inference helper (for new, unseen query-document pairs)
# ---------------------------------------------------------------------------

def rank_documents(
    model: Any,
    scaler: StandardScaler,
    pairs: List[QueryDocumentPair],
) -> pd.DataFrame:
    """
    Rank a list of query-document pairs using a trained LTR model.

    Parameters
    ----------
    model  : trained regression model
    scaler : fitted StandardScaler (from training)
    pairs  : new query-document pairs to rank

    Returns
    -------
    DataFrame sorted by (query_id, predicted_score desc)
    """
    df = pairs_to_dataframe(pairs)
    df["predicted_score"] = predict_scores(model, scaler, df)
    df = df.sort_values(["query_id", "predicted_score"], ascending=[True, False])
    df["rank"] = df.groupby("query_id").cumcount() + 1
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing / demonstration)
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_queries: int = 20,
    docs_per_query: int = 50,
    random_state: int = 42,
) -> List[QueryDocumentPair]:
    """
    Generate synthetic query-document pairs for demonstration purposes.

    Relevance is determined by a noisy linear combination of features so
    that the LTR model has something meaningful to learn.
    """
    rng = np.random.RandomState(random_state)
    pairs: List[QueryDocumentPair] = []

    for q_idx in range(n_queries):
        qid = f"q{q_idx:03d}"
        for d_idx in range(docs_per_query):
            did = f"d{q_idx:03d}_{d_idx:04d}"

            bm25 = float(rng.exponential(scale=3.0))
            tfidf = float(rng.beta(2, 5))
            doc_len = float(rng.lognormal(mean=5.5, sigma=1.0))
            qtf = float(rng.poisson(lam=2.0))
            authority = float(rng.beta(1, 9))

            # Noisy relevance signal
            score = (
                0.4 * bm25 / 10.0
                + 0.3 * tfidf
                + 0.1 * authority
                + 0.1 * qtf / 5.0
                - 0.05 * doc_len / 500.0
                + rng.normal(0, 0.15)
            )
            # Map to 0/1/2
            if score < 0.25:
                label = 0
            elif score < 0.55:
                label = 1
            else:
                label = 2

            pairs.append(QueryDocumentPair(
                query_id=qid,
                doc_id=did,
                bm25_score=bm25,
                tfidf_similarity=tfidf,
                doc_length=doc_len,
                query_term_frequency=qtf,
                authority_score=authority,
                relevance_label=label,
            ))

    return pairs


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic data …")
    data = generate_synthetic_data(n_queries=30, docs_per_query=40, random_state=0)

    print(f"Total pairs: {len(data)}")
    label_counts = pd.Series([p.relevance_label for p in data]).value_counts().sort_index()
    print(f"Label distribution:\n{label_counts.to_string()}\n")

    result = build_ranking_system(
        pairs=data,
        test_size=0.25,
        n_cv_folds=3,
        k_values=(5, 10),
        random_state=42,
    )

    print("\n[LTR] Per-query NDCG@10 (first 5 test queries):")
    for qid in list(result.per_query_ltr.keys())[:5]:
        ltr_n = result.per_query_ltr[qid].get("ndcg@10", float("nan"))
        bm25_n = result.per_query_baseline[qid].get("ndcg@10", float("nan"))
        print(f"  {qid}: LTR={ltr_n:.4f}  BM25={b