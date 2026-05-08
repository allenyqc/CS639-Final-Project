```python
"""
ML Data Processing Pipeline
============================
A configurable, reusable pipeline that reads data from multiple sources,
merges, cleans, and prepares it for model training.

Best practices enforced:
- Train/test split BEFORE any preprocessing
- Transformers fitted ONLY on training data, applied to test data
- No credentials hardcoded; secrets loaded from environment variables
- Comprehensive logging with timestamps and row counts
"""

import os
import json
import logging
import hashlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _build_logger(name: str = "ml_pipeline", log_file: Optional[str] = None) -> logging.Logger:
    """Create a logger that writes to stdout and optionally to a file."""
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


logger = _build_logger()


# ---------------------------------------------------------------------------
# Helper: secrets from environment
# ---------------------------------------------------------------------------

def _get_secret(env_var: str, required: bool = True) -> Optional[str]:
    """
    Retrieve a secret from an environment variable.
    Raises RuntimeError if *required* and the variable is not set.
    """
    value = os.environ.get(env_var)
    if required and not value:
        raise RuntimeError(
            f"Required environment variable '{env_var}' is not set. "
            "Set it before running the pipeline."
        )
    return value


# ---------------------------------------------------------------------------
# Step 1 – Read CSV
# ---------------------------------------------------------------------------

def read_csv_source(
    csv_path: str,
    *,
    encoding: str = "utf-8",
    sep: str = ",",
    dtype: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Read a CSV file from a local path and return a DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("Reading CSV from '%s'", csv_path)
    df = pd.read_csv(path, encoding=encoding, sep=sep, dtype=dtype)
    logger.info("CSV loaded  → %d rows × %d cols", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 2 – Fetch from REST API
# ---------------------------------------------------------------------------

def fetch_api_source(
    endpoint: str,
    *,
    api_key_env_var: str = "API_KEY",
    auth_header: str = "X-API-Key",
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch data from a REST API endpoint using an API key loaded from an
    environment variable.  Returns a DataFrame.
    """
    api_key = _get_secret(api_key_env_var)
    headers = {auth_header: api_key, "Accept": "application/json"}

    logger.info("Fetching API data from '%s'", endpoint)
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                endpoint, headers=headers, params=params, timeout=timeout
            )
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("API attempt %d/%d failed: %s", attempt, max_retries, exc)
    else:
        raise RuntimeError(
            f"All {max_retries} API attempts failed. Last error: {last_exc}"
        )

    # Support both list-of-records and {"data": [...]} shapes
    if isinstance(payload, list):
        df = pd.DataFrame(payload)
    elif isinstance(payload, dict) and "data" in payload:
        df = pd.DataFrame(payload["data"])
    else:
        df = pd.json_normalize(payload)

    logger.info("API data loaded → %d rows × %d cols", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 3 – Read JSON config / DB params
# ---------------------------------------------------------------------------

def read_json_config(config_path: str) -> Dict[str, Any]:
    """
    Read a JSON configuration file.
    Sensitive values (passwords, tokens) should be stored as environment
    variable *names* in the JSON, e.g. {"password": "DB_PASSWORD"}, and
    this function resolves them automatically.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info("Reading JSON config from '%s'", config_path)
    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    # Resolve any env-var placeholders (values that look like UPPER_SNAKE_CASE)
    _resolve_env_vars(config)
    logger.info("JSON config loaded with keys: %s", list(config.keys()))
    return config


def _resolve_env_vars(obj: Any) -> None:
    """Recursively replace env-var placeholder strings with their values."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and value.isupper() and "_" in value:
                resolved = os.environ.get(value)
                if resolved is not None:
                    obj[key] = resolved
                    logger.debug("Resolved env var '%s' for config key '%s'", value, key)
            else:
                _resolve_env_vars(value)
    elif isinstance(obj, list):
        for item in obj:
            _resolve_env_vars(item)


# ---------------------------------------------------------------------------
# Step 4 – Merge data sources
# ---------------------------------------------------------------------------

def merge_sources(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    merge_key: str,
    how: str = "inner",
    suffixes: Tuple[str, str] = ("_csv", "_api"),
) -> pd.DataFrame:
    """Merge two DataFrames on a common key."""
    for name, df in [("primary", primary_df), ("secondary", secondary_df)]:
        if merge_key not in df.columns:
            raise KeyError(f"Merge key '{merge_key}' not found in {name} DataFrame. "
                           f"Available columns: {list(df.columns)}")

    logger.info(
        "Merging sources on '%s' (how='%s') | primary=%d rows, secondary=%d rows",
        merge_key, how, len(primary_df), len(secondary_df),
    )
    merged = pd.merge(primary_df, secondary_df, on=merge_key, how=how, suffixes=suffixes)
    logger.info("Merged DataFrame → %d rows × %d cols", *merged.shape)
    return merged


# ---------------------------------------------------------------------------
# Step 5 – Clean data  (fitted ONLY on train split)
# ---------------------------------------------------------------------------

class DataCleaner:
    """
    Stateful cleaner that fits imputers / encoders on training data and
    applies the same transformations to test data.

    Usage
    -----
    cleaner = DataCleaner(...)
    X_train_clean = cleaner.fit_transform(X_train)
    X_test_clean  = cleaner.transform(X_test)
    """

    def __init__(
        self,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
        scale_numerics: bool = True,
        drop_duplicate_subset: Optional[List[str]] = None,
    ):
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.scale_numerics = scale_numerics
        self.drop_duplicate_subset = drop_duplicate_subset

        self._numeric_cols: List[str] = []
        self._categorical_cols: List[str] = []
        self._numeric_imputer: Optional[SimpleImputer] = None
        self._categorical_imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[StandardScaler] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *df* (training data) and return the cleaned version."""
        logger.info("[Cleaner] fit_transform on %d rows", len(df))
        df = self._drop_duplicates(df, label="train")
        df = df.reset_index(drop=True)

        self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Numeric imputation
        if self._numeric_cols:
            self._numeric_imputer = SimpleImputer(strategy=self.numeric_impute_strategy)
            df[self._numeric_cols] = self._numeric_imputer.fit_transform(
                df[self._numeric_cols]
            )

        # Categorical imputation
        if self._categorical_cols:
            self._categorical_imputer = SimpleImputer(
                strategy=self.categorical_impute_strategy
            )
            df[self._categorical_cols] = self._categorical_imputer.fit_transform(
                df[self._categorical_cols]
            )

        # Label-encode categoricals
        for col in self._categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self._label_encoders[col] = le

        # Scale numerics
        if self.scale_numerics and self._numeric_cols:
            self._scaler = StandardScaler()
            df[self._numeric_cols] = self._scaler.fit_transform(df[self._numeric_cols])

        self._fitted = True
        logger.info("[Cleaner] fit_transform complete → %d rows × %d cols", *df.shape)
        return df

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations to *df* (test / validation data)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data first.")

        logger.info("[Cleaner] transform on %d rows (test/val)", len(df))
        df = df.copy()

        # Numeric imputation
        if self._numeric_cols and self._numeric_imputer:
            present = [c for c in self._numeric_cols if c in df.columns]
            df[present] = self._numeric_imputer.transform(df[present])

        # Categorical imputation
        if self._categorical_cols and self._categorical_imputer:
            present = [c for c in self._categorical_cols if c in df.columns]
            df[present] = self._categorical_imputer.transform(df[present])

        # Label-encode categoricals (handle unseen labels gracefully)
        for col, le in self._label_encoders.items():
            if col not in df.columns:
                continue
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])

        # Scale numerics
        if self.scale_numerics and self._scaler and self._numeric_cols:
            present = [c for c in self._numeric_cols if c in df.columns]
            df[present] = self._scaler.transform(df[present])

        logger.info("[Cleaner] transform complete → %d rows × %d cols", *df.shape)
        return df

    # ------------------------------------------------------------------
    def _drop_duplicates(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=self.drop_duplicate_subset)
        removed = before - len(df)
        if removed:
            logger.info("[Cleaner] Removed %d duplicate rows from %s set", removed, label)
        return df


# ---------------------------------------------------------------------------
# Step 6 – Train / test split and save
# ---------------------------------------------------------------------------

def split_and_save(
    df: pd.DataFrame,
    target_col: str,
    output_dir: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    file_prefix: str = "dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train/test BEFORE any preprocessing.
    Returns (X_train, X_test, y_train, y_test).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    stratify_arr = y[target_col] if stratify else None

    logger.info(
        "Splitting data → test_size=%.2f, random_state=%d, stratify=%s",
        test_size, random_state, stratify,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arr
    )
    logger.info(
        "Split complete | train=%d rows, test=%d rows", len(X_train), len(X_test)
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {
        "X_train": out / f"{file_prefix}_X_train_{ts}.csv",
        "X_test":  out / f"{file_prefix}_X_test_{ts}.csv",
        "y_train": out / f"{file_prefix}_y_train_{ts}.csv",
        "y_test":  out / f"{file_prefix}_y_test_{ts}.csv",
    }

    X_train.to_csv(paths["X_train"], index=False)
    X_test.to_csv(paths["X_test"],   index=False)
    y_train.to_csv(paths["y_train"], index=False)
    y_test.to_csv(paths["y_test"],   index=False)

    for name, path in paths.items():
        logger.info("Saved %s → %s", name, path)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 7 & 8 – Pipeline orchestrator
# ---------------------------------------------------------------------------

class MLPipeline:
    """
    Configurable, reusable ML data processing pipeline.

    Parameters
    ----------
    config : dict
        Pipeline configuration (see ``default_config()`` for keys).

    Example
    -------
    >>> pipeline = MLPipeline(config)
    >>> result = pipeline.run()
    >>> result["X_train_clean"]   # ready for model training
    >>> result["X_test_clean"]    # held-out; never used during fitting
    >>> result["summary"]         # processing log
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._summary: List[Dict[str, Any]] = []
        self._cleaner: Optional[DataCleaner] = None

    # ------------------------------------------------------------------
    @staticmethod
    def default_config() -> Dict[str, Any]:
        """Return a template configuration dictionary."""
        return {
            # Data sources
            "csv_path": "data/raw/primary.csv",
            "api_endpoint": "https://api.example.com/v1/records",
            "api_key_env_var": "API_KEY",          # env var name, NOT the key itself
            "api_params": {"limit": 10000},
            "json_config_path": "config/db_config.json",

            # Merge
            "merge_key": "id",
            "merge_how": "inner",

            # Target
            "target_col": "label",

            # Cleaning
            "numeric_impute_strategy": "median",
            "categorical_impute_strategy": "most_frequent",
            "scale_numerics": True,
            "drop_duplicate_subset": None,

            # Split
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,

            # Output
            "output_dir": "data/processed",
            "file_prefix": "dataset",
            "log_file": None,   # e.g. "logs/pipeline.log"
        }

    # ------------------------------------------------------------------
    def _log_step(self, step: str, details: Dict[str, Any]) -> None:
        entry = {"timestamp": datetime.now().isoformat(), "step": step, **details}
        self._summary.append(entry)
        logger.info("[%s] %s", step, details)

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """
        Execute the full pipeline and return a results dictionary containing:
        - X_train_clean, X_test_clean, y_train, y_test  (DataFrames)
        - merged_raw                                     (DataFrame before cleaning)
        - db_config                                      (dict from JSON config)
        - summary                                        (list of processing steps)
        - cleaner                                        (fitted DataCleaner instance)
        """
        cfg = self.config

        # Optional per-run log file
        if cfg.get("log_file"):
            _build_logger(log_file=cfg["log_file"])

        self._log_step("START", {"config_keys": list(cfg.keys())})

        # ── Step 1: CSV ────────────────────────────────────────────────
        csv_df = read_csv_source(
            cfg["csv_path"],
            encoding=cfg.get("csv_encoding", "utf-8"),
            sep=cfg.get("csv_sep", ","),
        )
        self._log_step("READ_CSV", {"rows": len(csv_df), "cols": list(csv_df.columns)})

        # ── Step 2: API ────────────────────────────────────────────────
        api_df = fetch_api_source(
            cfg["api_endpoint"],
            api_key_env_var=cfg.get("api_key_env_var", "API_KEY"),
            params=cfg.get("api_params"),
            timeout=cfg.get("api_timeout", 30),
            max_retries=cfg.get("api_max_retries", 3),
        )
        self._log_step("FETCH_API", {"rows": len(api_df), "cols": list(api_df.columns)})

        # ── Step 3: JSON config ────────────────────────────────────────
        db_config = read_json_config(cfg["json_config_path"])
        self._log_step("READ_JSON_CONFIG", {"keys": list(db_config.keys())})

        # ── Step 4: Merge ──────────────────────────────────────────────
        merged = merge_sources(
            csv_df, api_df,
            merge_key=cfg["merge_key"],
            how=cfg.get("merge_how", "inner"),
        )
        self._log_step("MERGE", {"rows": len(merged), "cols": list(merged.columns)})

        # ── Step 5a: Train / test split BEFORE cleaning ────────────────
        X_train_raw, X_test_raw, y