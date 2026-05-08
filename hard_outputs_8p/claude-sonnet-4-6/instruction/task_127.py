```python
"""
ML Data Processing Pipeline Module

A configurable and reusable pipeline for reading, merging, cleaning,
and preparing data from multiple sources for model training.
"""

import os
import json
import logging
import hashlib
import hmac
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _build_logger(name: str = "ml_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = _build_logger()


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CSVSourceConfig:
    path: str
    encoding: str = "utf-8"
    sep: str = ","


@dataclass
class APISourceConfig:
    url: str
    # API key is read from an environment variable — never hardcoded
    api_key_env_var: str = "ML_PIPELINE_API_KEY"
    timeout_seconds: int = 30
    record_path: Optional[str] = None          # JSON path to records list
    params: dict = field(default_factory=dict)  # extra query params


@dataclass
class DBConfigFile:
    path: str  # path to JSON file with DB connection params


@dataclass
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15          # fraction of *training* data used for val
    random_state: int = 42
    stratify_col: Optional[str] = None
    group_col: Optional[str] = None  # for group-aware splitting
    time_col: Optional[str] = None   # for chronological splitting


@dataclass
class PipelineConfig:
    merge_key: str
    csv_source: CSVSourceConfig
    api_source: APISourceConfig
    db_config_file: DBConfigFile
    split: SplitConfig = field(default_factory=SplitConfig)
    output_dir: str = "processed_data"
    categorical_cols: list = field(default_factory=list)
    numeric_cols: list = field(default_factory=list)
    drop_cols: list = field(default_factory=list)
    missing_numeric_strategy: str = "median"   # mean | median | most_frequent
    missing_categorical_strategy: str = "most_frequent"
    duplicate_subset: Optional[list] = None


# ---------------------------------------------------------------------------
# Processing summary
# ---------------------------------------------------------------------------

@dataclass
class ProcessingSummary:
    steps: list = field(default_factory=list)

    def log_step(
        self,
        step: str,
        rows_before: Optional[int] = None,
        rows_after: Optional[int] = None,
        extra: Optional[dict] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "step": step,
        }
        if rows_before is not None:
            entry["rows_before"] = rows_before
        if rows_after is not None:
            entry["rows_after"] = rows_after
        if extra:
            entry.update(extra)
        self.steps.append(entry)
        msg = f"[{step}]"
        if rows_before is not None:
            msg += f"  rows_before={rows_before}"
        if rows_after is not None:
            msg += f"  rows_after={rows_after}"
        if extra:
            msg += f"  {extra}"
        logger.info(msg)


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------

def read_csv_source(config: CSVSourceConfig, summary: ProcessingSummary) -> pd.DataFrame:
    """Read a CSV file from a local path."""
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, encoding=config.encoding, sep=config.sep)
    summary.log_step("read_csv", rows_after=len(df), extra={"path": str(path)})
    return df


def fetch_api_source(config: APISourceConfig, summary: ProcessingSummary) -> pd.DataFrame:
    """
    Fetch data from a REST API endpoint.
    The API key is loaded from an environment variable — never hardcoded.
    """
    api_key = os.environ.get(config.api_key_env_var)
    if not api_key:
        raise EnvironmentError(
            f"API key not found. Set the environment variable '{config.api_key_env_var}'."
        )

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(
            config.url,
            headers=headers,
            params=config.params,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise TimeoutError(f"API request timed out: {config.url}") from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"HTTP error from API: {exc}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(f"Cannot connect to API: {config.url}") from exc

    payload = response.json()

    if config.record_path:
        # Navigate nested JSON to the list of records
        keys = config.record_path.split(".")
        for k in keys:
            payload = payload[k]

    if isinstance(payload, list):
        df = pd.DataFrame(payload)
    elif isinstance(payload, dict):
        df = pd.DataFrame([payload])
    else:
        raise ValueError(f"Unexpected API payload type: {type(payload)}")

    summary.log_step(
        "fetch_api",
        rows_after=len(df),
        extra={"url": config.url},
    )
    return df


def read_db_config(config: DBConfigFile, summary: ProcessingSummary) -> dict:
    """
    Read database connection parameters from a JSON config file.
    Passwords / secrets inside the file should themselves reference
    environment variables (pattern: {"password": "$DB_PASSWORD"}).
    This function resolves such references automatically.
    """
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"DB config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        db_params: dict = json.load(fh)

    # Resolve environment-variable references (values starting with "$")
    resolved: dict = {}
    for k, v in db_params.items():
        if isinstance(v, str) and v.startswith("$"):
            env_var = v[1:]
            env_val = os.environ.get(env_var)
            if env_val is None:
                raise EnvironmentError(
                    f"DB config references env var '{env_var}' which is not set."
                )
            resolved[k] = env_val
        else:
            resolved[k] = v

    summary.log_step("read_db_config", extra={"path": str(path)})
    return resolved


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_sources(
    df_csv: pd.DataFrame,
    df_api: pd.DataFrame,
    merge_key: str,
    summary: ProcessingSummary,
) -> pd.DataFrame:
    """Merge CSV and API DataFrames on a common key (outer join)."""
    rows_before = len(df_csv)
    merged = pd.merge(df_csv, df_api, on=merge_key, how="outer", suffixes=("_csv", "_api"))
    summary.log_step(
        "merge_sources",
        rows_before=rows_before,
        rows_after=len(merged),
        extra={"merge_key": merge_key, "how": "outer"},
    )
    return merged


# ---------------------------------------------------------------------------
# Train / val / test splitting  (BEFORE any preprocessing)
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    config: SplitConfig,
    summary: ProcessingSummary,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train / val / test partitions.

    Strategy selection (in priority order):
      1. Chronological split  — when config.time_col is set
      2. Group-aware split    — when config.group_col is set
      3. Stratified random    — when config.stratify_col is set
      4. Plain random split
    """
    n = len(df)

    # ---- 1. Chronological split ----
    if config.time_col:
        if config.time_col not in df.columns:
            raise ValueError(f"time_col '{config.time_col}' not found in DataFrame.")
        df = df.sort_values(config.time_col).reset_index(drop=True)
        test_cutoff = int(n * (1 - config.test_size))
        val_cutoff = int(test_cutoff * (1 - config.val_size))
        df_train = df.iloc[:val_cutoff].copy()
        df_val = df.iloc[val_cutoff:test_cutoff].copy()
        df_test = df.iloc[test_cutoff:].copy()
        split_type = "chronological"

    # ---- 2. Group-aware split ----
    elif config.group_col:
        if config.group_col not in df.columns:
            raise ValueError(f"group_col '{config.group_col}' not found in DataFrame.")
        groups = df[config.group_col].values

        gss_test = GroupShuffleSplit(
            n_splits=1,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        train_val_idx, test_idx = next(gss_test.split(df, groups=groups))
        df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].copy()

        groups_tv = df_train_val[config.group_col].values
        gss_val = GroupShuffleSplit(
            n_splits=1,
            test_size=config.val_size,
            random_state=config.random_state,
        )
        train_idx, val_idx = next(gss_val.split(df_train_val, groups=groups_tv))
        df_train = df_train_val.iloc[train_idx].copy()
        df_val = df_train_val.iloc[val_idx].copy()
        split_type = "group_aware"

    # ---- 3. Stratified / plain random split ----
    else:
        stratify = df[config.stratify_col] if config.stratify_col else None
        df_train_val, df_test = train_test_split(
            df,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify,
        )
        stratify_tv = (
            df_train_val[config.stratify_col] if config.stratify_col else None
        )
        df_train, df_val = train_test_split(
            df_train_val,
            test_size=config.val_size,
            random_state=config.random_state,
            stratify=stratify_tv,
        )
        split_type = "stratified" if config.stratify_col else "random"

    summary.log_step(
        "split_data",
        rows_before=n,
        extra={
            "split_type": split_type,
            "train_rows": len(df_train),
            "val_rows": len(df_val),
            "test_rows": len(df_test),
        },
    )
    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Preprocessing  (fit on train, transform val/test)
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """
    Fits imputers, scalers, and encoders on training data only,
    then transforms validation and test sets consistently.
    """

    def __init__(
        self,
        numeric_cols: list,
        categorical_cols: list,
        missing_numeric_strategy: str = "median",
        missing_categorical_strategy: str = "most_frequent",
    ) -> None:
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.missing_numeric_strategy = missing_numeric_strategy
        self.missing_categorical_strategy = missing_categorical_strategy

        self._num_imputer: Optional[SimpleImputer] = None
        self._cat_imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[StandardScaler] = None
        self._cat_encoder: Optional[OrdinalEncoder] = None
        self._fitted = False

    def fit_transform(
        self, df_train: pd.DataFrame, summary: ProcessingSummary
    ) -> pd.DataFrame:
        """Fit all transformers on training data and return transformed copy."""
        df = df_train.copy()

        # --- Numeric imputation ---
        if self.numeric_cols:
            present_num = [c for c in self.numeric_cols if c in df.columns]
            if present_num:
                self._num_imputer = SimpleImputer(strategy=self.missing_numeric_strategy)
                df[present_num] = self._num_imputer.fit_transform(df[present_num])

                self._scaler = StandardScaler()
                df[present_num] = self._scaler.fit_transform(df[present_num])

        # --- Categorical imputation + encoding ---
        if self.categorical_cols:
            present_cat = [c for c in self.categorical_cols if c in df.columns]
            if present_cat:
                self._cat_imputer = SimpleImputer(strategy=self.missing_categorical_strategy)
                df[present_cat] = self._cat_imputer.fit_transform(df[present_cat])

                self._cat_encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                df[present_cat] = self._cat_encoder.fit_transform(df[present_cat])

        self._fitted = True
        summary.log_step(
            "fit_transform_train",
            rows_after=len(df),
            extra={
                "numeric_cols": self.numeric_cols,
                "categorical_cols": self.categorical_cols,
            },
        )
        return df

    def transform(
        self, df: pd.DataFrame, partition_name: str, summary: ProcessingSummary
    ) -> pd.DataFrame:
        """Apply fitted transformers to val or test data."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        df = df.copy()

        if self._num_imputer is not None:
            present_num = [c for c in self.numeric_cols if c in df.columns]
            if present_num:
                df[present_num] = self._num_imputer.transform(df[present_num])
                df[present_num] = self._scaler.transform(df[present_num])

        if self._cat_imputer is not None:
            present_cat = [c for c in self.categorical_cols if c in df.columns]
            if present_cat:
                df[present_cat] = self._cat_imputer.transform(df[present_cat])
                df[present_cat] = self._cat_encoder.transform(df[present_cat])

        summary.log_step(
            f"transform_{partition_name}",
            rows_after=len(df),
        )
        return df


# ---------------------------------------------------------------------------
# Cleaning helpers  (applied before split — only structural cleaning)
# ---------------------------------------------------------------------------

def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[list],
    summary: ProcessingSummary,
) -> pd.DataFrame:
    rows_before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    summary.log_step(
        "remove_duplicates",
        rows_before=rows_before,
        rows_after=len(df),
        extra={"subset": subset},
    )
    return df


def drop_columns(
    df: pd.DataFrame,
    cols: list,
    summary: ProcessingSummary,
) -> pd.DataFrame:
    cols_to_drop = [c for c in cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        summary.log_step("drop_columns", extra={"dropped": cols_to_drop})
    return df


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save_partition(
    df: pd.DataFrame,
    output_dir: Path,
    name: str,
    summary: ProcessingSummary,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    checksum = _compute_sha256(out_path)
    summary.log_step(
        f"save_{name}",
        rows_after=len(df),
        extra={"path": str(out_path), "sha256": checksum},
    )
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    db_params: dict
    summary: ProcessingSummary
    preprocessor: DataPreprocessor


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """
    Execute the full ML data processing pipeline.

    Steps
    -----
    1.  Read CSV source
    2.  Fetch API source
    3.  Read DB config file
    4.  Merge sources on common key
    5.  Structural cleaning (drop cols, remove duplicates)
    6.  Train / val / test split  ← BEFORE any feature engineering
    7.  Fit preprocessors on train; transform val & test
    8.  Save processed partitions
    9.  Return results + summary
    """
    summary = ProcessingSummary()
    summary.log_step("pipeline_start", extra={"merge_key": config.merge_key})

    # ---- 1. Read CSV ----
    df_csv = read_csv_source(config.csv_source, summary)

    # ---- 2. Fetch API ----
    df_api = fetch_api_source(config.api_source, summary)

    # ---- 3. Read DB config ----
    db_params = read_db_config(config.db_config_file, summary)

    # ---- 4. Merge ----
    df_merged = merge_sources(df_csv, df_api, config.merge_key, summary)

    # ---- 5. Structural cleaning (safe to do before split) ----
    df_merged = drop_columns(df_merged, config.drop_cols, summary)
    df_merged = remove_duplicates(df_merged, config.duplicate_subset, summary)

    # ---- 6. Split BEFORE any preprocessing ----
    df_train, df_val, df_test = split_data(df_merged, config.split, summary)

    # ---- 7. Fit on train, transform val & test ----
    preprocessor = DataPreprocessor(
        numeric_cols=config.numeric_cols,
        categorical_cols=config.categorical_cols,
        missing_numeric_strategy=config.missing_numeric_strategy,
        missing_categorical_strategy