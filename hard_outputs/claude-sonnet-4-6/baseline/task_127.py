```python
"""
ML Data Processing Pipeline Module

A configurable and reusable pipeline for reading, merging, cleaning,
and preparing data from multiple sources for model training.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


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


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CSVSourceConfig:
    path: str
    sep: str = ","
    encoding: str = "utf-8"


@dataclass
class APISourceConfig:
    url: str
    api_key: str
    api_key_header: str = "X-API-Key"
    params: dict = field(default_factory=dict)
    timeout: int = 30
    records_path: Optional[str] = None   # dot-separated path inside JSON response


@dataclass
class DBConfig:
    host: str = ""
    port: int = 5432
    database: str = ""
    user: str = ""
    password: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class PipelineConfig:
    # Sources
    csv_source: CSVSourceConfig = field(default_factory=lambda: CSVSourceConfig(path="data.csv"))
    api_source: APISourceConfig = field(
        default_factory=lambda: APISourceConfig(url="", api_key="")
    )
    db_config_path: str = "db_config.json"

    # Merge
    merge_key: str = "id"
    merge_how: str = "inner"          # inner | left | right | outer

    # Cleaning
    drop_duplicate_subset: Optional[list] = None
    numeric_fill_strategy: str = "median"   # median | mean | zero
    categorical_fill_value: str = "unknown"
    categorical_columns: list = field(default_factory=list)
    columns_to_drop: list = field(default_factory=list)
    scale_numeric: bool = False

    # Split
    test_size: float = 0.2
    random_state: int = 42
    stratify_column: Optional[str] = None

    # Output
    output_dir: str = "processed_data"
    train_filename: str = "train.csv"
    test_filename: str = "test.csv"
    log_file: Optional[str] = "pipeline.log"


# ---------------------------------------------------------------------------
# Processing summary
# ---------------------------------------------------------------------------

@dataclass
class ProcessingSummary:
    steps: list = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    elapsed_seconds: float = 0.0
    final_train_rows: int = 0
    final_test_rows: int = 0
    final_columns: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def add_step(self, name: str, rows: int, notes: str = "") -> None:
        self.steps.append(
            {
                "step": name,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "rows": rows,
                "notes": notes,
            }
        )

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "steps": self.steps,
            "final_train_rows": self.final_train_rows,
            "final_test_rows": self.final_test_rows,
            "final_columns": self.final_columns,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Individual pipeline steps
# ---------------------------------------------------------------------------

def _read_csv(config: CSVSourceConfig, logger: logging.Logger, summary: ProcessingSummary) -> pd.DataFrame:
    """Step 1 – Read CSV from local path."""
    logger.info("Reading CSV from: %s", config.path)
    df = pd.read_csv(config.path, sep=config.sep, encoding=config.encoding)
    msg = f"path={config.path}"
    summary.add_step("read_csv", len(df), msg)
    logger.info("CSV loaded: %d rows × %d cols", *df.shape)
    return df


def _fetch_api_data(
    config: APISourceConfig,
    logger: logging.Logger,
    summary: ProcessingSummary,
) -> pd.DataFrame:
    """Step 2 – Fetch data from REST API with API-key authentication."""
    logger.info("Fetching data from API: %s", config.url)
    headers = {config.api_key_header: config.api_key}

    try:
        response = requests.get(
            config.url,
            headers=headers,
            params=config.params,
            timeout=config.timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("API request failed (%s). Returning empty DataFrame.", exc)
        summary.errors.append(f"API fetch failed: {exc}")
        summary.add_step("fetch_api", 0, f"FAILED: {exc}")
        return pd.DataFrame()

    payload = response.json()

    # Navigate nested JSON if records_path is provided (e.g. "data.items")
    if config.records_path:
        for key in config.records_path.split("."):
            payload = payload[key]

    df = pd.json_normalize(payload) if isinstance(payload, list) else pd.json_normalize([payload])
    summary.add_step("fetch_api", len(df), f"url={config.url}")
    logger.info("API data loaded: %d rows × %d cols", *df.shape)
    return df


def _read_db_config(path: str, logger: logging.Logger, summary: ProcessingSummary) -> DBConfig:
    """Step 3 – Read JSON configuration file for database connection parameters."""
    logger.info("Reading DB config from: %s", path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict = json.load(fh)
    except FileNotFoundError:
        logger.warning("DB config file not found at '%s'. Using defaults.", path)
        summary.errors.append(f"DB config not found: {path}")
        return DBConfig()
    except json.JSONDecodeError as exc:
        logger.warning("DB config JSON parse error: %s. Using defaults.", exc)
        summary.errors.append(f"DB config parse error: {exc}")
        return DBConfig()

    db_cfg = DBConfig(
        host=raw.get("host", ""),
        port=int(raw.get("port", 5432)),
        database=raw.get("database", ""),
        user=raw.get("user", ""),
        password=raw.get("password", ""),
        extra={k: v for k, v in raw.items() if k not in {"host", "port", "database", "user", "password"}},
    )
    summary.add_step("read_db_config", 0, f"host={db_cfg.host}, db={db_cfg.database}")
    logger.info("DB config loaded: host=%s, database=%s", db_cfg.host, db_cfg.database)
    return db_cfg


def _merge_sources(
    df_csv: pd.DataFrame,
    df_api: pd.DataFrame,
    merge_key: str,
    how: str,
    logger: logging.Logger,
    summary: ProcessingSummary,
) -> pd.DataFrame:
    """Step 4 – Merge CSV and API DataFrames on a common key."""
    if df_api.empty:
        logger.warning("API DataFrame is empty; skipping merge, using CSV data only.")
        summary.add_step("merge", len(df_csv), "API empty – CSV only")
        return df_csv.copy()

    if merge_key not in df_csv.columns:
        raise KeyError(f"Merge key '{merge_key}' not found in CSV DataFrame.")
    if merge_key not in df_api.columns:
        raise KeyError(f"Merge key '{merge_key}' not found in API DataFrame.")

    df_merged = pd.merge(df_csv, df_api, on=merge_key, how=how, suffixes=("_csv", "_api"))
    summary.add_step(
        "merge",
        len(df_merged),
        f"key={merge_key}, how={how}, csv_rows={len(df_csv)}, api_rows={len(df_api)}",
    )
    logger.info(
        "Merge complete: %d rows × %d cols (key=%s, how=%s)",
        *df_merged.shape,
        merge_key,
        how,
    )
    return df_merged


def _clean_data(
    df: pd.DataFrame,
    config: PipelineConfig,
    logger: logging.Logger,
    summary: ProcessingSummary,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Step 5 – Clean merged data:
      - Drop unwanted columns
      - Remove duplicates
      - Handle missing values (numeric & categorical)
      - Encode categorical columns
      - Optionally scale numeric columns
    Returns cleaned DataFrame and a dict of fitted encoders/scalers.
    """
    artifacts: dict[str, Any] = {}
    df = df.copy()

    # --- Drop specified columns ---
    cols_to_drop = [c for c in config.columns_to_drop if c in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        logger.info("Dropped columns: %s", cols_to_drop)

    # --- Remove duplicates ---
    before = len(df)
    df.drop_duplicates(subset=config.drop_duplicate_subset, inplace=True)
    removed = before - len(df)
    summary.add_step("drop_duplicates", len(df), f"removed={removed}")
    logger.info("Duplicates removed: %d (remaining: %d rows)", removed, len(df))

    # --- Handle missing values – numeric ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        if config.numeric_fill_strategy == "median":
            fill_vals = df[numeric_cols].median()
        elif config.numeric_fill_strategy == "mean":
            fill_vals = df[numeric_cols].mean()
        else:  # zero
            fill_vals = pd.Series(0, index=numeric_cols)

        df[numeric_cols] = df[numeric_cols].fillna(fill_vals)
        artifacts["numeric_fill_values"] = fill_vals.to_dict()
        logger.info(
            "Filled missing numerics (%s strategy) in %d columns.",
            config.numeric_fill_strategy,
            len(numeric_cols),
        )

    # --- Handle missing values – categorical ---
    cat_cols_in_df = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols_in_df:
        df[cat_cols_in_df] = df[cat_cols_in_df].fillna(config.categorical_fill_value)
        logger.info(
            "Filled missing categoricals with '%s' in %d columns.",
            config.categorical_fill_value,
            len(cat_cols_in_df),
        )

    summary.add_step("fill_missing", len(df), f"numeric_strategy={config.numeric_fill_strategy}")

    # --- Encode categorical columns ---
    encode_cols = config.categorical_columns or cat_cols_in_df
    encode_cols = [c for c in encode_cols if c in df.columns]
    encoders: dict[str, LabelEncoder] = {}
    for col in encode_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.debug("Label-encoded column: %s (%d classes)", col, len(le.classes_))

    artifacts["label_encoders"] = encoders
    summary.add_step("encode_categoricals", len(df), f"columns={encode_cols}")
    logger.info("Encoded %d categorical column(s).", len(encode_cols))

    # --- Optional numeric scaling ---
    if config.scale_numeric and numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        artifacts["scaler"] = scaler
        summary.add_step("scale_numeric", len(df), f"columns={numeric_cols}")
        logger.info("Scaled %d numeric column(s) with StandardScaler.", len(numeric_cols))

    return df, artifacts


def _split_and_save(
    df: pd.DataFrame,
    config: PipelineConfig,
    logger: logging.Logger,
    summary: ProcessingSummary,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 6 – Train/test split and save processed datasets."""
    stratify = None
    if config.stratify_column and config.stratify_column in df.columns:
        stratify = df[config.stratify_column]
        logger.info("Stratifying split on column: %s", config.stratify_column)

    df_train, df_test = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    summary.add_step(
        "train_test_split",
        len(df),
        f"train={len(df_train)}, test={len(df_test)}, test_size={config.test_size}",
    )
    logger.info("Split: train=%d rows, test=%d rows", len(df_train), len(df_test))

    # Save
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / config.train_filename
    test_path = out_dir / config.test_filename

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    summary.add_step("save_datasets", len(df), f"train={train_path}, test={test_path}")
    logger.info("Saved train dataset to: %s", train_path)
    logger.info("Saved test dataset to:  %s", test_path)

    return df_train, df_test


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    config: Optional[PipelineConfig] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Execute the full ML data processing pipeline.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration object. If None, a default config is created.
        Any keyword arguments override fields on the default config.

    Returns
    -------
    dict with keys:
        - "train"      : pd.DataFrame  – training set
        - "test"       : pd.DataFrame  – test set
        - "artifacts"  : dict          – encoders, scalers, fill values
        - "db_config"  : DBConfig      – parsed database configuration
        - "summary"    : dict          – processing summary with step log
    """
    if config is None:
        config = PipelineConfig(**{k: v for k, v in kwargs.items() if hasattr(PipelineConfig, k)})

    logger = _build_logger(log_file=config.log_file)
    summary = ProcessingSummary()
    summary.start_time = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Pipeline started at %s", summary.start_time)
    logger.info("=" * 60)

    try:
        # Step 1 – CSV
        df_csv = _read_csv(config.csv_source, logger, summary)

        # Step 2 – API
        df_api = _fetch_api_data(config.api_source, logger, summary)

        # Step 3 – DB config
        db_cfg = _read_db_config(config.db_config_path, logger, summary)

        # Step 4 – Merge
        df_merged = _merge_sources(
            df_csv, df_api, config.merge_key, config.merge_how, logger, summary
        )

        # Step 5 – Clean
        df_clean, artifacts = _clean_data(df_merged, config, logger, summary)

        # Step 6 – Split & save
        df_train, df_test = _split_and_save(df_clean, config, logger, summary)

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        summary.errors.append(str(exc))
        summary.end_time = datetime.now().isoformat(timespec="seconds")
        summary.elapsed_seconds = time.perf_counter() - t0
        raise

    summary.end_time = datetime.now().isoformat(timespec="seconds")
    summary.elapsed_seconds = time.perf_counter() - t0
    summary.final_train_rows = len(df_train)
    summary.final_test_rows = len(df_test)
    summary.final_columns = df_train.columns.tolist()

    logger.info("=" * 60)
    logger.info(
        "Pipeline finished at %s (%.2fs elapsed)",
        summary.end_time,
        summary.elapsed_seconds,
    )
    logger.info(
        "Final datasets – train: %d rows, test: %d rows, columns: %d",
        summary.final_train_rows,
        summary.final_test_rows,
        len(summary.final_columns),
    )
    logger.info("=" * 60)

    return {
        "train": df_train,
        "test": df_test,
        "artifacts": artifacts,
        "db_config": db_cfg,
        "summary": summary.to_dict(),
    }


# ---------------------------------------------------------------------------
# Convenience factory helpers
# ---------------------------------------------------------------------------

def pipeline_from_dict(cfg_dict: dict) -> dict[str, Any]:
    """
    Build and run a pipeline from a plain dictionary (e.g. loaded from YAML/JSON).

    Expected top-level keys (all optional):
        csv_source, api_source, db_config_path, merge_key, merge_how,
        drop_duplicate_subset, numeric_fill_strategy, categorical_fill_value,
        categorical_columns, columns_to_drop, scale_numeric,
        test_size, random_state, stratify_column,
        output_dir, train_filename, test_filename, log_file
    """
    csv_cfg = CSVSourceConfig(**cfg_dict.pop("csv_source", {}))
    api_cfg = APISourceConfig(**cfg_dict.pop("api_source", {"url": "", "api_key": ""}))
    config = PipelineConfig(csv_source=csv_cfg, api_source=api_cfg, **cfg_dict)
    return run_pipeline(config)


def pipeline_from_json(json_path: str) -> dict[str, Any]:
    """Load pipeline configuration from a JSON file and run the pipeline."""
    with open(json_path, "r", encoding="utf-8") as fh:
        cfg_dict = json.load(fh)
    return pipeline_from_dict(cfg_dict)


# ---------------------------------------------------------------------------
# Demo / smoke-test (runs when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main