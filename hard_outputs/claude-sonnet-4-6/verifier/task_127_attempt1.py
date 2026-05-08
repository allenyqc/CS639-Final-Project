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
import requests
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logger(name: str = "ml_pipeline", log_file: Optional[str] = None) -> logging.Logger:
    """Configure and return a logger with console (and optional file) handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
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
    record_path: Optional[str] = None   # JSON path to the records list


@dataclass
class DBConfig:
    """Parsed from a JSON config file; used for documentation / future DB reads."""
    host: str = ""
    port: int = 5432
    database: str = ""
    user: str = ""
    password: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class CleaningConfig:
    merge_key: str = "id"
    drop_duplicate_subset: Optional[list] = None
    numeric_fill_strategy: str = "median"   # "median" | "mean" | "zero"
    categorical_fill_value: str = "UNKNOWN"
    categorical_columns: list = field(default_factory=list)
    columns_to_drop: list = field(default_factory=list)
    scale_numeric: bool = True


@dataclass
class PipelineConfig:
    csv_source: CSVSourceConfig
    api_source: APISourceConfig
    db_config_path: str
    cleaning: CleaningConfig
    output_dir: str = "processed_data"
    test_size: float = 0.2
    random_state: int = 42
    log_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Processing summary
# ---------------------------------------------------------------------------

@dataclass
class ProcessingSummary:
    steps: list = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    errors: list = field(default_factory=list)

    def add_step(self, name: str, details: dict):
        self.steps.append({
            "step": name,
            "timestamp": datetime.now().isoformat(),
            **details,
        })

    def finalize(self):
        self.end_time = datetime.now()

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time else None
            ),
            "steps": self.steps,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Individual pipeline steps
# ---------------------------------------------------------------------------

class DataPipeline:
    """
    Configurable ML data processing pipeline.

    Usage
    -----
    >>> config = PipelineConfig(...)
    >>> pipeline = DataPipeline(config)
    >>> result = pipeline.run()
    >>> train_df = result["train"]
    >>> test_df  = result["test"]
    >>> summary  = result["summary"]
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = _setup_logger(log_file=config.log_file)
        self.summary = ProcessingSummary()
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Step 1 – Read CSV
    # ------------------------------------------------------------------

    def _read_csv(self) -> pd.DataFrame:
        cfg = self.config.csv_source
        self.logger.info("Reading CSV from: %s", cfg.path)

        df = pd.read_csv(cfg.path, sep=cfg.sep, encoding=cfg.encoding)

        self.summary.add_step(
            "read_csv",
            {"path": cfg.path, "rows": len(df), "columns": list(df.columns)},
        )
        self.logger.info("CSV loaded – rows: %d, cols: %d", len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Step 2 – Fetch API data
    # ------------------------------------------------------------------

    def _fetch_api_data(self) -> pd.DataFrame:
        cfg = self.config.api_source
        self.logger.info("Fetching data from API: %s", cfg.url)

        headers = {cfg.api_key_header: cfg.api_key}
        try:
            response = requests.get(
                cfg.url,
                headers=headers,
                params=cfg.params,
                timeout=cfg.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            self.logger.error("API request failed: %s", exc)
            self.summary.errors.append({"step": "fetch_api_data", "error": str(exc)})
            raise

        payload = response.json()

        # Navigate to the records list if a path is specified
        if cfg.record_path:
            for key in cfg.record_path.split("."):
                payload = payload[key]

        df = pd.json_normalize(payload) if isinstance(payload, list) else pd.DataFrame([payload])

        self.summary.add_step(
            "fetch_api_data",
            {"url": cfg.url, "rows": len(df), "columns": list(df.columns)},
        )
        self.logger.info("API data fetched – rows: %d, cols: %d", len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Step 3 – Read DB config JSON
    # ------------------------------------------------------------------

    def _read_db_config(self) -> DBConfig:
        path = self.config.db_config_path
        self.logger.info("Reading DB config from: %s", path)

        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        db_cfg = DBConfig(
            host=raw.get("host", ""),
            port=int(raw.get("port", 5432)),
            database=raw.get("database", ""),
            user=raw.get("user", ""),
            password=raw.get("password", ""),
            extra={k: v for k, v in raw.items()
                   if k not in ("host", "port", "database", "user", "password")},
        )

        self.summary.add_step(
            "read_db_config",
            {"path": path, "host": db_cfg.host, "database": db_cfg.database},
        )
        self.logger.info("DB config loaded – host: %s, db: %s", db_cfg.host, db_cfg.database)
        return db_cfg

    # ------------------------------------------------------------------
    # Step 4 – Merge data sources
    # ------------------------------------------------------------------

    def _merge_sources(self, csv_df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
        key = self.config.cleaning.merge_key
        self.logger.info("Merging CSV and API data on key: '%s'", key)

        for name, df in (("CSV", csv_df), ("API", api_df)):
            if key not in df.columns:
                raise KeyError(f"Merge key '{key}' not found in {name} DataFrame. "
                               f"Available columns: {list(df.columns)}")

        merged = pd.merge(csv_df, api_df, on=key, how="outer", suffixes=("_csv", "_api"))

        self.summary.add_step(
            "merge_sources",
            {
                "merge_key": key,
                "csv_rows": len(csv_df),
                "api_rows": len(api_df),
                "merged_rows": len(merged),
            },
        )
        self.logger.info(
            "Merge complete – csv: %d rows, api: %d rows → merged: %d rows",
            len(csv_df), len(api_df), len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Step 5 – Clean data
    # ------------------------------------------------------------------

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.cleaning
        self.logger.info("Starting data cleaning – initial rows: %d", len(df))
        rows_before = len(df)

        # 5a – Drop unwanted columns
        if cfg.columns_to_drop:
            existing = [c for c in cfg.columns_to_drop if c in df.columns]
            df = df.drop(columns=existing)
            self.logger.debug("Dropped columns: %s", existing)

        # 5b – Remove duplicates
        subset = cfg.drop_duplicate_subset
        df = df.drop_duplicates(subset=subset)
        rows_after_dedup = len(df)
        self.logger.info(
            "Duplicates removed: %d rows dropped", rows_before - rows_after_dedup
        )

        # 5c – Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if cfg.numeric_fill_strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif cfg.numeric_fill_strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:  # "zero"
            df[numeric_cols] = df[numeric_cols].fillna(0)

        df[object_cols] = df[object_cols].fillna(cfg.categorical_fill_value)

        missing_after = df.isnull().sum().sum()
        self.logger.info(
            "Missing values filled – strategy: %s | remaining nulls: %d",
            cfg.numeric_fill_strategy, missing_after,
        )

        # 5d – Encode categorical columns
        cat_cols = cfg.categorical_columns or object_cols
        for col in cat_cols:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self._label_encoders[col] = le
            self.logger.debug("Label-encoded column: %s (%d classes)", col, len(le.classes_))

        # 5e – Scale numeric features (optional)
        if cfg.scale_numeric and numeric_cols:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self._scaler = scaler
            self.logger.info("Numeric columns scaled with StandardScaler")

        self.summary.add_step(
            "clean_data",
            {
                "rows_before": rows_before,
                "rows_after": len(df),
                "duplicates_removed": rows_before - rows_after_dedup,
                "numeric_fill_strategy": cfg.numeric_fill_strategy,
                "encoded_columns": list(self._label_encoders.keys()),
                "scaled_numeric": cfg.scale_numeric,
            },
        )
        self.logger.info("Cleaning complete – final rows: %d", len(df))
        return df

    # ------------------------------------------------------------------
    # Step 6 – Train / test split and save
    # ------------------------------------------------------------------

    def _split_and_save(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        test_size    = self.config.test_size
        random_state = self.config.random_state
        output_dir   = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Splitting data – test_size: %.2f, random_state: %d", test_size, random_state
        )

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )

        train_path = output_dir / "train.csv"
        test_path  = output_dir / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        # Compute a simple checksum for data integrity
        def _checksum(path: Path) -> str:
            h = hashlib.md5()
            h.update(path.read_bytes())
            return h.hexdigest()

        self.summary.add_step(
            "split_and_save",
            {
                "train_rows": len(train_df),
                "test_rows":  len(test_df),
                "train_path": str(train_path),
                "test_path":  str(test_path),
                "train_md5":  _checksum(train_path),
                "test_md5":   _checksum(test_path),
            },
        )
        self.logger.info(
            "Saved – train: %d rows → %s | test: %d rows → %s",
            len(train_df), train_path, len(test_df), test_path,
        )
        return train_df, test_df

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """
        Execute the full pipeline.

        Returns
        -------
        dict with keys:
            "train"    – training DataFrame
            "test"     – test DataFrame
            "merged"   – full cleaned DataFrame (before split)
            "summary"  – ProcessingSummary instance
            "encoders" – dict of fitted LabelEncoders
            "scaler"   – fitted StandardScaler (or None)
        """
        self.logger.info("=" * 60)
        self.logger.info("Pipeline started at %s", self.summary.start_time.isoformat())
        self.logger.info("=" * 60)

        try:
            csv_df   = self._read_csv()
            api_df   = self._fetch_api_data()
            _        = self._read_db_config()          # loaded for side-effects / logging
            merged   = self._merge_sources(csv_df, api_df)
            cleaned  = self._clean_data(merged)
            train_df, test_df = self._split_and_save(cleaned)

        except Exception as exc:
            self.logger.exception("Pipeline failed: %s", exc)
            self.summary.errors.append({"step": "pipeline.run", "error": str(exc)})
            self.summary.finalize()
            raise

        self.summary.finalize()
        self.logger.info("=" * 60)
        self.logger.info(
            "Pipeline finished at %s (%.2fs)",
            self.summary.end_time.isoformat(),
            (self.summary.end_time - self.summary.start_time).total_seconds(),
        )
        self.logger.info("=" * 60)

        return {
            "train":    train_df,
            "test":     test_df,
            "merged":   cleaned,
            "summary":  self.summary,
            "encoders": self._label_encoders,
            "scaler":   self._scaler,
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_pipeline_from_dict(config_dict: dict) -> DataPipeline:
    """
    Build a DataPipeline from a plain dictionary (e.g. loaded from YAML/JSON).

    Expected top-level keys: csv_source, api_source, db_config_path,
                             cleaning, output_dir, test_size, random_state,
                             log_file (optional).
    """
    csv_cfg = CSVSourceConfig(**config_dict["csv_source"])

    api_raw = config_dict["api_source"]
    api_cfg = APISourceConfig(**api_raw)

    cleaning_raw = config_dict.get("cleaning", {})
    cleaning_cfg = CleaningConfig(**cleaning_raw)

    pipeline_cfg = PipelineConfig(
        csv_source      = csv_cfg,
        api_source      = api_cfg,
        db_config_path  = config_dict["db_config_path"],
        cleaning        = cleaning_cfg,
        output_dir      = config_dict.get("output_dir", "processed_data"),
        test_size       = float(config_dict.get("test_size", 0.2)),
        random_state    = int(config_dict.get("random_state", 42)),
        log_file        = config_dict.get("log_file"),
    )
    return DataPipeline(pipeline_cfg)


# ---------------------------------------------------------------------------
# Demo / smoke-test (runs only when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, io

    # ── Create temporary artefacts ──────────────────────────────────────────

    tmp = tempfile.mkdtemp()

    # 1. CSV file
    csv_content = """id,age,salary,department,churn
1,25,50000,Engineering,no
2,32,62000,Marketing,yes
3,28,55000,Engineering,no
4,45,80000,HR,yes
5,38,70000,Marketing,no
6,29,52000,Engineering,no
7,50,90000,HR,yes
8,33,65000,Marketing,no
9,27,48000,Engineering,no
10,41,75000,HR,yes
"""
    csv_path = os.path.join(tmp, "employees.csv")
    with open(csv_path, "w") as f:
        f.write(csv_content)

    # 2. DB config JSON
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "ml_db",
        "user": "admin",
        "password": "secret",
    }
    db_config_path = os.path.join(tmp, "db_config.json")
    with open(db_config_path, "w") as f:
        json.dump(db_config, f)

    # 3. Mock API response (monkey-patch requests.get)
    api_payload = [
        {"id": i, "performance_score": round(3.0 + i * 0.1, 1), "remote": i % 2 == 0}
        for i in range(1, 11)
    ]

    class _MockResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return api_payload

    original_get = requests.get
    requests.get = lambda *a, **kw: _MockResponse()

    # ── Build and run the pipeline ──────────────────────────────────────────

    config_dict = {
        "csv_source": {"path": csv_path},
        "api_source": {
            "url": "https://api.example.com/employees",
            "api_key": "demo-key-123",
        },
        "db_config_path":