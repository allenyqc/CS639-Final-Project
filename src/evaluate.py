# =============================================================================
# src/evaluate.py
#
# PURPOSE:
#   Scores each generated code sample on two axes:
#     1. Functional success — does the code solve the stated task?
#     2. Protocol violation — does it commit the hidden pitfall?
#   Aggregates per-task results into summary statistics across modes.
#
# TODO:
#   - Define the rubric for "functional success" (human label? execution? AST?)
#   - Define per-violation-type detection logic (may reuse verifiers.py)
#   - Implement compute_metrics() to produce per-mode summary statistics
#   - Generate result plots (pass/fail rates, violation rates by mode)
# =============================================================================

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

RESULTS_PATH = Path("results/results.csv")


def evaluate_result(task_id: str, method: str, code: str) -> dict:
    """
    Score a single generated code sample.

    Parameters
    ----------
    task_id : str
        The identifier of the task being evaluated.
    method : str
        The agent mode that produced the code ("baseline", "instruction",
        or "verifier").
    code : str
        The generated (and possibly repaired) Python code.

    Returns
    -------
    dict
        {
            "task_id": str,
            "method": str,
            "functional_success": bool,   # TODO: define this rubric
            "protocol_violation": bool,   # True if hidden pitfall is present
            "repaired": bool,             # whether repair was applied
        }

    TODO:
        - functional_success: start with human annotation; later try execution.
        - protocol_violation: re-run verifiers on the final code, or use a
          separate gold-label check per task.
        - Add a "notes" field for qualitative observations.
    """
    raise NotImplementedError("TODO: implement evaluate_result()")


def compute_metrics(results: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-task results into per-mode summary statistics.

    Parameters
    ----------
    results : list[dict]
        List of result dicts, one per (task, mode) pair.

    Returns
    -------
    pd.DataFrame
        Columns: method, functional_success_rate, violation_rate, repair_rate
        Rows: one per mode.

    TODO:
        - Add confidence intervals (bootstrap or Wilson) once n >= 10 tasks.
        - Break down violation_rate by category (data_leakage, test_misuse, etc.)
    """
    raise NotImplementedError("TODO: implement compute_metrics()")


def load_results() -> pd.DataFrame:
    """
    Load the results CSV into a DataFrame.

    Returns
    -------
    pd.DataFrame
        All recorded results.

    TODO: add validation that expected columns are present.
    """
    return pd.read_csv(RESULTS_PATH)


def append_result(row: dict) -> None:
    """
    Append a single result row to results.csv.

    Parameters
    ----------
    row : dict
        A result dict as returned by evaluate_result().

    TODO: make this atomic / thread-safe if tasks run in parallel.
    """
    fieldnames = ["task_id", "method", "functional_success", "protocol_violation", "repaired"]
    file_exists = RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
