# =============================================================================
# src/utils.py
#
# PURPOSE:
#   Shared helper functions used across the project. Keeps agent.py, verifiers.py,
#   and evaluate.py free of boilerplate I/O and string-processing code.
#
# TODO:
#   - Implement save_code() to write outputs to the correct outputs/ subdirectory
#   - Implement load_tasks() to parse tasks/tasks.json
#   - Implement clean_llm_output() to strip markdown fences from LLM responses
#   - Consider adding a simple logger setup here
# =============================================================================

from __future__ import annotations

import json
import re
from pathlib import Path

TASKS_PATH = Path("tasks/tasks.json")
OUTPUTS_DIR = Path("outputs")


def load_tasks(path: Path = TASKS_PATH) -> list[dict]:
    """
    Load and return all benchmark task records from tasks.json.

    Parameters
    ----------
    path : Path
        Path to the tasks JSON file.

    Returns
    -------
    list[dict]
        Parsed list of task records.

    TODO: add schema validation (check required fields are present).
    """
    raise NotImplementedError("TODO: implement load_tasks()")


def save_code(task_id: str, mode: str, code: str) -> Path:
    """
    Save generated (or repaired) code to outputs/<mode>/<task_id>.py.

    Parameters
    ----------
    task_id : str
        The task identifier (used as the filename stem).
    mode : str
        Agent mode ("baseline", "instruction", or "verifier").
    code : str
        Python source code to write.

    Returns
    -------
    Path
        The path where the file was written.

    TODO: create parent directory if it does not exist.
    """
    raise NotImplementedError("TODO: implement save_code()")


def clean_llm_output(raw: str) -> str:
    """
    Strip markdown code fences from an LLM response, returning only the
    Python source code inside the first ```python ... ``` block.

    If no fenced block is found, return the raw string with leading/trailing
    whitespace removed (graceful fallback).

    Parameters
    ----------
    raw : str
        The raw text response from the LLM.

    Returns
    -------
    str
        Cleaned Python source code.

    TODO:
        - Handle responses with multiple code blocks (take the first? largest?)
        - Log a warning when no fence is found, so we can track LLM formatting issues.
    """
    raise NotImplementedError("TODO: implement clean_llm_output()")
