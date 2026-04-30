from __future__ import annotations

import json
import re
from pathlib import Path

TASKS_PATH = Path("tasks/tasks.json")
OUTPUTS_DIR = Path("outputs")


def load_tasks(path: Path = TASKS_PATH) -> list[dict]:
    """Load and return all benchmark task records from tasks.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_code(task_id: str, mode: str, code: str) -> Path:
    """Save generated (or repaired) code to outputs/<mode>/<task_id>.py."""
    out_dir = OUTPUTS_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_id}.py"
    out_path.write_text(code, encoding="utf-8")
    return out_path


def clean_llm_output(raw: str) -> str:
    """Strip markdown code fences, returning only the Python source inside the first ```python ... ``` block."""
    match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()
