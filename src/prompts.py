from __future__ import annotations

_SYSTEM_MSG = (
    "You are an expert Python programmer specializing in machine learning. "
    "Return only valid Python code, no prose. "
    "Wrap your code in a single ```python ... ``` block."
)

_ML_INSTRUCTIONS = (
    "\n\nIMPORTANT — follow these ML best practices:\n"
    "1. Fit all preprocessors (scalers, encoders) ONLY on training data.\n"
    "2. Never use test data to select hyperparameters or thresholds.\n"
    "3. For imbalanced datasets, prefer F1, AUC, or MCC over plain accuracy.\n"
    "4. Apply train/test splits BEFORE any feature engineering.\n"
)

_CODING_INSTRUCTIONS = (
    "\n\nIMPORTANT — follow these Python best practices:\n"
    "1. Never use mutable objects (lists, dicts) as default argument values; use None and create them inside the function body.\n"
    "2. Catch only the specific exceptions you expect; avoid bare `except:` and broad `except Exception:` clauses.\n"
    "3. Always release resources with context managers (`with` statements) rather than manual open/close.\n"
    "4. Avoid shadowing built-in names (e.g., `list`, `dict`, `id`) as variable names.\n"
)


def get_system_message() -> str:
    return _SYSTEM_MSG


def get_baseline_prompt(task: dict) -> str:
    return task["prompt"]


def get_instruction_prompt(task: dict) -> str:
    if task.get("category") == "coding":
        return task["prompt"] + _CODING_INSTRUCTIONS
    return task["prompt"] + _ML_INSTRUCTIONS


def get_repair_prompt(task: dict, code: str, feedback: str) -> str:
    return (
        f"The previous solution violates ML/Python protocol.\n\n"
        f"TASK: {task['prompt']}\n\n"
        f"Violation:\n{feedback}\n\n"
        f"Original code:\n```python\n{code}\n```\n\n"
        f"Please fix it. Return Python code only in a single ```python ... ``` block."
    )
