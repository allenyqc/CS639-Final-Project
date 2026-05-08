from __future__ import annotations

import random

_SYSTEM_MSG = (
    "You are an expert Python programmer specializing in machine learning. "
    "Return only valid Python code, no prose. "
    "Wrap your code in a single ```python ... ``` block."
)

# ---------------------------------------------------------------------------
# Hard-task instruction best practices: 4 dimensions × 2 rules each
# Aligned with the 4 verifier checks:
#   1. DATA SEPARATION
#   2. TEST ISOLATION
#   3. EVALUATION RIGOR
#   4. CODE SAFETY
# ---------------------------------------------------------------------------

_HARD_PRACTICES: list[tuple[str, str]] = [
    # Dimension 1: DATA SEPARATION
    (
        "Split data into train/test BEFORE any preprocessing, feature engineering, "
        "or augmentation; fit all transformers (scalers, vectorizers, embeddings) "
        "ONLY on the training partition.",
        "For time-series data, use chronological splits (never random shuffle); "
        "for grouped data (same patient/user appearing multiple times), use "
        "group-aware splitting (GroupKFold) so the same entity never appears in "
        "both train and test.",
    ),
    # Dimension 2: TEST ISOLATION
    (
        "Never use test data for anything except final evaluation — do not fit "
        "models, select thresholds, tune hyperparameters, or calibrate "
        "probabilities on the test set.",
        "Create a separate validation split for threshold tuning, early stopping "
        "callbacks, and probability calibration; the test set must remain "
        "completely untouched until final metric reporting.",
    ),
    # Dimension 3: EVALUATION RIGOR
    (
        "Choose metrics appropriate to the task type: concordance index for "
        "survival data, NDCG/MAP for ranking, sample-averaged F1 or hamming loss "
        "for multi-label; never report only accuracy on imbalanced data.",
        "When performing hyperparameter search (GridSearchCV), report performance "
        "on a separate held-out test set, not the inner CV best_score_; when "
        "running multiple statistical tests, apply Bonferroni or Benjamini-Hochberg "
        "correction.",
    ),
    # Dimension 4: CODE SAFETY
    (
        "Never hardcode credentials, API keys, or connection strings in source "
        "code; load secrets from environment variables or a dedicated secrets "
        "manager.",
        "Avoid unsafe deserialization (pickle.load / joblib.load) without "
        "integrity verification; use None for mutable default arguments; catch "
        "only specific exceptions instead of bare except clauses.",
    ),
]


def _build_instructions(n_per_dim: int = 2, seed: int | None = None) -> str:
    """
    Build the instruction string.

    n_per_dim=2 → all 8 practices (full)
    n_per_dim=1 → randomly pick 1 per dimension using `seed` (ablation)
    """
    rng = random.Random(seed) if seed is not None else None
    lines = ["\n\nIMPORTANT — follow these best practices:\n"]
    idx = 1
    for pair in _HARD_PRACTICES:
        if n_per_dim == 1 and rng is not None:
            selected = [rng.choice(pair)]
        else:
            selected = list(pair)
        for rule in selected:
            lines.append(f"{idx}. {rule}\n")
            idx += 1
    return "".join(lines)


INSTRUCTION_8 = _build_instructions(n_per_dim=2)
INSTRUCTION_4 = _build_instructions(n_per_dim=1, seed=42)

# Legacy instructions for standard tasks (tasks 001-016)
_ML_INSTRUCTIONS = (
    "\n\nIMPORTANT — follow these ML best practices:\n"
    "1. Fit all preprocessors (scalers, encoders) ONLY on training data.\n"
    "2. Never use test data to select hyperparameters or thresholds.\n"
    "3. For imbalanced datasets, prefer F1, AUC, or MCC over plain accuracy.\n"
    "4. Apply train/test splits BEFORE any feature engineering.\n"
)

_CODING_INSTRUCTIONS = (
    "\n\nIMPORTANT — follow these Python best practices:\n"
    "1. Never use mutable objects (lists, dicts) as default argument values; "
    "use None and create them inside the function body.\n"
    "2. Catch only the specific exceptions you expect; avoid bare `except:` "
    "and broad `except Exception:` clauses.\n"
    "3. Always release resources with context managers (`with` statements) "
    "rather than manual open/close.\n"
    "4. Avoid shadowing built-in names (e.g., `list`, `dict`, `id`) as "
    "variable names.\n"
)


def get_system_message() -> str:
    return _SYSTEM_MSG


def get_baseline_prompt(task: dict) -> str:
    return task["prompt"]


def get_instruction_prompt(task: dict, n_practices: int = 8) -> str:
    """
    Return task prompt + best-practice instructions.

    For hard tasks (id starts with 'task_1'):
      n_practices=8 → full 8 rules (2 per dimension)
      n_practices=4 → ablation: 1 randomly selected rule per dimension (seed=42)

    For standard tasks: uses legacy _ML_INSTRUCTIONS / _CODING_INSTRUCTIONS.
    """
    task_id = task.get("id", "")
    is_hard = task_id.startswith("task_1")

    if is_hard:
        suffix = INSTRUCTION_8 if n_practices == 8 else INSTRUCTION_4
        return task["prompt"] + suffix

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
