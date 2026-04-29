# =============================================================================
# src/prompts.py
#
# PURPOSE:
#   Defines all prompt templates used by the agent. Keeping prompts here (not
#   in agent.py) makes it easy to iterate on wording without touching logic.
#
# TODO:
#   - Refine baseline prompt to be realistic but pitfall-neutral
#   - Strengthen instruction prompt: which pitfalls to list? how explicit?
#   - Experiment with few-shot examples in instruction prompt
#   - Evaluate whether system-message vs user-message placement matters
# =============================================================================

from __future__ import annotations


_SYSTEM_MSG = (
    "You are an expert Python programmer specializing in machine learning. "
    "Return only valid Python code, no prose. "
    "Wrap your code in a single ```python ... ``` block."
)


def get_system_message() -> str:
    """Return the shared system message used across all modes."""
    return _SYSTEM_MSG


def get_baseline_prompt(task: dict) -> str:
    """
    Build a plain prompt for the baseline mode.

    The prompt contains no hints about common ML pitfalls — the LLM must
    rely solely on its pre-trained knowledge.

    Parameters
    ----------
    task : dict
        A task record with at least a "prompt" field.

    Returns
    -------
    str
        The full user-turn prompt string.

    TODO: decide whether to include task category as context.
    """
    return task["prompt"]


def get_instruction_prompt(task: dict) -> str:
    """
    Build an instruction-augmented prompt that warns the model about common
    ML coding pitfalls without naming the specific violation in this task.

    Parameters
    ----------
    task : dict
        A task record with at least a "prompt" field.

    Returns
    -------
    str
        The full user-turn prompt string with appended instructions.

    TODO:
        - Decide on the right level of specificity (generic list vs targeted).
        - A/B test different instruction phrasings.
    """
    instructions = (
        "\n\nIMPORTANT — follow these ML best practices:\n"
        "1. Fit all preprocessors (scalers, encoders) ONLY on training data.\n"
        "2. Never use test data to select hyperparameters or thresholds.\n"
        "3. For imbalanced datasets, prefer F1, AUC, or MCC over plain accuracy.\n"
        "4. Apply train/test splits BEFORE any feature engineering.\n"
    )
    return task["prompt"] + instructions


def get_repair_prompt(task: dict, code: str, feedback: str) -> str:
    """
    Build a repair prompt that asks the LLM to fix a specific violation in
    previously generated code.

    Parameters
    ----------
    task : dict
        The original task record.
    code : str
        The code that was flagged by the verifier.
    feedback : str
        Human-readable description of the violation(s) detected.

    Returns
    -------
    str
        A prompt asking the LLM to produce corrected code.

    TODO:
        - Test whether including the original task prompt helps.
        - Consider chain-of-thought before the fix to improve repair quality.
    """
    return (
        f"The following Python code was written for this task:\n"
        f"TASK: {task['prompt']}\n\n"
        f"CODE:\n```python\n{code}\n```\n\n"
        f"The code has the following problem(s):\n{feedback}\n\n"
        f"Please rewrite the code to fix these issues. "
        f"Return only the corrected Python code in a single ```python ... ``` block."
    )
