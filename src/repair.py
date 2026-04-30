from __future__ import annotations

from src import prompts, utils
from src.llm import call_llm


def repair_code(task: dict, code: str, feedback: str) -> str:
    """Ask the LLM to fix flagged violations and return cleaned code."""
    repair_prompt = prompts.get_repair_prompt(task, code, feedback)
    raw = call_llm(prompts.get_system_message(), repair_prompt)
    return utils.clean_llm_output(raw)
