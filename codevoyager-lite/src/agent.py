# =============================================================================
# src/agent.py
#
# PURPOSE:
#   Main agent loop. Iterates over all benchmark tasks and runs each one in
#   a specified mode (baseline, instruction, or verifier).
#
# TODO:
#   - Integrate LLM API (OpenAI / Anthropic) in _call_llm()
#   - Wire verifier mode: call verifiers, conditionally call repair
#   - Save outputs to outputs/<mode>/<task_id>.py via utils.save_code()
#   - Add CLI entry point (argparse) so modes can be selected at runtime
# =============================================================================

from __future__ import annotations

import json
from typing import Literal

from src import prompts, verifiers, repair, utils

Mode = Literal["baseline", "instruction", "verifier"]


def _call_llm(prompt: str) -> str:
    """
    Send a prompt to the LLM and return the raw text response.

    TODO: implement API call (OpenAI ChatCompletion or Anthropic Messages).
    """
    raise NotImplementedError("TODO: implement _call_llm()")


def run_agent(task: dict, mode: Mode) -> dict:
    """
    Run a single task in the given mode.

    Parameters
    ----------
    task : dict
        A task record loaded from tasks.json.
    mode : Mode
        One of "baseline", "instruction", or "verifier".

    Returns
    -------
    dict
        {
            "task_id": str,
            "mode": str,
            "code": str,          # final generated (and possibly repaired) code
            "violations": list,   # list of violation labels found (verifier mode only)
            "repaired": bool,     # whether repair was triggered
        }

    TODO: implement each mode branch below.
    """
    task_id = task["id"]
    result = {
        "task_id": task_id,
        "mode": mode,
        "code": "",
        "violations": [],
        "repaired": False,
    }

    if mode == "baseline":
        # TODO: build baseline prompt, call LLM, save output
        prompt = prompts.get_baseline_prompt(task)
        code = _call_llm(prompt)
        result["code"] = utils.clean_llm_output(code)

    elif mode == "instruction":
        # TODO: build instruction prompt, call LLM, save output
        prompt = prompts.get_instruction_prompt(task)
        code = _call_llm(prompt)
        result["code"] = utils.clean_llm_output(code)

    elif mode == "verifier":
        # TODO: baseline generation → verify → repair if violations found
        prompt = prompts.get_baseline_prompt(task)
        code = utils.clean_llm_output(_call_llm(prompt))
        violations = verifiers.run_verifiers(code)
        result["violations"] = violations
        if violations:
            feedback = "\n".join(violations)
            code = repair.repair_code(task, code, feedback)
            result["repaired"] = True
        result["code"] = code

    utils.save_code(task_id, mode, result["code"])
    return result


def run_all_tasks(tasks: list[dict], mode: Mode) -> list[dict]:
    """
    Run every task in the benchmark under the specified mode.

    Parameters
    ----------
    tasks : list[dict]
        All task records (output of utils.load_tasks()).
    mode : Mode
        Which agent mode to use.

    Returns
    -------
    list[dict]
        One result dict per task (same format as run_agent).

    TODO: optionally add progress logging.
    """
    results = []
    for task in tasks:
        result = run_agent(task, mode)
        results.append(result)
    return results


if __name__ == "__main__":
    # TODO: replace with argparse for mode selection
    tasks = utils.load_tasks()
    for mode in ("baseline", "instruction", "verifier"):
        run_all_tasks(tasks, mode)
