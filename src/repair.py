# =============================================================================
# src/repair.py
#
# PURPOSE:
#   Implements the LLM-based repair loop. When the verifier flags violations
#   in generated code, this module builds a corrective prompt and calls the
#   LLM to produce a fixed version.
#
# TODO:
#   - Implement the repair call using the same LLM client as agent.py
#   - Decide on max repair attempts (currently 1; consider a retry loop)
#   - Re-run verifiers after repair to confirm the fix worked
#   - Log repair attempts (original code, feedback, repaired code) for analysis
# =============================================================================

from __future__ import annotations

from src import prompts


def repair_code(task: dict, code: str, feedback: str) -> str:
    """
    Ask the LLM to fix flagged violations in generated code.

    This function constructs a repair prompt using the original task context,
    the flagged code, and the verifier's feedback, then calls the LLM to
    produce a corrected version.

    Parameters
    ----------
    task : dict
        The original task record (used to reconstruct context for the LLM).
    code : str
        The code that was flagged by one or more verifiers.
    feedback : str
        Human-readable description of all violations found, one per line.

    Returns
    -------
    str
        The repaired code string (cleaned of markdown fences).

    TODO:
        - Import and call _call_llm from agent (or move _call_llm to utils
          to avoid circular imports).
        - Optionally loop: re-verify the repaired code and retry if violations
          persist (up to MAX_REPAIR_ATTEMPTS).
        - Save both the pre- and post-repair code for error analysis.
    """
    MAX_REPAIR_ATTEMPTS = 1  # TODO: increase and add loop once basic repair works

    repair_prompt = prompts.get_repair_prompt(task, code, feedback)

    # TODO: call LLM with repair_prompt and return cleaned output
    # from src.agent import _call_llm  (move _call_llm to utils to avoid circular import)
    # from src.utils import clean_llm_output
    # repaired = clean_llm_output(_call_llm(repair_prompt))
    # return repaired

    raise NotImplementedError("TODO: implement repair_code()")
