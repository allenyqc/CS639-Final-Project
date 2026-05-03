from __future__ import annotations

import argparse
import json
from typing import Literal

from . import prompts, verifiers, repair, utils
from .llm import call_llm

Mode = Literal["baseline", "instruction", "verifier"]

MAX_REPAIR_ATTEMPTS = 2


def _call_llm(prompt: str) -> str:
    return call_llm(prompts.get_system_message(), prompt)


def run_agent(task: dict, mode: Mode) -> dict:
    task_id = task["id"]
    result = {
        "task_id": task_id,
        "mode": mode,
        "code": "",
        "violations": [],
        "initial_violations": [],
        "repaired": False,
        "functional_success": None,
    }

    if mode == "baseline":
        code = utils.clean_llm_output(_call_llm(prompts.get_baseline_prompt(task)))
        result["code"] = code
        ver = verifiers.run_verifiers(code, task.get("violation_type"))
        result["violations"] = ver["violations"]

    elif mode == "instruction":
        code = utils.clean_llm_output(_call_llm(prompts.get_instruction_prompt(task)))
        result["code"] = code
        ver = verifiers.run_verifiers(code, task.get("violation_type"))
        result["violations"] = ver["violations"]

    elif mode == "verifier":
        code = utils.clean_llm_output(_call_llm(prompts.get_baseline_prompt(task)))
        attempt = 1
        while True:
            utils.save_code(f"{task_id}_attempt{attempt}", mode, code)
            ver = verifiers.run_verifiers(code, task.get("violation_type"))
            if attempt == 1:
                result["initial_violations"] = ver["violations"]
            result["violations"] = ver["violations"]
            if not ver["has_violation"] or attempt > MAX_REPAIR_ATTEMPTS:
                break
            code = repair.repair_code(task, code, ver["feedback"])
            result["repaired"] = True
            attempt += 1
        result["code"] = code

    utils.save_code(task_id, mode, result["code"])
    return result


def run_all_tasks(tasks: list[dict], mode: Mode) -> list[dict]:
    results = []
    for task in tasks:
        print(f"[{mode}] {task['id']} ...")
        results.append(run_agent(task, mode))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the codevoyager-lite benchmark.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "instruction", "verifier", "all"],
        default="all",
    )
    parser.add_argument("--tasks", default=None, help="Path to tasks JSON file.")
    args = parser.parse_args()

    task_path = utils.TASKS_PATH if args.tasks is None else args.tasks
    tasks = utils.load_tasks(task_path)
    modes: list[Mode] = (
        ["baseline", "instruction", "verifier"] if args.mode == "all" else [args.mode]
    )

    summary_path = utils.OUTPUTS_DIR / "results.json"
    utils.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results so re-runs of individual modes don't overwrite other modes.
    existing: list[dict] = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))

    new_results: list[dict] = []
    for mode in modes:
        new_results.extend(run_all_tasks(tasks, mode))

    # Replace entries for the modes we just ran; keep everything else.
    ran_modes = set(modes)
    merged = [r for r in existing if r["mode"] not in ran_modes] + new_results
    summary_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"\nDone. Results saved to {summary_path}")
