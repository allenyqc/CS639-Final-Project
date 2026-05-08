from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal

from . import prompts, verifiers, repair, utils
from .llm import call_llm

Mode = Literal["baseline", "instruction", "verifier"]

MAX_REPAIR_ATTEMPTS = 2


def _call_llm(prompt: str) -> str:
    return call_llm(prompts.get_system_message(), prompt)


def run_agent(task: dict, mode: Mode, model: str, n_practices: int = 8) -> dict:
    task_id = task["id"]
    result = {
        "task_id": task_id,
        "mode": mode,
        "model": model,
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
        code = utils.clean_llm_output(
            _call_llm(prompts.get_instruction_prompt(task, n_practices=n_practices)))
        result["code"] = code
        ver = verifiers.run_verifiers(code, task.get("violation_type"))
        result["violations"] = ver["violations"]

    elif mode == "verifier":
        code = utils.clean_llm_output(_call_llm(prompts.get_baseline_prompt(task)))
        attempt = 1
        while True:
            utils.save_code(f"{task_id}_attempt{attempt}", mode, code, model)
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

    utils.save_code(task_id, mode, result["code"], model)
    return result


def _save_results(all_results: list[dict], path: Path) -> None:
    """Atomically write results to disk (write to tmp then rename)."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    tmp.replace(path)


def _rebuild_from_code_files(
    tasks: list[dict], modes: list[Mode], model: str,
) -> list[dict]:
    """Re-derive results from saved .py files when results.json is missing.

    For each (task, mode) where a code file exists, re-run the verifiers to
    reconstruct the result record.  This recovers progress lost due to an
    interrupted run that never wrote results.json.
    """
    recovered: list[dict] = []
    task_map = {t["id"]: t for t in tasks}
    for mode in modes:
        code_dir = utils.OUTPUTS_DIR / model / mode
        if not code_dir.is_dir():
            continue
        for py_file in sorted(code_dir.glob("task_*.py")):
            task_id = py_file.stem
            if "_attempt" in task_id:
                continue
            task = task_map.get(task_id)
            if task is None:
                continue
            code = py_file.read_text(encoding="utf-8")
            ver = verifiers.run_verifiers(code, task.get("violation_type"))
            recovered.append({
                "task_id": task_id,
                "mode": mode,
                "model": model,
                "code": code,
                "violations": ver["violations"],
                "initial_violations": [],
                "repaired": False,
                "functional_success": None,
            })
            print(f"  [recovered] {task_id} / {mode} from {py_file.name}")
    return recovered


def run_all_tasks(tasks: list[dict], mode: Mode, model: str) -> list[dict]:
    results = []
    for task in tasks:
        print(f"[{model}] [{mode}] {task['id']} ...")
        results.append(run_agent(task, mode, model))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the codevoyager-lite benchmark.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "instruction", "verifier", "all"],
        default="all",
    )
    parser.add_argument("--tasks", default=None, help="Path to tasks JSON file.")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: outputs).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks already in results.json and continue.")
    parser.add_argument("--n-practices", type=int, default=8, choices=[4, 8],
                        help="Number of best practices for instruction mode (4=ablation, 8=full).")
    args = parser.parse_args()

    if args.output_dir:
        utils.set_outputs_dir(args.output_dir)

    task_path = utils.TASKS_PATH if args.tasks is None else args.tasks
    tasks = utils.load_tasks(task_path)
    modes: list[Mode] = (
        ["baseline", "instruction", "verifier"] if args.mode == "all" else [args.mode]
    )

    model_name = os.getenv("LLM_MODEL", 
        "claude-sonnet-4-6" if os.getenv("ANTHROPIC_API_KEY") else "gpt-4o")

    summary_path = utils.OUTPUTS_DIR / "results.json"
    utils.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load or rebuild existing results ---
    all_results: list[dict] = []
    if summary_path.exists():
        all_results = json.loads(summary_path.read_text(encoding="utf-8"))
    elif args.resume:
        print("[resume] results.json not found — rebuilding from saved code files ...")
        all_results = _rebuild_from_code_files(tasks, modes, model_name)
        if all_results:
            _save_results(all_results, summary_path)
            print(f"[resume] recovered {len(all_results)} result(s)\n")

    # --- Build completed set for resume ---
    completed: set[tuple[str, str, str]] = set()
    if args.resume:
        for r in all_results:
            completed.add((r.get("model", ""), r["mode"], r["task_id"]))

    # --- Run tasks, skip completed, save incrementally ---
    n_skipped = 0
    n_ran = 0
    for mode in modes:
        for task in tasks:
            key = (model_name, mode, task["id"])
            if key in completed:
                print(f"[{model_name}] [{mode}] {task['id']} ... SKIPPED (done)")
                n_skipped += 1
                continue

            print(f"[{model_name}] [{mode}] {task['id']} ...")
            result = run_agent(task, mode, model_name, n_practices=args.n_practices)
            n_ran += 1

            all_results = [
                r for r in all_results
                if not (r.get("model") == model_name
                        and r["mode"] == mode
                        and r["task_id"] == task["id"])
            ]
            all_results.append(result)

            _save_results(all_results, summary_path)

    print(f"\nDone. Ran {n_ran} task(s), skipped {n_skipped}.")
    print(f"Results saved to {summary_path}")