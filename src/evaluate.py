from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_DIR = Path("results/plots")
ANNOTATIONS_PATH = Path("results/opus_annotations.json")
TASKS_PATH = Path("tasks/tasks_hard.json")

CONDITIONS = ["baseline", "verifier", "instruction_4p", "instruction_8p"]
CONDITION_LABELS = {
    "baseline": "Baseline",
    "verifier": "Verifier",
    "instruction_4p": "Instr. 4p",
    "instruction_8p": "Instr. 8p",
}
CONDITION_COLORS = {
    "baseline": "#e07b7b",
    "verifier": "#e0c07b",
    "instruction_4p": "#7bb8e0",
    "instruction_8p": "#7bc77b",
}

DIMENSIONS = {
    "Data Separation": [
        "temporal_leakage", "stacking_leakage", "group_leakage",
        "tfidf_leakage", "embedding_leakage", "augmentation_leakage",
        "data_leakage",
    ],
    "Test Isolation": ["test_misuse", "early_stopping_leak", "calibration_leakage"],
    "Evaluation Rigor": [
        "nested_cv_missing", "multiple_testing",
        "survival_metric_misuse", "multilabel_metric_misuse",
        "ranking_metric_misuse",
    ],
    "Code Safety": ["unsafe_deserialization", "credential_exposure", "broad_exception"],
}


def load_annotations() -> dict[str, dict[str, int]]:
    return json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))


def load_tasks() -> list[dict]:
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


def plot_error_rate_bar(ann: dict, tasks: list[dict]) -> Path:
    """Bar chart: error rate per condition (Opus ground truth)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(tasks)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rates = []
    labels = []
    colors = []
    for cond in CONDITIONS:
        errs = sum(ann[cond].values())
        rate = errs / n
        rates.append(rate)
        labels.append(CONDITION_LABELS[cond])
        colors.append(CONDITION_COLORS[cond])

    bars = ax.bar(labels, rates, color=colors, width=0.55, edgecolor="white")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(0, max(rates) + 0.1)
    ax.set_ylabel("Error Rate (Opus 4.7 Ground Truth)")
    ax.set_title("Hard Tasks — Error Rate by Condition")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(y=rates[0], color="#e07b7b", linestyle="--", alpha=0.4, linewidth=1)

    out = PLOTS_DIR / "hard_error_rate.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_dimension(ann: dict, tasks: list[dict]) -> Path:
    """Grouped bar chart: error rate per dimension per condition."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dims = list(DIMENSIONS.keys())
    x = np.arange(len(dims))
    n_conds = len(CONDITIONS)
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cond in enumerate(CONDITIONS):
        rates = []
        for dim_name, vtypes in DIMENSIONS.items():
            dim_tasks = [t for t in tasks if t["violation_type"] in vtypes]
            n = len(dim_tasks)
            errs = sum(ann[cond].get(t["id"], 0) for t in dim_tasks) if n else 0
            rates.append(errs / n if n else 0)
        offset = (i - n_conds / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width=width,
                      label=CONDITION_LABELS[cond],
                      color=CONDITION_COLORS[cond], edgecolor="white")
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{rate:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylabel("Error Rate")
    ax.set_title("Hard Tasks — Error Rate by Dimension (Opus 4.7 GT)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    out = PLOTS_DIR / "hard_error_by_dimension.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_task_heatmap(ann: dict, tasks: list[dict]) -> Path:
    """Heatmap: per-task × per-condition error matrix."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    task_ids = [t["id"] for t in tasks]
    matrix = np.array([
        [ann[cond].get(tid, 0) for cond in CONDITIONS]
        for tid in task_ids
    ])

    fig, ax = plt.subplots(figsize=(6, 12))
    cmap = plt.cm.colors.ListedColormap(["#d4edda", "#f8d7da"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS], fontsize=9)
    ax.set_yticks(range(len(task_ids)))
    ax.set_yticklabels(task_ids, fontsize=7)

    for i in range(len(task_ids)):
        for j in range(len(CONDITIONS)):
            val = matrix[i, j]
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=7, color="#c0392b" if val == 1 else "#27ae60")

    ax.set_title("Per-Task Error Matrix (0=clean, 1=error)", fontsize=11)
    ax.set_xlabel("Condition")

    out = PLOTS_DIR / "hard_task_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_improvement_waterfall(ann: dict, tasks: list[dict]) -> Path:
    """Show how instruction_8p fixes/breaks tasks relative to baseline."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    base_errors = sum(ann["baseline"].values())
    fixed_8p = sum(1 for t in tasks
                   if ann["baseline"].get(t["id"]) == 1
                   and ann["instruction_8p"].get(t["id"]) == 0)
    broken_8p = sum(1 for t in tasks
                    if ann["baseline"].get(t["id"]) == 0
                    and ann["instruction_8p"].get(t["id"]) == 1)
    final_8p = sum(ann["instruction_8p"].values())

    fig, ax = plt.subplots(figsize=(8, 4.5))

    categories = ["Baseline\nErrors", "Fixed by\n8p Instr.", "New Errors\nfrom 8p", "Final\n8p Errors"]
    values = [base_errors, -fixed_8p, broken_8p, final_8p]
    bottoms = [0, base_errors, base_errors - fixed_8p, 0]
    colors_list = ["#e07b7b", "#7bc77b", "#f39c12", "#7bb8e0"]

    bars = ax.bar(categories, [abs(v) for v in values], bottom=bottoms,
                  color=colors_list, width=0.5, edgecolor="white")

    ax.text(0, base_errors + 0.3, str(base_errors), ha="center", fontsize=12, fontweight="bold")
    ax.text(1, base_errors + 0.3, f"-{fixed_8p}", ha="center", fontsize=12,
            fontweight="bold", color="#27ae60")
    ax.text(2, base_errors - fixed_8p + broken_8p + 0.3, f"+{broken_8p}",
            ha="center", fontsize=12, fontweight="bold", color="#e67e22")
    ax.text(3, final_8p + 0.3, str(final_8p), ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Number of Errors")
    ax.set_title("Instruction 8p: Error Reduction Breakdown")
    ax.set_ylim(0, base_errors + 3)
    ax.spines[["top", "right"]].set_visible(False)

    out = PLOTS_DIR / "hard_improvement_waterfall.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_ablation_comparison(ann: dict, tasks: list[dict]) -> Path:
    """Compare 4p vs 8p instruction with per-dimension breakdown."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dims = list(DIMENSIONS.keys())
    x = np.arange(len(dims))
    width = 0.3

    rates_4p = []
    rates_8p = []
    for dim_name, vtypes in DIMENSIONS.items():
        dim_tasks = [t for t in tasks if t["violation_type"] in vtypes]
        n = len(dim_tasks)
        e4 = sum(ann["instruction_4p"].get(t["id"], 0) for t in dim_tasks)
        e8 = sum(ann["instruction_8p"].get(t["id"], 0) for t in dim_tasks)
        rates_4p.append(e4 / n if n else 0)
        rates_8p.append(e8 / n if n else 0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar(x - width / 2, rates_4p, width, label="4 Practices (ablation)",
                   color="#7bb8e0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, rates_8p, width, label="8 Practices (full)",
                   color="#7bc77b", edgecolor="white")

    for bar, val in zip(bars1, rates_4p):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, rates_8p):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylabel("Error Rate")
    ax.set_title("Ablation: 4 vs 8 Best Practices per Dimension")
    ax.set_ylim(0, max(max(rates_4p), max(rates_8p)) + 0.15)
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    out = PLOTS_DIR / "hard_ablation_4v8.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_summary_csv(ann: dict, tasks: list[dict]) -> Path:
    """Save a CSV with per-task annotations across all conditions."""
    rows = []
    for t in tasks:
        row = {"task_id": t["id"], "violation_type": t["violation_type"]}
        for cond in CONDITIONS:
            row[cond] = ann[cond].get(t["id"], -1)
        rows.append(row)
    df = pd.DataFrame(rows)

    summary_row = {"task_id": "TOTAL_ERRORS", "violation_type": ""}
    for cond in CONDITIONS:
        summary_row[cond] = sum(ann[cond].values())
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    out = Path("results") / "hard_opus_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ANNOTATIONS_PATH.exists():
        print(f"Error: {ANNOTATIONS_PATH} not found. Run src.annotate_opus first.")
        raise SystemExit(1)

    ann = load_annotations()
    tasks = load_tasks()

    print("=" * 60)
    print("  HARD TASKS — Opus 4.7 Ground Truth Evaluation")
    print("=" * 60)
    print()
    print(f"{'Condition':<25} {'Errors':<10} {'Rate'}")
    print("-" * 50)
    for cond in CONDITIONS:
        errs = sum(ann[cond].values())
        print(f"{CONDITION_LABELS[cond]:<25} {errs}/30       {errs/30*100:.1f}%")

    plots = []
    plots.append(plot_error_rate_bar(ann, tasks))
    plots.append(plot_per_dimension(ann, tasks))
    plots.append(plot_per_task_heatmap(ann, tasks))
    plots.append(plot_improvement_waterfall(ann, tasks))
    plots.append(plot_ablation_comparison(ann, tasks))

    csv_path = save_summary_csv(ann, tasks)

    print(f"\nPlots saved to {PLOTS_DIR}/:")
    for p in plots:
        print(f"  {p.name}")
    print(f"\nCSV saved to {csv_path}")
