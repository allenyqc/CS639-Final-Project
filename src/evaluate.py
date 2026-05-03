from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUTS_DIR = Path("outputs")
RESULTS_PATH = OUTPUTS_DIR / "results.json"
PLOTS_DIR = Path("results/plots")


def load_results() -> list[dict]:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def compute_metrics(results: list[dict]) -> pd.DataFrame:
    """Per-mode summary: violation rate and repair rate."""
    modes = ["baseline", "instruction", "verifier"]
    rows = []
    for mode in modes:
        subset = [r for r in results if r["mode"] == mode]
        n = len(subset)
        violated = sum(1 for r in subset if r.get("violations"))
        repaired = sum(1 for r in subset if r.get("repaired"))
        rows.append({
            "mode": mode,
            "n_tasks": n,
            "n_violated": violated,
            "violation_rate": round(violated / n, 3) if n else 0,
            "n_repaired": repaired,
            "repair_rate": round(repaired / n, 3) if n else 0,
        })
    return pd.DataFrame(rows)


def plot_violation_rate(df: pd.DataFrame) -> Path:
    """Bar chart: violation rate per mode."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e07b7b", "#e0c07b", "#7bb8e0"]
    bars = ax.bar(df["mode"], df["violation_rate"], color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, df["violation_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Violation Rate")
    ax.set_title("Violation Rate by Agent Mode")
    ax.spines[["top", "right"]].set_visible(False)
    out = PLOTS_DIR / "violation_rate.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_category(results: list[dict]) -> Path:
    """Grouped bar chart: violation rate per category per mode."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(Path("tasks/tasks.json").read_text(encoding="utf-8"))
    task_vtype = {t["id"]: t["violation_type"] for t in tasks}

    counts: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for r in results:
        cat = task_vtype.get(r["task_id"], "unknown")
        mode = r["mode"]
        totals[(mode, cat)] += 1
        if r.get("violations"):
            counts[(mode, cat)] += 1

    categories = sorted({t["violation_type"] for t in tasks})
    modes = ["baseline", "instruction", "verifier"]
    x = range(len(categories))
    width = 0.25
    colors = ["#e07b7b", "#e0c07b", "#7bb8e0"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (mode, color) in enumerate(zip(modes, colors)):
        rates = [
            counts[(mode, cat)] / totals[(mode, cat)] if totals[(mode, cat)] else 0
            for cat in categories
        ]
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], rates, width=width, label=mode,
               color=color, edgecolor="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylabel("Violation Rate")
    ax.set_title("Violation Rate by Category and Mode")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    out = PLOTS_DIR / "violation_by_category.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_csv(df: pd.DataFrame) -> Path:
    out = Path("results/results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    results = load_results()
    df = compute_metrics(results)

    print("\n=== Summary ===")
    print(df.to_string(index=False))

    csv_path = save_csv(df)
    print(f"\nCSV saved to {csv_path}")

    p1 = plot_violation_rate(df)
    p2 = plot_per_category(results)
    print(f"Plots saved to:\n  {p1}\n  {p2}")
