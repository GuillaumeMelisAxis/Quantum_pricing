"""Generate the result figures used in the manuscript.

The script reads the archived experiment outputs rather than duplicating the
reported values.  European runs are stored in concatenated console logs,
whereas the American robustness experiment is a standalone JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
UPLOAD = ROOT / "upload"
FIGURES = ROOT / "stn-gpr-replication" / "figures"


def _json_objects(path: Path) -> list[dict]:
    """Decode every top-level JSON object embedded in a console log."""

    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        index = end
    return objects


def _european_runs() -> tuple[list[dict], list[dict]]:
    baseline_candidates = _json_objects(UPLOAD / "Texte collé(2).txt")
    adaptive_candidates = _json_objects(UPLOAD / "Texte collé(3).txt")

    baseline = next(
        obj
        for obj in baseline_candidates
        if len(obj.get("tt", [])) == 4
        and "coordinate_grid" not in obj.get("assumptions", {})
    )
    adaptive = next(
        obj
        for obj in adaptive_candidates
        if len(obj.get("tt", [])) == 4
        and obj.get("assumptions", {})
        .get("coordinate_grid", {})
        .get("mode")
        == "moneyness_adaptive"
    )
    return baseline["tt"], adaptive["tt"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 180,
        }
    )


def plot_european() -> None:
    baseline, adaptive = _european_runs()
    budgets = np.asarray([run["budget"] for run in baseline], dtype=float) / 1000

    base_rank = [run["effective_rank"] for run in baseline]
    adapt_rank = [run["effective_rank"] for run in adaptive]
    base_mae = [run["metrics"]["global"]["mae"] for run in baseline]
    adapt_mae = [run["metrics"]["global"]["mae"] for run in adaptive]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    colors = {"baseline": "#355C7D", "adaptive": "#C44E52"}

    axes[0].plot(
        budgets,
        base_rank,
        "o-",
        color=colors["baseline"],
        linewidth=1.7,
        label="Original grid",
    )
    axes[0].plot(
        budgets,
        adapt_rank,
        "o-",
        color=colors["adaptive"],
        linewidth=1.7,
        label="Adaptive grid",
    )
    axes[0].set_xlabel("Nominal TT-cross budget (thousands)")
    axes[0].set_ylabel("Effective QTT rank")
    axes[0].set_xticks(budgets)
    axes[0].legend(frameon=False)

    axes[1].plot(
        budgets,
        base_mae,
        "o-",
        color=colors["baseline"],
        linewidth=1.7,
        label="Original grid",
    )
    axes[1].plot(
        budgets,
        adapt_mae,
        "o-",
        color=colors["adaptive"],
        linewidth=1.7,
        label="Adaptive grid",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Nominal TT-cross budget (thousands)")
    axes[1].set_ylabel("Out-of-sample price MAE")
    axes[1].set_xticks(budgets)
    axes[1].legend(frameon=False)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(FIGURES / "results-european-grid.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "results-european-grid.png", bbox_inches="tight")
    plt.close(fig)


def plot_american() -> None:
    data = json.loads(
        (UPLOAD / "american_grid_robustness.json").read_text(encoding="utf-8")
    )
    grouped = {
        (row["mode"], row["budget"]): row
        for row in data["summary"]["by_mode_and_budget"]
    }
    budgets = np.asarray([7500, 9000, 12000], dtype=float) / 1000
    uniform_mode = "moneyness_adaptive_uniform_maturity"
    adaptive_mode = "moneyness_adaptive"

    def values(mode: str, metric: str, moment: str) -> list[float]:
        return [
            grouped[(mode, int(1000 * budget))][metric][moment]
            for budget in budgets
        ]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    colors = {"uniform": "#355C7D", "adaptive": "#C44E52"}

    axes[0].errorbar(
        budgets,
        values(uniform_mode, "mae", "mean"),
        yerr=values(uniform_mode, "mae", "std"),
        fmt="o-",
        capsize=3,
        color=colors["uniform"],
        linewidth=1.7,
        label="Uniform maturity",
    )
    axes[0].errorbar(
        budgets,
        values(adaptive_mode, "mae", "mean"),
        yerr=values(adaptive_mode, "mae", "std"),
        fmt="o-",
        capsize=3,
        color=colors["adaptive"],
        linewidth=1.7,
        label="Adaptive maturity",
    )
    axes[0].set_xlabel("Nominal TT-cross budget (thousands)")
    axes[0].set_ylabel("Price MAE (mean +/- std.)")
    axes[0].set_xticks(budgets)
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        budgets,
        values(uniform_mode, "effective_rank", "mean"),
        yerr=values(uniform_mode, "effective_rank", "std"),
        fmt="o-",
        capsize=3,
        color=colors["uniform"],
        linewidth=1.7,
        label="Uniform maturity",
    )
    axes[1].errorbar(
        budgets,
        values(adaptive_mode, "effective_rank", "mean"),
        yerr=values(adaptive_mode, "effective_rank", "std"),
        fmt="o-",
        capsize=3,
        color=colors["adaptive"],
        linewidth=1.7,
        label="Adaptive maturity",
    )
    axes[1].set_xlabel("Nominal TT-cross budget (thousands)")
    axes[1].set_ylabel("Effective QTT rank (mean +/- std.)")
    axes[1].set_xticks(budgets)
    axes[1].legend(frameon=False)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(FIGURES / "results-american-grid.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "results-american-grid.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_european()
    plot_american()
