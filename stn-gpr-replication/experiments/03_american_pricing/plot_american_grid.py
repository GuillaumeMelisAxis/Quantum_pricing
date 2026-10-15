"""American arithmetic-basket put: uniform versus adaptive maturity (paper
figure ``results-american-grid``).

Reads the JSON written by ``validate_american_grid_robustness.py`` and plots
price MAE and effective rank (mean +/- std. across paired TT seeds) against the
nominal TT-cross budget.

    .venv/Scripts/python.exe experiments/03_american_pricing/plot_american_grid.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UNIFORM_MODE = "moneyness_adaptive_uniform_maturity"
ADAPTIVE_MODE = "moneyness_adaptive"


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


def plot_american(source: Path, output_dir: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    grouped = {
        (row["mode"], row["budget"]): row
        for row in data["summary"]["by_mode_and_budget"]
    }
    budget_values = sorted({budget for _, budget in grouped})
    budgets = np.asarray(budget_values, dtype=float) / 1000

    def values(mode: str, metric: str, moment: str) -> list[float]:
        return [grouped[(mode, budget)][metric][moment] for budget in budget_values]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    colors = {"uniform": "#355C7D", "adaptive": "#C44E52"}

    for axis, metric, ylabel in (
        (axes[0], "mae", "Price MAE (mean +/- std.)"),
        (axes[1], "effective_rank", "Effective QTT rank (mean +/- std.)"),
    ):
        axis.errorbar(
            budgets,
            values(UNIFORM_MODE, metric, "mean"),
            yerr=values(UNIFORM_MODE, metric, "std"),
            fmt="o-",
            capsize=3,
            color=colors["uniform"],
            linewidth=1.7,
            label="Uniform maturity",
        )
        axis.errorbar(
            budgets,
            values(ADAPTIVE_MODE, metric, "mean"),
            yerr=values(ADAPTIVE_MODE, metric, "std"),
            fmt="o-",
            capsize=3,
            color=colors["adaptive"],
            linewidth=1.7,
            label="Adaptive maturity",
        )
        axis.set_xlabel("Nominal TT-cross budget (thousands)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(budgets)
        axis.legend(frameon=False)

    fig.tight_layout(w_pad=2.2)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_dir / f"results-american-grid.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/03_american_pricing/american_grid_robustness.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    _style()
    plot_american(args.input, args.output_dir)


if __name__ == "__main__":
    main()
