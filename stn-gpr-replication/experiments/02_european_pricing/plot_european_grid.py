"""European geometric-basket put: original versus adaptive grid (paper figure
``results-european-grid``).

Reads the two JSON files written by ``reproduce_european.py`` (grid modes
``paper`` and ``moneyness_adaptive``) and plots effective rank and out-of-sample
price MAE against the nominal TT-cross budget.

    .venv/Scripts/python.exe experiments/02_european_pricing/plot_european_grid.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results/02_european_pricing")


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


def _tt_runs(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["tt"]


def plot_european(original: Path, adaptive: Path, output_dir: Path) -> None:
    baseline = _tt_runs(original)
    adaptive_runs = _tt_runs(adaptive)
    budgets = np.asarray([run["budget"] for run in baseline], dtype=float) / 1000

    base_rank = [run["effective_rank"] for run in baseline]
    adapt_rank = [run["effective_rank"] for run in adaptive_runs]
    base_mae = [run["metrics"]["global"]["mae"] for run in baseline]
    adapt_mae = [run["metrics"]["global"]["mae"] for run in adaptive_runs]

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
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_dir / f"results-european-grid.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original", type=Path, default=RESULTS / "european_paper.json"
    )
    parser.add_argument(
        "--adaptive", type=Path, default=RESULTS / "european_moneyness_adaptive.json"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    _style()
    plot_european(args.original, args.adaptive, args.output_dir)


if __name__ == "__main__":
    main()
