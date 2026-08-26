from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("price", "delta", "gamma_diagonal", "cross_gamma")
LABELS = ("Price", r"$\Delta$", r"$\Gamma_{ii}$", r"$\Gamma_{ij}$")


def normalized_errors(payload):
    return np.asarray([
        100.0 * payload[name]["normalized_mae"]
        for name in COMPONENTS
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v82_american_decomposition"),
    )
    parser.add_argument("--budget", type=int, default=None)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not data.get("tt"):
        raise RuntimeError("the input JSON does not contain a completed TT stage")
    budgets = sorted(int(value) for value in data["tt"])
    selected = max(budgets) if args.budget is None else args.budget
    if str(selected) not in data["tt"]:
        raise ValueError(f"budget {selected} is absent from the input file")

    run = data["tt"][str(selected)]
    layers = {
        "LSMC labels": normalized_errors(
            data["training_labels"]["error_against_independent_reference"]
        ),
        "Grid interpolation": normalized_errors(
            data["grid_only"]["error_against_direct_training_labels"]
        ),
        "TT reconstruction": normalized_errors(run["error_against_grid_only"]),
        "Total TT error": normalized_errors(
            run["error_against_independent_reference"]
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    colors = ("#555555", "#8c6d46", "#e8751a", "#2f2f2f")

    ax = axes[0, 0]
    x = np.arange(len(COMPONENTS))
    width = 0.19
    for offset, ((label, values), color) in enumerate(zip(layers.items(), colors)):
        ax.bar(
            x + (offset - 1.5) * width,
            np.maximum(values, 1e-5),
            width,
            label=label,
            color=color,
        )
    ax.set_yscale("log")
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Normalized MAE (%)")
    ax.set_title(f"Error layers at budget {selected:,}")
    ax.grid(axis="y", alpha=0.22, which="both")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for component, label, color in zip(COMPONENTS, LABELS, colors):
        total = [
            100.0
            * data["tt"][str(budget)]["error_against_independent_reference"]
            [component]["normalized_mae"]
            for budget in budgets
        ]
        reconstruction = [
            100.0
            * data["tt"][str(budget)]["error_against_grid_only"]
            [component]["normalized_mae"]
            for budget in budgets
        ]
        ax.plot(budgets, total, marker="o", color=color, label=f"{label} total")
        ax.plot(
            budgets,
            reconstruction,
            marker="x",
            linestyle="--",
            color=color,
            alpha=0.75,
            label=f"{label} TT-grid",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("TT-cross budget")
    ax.set_ylabel("Normalized MAE (%)")
    ax.set_title("TT convergence")
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    ranks = [data["tt"][str(b)]["fit"]["effective_rank"] for b in budgets]
    evaluations = [
        data["tt"][str(b)]["fit"]["function_evaluations"] for b in budgets
    ]
    parameters = [data["tt"][str(b)]["fit"]["parameter_count"] for b in budgets]
    ax.plot(budgets, ranks, marker="o", color="#555555", label="Effective rank")
    ax.set_xscale("log")
    ax.set_xlabel("TT-cross budget")
    ax.set_ylabel("Effective QTT rank")
    ax.grid(alpha=0.22)
    twin = ax.twinx()
    twin.plot(
        budgets,
        evaluations,
        marker="s",
        color="#e8751a",
        label="Oracle evaluations",
    )
    twin.plot(
        budgets,
        parameters,
        marker="^",
        linestyle="--",
        color="#8c6d46",
        label="TT parameters",
    )
    twin.set_ylabel("Count")
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)
    ax.set_title("Compression cost")

    ax = axes[1, 1]
    names = ("Reference", "LSMC labels", "Grid", "TT")
    minimum_eigenvalues = (
        data["independent_reference"]["hessian_shape"]["minimum_eigenvalue"],
        data["training_labels"]["hessian_shape"]["minimum_eigenvalue"],
        data["grid_only"]["hessian_shape"]["minimum_eigenvalue"],
        run["hessian_shape"]["minimum_eigenvalue"],
    )
    bars = ax.bar(names, minimum_eigenvalues, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylabel("Minimum Hessian eigenvalue")
    ax.set_title("Convexity diagnostic")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, minimum_eigenvalues):
        ax.annotate(
            f"{value:.1e}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4 if value >= 0 else -12),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    fig.suptitle(
        "American arithmetic-basket Greek error decomposition",
        fontsize=15,
    )
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "american_greek_error_decomposition"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
