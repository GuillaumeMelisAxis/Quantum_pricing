from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COMPONENTS = (
    ("price", "Price"),
    ("delta", r"$\Delta$"),
    ("gamma_diagonal", r"$\Gamma_{ii}$"),
    ("cross_gamma", r"$\Gamma_{ij}$"),
)


def _save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _runs(payload):
    return list(payload["runs"].values())


def _run(payload, budget, seed=None):
    candidates = [
        record
        for record in _runs(payload)
        if int(record["budget"]) == int(budget)
        and (seed is None or int(record["tt_seed"]) == int(seed))
    ]
    if not candidates:
        raise ValueError(f"no stored run for budget={budget}, seed={seed}")
    return min(candidates, key=lambda record: int(record["tt_seed"]))


def _budget_runs(payload):
    seed = int(payload["settings"]["tt_seeds"][0])
    records = [
        record
        for record in _runs(payload)
        if int(record["tt_seed"]) == seed
    ]
    return sorted(records, key=lambda record: int(record["budget"]))


def plot_budget_convergence(payload, output_dir):
    records = _budget_runs(payload)
    if not records:
        return
    budgets = np.asarray([record["budget"] for record in records], dtype=float)
    grid = payload["grid_oracle"]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))
    for color_index, (name, label) in enumerate(COMPONENTS):
        total = 100.0 * np.asarray([
            record["global_error_against_analytical"][name]["normalized_mae"]
            for record in records
        ])
        grid_floor = 100.0 * grid["global_error_against_analytical"][name][
            "normalized_mae"
        ]
        axes[0, 0].plot(
            budgets,
            total,
            marker="o",
            color=f"C{color_index}",
            label=label,
        )
        axes[0, 0].axhline(
            grid_floor,
            color=f"C{color_index}",
            linestyle=":",
            linewidth=1.1,
            alpha=0.75,
        )
    axes[0, 0].set_ylabel("Normalized MAE (%)")
    axes[0, 0].set_title("Total error and grid floor")
    axes[0, 0].legend(ncol=2, frameon=False)

    hessian = 100.0 * np.asarray([
        record["full_hessian_error_against_analytical"][
            "aggregate_relative_frobenius_error"
        ]
        for record in records
    ])
    negative = 100.0 * np.asarray([
        record["full_hessian_error_against_analytical"][
            "negative_part_relative_frobenius_norm"
        ]
        for record in records
    ])
    axes[0, 1].plot(budgets, hessian, marker="o", label="Hessian error")
    axes[0, 1].plot(
        budgets,
        negative,
        marker="s",
        label="Negative Hessian part",
    )
    axes[0, 1].axhline(
        100.0
        * grid["full_hessian_error_against_analytical"][
            "aggregate_relative_frobenius_error"
        ],
        color="C0",
        linestyle=":",
        linewidth=1.1,
    )
    axes[0, 1].set_ylabel("Relative norm (%)")
    axes[0, 1].set_title("Full-Hessian diagnostics")
    axes[0, 1].legend(frameon=False)

    rank = [record["fit"]["effective_rank"] for record in records]
    evaluations = [record["fit"]["function_evaluations"] for record in records]
    axes[1, 0].plot(budgets, rank, color="C0", marker="o")
    axes[1, 0].set_ylabel("Effective QTT rank", color="C0")
    axes[1, 0].tick_params(axis="y", labelcolor="C0")
    cost_axis = axes[1, 0].twinx()
    cost_axis.plot(budgets, evaluations, color="C1", marker="s")
    cost_axis.set_ylabel("Oracle evaluations", color="C1")
    cost_axis.tick_params(axis="y", labelcolor="C1")
    axes[1, 0].set_title("Compression cost")

    selected_budget = int(payload["settings"]["retained_budget"])
    available = [int(record["budget"]) for record in records]
    if selected_budget not in available:
        selected_budget = max(available)
    selected = _run(
        payload,
        selected_budget,
        payload["settings"]["tt_seeds"][0],
    )
    layers = (
        ("finite_difference", "Finite difference"),
        ("grid_interpolation", "Grid interpolation"),
        ("tt_reconstruction", "TT reconstruction"),
        ("total", "Total"),
    )
    width = 0.19
    positions = np.arange(len(COMPONENTS), dtype=float)
    for layer_index, (layer, label) in enumerate(layers):
        values = [
            100.0
            * selected["deterministic_error_decomposition"][layer][name][
                "normalized_mae"
            ]
            for name, _ in COMPONENTS
        ]
        axes[1, 1].bar(
            positions + (layer_index - 1.5) * width,
            values,
            width=width,
            label=label,
        )
    axes[1, 1].set_xticks(positions, [label for _, label in COMPONENTS])
    axes[1, 1].set_ylabel("Layer magnitude: normalized MAE (%)")
    axes[1, 1].set_title(f"Error decomposition at budget {selected_budget:,}")
    axes[1, 1].legend(fontsize=8, frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        if axis is not axes[1, 1]:
            axis.set_xlabel("Oracle budget")
    fig.suptitle("v9.4 QTT Greek budget convergence")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v94_budget_convergence")


def plot_seed_robustness(payload, output_dir):
    retained = int(payload["settings"]["retained_budget"])
    records = sorted(
        [record for record in _runs(payload) if int(record["budget"]) == retained],
        key=lambda record: int(record["tt_seed"]),
    )
    if len(records) < 2:
        return
    seeds = [str(record["tt_seed"]) for record in records]
    positions = np.arange(len(records), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for name, label in COMPONENTS:
        values = [
            100.0
            * record["global_error_against_analytical"][name]["normalized_mae"]
            for record in records
        ]
        axes[0].plot(positions, values, marker="o", label=label)
    axes[0].set_xticks(positions, seeds, rotation=25)
    axes[0].set_ylabel("Normalized MAE (%)")
    axes[0].set_title(f"Accuracy at budget {retained:,}")
    axes[0].legend(ncol=2, frameon=False)

    width = 0.36
    ranks = [record["fit"]["effective_rank"] for record in records]
    times = [record["fit"]["total_time_seconds"] for record in records]
    axes[1].bar(positions - width / 2, ranks, width=width, label="Effective rank")
    time_axis = axes[1].twinx()
    time_axis.bar(
        positions + width / 2,
        times,
        width=width,
        color="C1",
        label="Fit time",
    )
    axes[1].set_xticks(positions, seeds, rotation=25)
    axes[1].set_ylabel("Effective QTT rank")
    time_axis.set_ylabel("Fit time (s)")
    axes[1].set_title("Seed sensitivity of cost")
    handles_left, labels_left = axes[1].get_legend_handles_labels()
    handles_right, labels_right = time_axis.get_legend_handles_labels()
    axes[1].legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("v9.4 QTT Greek seed robustness")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v94_seed_robustness")


def _curve_indices(payload, panel, maturity_days):
    design = payload["test_design"]
    labels = np.asarray(design["spot_panel_labels"])
    maturities = np.asarray(design["maturity_days"], dtype=float)
    moneyness = np.asarray(design["log_moneyness"], dtype=float)
    indices = np.flatnonzero(
        (labels == panel) & np.isclose(maturities, float(maturity_days))
    )
    if not indices.size:
        raise ValueError(f"curve {panel}, T={maturity_days:g}d was not stored")
    order = np.argsort(moneyness[indices])
    return indices[order], moneyness[indices][order]


def plot_retained_profiles(payload, output_dir, panel, maturity_days):
    retained = int(payload["settings"]["retained_budget"])
    selected = _run(payload, retained)
    indices, moneyness = _curve_indices(payload, panel, maturity_days)
    sources = (
        ("Analytical", payload["analytical_reference"]["components"], "black", "-"),
        ("Grid floor", payload["grid_oracle"]["components"], "C0", "--"),
        ("QTT + interpolation", selected["components"], "C1", "-"),
    )
    components = (
        ("delta", 0, r"$\Delta_1$"),
        ("gamma_diagonal", 0, r"$\Gamma_{1,1}$"),
        ("cross_gamma", 0, r"$\Gamma_{1,2}$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharex=True)
    for axis, (component, column, title) in zip(axes, components, strict=True):
        for label, values, color, linestyle in sources:
            data = np.asarray(values[component], dtype=float)[indices, column]
            axis.plot(
                moneyness,
                data,
                color=color,
                linestyle=linestyle,
                marker="o" if label != "Analytical" else None,
                linewidth=2.0 if label == "Analytical" else 1.5,
                label=label,
            )
        axis.axvline(0.0, color="0.75", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Log-moneyness $m$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Greek value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"Retained QTT audit — {panel.replace('_', ' ')}, "
        rf"$T={float(maturity_days):g}$d, budget {retained:,}",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    _save(fig, output_dir, "greeks_v94_retained_profiles")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v94_tt_convergence"),
    )
    parser.add_argument("--panel", default=None)
    parser.add_argument("--maturity-days", type=float, default=30.0)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-tt-convergence-v9.4":
        raise ValueError("the input is not a v9.4 TT-convergence result")
    if not payload["runs"]:
        raise ValueError("the input does not contain any completed TT run")
    stored_panels = list(dict.fromkeys(
        payload["test_design"]["spot_panel_labels"]
    ))
    panel = args.panel or (
        "oos_mid" if "oos_mid" in stored_panels else stored_panels[0]
    )
    plot_budget_convergence(payload, args.output_dir)
    plot_seed_robustness(payload, args.output_dir)
    plot_retained_profiles(
        payload,
        args.output_dir,
        panel,
        args.maturity_days,
    )
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
