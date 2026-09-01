from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GRID_LABELS = {
    "pricing_grid": "Pricing grid",
    "risk_hybrid_grid": "Risk-hybrid grid",
}


def _save(fig, output_directory, stem):
    output_directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_directory / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _metric(record, name):
    global_error = record["global_error_against_analytical"]
    if name == "price":
        return global_error["price"]["normalized_mae"]
    if name == "delta":
        return global_error["delta"]["normalized_mae"]
    if name == "gamma_diagonal":
        return global_error["gamma_diagonal"]["normalized_mae"]
    if name == "cross_gamma":
        return global_error["cross_gamma"]["normalized_mae"]
    return record["full_hessian_error_against_analytical"][
        "aggregate_relative_frobenius_error"
    ]


def plot_accuracy(payload, output_directory):
    grids = list(GRID_LABELS)
    metrics = (
        ("price", "Price"),
        ("delta", r"$\Delta$"),
        ("gamma_diagonal", r"$\Gamma_{ii}$"),
        ("cross_gamma", r"$\Gamma_{ij}$"),
        ("hessian", "Hessian"),
    )
    positions = np.arange(len(metrics), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))

    controlled = payload["controlled_coordinate_ablation"]["grids"]
    for offset, grid_name in zip((-0.5, 0.5), grids, strict=True):
        if grid_name not in controlled:
            continue
        record = controlled[grid_name]["record"]
        axes[0].bar(
            positions + offset * width,
            [100.0 * _metric(record, name) for name, _ in metrics],
            width,
            label=GRID_LABELS[grid_name],
        )

    production = payload["production_comparison"]["summary"]["grids"]
    summary_names = {
        "price": "price_normalized_mae",
        "delta": "delta_normalized_mae",
        "gamma_diagonal": "gamma_diagonal_normalized_mae",
        "cross_gamma": "cross_gamma_normalized_mae",
        "hessian": "hessian_relative_frobenius_error",
    }
    for offset, grid_name in zip((-0.5, 0.5), grids, strict=True):
        if grid_name not in production or not production[grid_name]["metrics"]:
            continue
        summaries = production[grid_name]["metrics"]
        means = [
            100.0 * summaries[summary_names[name]]["mean"] for name, _ in metrics
        ]
        deviations = [
            100.0 * summaries[summary_names[name]]["standard_deviation"]
            for name, _ in metrics
        ]
        axes[1].bar(
            positions + offset * width,
            means,
            width,
            yerr=deviations,
            capsize=3,
            label=GRID_LABELS[grid_name],
        )

    for axis, title in zip(
        axes,
        ("Matched-resolution exact-grid floor", "Native raw-TT architectures"),
        strict=True,
    ):
        axis.set_yscale("log")
        axis.set_xticks(positions, [label for _, label in metrics])
        axis.set_ylabel("Normalized error (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Pricing-grid versus risk-hybrid accuracy")
    fig.tight_layout()
    _save(fig, output_directory, "greeks_v98_accuracy")


def plot_complexity(payload, output_directory):
    grids = list(GRID_LABELS)
    summary = payload["production_comparison"]["summary"]["grids"]
    positions = np.arange(len(grids), dtype=float)
    metrics = (
        ("effective_rank", "Effective QTT rank"),
        ("parameter_count", "TT parameters"),
        ("fit_time_seconds", "Fit time (s)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        means = [summary[name]["metrics"][metric]["mean"] for name in grids]
        deviations = [
            summary[name]["metrics"][metric]["standard_deviation"] for name in grids
        ]
        axis.bar(
            positions,
            means,
            yerr=deviations,
            capsize=4,
            color=["C0", "C1"],
        )
        axis.set_xticks(
            positions,
            [GRID_LABELS[name] for name in grids],
            rotation=18,
            ha="right",
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Native-architecture TT cost across seeds")
    fig.tight_layout()
    _save(fig, output_directory, "greeks_v98_complexity")


def plot_seed_gate(payload, output_directory):
    grids = list(GRID_LABELS)
    seeds = payload["settings"]["tt_seeds"]
    production = payload["production_comparison"]["grids"]
    accepted = np.asarray(
        [
            [
                float(
                    production[grid_name]["tt_runs"][str(seed)]["record"][
                        "greek_aware_local_acceptance"
                    ]["passed"]
                )
                for grid_name in grids
            ]
            for seed in seeds
        ]
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    image = axis.imshow(accepted, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for row, seed in enumerate(seeds):
        for column, grid_name in enumerate(grids):
            record = production[grid_name]["tt_runs"][str(seed)]["record"]
            error = 100.0 * record["full_hessian_error_against_analytical"][
                "aggregate_relative_frobenius_error"
            ]
            axis.text(column, row, f"{error:.2f}%", ha="center", va="center")
    axis.set_xticks(np.arange(len(grids)), [GRID_LABELS[name] for name in grids])
    axis.set_yticks(np.arange(len(seeds)), [str(seed) for seed in seeds])
    axis.set_xlabel("Native grid")
    axis.set_ylabel("TT-cross seed")
    axis.set_title("Greek-aware gate; cells show Hessian error")
    fig.colorbar(image, ax=axis, ticks=(0, 1), label="Fail / pass")
    fig.tight_layout()
    _save(fig, output_directory, "greeks_v98_seed_gate")


def plot_dense_profiles(payload, output_directory):
    profiles = payload["dense_profiles"]["grids"]
    if any(name not in profiles for name in GRID_LABELS):
        return
    first = profiles["pricing_grid"]
    moneyness = np.asarray(first["log_moneyness"], dtype=float)
    reference = first["analytical_components"]
    definitions = (
        ("delta", 0, r"$\Delta_1$"),
        ("gamma_diagonal", 0, r"$\Gamma_{1,1}$"),
        ("cross_gamma", 0, r"$\Gamma_{1,2}$"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.4), sharex="col")
    for column, (component, index, title) in enumerate(definitions):
        truth = np.asarray(reference[component], dtype=float)
        if truth.ndim > 1:
            truth = truth[:, index]
        axes[0, column].plot(moneyness, truth, color="black", label="Analytical")
        for grid_name, grid_label in GRID_LABELS.items():
            values = np.asarray(
                profiles[grid_name]["record"]["components"][component],
                dtype=float,
            )
            if values.ndim > 1:
                values = values[:, index]
            axes[0, column].plot(
                moneyness,
                values,
                label=grid_label,
            )
            axes[1, column].plot(
                moneyness,
                np.maximum(np.abs(values - truth), 1e-16),
                label=grid_label,
            )
        axes[0, column].set_title(title)
        axes[0, column].grid(alpha=0.23)
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("Log-moneyness $m$")
        axes[1, column].set_ylabel("Absolute error")
        axes[1, column].grid(alpha=0.23)
    axes[0, 0].set_ylabel("Greek value")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Dense fixed-strike Greek profiles, "
        f"$T={payload['dense_profiles']['maturity_days']:.0f}$d"
    )
    fig.tight_layout()
    _save(fig, output_directory, "greeks_v98_dense_profiles")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v98_pricing_vs_risk_grid"),
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-pricing-risk-grid-v9.8":
        raise ValueError("input is not a v9.8 pricing/risk-grid comparison")
    plot_accuracy(payload, args.output_dir)
    plot_complexity(payload, args.output_dir)
    plot_seed_gate(payload, args.output_dir)
    plot_dense_profiles(payload, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
