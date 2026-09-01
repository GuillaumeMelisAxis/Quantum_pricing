from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DISPLAY_NAMES = {
    "multilinear": "Multilinear",
    "risk_coordinate_cubic": "Risk-axis cubic",
    "moneyness_cubic": "Risk-axis cubic",
    "component_hybrid": "Component hybrid",
    "unified_risk_cubic": "Unified risk cubic",
}


def _save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _variant(payload, requested):
    if requested in payload["variants"]:
        return requested, payload["variants"][requested]
    matches = [
        (label, value)
        for label, value in payload["variants"].items()
        if label.startswith("truncated_")
    ]
    if requested == "compressed" and matches:
        return matches[0]
    raise KeyError(f"variant {requested!r} is unavailable")


def _percent(record, component):
    return (
        100.0 * record["global_error_against_analytical"][component]["normalized_mae"]
    )


def _hessian_percent(record):
    return (
        100.0
        * record["full_hessian_error_against_analytical"][
            "aggregate_relative_frobenius_error"
        ]
    )


def plot_summary(payload, output_dir, variant_label):
    resolved_label, variant = _variant(payload, variant_label)
    modes = list(variant["modes"])
    positions = np.arange(len(modes), dtype=float)
    display = [DISPLAY_NAMES.get(mode, mode) for mode in modes]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))

    global_axis = axes[0, 0]
    global_metrics = (
        ("delta", r"$\Delta$"),
        ("gamma_diagonal", r"$\Gamma_{ii}$"),
        ("cross_gamma", r"$\Gamma_{ij}$"),
    )
    width = 0.2
    for index, (component, label) in enumerate(global_metrics):
        values = [_percent(variant["modes"][mode], component) for mode in modes]
        global_axis.bar(
            positions + (index - 1.5) * width,
            values,
            width,
            label=label,
            color=f"C{index}",
        )
    global_axis.bar(
        positions + 1.5 * width,
        [_hessian_percent(variant["modes"][mode]) for mode in modes],
        width,
        label="Full Hessian",
        color="C3",
    )
    global_axis.set_yscale("log")
    global_axis.set_ylabel("Normalized error (%)")
    global_axis.set_title("Global Greek accuracy")
    global_axis.legend(frameon=False, fontsize=8)

    local_axis = axes[0, 1]
    local_metrics = (
        ("gamma_curve_normalized_mae", r"$\Gamma_{ii}$ curve MAE"),
        ("cross_gamma_curve_normalized_mae", r"$\Gamma_{ij}$ curve MAE"),
        ("gamma_point_normalized_max_error", r"$\Gamma_{ii}$ point max"),
        ("cross_gamma_point_normalized_max_error", r"$\Gamma_{ij}$ point max"),
    )
    for index, (metric, label) in enumerate(local_metrics):
        values = [
            100.0 * variant["modes"][mode]["local_error_audit"][metric]
            for mode in modes
        ]
        local_axis.plot(
            positions,
            values,
            marker="o",
            color=f"C{index}",
            label=label,
        )
    local_axis.axhline(2.5, color="0.4", linestyle=":", linewidth=1.1)
    local_axis.axhline(10.0, color="0.4", linestyle="--", linewidth=1.1)
    local_axis.set_yscale("log")
    local_axis.set_ylabel("Local normalized error (%)")
    local_axis.set_title("Worst maturity–moneyness slice")
    local_axis.legend(frameon=False, fontsize=8)

    cost_axis = axes[1, 0]
    cost_axis.bar(
        positions,
        [variant["modes"][mode]["wall_time_seconds"] for mode in modes],
        color=[f"C{index}" for index in range(len(modes))],
    )
    cost_axis.set_yscale("log")
    cost_axis.set_ylabel("Greek evaluation wall time (s)")
    cost_axis.set_title("Evaluation cost after TT fitting")

    consistency_axis = axes[1, 1]
    component_modes = [mode for mode in modes if mode != "unified_risk_cubic"]
    component_positions = np.arange(len(component_modes), dtype=float)
    for index, (component, label) in enumerate(
        (
            ("price", "Price"),
            ("delta", r"$\Delta$"),
            ("gamma_diagonal", r"$\Gamma_{ii}$"),
            ("cross_gamma", r"$\Gamma_{ij}$"),
        )
    ):
        values = [
            100.0
            * variant["modes"][mode]["difference_from_unified_risk_cubic"][
                "components"
            ][component]["normalized_mae"]
            for mode in component_modes
        ]
        consistency_axis.plot(
            component_positions,
            values,
            marker="o",
            color=f"C{index}",
            label=label,
        )
    consistency_axis.set_yscale("log")
    consistency_axis.set_ylabel("Difference from unified surface (%)")
    consistency_axis.set_title("Operator-consistency audit")
    consistency_axis.set_xticks(
        component_positions,
        [DISPLAY_NAMES.get(mode, mode) for mode in component_modes],
        rotation=16,
        ha="right",
    )
    consistency_axis.legend(frameon=False, fontsize=8)

    for axis in (global_axis, local_axis, cost_axis):
        axis.set_xticks(positions, display, rotation=16, ha="right")
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    fig.suptitle(f"v9.6 price–Greek interpolant audit — {resolved_label}")
    fig.tight_layout()
    _save(fig, output_dir, f"greeks_v96_interpolant_audit_{resolved_label}")


def plot_grid_oracle(payload, output_dir):
    records = payload.get("grid_oracle", {})
    if not records:
        return
    modes = list(records)
    positions = np.arange(len(modes), dtype=float)
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    for index, (component, label) in enumerate(
        (
            ("price", "Price"),
            ("delta", r"$\Delta$"),
            ("gamma_diagonal", r"$\Gamma_{ii}$"),
            ("cross_gamma", r"$\Gamma_{ij}$"),
        )
    ):
        axis.plot(
            positions,
            [_percent(records[mode], component) for mode in modes],
            marker="o",
            color=f"C{index}",
            label=label,
        )
    axis.plot(
        positions,
        [_hessian_percent(records[mode]) for mode in modes],
        marker="D",
        color="C4",
        label="Full Hessian",
    )
    axis.set_yscale("log")
    axis.set_xticks(
        positions,
        [DISPLAY_NAMES.get(mode, mode) for mode in modes],
        rotation=16,
        ha="right",
    )
    axis.set_ylabel("Normalized error against analytical Greeks (%)")
    axis.set_title("Exact-grid interpolation error floor")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v96_grid_oracle_error_floor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v96_interpolant_consistency"),
    )
    parser.add_argument(
        "--variant",
        default="compressed",
        help="Variant key, or 'compressed' for the stored truncated variant.",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-interpolant-consistency-v9.6":
        raise ValueError("input is not a v9.6 interpolant-consistency result")
    plot_summary(payload, args.output_dir, args.variant)
    plot_grid_oracle(payload, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
