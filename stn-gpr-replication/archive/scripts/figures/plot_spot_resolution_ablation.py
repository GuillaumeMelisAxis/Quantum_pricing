from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = ("price", "delta", "gamma_diagonal", "cross_gamma")
LABELS = {
    "price": "Price",
    "delta": r"$\Delta$",
    "gamma_diagonal": r"$\Gamma_{ii}$",
    "cross_gamma": r"$\Gamma_{ij}$",
}


def _selected_profile(payload, maturity_days, risk_column):
    design = payload["test_design"]
    m_values = np.asarray(design["log_moneyness"], dtype=float)
    point_maturities = np.asarray(design["maturity_days"], dtype=float)
    if point_maturities.size != m_values.size:
        market_parameters = np.asarray(design["market_parameters"], dtype=float)
        point_maturities = 365.0 * market_parameters[:, -1]
    panel_labels = np.asarray(design["spot_panel_labels"])
    risk_columns = [int(column) for column in payload["settings"]["risk_columns"]]
    if risk_column not in risk_columns:
        choices = ", ".join(str(column) for column in risk_columns)
        raise ValueError(f"risk column {risk_column} is unavailable; choose from {choices}")
    risk_position = risk_columns.index(risk_column)
    available_maturities = np.unique(point_maturities)
    selected_maturity = float(
        available_maturities[
            np.argmin(np.abs(available_maturities - float(maturity_days)))
        ]
    )
    panels = list(design["spot_panels"])
    profiles = []
    for panel in panels:
        mask = (panel_labels == panel) & np.isclose(
            point_maturities,
            selected_maturity,
        )
        order = np.argsort(m_values[mask])
        profiles.append((panel, np.flatnonzero(mask)[order]))
    return profiles, risk_position, selected_maturity


def _profile_label(panel):
    return panel.replace("_", " ").title()


def _save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_global_convergence(payload, output_dir):
    nodes = np.asarray(payload["settings"]["spot_nodes"], dtype=int)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), sharex=True)
    for ax, component in zip(axes.flat, COMPONENTS):
        errors = np.asarray([
            100.0
            * payload["spot_grids"][str(int(n))]["error_against_analytical"][
                component
            ]["normalized_mae"]
            for n in nodes
        ])
        ax.plot(nodes, errors, marker="o", linewidth=2.0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(LABELS[component])
        ax.set_ylabel("Normalized MAE (%)")
        ax.grid(which="both", alpha=0.25)
        for x, y in zip(nodes, errors):
            ax.annotate(
                f"{y:.3g}%",
                (x, y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    for ax in axes[-1, :]:
        ax.set_xlabel("Nodes per spot axis")
        ax.set_xticks(nodes, [str(int(value)) for value in nodes])
    fig.suptitle("Standardized-risk Greek convergence with spot resolution")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v91_spot_resolution_convergence")


def plot_gamma_regions(payload, output_dir):
    nodes = np.asarray(payload["settings"]["spot_nodes"], dtype=int)
    regions = (
        ("global", "Global"),
        ("risk_core_abs_m_le_0p15", r"$|m|\leq0.15$"),
        ("short_risk_core", r"$T\leq30$d, $|m|\leq0.15$"),
        ("put_deep_itm_m_ge_0p35", r"Deep ITM, $m\geq0.35$"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)
    for ax, component in zip(axes, ("gamma_diagonal", "cross_gamma")):
        for region, label in regions:
            if region != "global" and not all(
                region
                in payload["spot_grids"][str(int(n))][
                    "regional_error_against_analytical"
                ]
                for n in nodes
            ):
                continue
            values = []
            for n in nodes:
                grid = payload["spot_grids"][str(int(n))]
                metrics = (
                    grid["error_against_analytical"]
                    if region == "global"
                    else grid["regional_error_against_analytical"][region]
                )
                values.append(100.0 * metrics[component]["normalized_mae"])
            ax.plot(nodes, values, marker="o", label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(nodes, [str(int(value)) for value in nodes])
        ax.set_xlabel("Nodes per spot axis")
        ax.set_ylabel("Normalized MAE (%)")
        ax.set_title(LABELS[component])
        ax.grid(which="both", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Where the remaining Hessian error is removed")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v91_gamma_regions")


def plot_cost(payload, output_dir):
    nodes = np.asarray(payload["settings"]["spot_nodes"], dtype=int)
    cores = np.asarray([
        payload["spot_grids"][str(int(n))]["qtt_core_count"] for n in nodes
    ])
    evaluations = np.asarray([
        payload["spot_grids"][str(int(n))]["cache"][
            "unique_function_evaluations"
        ]
        for n in nodes
    ])
    seconds = np.asarray([
        payload["spot_grids"][str(int(n))]["wall_time_seconds"] for n in nodes
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    axes[0].plot(nodes, cores, marker="o")
    axes[0].set_ylabel("QTT core count")
    axes[0].set_title("Architecture size")
    axes[1].plot(nodes, evaluations, marker="o", label="Oracle evaluations")
    secondary = axes[1].twinx()
    secondary.plot(
        nodes,
        seconds,
        marker="s",
        linestyle="--",
        color="tab:orange",
        label="Wall time",
    )
    axes[1].set_ylabel("Unique oracle evaluations")
    secondary.set_ylabel("Wall time (s)")
    axes[1].set_title("Grid-only diagnostic cost")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(nodes, [str(int(value)) for value in nodes])
        ax.set_xlabel("Nodes per spot axis")
        ax.grid(alpha=0.25)
    lines = axes[1].get_lines() + secondary.get_lines()
    axes[1].legend(lines, [line.get_label() for line in lines], frameon=False)
    fig.suptitle("Cost of spot-axis refinement")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v91_spot_resolution_cost")


def plot_delta_gamma_profiles(payload, output_dir, maturity_days, risk_column):
    profiles, risk_position, selected_maturity = _selected_profile(
        payload,
        maturity_days,
        risk_column,
    )
    nodes = np.asarray(payload["settings"]["spot_nodes"], dtype=int)
    m_values = np.asarray(payload["test_design"]["log_moneyness"], dtype=float)
    reference = payload["analytical_reference"]["components"]
    components = ("delta", "gamma_diagonal")
    fig, axes = plt.subplots(
        2,
        len(profiles),
        figsize=(4.25 * len(profiles), 7.2),
        sharex="col",
        squeeze=False,
    )
    for column, (panel, indices) in enumerate(profiles):
        for row, component in enumerate(components):
            ax = axes[row, column]
            truth = np.asarray(reference[component], dtype=float)[
                indices, risk_position
            ]
            ax.plot(
                m_values[indices],
                truth,
                color="black",
                marker="o",
                linewidth=2.4,
                markersize=4.5,
                label="Analytical reference",
                zorder=5,
            )
            for method_position, node_count in enumerate(nodes):
                estimate = np.asarray(
                    payload["spot_grids"][str(int(node_count))]["components"][
                        component
                    ],
                    dtype=float,
                )[indices, risk_position]
                ax.plot(
                    m_values[indices],
                    estimate,
                    color=f"C{method_position}",
                    marker="s",
                    linewidth=1.35,
                    markersize=3.5,
                    alpha=0.9,
                    label=rf"Grid $N_S={int(node_count)}$",
                )
            ax.axvline(0.0, color="0.75", linewidth=0.9, zorder=0)
            ax.grid(alpha=0.25)
            ax.set_title(_profile_label(panel))
            if column == 0:
                ax.set_ylabel(LABELS[component])
            if row == 1:
                ax.set_xlabel("Log-moneyness $m$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=min(5, len(labels)),
        frameon=False,
    )
    fig.suptitle(
        rf"Spot Greek profiles at $T={selected_maturity:.1f}$ days "
        rf"(risk factor $S_{{{risk_column + 1}}}$)",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    _save(fig, output_dir, "greeks_v91_delta_gamma_profiles")


def plot_delta_gamma_absolute_errors(
    payload,
    output_dir,
    maturity_days,
    risk_column,
):
    profiles, risk_position, selected_maturity = _selected_profile(
        payload,
        maturity_days,
        risk_column,
    )
    nodes = np.asarray(payload["settings"]["spot_nodes"], dtype=int)
    m_values = np.asarray(payload["test_design"]["log_moneyness"], dtype=float)
    reference = payload["analytical_reference"]["components"]
    finite_difference = payload["exact_price_finite_difference_control"][
        "components"
    ]
    components = ("delta", "gamma_diagonal")
    fig, axes = plt.subplots(
        2,
        len(profiles),
        figsize=(4.25 * len(profiles), 7.2),
        sharex="col",
        squeeze=False,
    )
    for column, (panel, indices) in enumerate(profiles):
        for row, component in enumerate(components):
            ax = axes[row, column]
            truth = np.asarray(reference[component], dtype=float)[
                indices, risk_position
            ]
            fd_value = np.asarray(finite_difference[component], dtype=float)[
                indices, risk_position
            ]
            ax.semilogy(
                m_values[indices],
                np.maximum(np.abs(fd_value - truth), 1.0e-12),
                color="0.35",
                marker="o",
                linestyle=":",
                linewidth=1.8,
                markersize=4.0,
                label="Exact-price FD",
            )
            for method_position, node_count in enumerate(nodes):
                estimate = np.asarray(
                    payload["spot_grids"][str(int(node_count))]["components"][
                        component
                    ],
                    dtype=float,
                )[indices, risk_position]
                ax.semilogy(
                    m_values[indices],
                    np.maximum(np.abs(estimate - truth), 1.0e-12),
                    color=f"C{method_position}",
                    marker="s",
                    linewidth=1.35,
                    markersize=3.5,
                    label=rf"Grid $N_S={int(node_count)}$",
                )
            ax.axvline(0.0, color="0.75", linewidth=0.9, zorder=0)
            ax.set_ylim(bottom=1.0e-12)
            ax.grid(which="both", alpha=0.25)
            ax.set_title(_profile_label(panel))
            if column == 0:
                ax.set_ylabel(rf"Absolute error in {LABELS[component]}")
            if row == 1:
                ax.set_xlabel("Log-moneyness $m$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=min(5, len(labels)),
        frameon=False,
    )
    fig.suptitle(
        rf"Pointwise spot-Greek errors at $T={selected_maturity:.1f}$ days "
        rf"(risk factor $S_{{{risk_column + 1}}}$)",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    _save(fig, output_dir, "greeks_v91_delta_gamma_absolute_errors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v91_spot_resolution"),
    )
    parser.add_argument(
        "--maturity-days",
        type=float,
        default=30.0,
        help="target maturity for the pointwise profiles (nearest panel maturity)",
    )
    parser.add_argument(
        "--risk-column",
        type=int,
        default=0,
        help="zero-based spot column used for Delta and diagonal Gamma",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-standardized-spot-ablation-v9.1":
        parser.error("input is not a v9.1 spot-resolution JSON")
    plot_global_convergence(payload, args.output_dir)
    plot_gamma_regions(payload, args.output_dir)
    plot_cost(payload, args.output_dir)
    try:
        plot_delta_gamma_profiles(
            payload,
            args.output_dir,
            args.maturity_days,
            args.risk_column,
        )
        plot_delta_gamma_absolute_errors(
            payload,
            args.output_dir,
            args.maturity_days,
            args.risk_column,
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
