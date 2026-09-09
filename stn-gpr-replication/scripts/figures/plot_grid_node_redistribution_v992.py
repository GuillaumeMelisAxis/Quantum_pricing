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
GRID_COLORS = {
    "pricing_grid": "C0",
    "risk_hybrid_grid": "C1",
}


def _save(fig, output_directory: Path, stem: str) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_directory / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _maturity_summary(payload, extractor):
    maturities = sorted(
        {row["theta"]["maturity_days"] for row in payload["scenarios"]}
    )
    median, q05, q95 = [], [], []
    for maturity in maturities:
        values = np.asarray(
            [
                extractor(row)
                for row in payload["scenarios"]
                if row["theta"]["maturity_days"] == maturity
            ],
            dtype=float,
        )
        median.append(np.median(values))
        q05.append(np.quantile(values, 0.05))
        q95.append(np.quantile(values, 0.95))
    return (
        np.asarray(maturities),
        np.asarray(median),
        np.asarray(q05),
        np.asarray(q95),
    )


def plot_node_displacement(payload, output_directory):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    definitions = (
        (
            "D2",
            r"$D_2(\boldsymbol{\theta})$",
            "Normalized root-mean-square displacement",
        ),
        (
            "D_infinity",
            r"$D_\infty(\boldsymbol{\theta})$",
            "Normalized maximum displacement",
        ),
    )
    for axis, (key, ylabel, title) in zip(axes, definitions):
        maturity, median, q05, q95 = _maturity_summary(
            payload,
            lambda row, metric=key: row["node_displacement"][metric],
        )
        axis.plot(maturity, median, color="C0", marker="o", linewidth=1.8)
        axis.fill_between(maturity, q05, q95, color="C0", alpha=0.18)
        axis.set_xscale("log")
        axis.set_xlabel("Maturity (days)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.3)
    fig.suptitle("Physical node displacement between matched grids")
    fig.tight_layout()
    _save(fig, output_directory, "greeks_v992_node_displacement")


def plot_coverage_gain(payload, output_directory):
    fig, axis = plt.subplots(figsize=(7.2, 4.7))
    for kappa, linestyle in zip((1, 2, 3), ("-", "--", ":")):
        key = f"kappa_{kappa}"
        maturity, median, q05, q95 = _maturity_summary(
            payload,
            lambda row, band=key: row["bands"][band]["coverage_gain"],
        )
        line = axis.plot(
            maturity,
            median,
            marker="o",
            linestyle=linestyle,
            linewidth=1.8,
            label=rf"$\kappa={kappa}$",
        )[0]
        axis.fill_between(
            maturity,
            q05,
            q95,
            color=line.get_color(),
            alpha=0.12,
        )
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_xlabel("Maturity (days)")
    axis.set_ylabel(
        r"$G_\kappa^{\mathrm{fill}}=h_\kappa^{(p)}/h_\kappa^{(r)}$"
    )
    axis.set_title("Risk-band coverage gain")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    _save(fig, output_directory, "greeks_v992_coverage_gain")


def plot_node_concentration(payload, output_directory):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)
    for axis, kappa in zip(axes, (1, 2, 3)):
        key = f"kappa_{kappa}"
        for grid in GRID_LABELS:
            maturity, median, q05, q95 = _maturity_summary(
                payload,
                lambda row, band=key, name=grid: row["bands"][band][name][
                    "P_nodes"
                ],
            )
            axis.plot(
                maturity,
                median,
                color=GRID_COLORS[grid],
                marker="o",
                linewidth=1.8,
                label=GRID_LABELS[grid],
            )
            axis.fill_between(
                maturity,
                q05,
                q95,
                color=GRID_COLORS[grid],
                alpha=0.12,
            )
        axis.set_xscale("log")
        axis.set_xlabel("Maturity (days)")
        axis.set_title(rf"$\kappa={kappa}$")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel(r"$P_\kappa^{(g)}(\boldsymbol{\theta})$")
    axes[-1].legend(loc="best")
    fig.suptitle("Proportion of nodes inside the risk band")
    fig.tight_layout()
    _save(fig, output_directory, "greeks_v992_node_concentration")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v992_grid_geometry"),
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-grid-geometry-v9.9.2":
        raise ValueError("input is not a v9.9.2 grid-geometry result")
    plot_node_displacement(payload, args.output_dir)
    plot_coverage_gain(payload, args.output_dir)
    plot_node_concentration(payload, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
