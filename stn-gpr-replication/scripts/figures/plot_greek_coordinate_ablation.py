from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


MODE_LABELS = {
    "m_uniform": "Uniform $m$",
    "price_adaptive": "Price-adaptive",
    "gamma_monitor": r"$\Gamma$-monitor",
    "standardized_risk": "Standardized risk",
}
COLORS = {
    "m_uniform": "tab:blue",
    "price_adaptive": "tab:orange",
    "gamma_monitor": "tab:green",
    "standardized_risk": "tab:red",
}


def _save(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_node_placement(payload, output_dir):
    modes = list(payload["grids"])
    maturity_days = np.unique(payload["test_design"]["maturity_days"])
    selected = [float(maturity_days[0]), float(maturity_days[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), sharey=True)
    for panel, days in zip(axes, selected):
        key = min(
            payload["grids"][modes[0]][
                "implied_moneyness_axes_at_test_maturities"
            ],
            key=lambda value: abs(float(value) - days),
        )
        for row, mode in enumerate(modes):
            nodes = np.asarray(
                payload["grids"][mode][
                    "implied_moneyness_axes_at_test_maturities"
                ][key],
                dtype=float,
            )
            visible = nodes[(nodes >= -0.55) & (nodes <= 0.55)]
            panel.vlines(
                visible,
                row - 0.30,
                row + 0.30,
                color=COLORS[mode],
                linewidth=0.8,
                alpha=0.85,
            )
        panel.axvline(0.0, color="black", linewidth=0.8, alpha=0.45)
        panel.set_xlim(-0.55, 0.55)
        panel.set_xlabel("Log-moneyness $m$")
        panel.set_title(f"T = {float(key):.1f} days")
        panel.grid(axis="x", alpha=0.18)
    axes[0].set_yticks(range(len(modes)), [MODE_LABELS[m] for m in modes])
    axes[0].invert_yaxis()
    fig.suptitle("Equal-node coordinate grids in the risk-relevant window")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v9_node_placement")


def plot_accuracy(payload, output_dir):
    modes = list(payload["grids"])
    components = ["price", "delta", "gamma_diagonal", "cross_gamma"]
    labels = ["Price", r"$\Delta$", r"$\Gamma_{ii}$", r"$\Gamma_{ij}$"]
    x = np.arange(len(components), dtype=float)
    width = 0.82 / len(modes)
    fig, ax = plt.subplots(figsize=(10.2, 5.3))
    for index, mode in enumerate(modes):
        values = [
            100.0
            * payload["grids"][mode]["error_against_analytical"][component][
                "normalized_mae"
            ]
            for component in components
        ]
        ax.bar(
            x + (index - (len(modes) - 1) / 2.0) * width,
            values,
            width,
            label=MODE_LABELS[mode],
            color=COLORS[mode],
        )
    ax.set_yscale("log")
    ax.set_ylabel("Normalized MAE (%)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", which="both", alpha=0.2)
    ax.legend(ncol=2, frameon=False)
    ax.set_title("Grid-only error against analytical fixed-strike Greeks")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v9_accuracy")


def plot_gamma_heatmaps(payload, output_dir):
    modes = list(payload["grids"])
    m_values = np.asarray(payload["test_design"]["log_moneyness"], dtype=float)
    maturity_days = np.asarray(payload["test_design"]["maturity_days"], dtype=float)
    m_levels = np.unique(m_values)
    t_levels = np.unique(maturity_days)
    reference = np.asarray(
        payload["analytical_reference"]["components"]["gamma_diagonal"],
        dtype=float,
    )[:, 0]
    scale = max(float(np.mean(np.abs(reference))), np.finfo(float).tiny)
    matrices = []
    positive = []
    for mode in modes:
        estimate = np.asarray(
            payload["grids"][mode]["components"]["gamma_diagonal"],
            dtype=float,
        )[:, 0]
        matrix = (100.0 * np.abs(estimate - reference) / scale).reshape(
            len(t_levels), len(m_levels)
        )
        matrices.append(matrix)
        positive.extend(matrix[matrix > 0.0].tolist())
    vmin = max(min(positive), 1e-5)
    vmax = max(positive)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex=True, sharey=True)
    image = None
    for ax, mode, matrix in zip(axes.flat, modes, matrices):
        image = ax.imshow(
            np.maximum(matrix, vmin),
            origin="lower",
            aspect="auto",
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        ax.set_title(MODE_LABELS[mode])
        ax.set_xticks(range(len(m_levels)), [f"{value:+.2f}" for value in m_levels])
        ax.set_yticks(range(len(t_levels)), [f"{value:.1f}" for value in t_levels])
    for ax in axes[-1, :]:
        ax.set_xlabel("Log-moneyness $m$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Maturity (days)")
    fig.suptitle("Localization of diagonal-Gamma interpolation error")
    fig.subplots_adjust(top=0.90, right=0.84, wspace=0.18, hspace=0.24)
    color_axis = fig.add_axes((0.87, 0.15, 0.025, 0.70))
    fig.colorbar(
        image,
        cax=color_axis,
        label=r"Absolute $\Gamma_{11}$ error / mean $|\Gamma_{11}|$ (%)",
    )
    _save(fig, output_dir, "greeks_v9_gamma_heatmaps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v9_coordinate_ablation"),
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-coordinate-ablation-v9":
        parser.error("input is not a v9 coordinate-ablation JSON")
    plot_node_placement(payload, args.output_dir)
    plot_accuracy(payload, args.output_dir)
    plot_gamma_heatmaps(payload, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
