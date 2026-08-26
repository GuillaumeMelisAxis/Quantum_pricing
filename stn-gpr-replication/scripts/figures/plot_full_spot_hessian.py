from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _filename_number(value):
    return f"{float(value):.10g}".replace("-", "minus_").replace(".", "p")


def _matrices(components):
    diagonal = np.asarray(components["gamma_diagonal"], dtype=float)
    cross = np.asarray(components["cross_gamma"], dtype=float)
    n_points, n_assets = diagonal.shape
    matrices = np.zeros((n_points, n_assets, n_assets), dtype=float)
    for column in range(n_assets):
        matrices[:, column, column] = diagonal[:, column]
    for pair_column, (left, right) in enumerate(
        itertools.combinations(range(n_assets), 2)
    ):
        matrices[:, left, right] = cross[:, pair_column]
        matrices[:, right, left] = cross[:, pair_column]
    return matrices


def _curve_indices(payload, panel, maturity_days):
    design = payload["test_design"]
    labels = np.asarray(design["spot_panel_labels"])
    maturities = np.asarray(design["maturity_days"], dtype=float)
    moneyness = np.asarray(design["log_moneyness"], dtype=float)
    indices = np.flatnonzero(
        (labels == panel) & np.isclose(maturities, float(maturity_days))
    )
    if indices.size == 0:
        raise ValueError(f"curve {panel}, T={maturity_days:g}d was not stored")
    order = np.argsort(moneyness[indices])
    return indices[order], moneyness[indices][order]


def plot_component_errors(payload, output_dir):
    diagnostics = payload["risk_hybrid_cubic"][
        "componentwise_error_against_analytical"
    ]
    delta_records = diagnostics["delta"]
    diagonal_records = diagnostics["gamma_diagonal"]
    cross_records = diagnostics["cross_gamma"]
    n_assets = len(delta_records)
    delta_error = np.asarray([
        100.0 * record["normalized_mae"] for record in delta_records
    ])
    hessian_error = np.zeros((n_assets, n_assets), dtype=float)
    for column, record in enumerate(diagonal_records):
        hessian_error[column, column] = 100.0 * record["normalized_mae"]
    for record, (left, right) in zip(
        cross_records,
        itertools.combinations(range(n_assets), 2),
        strict=True,
    ):
        hessian_error[left, right] = 100.0 * record["normalized_mae"]
        hessian_error[right, left] = hessian_error[left, right]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].bar(np.arange(n_assets), delta_error, color="C0")
    axes[0].set_xticks(np.arange(n_assets), [rf"$\Delta_{i + 1}$" for i in range(n_assets)])
    axes[0].set_ylabel("Normalized MAE (%)")
    axes[0].set_title("Delta component accuracy")
    axes[0].grid(axis="y", alpha=0.25)

    image = axes[1].imshow(hessian_error, cmap="viridis", aspect="equal")
    axes[1].set_xticks(np.arange(n_assets), [rf"$S_{i + 1}$" for i in range(n_assets)])
    axes[1].set_yticks(np.arange(n_assets), [rf"$S_{i + 1}$" for i in range(n_assets)])
    axes[1].set_title("Hessian component accuracy")
    for row in range(n_assets):
        for column in range(n_assets):
            color = "white" if hessian_error[row, column] > 0.55 * np.max(hessian_error) else "black"
            axes[1].text(
                column,
                row,
                f"{hessian_error[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    fig.colorbar(image, ax=axes[1], label="Normalized MAE (%)")
    fig.suptitle("v9.3 full five-asset Greek validation")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v93_component_accuracy")


def plot_atm_matrices(payload, output_dir, panel, maturity_days):
    indices, moneyness = _curve_indices(payload, panel, maturity_days)
    local_index = int(np.argmin(np.abs(moneyness)))
    point_index = int(indices[local_index])
    analytical = _matrices(payload["analytical_reference"]["components"])[
        point_index
    ]
    hybrid = _matrices(payload["risk_hybrid_cubic"]["components"])[point_index]
    error = hybrid - analytical
    value_limit = max(float(np.max(np.abs(analytical))), float(np.max(np.abs(hybrid))))
    error_limit = max(float(np.max(np.abs(error))), np.finfo(float).eps)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    records = (
        ("Analytical Hessian", analytical, value_limit),
        ("Risk-hybrid Hessian", hybrid, value_limit),
        ("Signed error", error, error_limit),
    )
    for axis, (title, matrix, limit) in zip(axes, records, strict=True):
        image = axis.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            aspect="equal",
        )
        axis.set_xticks(range(matrix.shape[0]), [rf"$S_{i + 1}$" for i in range(matrix.shape[0])])
        axis.set_yticks(range(matrix.shape[0]), [rf"$S_{i + 1}$" for i in range(matrix.shape[0])])
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.1e}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        rf"ATM spot Hessian — {panel.replace('_', ' ')}, "
        rf"$T={float(maturity_days):g}$d"
    )
    fig.tight_layout()
    stem = (
        f"greeks_v93_atm_hessian_{panel}_T_"
        f"{_filename_number(maturity_days)}d"
    )
    _save(fig, output_dir, stem)


def plot_minimum_eigenvalues(payload, output_dir, panel):
    maturities = payload["settings"]["maturity_days"]
    analytical = _matrices(payload["analytical_reference"]["components"])
    hybrid = _matrices(payload["risk_hybrid_cubic"]["components"])
    n_columns = 2
    n_rows = int(np.ceil(len(maturities) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(11.0, 3.5 * n_rows),
        squeeze=False,
        sharex=True,
    )
    for axis, maturity in zip(axes.flat, maturities, strict=False):
        indices, moneyness = _curve_indices(payload, panel, maturity)
        reference_minimum = np.linalg.eigvalsh(analytical[indices])[:, 0]
        hybrid_minimum = np.linalg.eigvalsh(hybrid[indices])[:, 0]
        axis.plot(moneyness, reference_minimum, color="black", linewidth=2.1, label="Analytical")
        axis.plot(moneyness, hybrid_minimum, color="C1", linewidth=1.5, label="Risk-hybrid cubic")
        axis.axhline(0.0, color="0.55", linewidth=0.8)
        axis.axvline(0.0, color="0.8", linewidth=0.8)
        axis.set_yscale("symlog", linthresh=1e-10)
        axis.set_title(rf"$T={float(maturity):g}$d")
        axis.set_xlabel("Log-moneyness $m$")
        axis.set_ylabel(r"Minimum eigenvalue $\lambda_{\min}$")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(maturities) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        f"Full-Hessian PSD audit — {panel.replace('_', ' ')}",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    _save(fig, output_dir, f"greeks_v93_psd_audit_{panel}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v93_full_hessian"),
    )
    parser.add_argument("--panel", default=None)
    parser.add_argument("--maturity-days", type=float, default=30.0)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-full-spot-hessian-v9.3":
        raise ValueError("the input is not a v9.3 full-Hessian result")
    panel = args.panel or payload["settings"]["panels"][0]
    if panel not in payload["settings"]["panels"]:
        raise ValueError(f"panel {panel!r} was not stored")
    if not any(
        np.isclose(args.maturity_days, maturity)
        for maturity in payload["settings"]["maturity_days"]
    ):
        raise ValueError("the requested maturity was not stored")

    plot_component_errors(payload, args.output_dir)
    plot_atm_matrices(
        payload,
        args.output_dir,
        panel,
        args.maturity_days,
    )
    plot_minimum_eigenvalues(payload, args.output_dir, panel)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
