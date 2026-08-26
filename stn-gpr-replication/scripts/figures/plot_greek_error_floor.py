from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

COMPONENTS = ("price", "delta", "gamma_diagonal", "cross_gamma")
GREEK_COMPONENTS = ("delta", "gamma_diagonal", "cross_gamma")
LABELS = {
    "price": "Price",
    "delta": r"$\Delta$",
    "gamma_diagonal": r"$\Gamma_{ii}$",
    "cross_gamma": r"$\Gamma_{ij}$",
}
METHOD_LABELS = {
    "multilinear": "Multilinear",
    "risk_hybrid_cubic": "Risk-hybrid cubic",
}


def _bump_key(value):
    return f"{float(value):.10g}"


def _save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_bump_convergence(payload, output_dir):
    bumps = np.asarray(payload["settings"]["relative_bumps"], dtype=float)
    methods = payload["settings"]["interpolation_modes"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), sharex=True)
    for ax, component in zip(axes.flat, COMPONENTS):
        fd_errors = np.asarray([
            payload["exact_price_finite_difference_controls"][_bump_key(bump)][
                "error_against_analytical"
            ][component]["normalized_mae"]
            for bump in bumps
        ])
        if np.any(fd_errors > 0.0):
            ax.plot(
                100.0 * bumps,
                100.0 * np.maximum(fd_errors, 1.0e-15),
                color="0.35",
                marker="o",
                linestyle=":",
                label="Exact-price FD",
            )
        for position, mode in enumerate(methods):
            values = np.asarray([
                payload["interpolation_results"][mode]["bump_results"][
                    _bump_key(bump)
                ]["decomposition"]["total"]["metrics"][component][
                    "normalized_mae"
                ]
                for bump in bumps
            ])
            ax.plot(
                100.0 * bumps,
                100.0 * np.maximum(values, 1.0e-15),
                color=f"C{position}",
                marker="s",
                label=METHOD_LABELS.get(mode, mode),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(LABELS[component])
        ax.set_ylabel("Normalized MAE (%)")
        ax.grid(which="both", alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Relative spot bump (%)")
        ax.set_xticks(100.0 * bumps, [f"{100.0 * value:g}" for value in bumps])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.suptitle("Greek accuracy as a function of the finite-difference bump", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    _save(fig, output_dir, "greeks_v92_bump_convergence")


def plot_error_decomposition(payload, output_dir):
    bumps = np.asarray(payload["settings"]["relative_bumps"], dtype=float)
    methods = payload["settings"]["interpolation_modes"]
    layer_styles = {
        "finite_difference": ("C0", "o", "Finite-difference layer"),
        "grid_interpolation": ("C1", "s", "Grid-interpolation layer"),
        "total": ("C2", "^", "Total error"),
    }
    fig, axes = plt.subplots(
        len(GREEK_COMPONENTS),
        len(methods),
        figsize=(5.1 * len(methods), 10.0),
        sharex=True,
        squeeze=False,
    )
    for column, mode in enumerate(methods):
        for row, component in enumerate(GREEK_COMPONENTS):
            ax = axes[row, column]
            for layer, (color, marker, label) in layer_styles.items():
                values = np.asarray([
                    payload["interpolation_results"][mode]["bump_results"][
                        _bump_key(bump)
                    ]["decomposition"][layer]["metrics"][component][
                        "normalized_mae"
                    ]
                    for bump in bumps
                ])
                ax.plot(
                    100.0 * bumps,
                    100.0 * np.maximum(values, 1.0e-15),
                    color=color,
                    marker=marker,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(which="both", alpha=0.25)
            ax.set_title(
                f"{METHOD_LABELS.get(mode, mode)} — {LABELS[component]}"
            )
            if column == 0:
                ax.set_ylabel("Normalized residual MAE (%)")
            if row == len(GREEK_COMPONENTS) - 1:
                ax.set_xlabel("Relative spot bump (%)")
                ax.set_xticks(
                    100.0 * bumps,
                    [f"{100.0 * value:g}" for value in bumps],
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Additive decomposition of the Greek error floor", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.963),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    _save(fig, output_dir, "greeks_v92_error_decomposition")


def _selected_indices(payload, maturity_days, panel):
    design = payload["test_design"]
    m_values = np.asarray(design["log_moneyness"], dtype=float)
    t_values = np.asarray(design["maturity_days"], dtype=float)
    labels = np.asarray(design["spot_panel_labels"])
    available = np.unique(t_values)
    selected_maturity = float(available[np.argmin(np.abs(available - maturity_days))])
    if panel is None:
        panel = str(np.unique(labels)[0])
    if panel not in set(labels):
        raise ValueError(f"unknown panel {panel!r}")
    mask = (labels == panel) & np.isclose(t_values, selected_maturity)
    indices = np.flatnonzero(mask)
    indices = indices[np.argsort(m_values[indices])]
    return indices, m_values, selected_maturity, panel


def plot_profiles(payload, output_dir, maturity_days, panel, risk_column):
    indices, m_values, selected_maturity, selected_panel = _selected_indices(
        payload,
        maturity_days,
        panel,
    )
    columns = [int(value) for value in payload["settings"]["risk_columns"]]
    if risk_column not in columns:
        raise ValueError(f"risk column {risk_column} was not stored")
    risk_position = columns.index(risk_column)
    bump = float(payload["settings"]["baseline_relative_bump"])
    key = _bump_key(bump)
    analytical = payload["analytical_reference"]["components"]
    exact_fd = payload["exact_price_finite_difference_controls"][key]["components"]
    methods = payload["settings"]["interpolation_modes"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharex=True)
    for ax, component in zip(axes, ("delta", "gamma_diagonal")):
        truth = np.asarray(analytical[component], dtype=float)[indices, risk_position]
        exact = np.asarray(exact_fd[component], dtype=float)[indices, risk_position]
        ax.plot(
            m_values[indices],
            truth,
            color="black",
            linewidth=2.5,
            marker="o",
            label="Analytical",
            zorder=5,
        )
        ax.plot(
            m_values[indices],
            exact,
            color="0.45",
            linewidth=1.7,
            linestyle=":",
            marker="d",
            label="Exact-price FD",
        )
        for position, mode in enumerate(methods):
            values = np.asarray(
                payload["interpolation_results"][mode]["bump_results"][key][
                    "components"
                ][component],
                dtype=float,
            )[indices, risk_position]
            ax.plot(
                m_values[indices],
                values,
                color=f"C{position}",
                marker="s",
                linewidth=1.5,
                label=METHOD_LABELS.get(mode, mode),
            )
        ax.axvline(0.0, color="0.8", linewidth=0.9)
        ax.set_title(LABELS[component])
        ax.set_xlabel("Log-moneyness $m$")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        rf"Fixed-strike profiles, $T={selected_maturity:.1f}$d, "
        rf"$S_{{{risk_column + 1}}}$, {selected_panel.replace('_', ' ')}",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    _save(fig, output_dir, "greeks_v92_delta_gamma_profiles")


def _heatmap(payload, mode, component, bump):
    design = payload["test_design"]
    m_values = np.asarray(design["log_moneyness"], dtype=float)
    t_values = np.asarray(design["maturity_days"], dtype=float)
    truth = np.asarray(
        payload["analytical_reference"]["components"][component],
        dtype=float,
    )
    estimate = np.asarray(
        payload["interpolation_results"][mode]["bump_results"][
            _bump_key(bump)
        ]["components"][component],
        dtype=float,
    )
    if truth.ndim == 1:
        truth = truth[:, None]
        estimate = estimate[:, None]
    scale = max(float(np.mean(np.abs(truth))), 1.0e-15)
    point_error = np.mean(np.abs(estimate - truth), axis=1) / scale
    unique_m = np.unique(m_values)
    unique_t = np.unique(t_values)
    matrix = np.empty((len(unique_t), len(unique_m)), dtype=float)
    for row, maturity in enumerate(unique_t):
        for column, moneyness in enumerate(unique_m):
            mask = np.isclose(t_values, maturity) & np.isclose(m_values, moneyness)
            matrix[row, column] = float(np.mean(point_error[mask]))
    return unique_m, unique_t, np.maximum(100.0 * matrix, 1.0e-12)


def plot_heatmaps(payload, output_dir):
    methods = payload["settings"]["interpolation_modes"]
    bump = float(payload["settings"]["baseline_relative_bump"])
    arrays = {
        (mode, component): _heatmap(payload, mode, component, bump)
        for mode in methods
        for component in GREEK_COMPONENTS
    }
    positive_values = np.concatenate([
        values[2].reshape(-1) for values in arrays.values()
    ])
    norm = LogNorm(
        vmin=max(float(np.min(positive_values)), 1.0e-4),
        vmax=max(float(np.max(positive_values)), 1.0e-3),
    )
    fig, axes = plt.subplots(
        len(methods),
        len(GREEK_COMPONENTS),
        figsize=(12.2, 3.8 * len(methods)),
        squeeze=False,
    )
    image = None
    for row, mode in enumerate(methods):
        for column, component in enumerate(GREEK_COMPONENTS):
            m_values, t_values, matrix = arrays[(mode, component)]
            ax = axes[row, column]
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                cmap="magma",
                norm=norm,
            )
            ax.set_xticks(range(len(m_values)), [f"{value:+g}" for value in m_values])
            ax.set_yticks(range(len(t_values)), [f"{value:.1f}" for value in t_values])
            ax.set_title(LABELS[component])
            ax.set_xlabel("Log-moneyness $m$")
            if column == 0:
                ax.set_ylabel(
                    f"{METHOD_LABELS.get(mode, mode)}\nMaturity (days)"
                )
    colorbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.018,
        pad=0.035,
    )
    colorbar.set_label("Pointwise normalized absolute error (%)")
    fig.suptitle(
        rf"Greek error localization at a {100.0 * bump:g}% spot bump",
        y=0.995,
    )
    fig.subplots_adjust(top=0.92, right=0.88, hspace=0.35, wspace=0.25)
    _save(fig, output_dir, "greeks_v92_error_heatmaps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v92_error_floor"),
    )
    parser.add_argument("--maturity-days", type=float, default=30.0)
    parser.add_argument("--panel", default=None)
    parser.add_argument("--risk-column", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-error-floor-decomposition-v9.2":
        raise ValueError("the input is not a v9.2 error-floor result")
    plot_bump_convergence(payload, args.output_dir)
    plot_error_decomposition(payload, args.output_dir)
    plot_profiles(
        payload,
        args.output_dir,
        args.maturity_days,
        args.panel,
        args.risk_column,
    )
    plot_heatmaps(payload, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
