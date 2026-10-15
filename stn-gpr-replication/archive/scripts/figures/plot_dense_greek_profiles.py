from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COMPONENTS = ("delta", "gamma_diagonal", "cross_gamma")
LABELS = {
    "delta": r"$\Delta$",
    "gamma_diagonal": r"$\Gamma_{ii}$",
    "cross_gamma": r"$\Gamma_{ij}$",
}
METHOD_LABELS = {
    "multilinear": "Multilinear",
    "risk_hybrid_cubic": "Risk-hybrid cubic",
}
METHOD_COLORS = {
    "multilinear": "C0",
    "risk_hybrid_cubic": "C1",
}


def _bump_key(value):
    return f"{float(value):.10g}"


def _filename_number(value):
    return f"{float(value):.10g}".replace("-", "minus_").replace(".", "p")


def _save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _risk_positions(payload, risk_column, cross_column):
    columns = [int(value) for value in payload["settings"]["risk_columns"]]
    if risk_column not in columns:
        raise ValueError(f"risk column {risk_column} was not stored")
    if cross_column is None:
        cross_column = next((value for value in columns if value != risk_column), None)
    if cross_column not in columns or cross_column == risk_column:
        raise ValueError("cross-column must be a different stored risk column")
    pairs = list(itertools.combinations(columns, 2))
    pair = tuple(sorted((risk_column, cross_column)))
    return columns.index(risk_column), pairs.index(pair), cross_column


def _curve_indices(payload, panel, maturity_days):
    design = payload["test_design"]
    labels = np.asarray(design["spot_panel_labels"])
    maturities = np.asarray(design["maturity_days"], dtype=float)
    moneyness = np.asarray(design["log_moneyness"], dtype=float)
    mask = (labels == panel) & np.isclose(maturities, maturity_days)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(f"curve {panel}, T={maturity_days:g}d was not stored")
    return indices[np.argsort(moneyness[indices])], moneyness


def _component_values(record, component, indices, risk_position, cross_position):
    values = np.asarray(record[component], dtype=float)
    position = cross_position if component == "cross_gamma" else risk_position
    return values[indices, position]


def plot_one_curve(
    payload,
    output_dir,
    panel,
    maturity_days,
    relative_bump,
    risk_column,
    cross_column,
    include_multilinear,
):
    risk_position, cross_position, cross_column = _risk_positions(
        payload,
        risk_column,
        cross_column,
    )
    indices, all_moneyness = _curve_indices(payload, panel, maturity_days)
    moneyness = all_moneyness[indices]
    bump_key = _bump_key(relative_bump)
    analytical = payload["analytical_reference"]["components"]
    exact = payload["exact_price_finite_difference_controls"][bump_key][
        "components"
    ]
    methods = [
        mode
        for mode in payload["settings"]["interpolation_modes"]
        if include_multilinear or mode != "multilinear"
    ]
    marker_every = max(1, len(indices) // 10)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.2, 8.0),
        sharex="col",
        squeeze=False,
    )
    for column, component in enumerate(COMPONENTS):
        top, bottom = axes[0, column], axes[1, column]
        truth = _component_values(
            analytical,
            component,
            indices,
            risk_position,
            cross_position,
        )
        exact_values = _component_values(
            exact,
            component,
            indices,
            risk_position,
            cross_position,
        )
        top.plot(
            moneyness,
            truth,
            color="black",
            linewidth=2.5,
            label="Analytical",
            zorder=5,
        )
        top.plot(
            moneyness,
            exact_values,
            color="0.45",
            linewidth=1.6,
            linestyle=":",
            marker="d",
            markevery=marker_every,
            markersize=4.0,
            label="Exact-price FD",
        )
        bottom.semilogy(
            moneyness,
            np.maximum(np.abs(exact_values - truth), 1.0e-15),
            color="0.45",
            linewidth=1.6,
            linestyle=":",
            label="Exact-price FD",
        )
        for mode in methods:
            record = payload["interpolation_results"][mode]["bump_results"][
                bump_key
            ]["components"]
            values = _component_values(
                record,
                component,
                indices,
                risk_position,
                cross_position,
            )
            color = METHOD_COLORS.get(mode, "C2")
            label = METHOD_LABELS.get(mode, mode)
            top.plot(
                moneyness,
                values,
                color=color,
                linewidth=1.55,
                marker="s",
                markevery=marker_every,
                markersize=3.5,
                label=label,
            )
            bottom.semilogy(
                moneyness,
                np.maximum(np.abs(values - truth), 1.0e-15),
                color=color,
                linewidth=1.55,
                label=label,
            )
        top.axvline(0.0, color="0.8", linewidth=0.9)
        bottom.axvline(0.0, color="0.8", linewidth=0.9)
        top.set_title(LABELS[component])
        top.grid(alpha=0.25)
        bottom.grid(which="both", alpha=0.25)
        bottom.set_xlabel("Log-moneyness $m$")
        if column == 0:
            top.set_ylabel("Greek value")
            bottom.set_ylabel("Absolute error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        rf"Dense fixed-strike Greek audit"
        rf"{' — hybrid zoom' if not include_multilinear else ''} — "
        rf"{panel.replace('_', ' ')}, "
        rf"$T={maturity_days:g}$d, bump={100.0 * relative_bump:g}%, "
        rf"$S_{{{risk_column + 1}}}$ and "
        rf"$\Gamma_{{{risk_column + 1},{cross_column + 1}}}$",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    stem = (
        f"greeks_v921_dense_{panel}_T_{_filename_number(maturity_days)}d_"
        f"bump_{_filename_number(relative_bump)}_"
        f"{'comparison' if include_multilinear else 'hybrid_zoom'}"
    )
    _save(fig, output_dir, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v921_dense_profiles"),
    )
    parser.add_argument("--panel", default=None)
    parser.add_argument("--maturity-days", nargs="+", type=float, default=None)
    parser.add_argument("--relative-bump", type=float, default=0.002)
    parser.add_argument("--all-bumps", action="store_true")
    parser.add_argument("--risk-column", type=int, default=0)
    parser.add_argument("--cross-column", type=int, default=None)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-dense-profile-audit-v9.2.1":
        raise ValueError("the input is not a v9.2.1 dense-profile result")
    panel = args.panel or payload["settings"]["panels"][0]
    maturities = args.maturity_days or payload["settings"]["maturity_days"]
    bumps = (
        payload["settings"]["relative_bumps"]
        if args.all_bumps
        else [args.relative_bump]
    )
    available_bumps = {
        _bump_key(value) for value in payload["settings"]["relative_bumps"]
    }
    if any(_bump_key(value) not in available_bumps for value in bumps):
        raise ValueError("a requested bump is unavailable in the JSON")
    for bump in bumps:
        for maturity in maturities:
            for include_multilinear in (True, False):
                plot_one_curve(
                    payload,
                    args.output_dir,
                    panel,
                    float(maturity),
                    float(bump),
                    args.risk_column,
                    args.cross_column,
                    include_multilinear,
                )
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
