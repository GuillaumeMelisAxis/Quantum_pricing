from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

from stngpr.diagnostics import (
    geometric_basket_log_moneyness_convexity,
    geometric_basket_normalized_put,
)
from stngpr.grid_geometry import local_cubic_lagrange


LABELS = {
    "pricing_grid": "Pricing grid",
    "risk_hybrid_grid": "Risk-hybrid grid",
}
COLORS = {
    "pricing_grid": "C0",
    "risk_hybrid_grid": "C1",
}


def _save(fig, output_directory: Path, stem: str) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_directory / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _analytical_curve(m, maturity, rate, basket_sigma, basket_carry):
    value = geometric_basket_normalized_put(
        m, maturity, rate, basket_sigma, basket_carry
    )
    std = basket_sigma * np.sqrt(maturity)
    d1 = (-m + (basket_carry + 0.5 * basket_sigma**2) * maturity) / std
    d2 = d1 - std
    first = np.exp(m - rate * maturity) * ndtr(-d2)
    convexity = geometric_basket_log_moneyness_convexity(
        m, maturity, rate, basket_sigma, basket_carry
    )
    return value, first, convexity


def plot_node_map(payload, output_directory):
    profiles = payload["representative_profiles"]
    fig, axes = plt.subplots(len(profiles), 1, figsize=(9.0, 2.15 * len(profiles)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, profile in zip(axes, profiles):
        lower, upper = profile["zoom_bounds"]
        for row, name in enumerate(("pricing_grid", "risk_hybrid_grid")):
            nodes = np.asarray(profile["grids"][name]["physical_nodes"], dtype=float)
            visible = nodes[(nodes >= lower) & (nodes <= upper)]
            axis.vlines(
                visible,
                row - 0.28,
                row + 0.28,
                color=COLORS[name],
                linewidth=0.9,
                alpha=0.9,
            )
        ridge = float(profile["convexity_ridge"])
        scale = float(profile["risk_scale"])
        axis.axvspan(ridge - 2.0 * scale, ridge + 2.0 * scale, color="0.88", zorder=0)
        axis.axvline(ridge, color="black", linestyle="--", linewidth=1.0)
        axis.set_xlim(lower, upper)
        axis.set_ylim(-0.55, 1.55)
        axis.set_yticks([0, 1], [LABELS["pricing_grid"], LABELS["risk_hybrid_grid"]])
        axis.set_title(
            f"$T={profile['maturity_days']:.0f}$d, $r={profile['rate']:.3f}$; "
            r"shaded band: $|m-m_\star|\leq2\sigma_G\sqrt{T}$"
        )
        axis.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("Physical log-moneyness $m$")
    fig.suptitle("v9.9 physical node redistribution at identical node count")
    fig.tight_layout()
    _save(fig, output_directory, "greeks_v99_node_redistribution")


def _by_maturity(payload, kappa_key, metric):
    maturities = sorted(set(row["maturity_days"] for row in payload["scenarios"]))
    result = {name: [] for name in LABELS}
    for maturity in maturities:
        rows = [row for row in payload["scenarios"] if row["maturity_days"] == maturity]
        for name in LABELS:
            result[name].append(
                np.median(
                    [row["grids"][name]["bands"][kappa_key][metric] for row in rows]
                )
            )
    return np.asarray(maturities), result


def plot_local_coverage(payload, output_directory):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for kappa, linestyle in zip((1, 2, 3), ("-", "--", ":")):
        key = f"kappa_{kappa}"
        maturity, fill = _by_maturity(payload, key, "fill_distance")
        _, fraction = _by_maturity(payload, key, "node_fraction")
        coverage_gain = np.asarray(fill["pricing_grid"]) / np.asarray(
            fill["risk_hybrid_grid"]
        )
        concentration_gain = np.asarray(fraction["risk_hybrid_grid"]) / np.asarray(
            fraction["pricing_grid"]
        )
        axes[0].plot(
            maturity,
            coverage_gain,
            linestyle=linestyle,
            marker="o",
            label=rf"$\kappa={kappa}$",
        )
        axes[1].plot(
            maturity,
            concentration_gain,
            linestyle=linestyle,
            marker="o",
            label=rf"$\kappa={kappa}$",
        )
    axes[0].axhline(1.0, color="black", linewidth=1.0)
    axes[1].axhline(1.0, color="black", linewidth=1.0)
    axes[0].set_ylabel(r"$G_\kappa^{\mathrm{fill}}=h_\kappa^P/h_\kappa^R$")
    axes[1].set_ylabel(r"$G_\kappa^{\mathrm{nodes}}=P_\kappa^R/P_\kappa^P$")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Maturity (days)")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[0].set_title("Local coverage gain (> 1 is better)")
    axes[1].set_title("Local concentration gain (> 1 is denser)")
    fig.suptitle("v9.9.1 ridge-restricted grid gains (median across rates)")
    fig.tight_layout()
    _save(fig, output_directory, "greeks_v99_local_coverage")


def plot_derivative_profiles(payload, output_directory):
    profiles = payload["representative_profiles"]
    names = ("pricing_grid", "risk_hybrid_grid")
    component_names = ("$u$", "$u_m$", r"$\chi=u_{mm}-u_m$")
    fig, axes = plt.subplots(len(profiles), 3, figsize=(14.0, 3.35 * len(profiles)), squeeze=False)
    for row_index, profile in enumerate(profiles):
        lower, upper = profile["zoom_bounds"]
        m = np.linspace(lower, upper, 1201)
        maturity = float(profile["maturity_days"]) / 365.0
        reference = _analytical_curve(
            m,
            maturity,
            float(profile["rate"]),
            float(profile["basket_volatility"]),
            float(profile["basket_carry"]),
        )
        for column, label in enumerate(component_names):
            axis = axes[row_index, column]
            axis.plot(m, reference[column], color="black", linewidth=2.0, label="Analytical")
            for name in names:
                nodes = np.asarray(profile["grids"][name]["physical_nodes"], dtype=float)
                node_values = geometric_basket_normalized_put(
                    nodes,
                    maturity,
                    float(profile["rate"]),
                    float(profile["basket_volatility"]),
                    float(profile["basket_carry"]),
                )
                interpolated = local_cubic_lagrange(nodes, node_values, m)
                estimate = (
                    interpolated[column]
                    if column < 2
                    else interpolated[2] - interpolated[1]
                )
                axis.plot(m, estimate, color=COLORS[name], linewidth=1.25, linestyle="--", label=LABELS[name])
            axis.axvline(profile["convexity_ridge"], color="0.5", linewidth=0.8)
            axis.grid(alpha=0.25)
            axis.set_title(f"{label}, $T={profile['maturity_days']:.0f}$d")
            if row_index == len(profiles) - 1:
                axis.set_xlabel("Log-moneyness $m$")
            if column == 0:
                axis.set_ylabel("Normalized value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("v9.9.1 local-cubic analytical derivative audit", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    _save(fig, output_directory, "greeks_v99_derivative_profiles")


def plot_fill_error_association(payload, output_directory):
    fig, axis = plt.subplots(figsize=(6.8, 5.0))
    coverage_gain = []
    error_reduction = []
    maturity = []
    for row in payload["scenarios"]:
        pricing = row["grids"]["pricing_grid"]["bands"]["kappa_2"]
        risk = row["grids"]["risk_hybrid_grid"]["bands"]["kappa_2"]
        coverage_gain.append(pricing["fill_distance"] / risk["fill_distance"])
        error_reduction.append(
            pricing["interpolation"]["chi"]["normalized_mae"]
            / risk["interpolation"]["chi"]["normalized_mae"]
        )
        maturity.append(row["maturity_days"])
    points = axis.scatter(
        coverage_gain,
        error_reduction,
        c=maturity,
        cmap="viridis",
        s=34,
        alpha=0.8,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.axvline(1.0, color="black", linewidth=1.0)
    axis.axhline(1.0, color="black", linewidth=1.0)
    axis.set_xlabel(r"Coverage gain $G_2^{\mathrm{fill}}=h_2^P/h_2^R$")
    axis.set_ylabel(r"Curvature-error reduction $G_\chi=E_\chi^P/E_\chi^R$")
    correlation = payload["summary"]["paired_gain_association"][
        "spearman_correlation"
    ]
    axis.set_title(
        rf"Coverage gain versus curvature-error reduction "
        rf"($\rho_S={correlation:.3f}$)"
    )
    axis.grid(alpha=0.3)
    colorbar = fig.colorbar(points, ax=axis)
    colorbar.set_label("Maturity (days)")
    fig.tight_layout()
    _save(fig, output_directory, "greeks_v99_fill_curvature_association")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v99_grid_geometry"),
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-grid-geometry-v9.9.1":
        raise ValueError("input is not a v9.9.1 grid-geometry result")
    plot_node_map(payload, args.output_dir)
    plot_local_coverage(payload, args.output_dir)
    plot_derivative_profiles(payload, args.output_dir)
    plot_fill_error_association(payload, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
