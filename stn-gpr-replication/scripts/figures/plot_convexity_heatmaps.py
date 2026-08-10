from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from stngpr.config import PaperConfig
from stngpr.coordinates import build_coordinate_grid
from stngpr.diagnostics import (
    geometric_basket_convexity_ridge,
    geometric_basket_effective_parameters,
    geometric_basket_log_moneyness_convexity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot geometric-basket convexity with uniform and adaptive QTT nodes."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--rate", type=float, default=0.03)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PaperConfig()
    uniform, _, _ = build_coordinate_grid(
        config, "moneyness_uniform", "geometric"
    )
    adaptive, _, _ = build_coordinate_grid(
        config, "moneyness_adaptive", "geometric"
    )
    m_uniform, t_uniform = uniform.axes[config.n_assets], uniform.axes[-1]
    m_adaptive, t_adaptive = adaptive.axes[config.n_assets], adaptive.axes[-1]

    basket_sigma, basket_carry = geometric_basket_effective_parameters(
        args.rate,
        config.volatilities,
        config.correlation,
        config.dividends,
    )
    m_dense = np.linspace(m_uniform[0], m_uniform[-1], 900)
    t_dense = np.geomspace(config.maturity_bounds[0], config.maturity_bounds[1], 500)
    mm, tt = np.meshgrid(m_dense, t_dense)
    convexity = geometric_basket_log_moneyness_convexity(
        mm, tt, args.rate, basket_sigma, basket_carry
    )
    positive = convexity[convexity > 0.0]
    vmin = max(float(np.quantile(positive, 0.01)), 1.0e-8)
    vmax = float(np.quantile(positive, 0.999))
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    plotted_convexity = np.maximum(convexity, vmin)
    ridge = geometric_basket_convexity_ridge(
        t_dense, basket_sigma, basket_carry
    )

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": args.dpi,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25), sharex=True, sharey=True)
    panels = (
        (axes[0], m_uniform, t_uniform, "(a) Uniform QTT grid"),
        (axes[1], m_adaptive, t_adaptive, "(b) Convexity-guided QTT grid"),
    )
    mesh = None
    for ax, m_nodes, t_nodes, title in panels:
        mesh = ax.pcolormesh(
            m_dense,
            t_dense,
            plotted_convexity,
            shading="auto",
            cmap="magma",
            norm=norm,
            rasterized=True,
        )
        node_m, node_t = np.meshgrid(m_nodes, t_nodes)
        ax.scatter(
            node_m,
            node_t,
            s=5,
            marker="o",
            facecolors="none",
            edgecolors="white",
            linewidths=0.35,
            alpha=0.72,
            label="Physical QTT nodes",
        )
        ax.plot(
            ridge,
            t_dense,
            color="cyan",
            linewidth=1.2,
            linestyle="--",
            label=r"Analytical ridge $m_\star(T)$",
        )
        ax.axvline(0.0, color="white", linewidth=0.7, alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(r"Basket log-moneyness $m=\log(K/B_0)$")
        ax.grid(False)
    axes[0].set_ylabel("Maturity $T$ (years, log scale)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.49, 0.005),
        ncol=2,
        frameon=False,
    )
    colorbar = fig.colorbar(mesh, ax=axes, pad=0.025, fraction=0.035)
    colorbar.set_label(r"Dimensionless strike convexity $\chi=(K^2/B_0)V_{KK}$")
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.20, top=0.91, wspace=0.08)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "convexity-grid-comparison"
    for extension in args.formats:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"basket_sigma={basket_sigma:.8f}")
    print(f"basket_carry={basket_carry:.8f}")
    for extension in args.formats:
        print(stem.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
