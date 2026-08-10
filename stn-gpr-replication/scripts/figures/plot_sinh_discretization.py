from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stngpr.config import PaperConfig
from stngpr.grids import sinh_centered_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the hyperbolic-sine log-moneyness discretization."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--concentration", type=float, default=3.0)
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
    n_nodes = config.physical_shape[config.n_assets]
    m_min = float(np.log(config.strike_bounds[0] / config.spot_bounds[1]))
    m_max = float(np.log(config.strike_bounds[1] / config.spot_bounds[0]))

    u = np.linspace(-1.0, 1.0, n_nodes)
    m_uniform = np.linspace(m_min, m_max, n_nodes)
    m_sinh = sinh_centered_axis(
        m_min, m_max, n_nodes, concentration=args.concentration
    )
    uniform_widths = np.diff(m_uniform)
    sinh_widths = np.diff(m_sinh)
    uniform_midpoints = 0.5 * (m_uniform[:-1] + m_uniform[1:])
    sinh_midpoints = 0.5 * (m_sinh[:-1] + m_sinh[1:])

    central_cell = n_nodes // 2 - 1
    ratio = float(sinh_widths[central_cell] / uniform_widths[central_cell])
    refinement = 1.0 / ratio

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": args.dpi,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))

    axes[0].plot(
        u,
        m_uniform,
        color="0.45",
        linewidth=1.5,
        linestyle="--",
        label="Uniform physical grid",
    )
    axes[0].plot(
        u,
        m_sinh,
        color="#1764ab",
        linewidth=1.8,
        label=rf"Sinh grid ($\gamma={args.concentration:g}$)",
    )
    axes[0].scatter(
        u,
        m_sinh,
        s=9,
        color="#1764ab",
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
        label=rf"Physical QTT nodes ($n_m={n_nodes}$)",
    )
    axes[0].axhline(0.0, color="#c43c39", linewidth=0.9, alpha=0.9)
    axes[0].axvline(0.0, color="0.75", linewidth=0.7)
    axes[0].set_title("(a) Computational-to-physical coordinate map")
    axes[0].set_xlabel(r"Uniform computational coordinate $u$")
    axes[0].set_ylabel(r"Physical log-moneyness $m$")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].plot(
        uniform_midpoints,
        uniform_widths,
        color="0.45",
        linewidth=1.5,
        linestyle="--",
        label="Uniform physical grid",
    )
    axes[1].plot(
        sinh_midpoints,
        sinh_widths,
        color="#1764ab",
        linewidth=1.6,
        marker="o",
        markersize=2.8,
        markeredgecolor="white",
        markeredgewidth=0.2,
        label=rf"Sinh grid ($\gamma={args.concentration:g}$)",
    )
    axes[1].axvline(0.0, color="#c43c39", linewidth=0.9, alpha=0.9)
    axes[1].annotate(
        rf"ATM-straddling cell: $h_{{\rm ATM}}/h_{{\rm unif}}={ratio:.3f}$"
        "\n"
        rf"$\approx {refinement:.2f}\times$ finer",
        xy=(sinh_midpoints[central_cell], sinh_widths[central_cell]),
        xytext=(0.12, 0.24),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "0.25", "lw": 0.8},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75"},
    )
    axes[1].set_title("(b) Local physical cell width")
    axes[1].set_xlabel(r"Cell-midpoint log-moneyness $\bar m_j$")
    axes[1].set_ylabel(r"Cell width $\Delta m_j$")
    axes[1].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=2.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "sinh-moneyness-discretization"
    for extension in args.formats:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    theoretical_ratio = (
        (n_nodes - 1)
        * np.sinh(args.concentration / (n_nodes - 1))
        / np.sinh(args.concentration)
    )
    print(f"m_min={m_min:.8f}")
    print(f"m_max={m_max:.8f}")
    print(f"atm_cell_ratio={ratio:.8f}")
    print(f"theoretical_ratio={theoretical_ratio:.8f}")
    for extension in args.formats:
        print(stem.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
