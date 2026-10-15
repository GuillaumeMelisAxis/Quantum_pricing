"""Illustrate the adaptive (m, T) grid against the uniform baseline.

Two panels only: the uniform lattice and the adaptive lattice, drawn as full
tensor-product meshes so the change of discretization is immediately visible.
The grids come from ``build_coordinate_grid``, so the figure always shows the
nodes a run actually uses. Only the strike/log-moneyness and the maturity axes
differ between modes; the five spot axes are the same uniform grid everywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from stngpr.config import PaperConfig
from stngpr.coordinates import GRID_MODES, build_coordinate_grid


BASELINE_COLOR = "#5b6472"
ADAPTIVE_COLOR = "#1764ab"
ATM_COLOR = "#c43c39"
HIGHLIGHT_COLOR = "#f2c14b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the adaptive QTT grid next to the uniform grid."
    )
    parser.add_argument("--baseline-mode", choices=GRID_MODES, default="moneyness_uniform")
    parser.add_argument("--new-mode", choices=GRID_MODES, default="moneyness_adaptive")
    parser.add_argument("--basket-kind", choices=("geometric", "arithmetic"), default="geometric")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--reference-spot",
        type=float,
        default=100.0,
        help="basket spot used to map a strike axis into log-moneyness",
    )
    parser.add_argument(
        "--m-limit",
        type=float,
        default=None,
        help="clip the log-moneyness view to [-M, M] (default: full grid span)",
    )
    parser.add_argument(
        "--atm-band",
        type=float,
        default=0.30,
        help="half-width of the highlighted ATM log-moneyness band",
    )
    parser.add_argument(
        "--short-maturity",
        type=float,
        default=0.25,
        help="maturity ceiling (years) of the highlighted short-maturity band",
    )
    parser.add_argument("--no-highlight", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png",))
    return parser.parse_args()


def extract_axes(config, mode, basket_kind, reference_spot):
    """Return (log-moneyness axis, maturity axis, converted flag) for one mode."""
    grid, _, metadata = build_coordinate_grid(config, mode, basket_kind)
    strike_axis = grid.axes[config.n_assets]
    converted = metadata["strike_coordinate"] == "strike"
    moneyness_axis = np.log(strike_axis / reference_spot) if converted else strike_axis
    return moneyness_axis, grid.axes[-1], converted


def draw_mesh(ax, moneyness, maturity, color, node_size, line_width, alpha):
    for value in moneyness:
        ax.axvline(value, color=color, linewidth=line_width, alpha=alpha, zorder=1)
    for value in maturity:
        ax.axhline(value, color=color, linewidth=line_width, alpha=alpha, zorder=1)
    mesh_m, mesh_t = np.meshgrid(moneyness, maturity, indexing="ij")
    ax.scatter(
        mesh_m.ravel(),
        mesh_t.ravel(),
        s=node_size,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax.axvline(0.0, color=ATM_COLOR, linewidth=1.4, alpha=0.95, zorder=2)


def lattice_panel(
    ax,
    moneyness,
    maturity,
    color,
    title,
    limits,
    band,
    t_band,
    highlight,
    show_ylabel=True,
):
    m_lo, m_hi, t_lo, t_hi = limits
    visible = moneyness[(moneyness >= m_lo) & (moneyness <= m_hi)]
    draw_mesh(ax, visible, maturity, color, 12, 0.45, 0.30)

    ax.set_xlim(m_lo, m_hi)
    ax.set_ylim(t_lo - 0.04 * (t_hi - t_lo), t_hi + 0.06 * (t_hi - t_lo))
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(r"Log-moneyness $m=\log(K/B)$", fontsize=11)
    if show_ylabel:
        ax.set_ylabel(r"Maturity $T$ (years)", fontsize=11)
    ax.tick_params(labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("0.6")

    if not highlight:
        return

    # Magnified ATM / short-maturity corner: the region the surrogate is graded on.
    inset = ax.inset_axes([0.05, 0.40, 0.34, 0.38], facecolor="white")
    keep_m = moneyness[np.abs(moneyness) <= band]
    keep_t = maturity[maturity <= t_band]
    draw_mesh(inset, keep_m, keep_t, color, 34, 0.9, 0.55)
    inset.set_xlim(-band, band)
    inset.set_ylim(t_lo - 0.06 * t_band, t_band)
    inset.set_xticks([-band, 0.0, band])
    inset.set_yticks([t_lo, t_band])
    inset.set_yticklabels([f"{t_lo:.3f}", f"{t_band:g}"])
    inset.tick_params(labelsize=8, length=2)
    inset.set_title(
        f"{keep_m.size * keep_t.size} nodes near ATM, short $T$",
        fontsize=10,
        fontweight="bold",
        pad=4,
        color="0.15",
    ).set_bbox({"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none"})
    for spine in inset.spines.values():
        spine.set_color(HIGHLIGHT_COLOR)
        spine.set_linewidth(1.6)
    ax.indicate_inset_zoom(inset, edgecolor=HIGHLIGHT_COLOR, linewidth=1.6, alpha=0.95)
    for patch in ax.patches:
        if isinstance(patch, Rectangle) and patch.get_width() > 0:
            patch.set_facecolor(HIGHLIGHT_COLOR)
            patch.set_alpha(0.30)


def main() -> None:
    args = parse_args()
    config = PaperConfig()

    m_base, t_base, converted_base = extract_axes(
        config, args.baseline_mode, args.basket_kind, args.reference_spot
    )
    m_new, t_new, converted_new = extract_axes(
        config, args.new_mode, args.basket_kind, args.reference_spot
    )

    m_lo = max(m_base[0], m_new[0])
    m_hi = min(m_base[-1], m_new[-1])
    if args.m_limit is not None:
        m_lo, m_hi = max(m_lo, -args.m_limit), min(m_hi, args.m_limit)
    t_lo = min(t_base[0], t_new[0])
    t_hi = max(t_base[-1], t_new[-1])
    limits = (m_lo, m_hi, t_lo, t_hi)
    highlight = not args.no_highlight

    plt.rcParams.update({"font.size": 10, "figure.dpi": args.dpi})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), sharex=True, sharey=True)

    lattice_panel(
        axes[0],
        m_base,
        t_base,
        BASELINE_COLOR,
        f"Uniform grid\n{m_base.size} x {t_base.size} = {m_base.size * t_base.size} nodes",
        limits,
        args.atm_band,
        args.short_maturity,
        highlight,
        show_ylabel=True,
    )
    lattice_panel(
        axes[1],
        m_new,
        t_new,
        ADAPTIVE_COLOR,
        f"Adaptive grid\n{m_new.size} x {t_new.size} = {m_new.size * t_new.size} nodes",
        limits,
        args.atm_band,
        args.short_maturity,
        highlight,
        show_ylabel=False,
    )

    axes[0].annotate(
        "ATM",
        xy=(0.0, t_hi),
        xytext=(5, -3),
        textcoords="offset points",
        color=ATM_COLOR,
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    subtitle = (
        "Same node budget, redistributed: sinh clustering around the money and "
        "quadratic refinement at the short maturity end."
    )
    if converted_base or converted_new:
        subtitle += f"  Strike axes mapped to $m=\\log(K/{args.reference_spot:g})$."
    fig.suptitle("Same budget, better placed nodes", fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.915, subtitle, ha="center", fontsize=10, color="0.3")
    fig.subplots_adjust(top=0.775, bottom=0.115, left=0.06, right=0.985, wspace=0.11)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / (
        args.output_name or f"grid-illustration-{args.new_mode}-vs-{args.baseline_mode}"
    )
    for extension in args.formats:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    band, t_band = args.atm_band, args.short_maturity
    corner_base = int(np.sum(np.abs(m_base) <= band)) * int(np.sum(t_base <= t_band))
    corner_new = int(np.sum(np.abs(m_new) <= band)) * int(np.sum(t_new <= t_band))
    print(f"baseline_mode={args.baseline_mode}")
    print(f"new_mode={args.new_mode}")
    print(f"view_moneyness=[{m_lo:.6f}, {m_hi:.6f}]")
    print(f"corner_nodes_baseline={corner_base}")
    print(f"corner_nodes_new={corner_new}")
    for extension in args.formats:
        print(stem.with_suffix(f".{extension}"))


if __name__ == "__main__":
    main()
