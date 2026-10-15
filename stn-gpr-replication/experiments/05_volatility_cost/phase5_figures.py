r"""Phase 5 - the three figures, as vector PDF and SVG.

No titles are drawn: the captions live in the manuscript. Everything is sized for
a single journal column and has to survive being printed in greyscale, so series
identity never rests on hue - every series carries its own dash pattern, its own
marker and a direct label, and the colors are a fourth, redundant channel.

Palette. Five categorical hues taken from the reference palette of the dataviz
skill, in its documented slot order, restricted to a five-hue subset that clears
the *all-pairs* gates rather than only the adjacent ones: figure 1 exists to let
the reader compare non-adjacent curves, so the adjacent pairlist is the wrong
test. The chosen subset (blue, yellow, magenta, green, violet) is the best of the
two that pass with CVD status PASS - worst all-pairs CVD Delta E 13.0 (protan),
worst normal-vision Delta E 16.3, both above their gates. Three of the five sit
below 3:1 contrast on a light surface, which is exactly why direct labels are not
optional here.

    .venv/Scripts/python.exe experiments/05_volatility_cost/phase5_figures.py --tag phase4
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUTPUT_DIR, UNRESOLVED_FACTOR

# d_sigma -> (color, dash, marker). Colors follow the entity, never its rank, and
# are assigned once here.
STYLE = {
    0: ("#2a78d6", (0, ()), "o"),
    1: ("#eda100", (0, (5, 1.5)), "s"),
    2: ("#e87ba4", (0, (1.5, 1.5)), "^"),
    3: ("#008300", (0, (5, 1.5, 1.5, 1.5)), "D"),
    5: ("#4a3aa7", (0, (7, 2, 1.5, 2, 1.5, 2)), "v"),
}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
SIGMA_BAND = "#e8e7e2"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

COLUMN_WIDTH = 3.4  # inches, one journal column


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def save(fig, out_dir: Path, name: str) -> list[Path]:
    paths = []
    # PNG as well: the manuscript includes the raster version.
    for suffix in ("pdf", "svg", "png"):
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def style_axes(axis, log_x=False, log_y=False):
    axis.grid(True, which="major", axis="both")
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    # Matplotlib labels log minor ticks whenever a decade axis spans less than
    # one decade per tick, which at column width turns the axis into an
    # unreadable run of overlapping 2x10^4 / 3x10^4 labels. Decades only.
    if log_x:
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
        axis.xaxis.set_minor_formatter(mticker.NullFormatter())
    if log_y:
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        axis.yaxis.set_minor_formatter(mticker.NullFormatter())


def log_ticks_with_subs(axis, which="y", subs=(1.0, 2.0, 3.0, 5.0)):
    """Label 1/2/3/5 within each decade - useful when the span is narrow."""
    target = axis.yaxis if which == "y" else axis.xaxis
    target.set_major_locator(mticker.LogLocator(base=10.0, subs=subs, numticks=20))
    target.set_major_formatter(
        mticker.FuncFormatter(
            lambda value, _: f"{value:g}" if value >= 1 else f"{value:.3g}"
        )
    )
    target.set_minor_formatter(mticker.NullFormatter())


# --------------------------------------------------------------------------
# figure 1 - statistical cost
# --------------------------------------------------------------------------


def figure_statistical_cost(rows, out_dir: Path, name: str):
    """NMAE on the common slice against realized oracle calls, one curve per d_sigma.

    The quantity of interest is the horizontal gap between curves at fixed
    precision, so the abscissa is log-scaled and the interpolation floors are
    drawn as dashed horizontals: a curve that has reached its floor has stopped
    being a statement about TT-cross.
    """
    by_config = defaultdict(list)
    floors = {}
    for row in rows:
        d_sigma = int(row["d_sigma"])
        by_config[(d_sigma, int(row["budget_nominal"]))].append(row)
        floors[d_sigma] = number(row["floor_nmae_common_slice"])

    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))
    style_axes(axis, log_x=True, log_y=True)

    for d_sigma in sorted({int(row["d_sigma"]) for row in rows}):
        color, dash, marker = STYLE[d_sigma]
        calls, nmae, error, unresolved = [], [], [], []
        budgets = sorted(
            {
                int(row["budget_nominal"])
                for row in rows
                if int(row["d_sigma"]) == d_sigma
            }
        )
        for budget in budgets:
            group = by_config[(d_sigma, budget)]
            calls.append(np.mean([number(row["oracle_calls_realized"]) for row in group]))
            values = np.array([number(row["nmae_common_slice"]) for row in group])
            nmae.append(float(np.mean(values)))
            error.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
            unresolved.append(
                any(row["resolution_status"] == "UNRESOLVED" for row in group)
            )
        axis.errorbar(
            calls,
            nmae,
            yerr=error,
            color=color,
            linestyle=dash,
            marker=marker,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            capsize=2,
            elinewidth=0.8,
            zorder=3,
        )
        # An unresolved fit is marked on the point itself, so a low-rank claim
        # can never be read off a curve without seeing that it is unresolved.
        for x, y, flag in zip(calls, nmae, unresolved):
            if flag:
                axis.plot(
                    x, y, marker="x", color=color, markersize=6, markeredgewidth=1.3,
                    zorder=4,
                )

    for d_sigma, floor in sorted(floors.items()):
        if floor is None:
            continue
        color, _, _ = STYLE[d_sigma]
        axis.axhline(floor, color=color, linestyle=(0, (1, 2)), linewidth=0.9, zorder=1)

    axis.set_xlabel("realized oracle calls")
    axis.set_ylabel("NMAE, common slice")
    handles = [
        Line2D(
            [], [], color=STYLE[d][0], linestyle=STYLE[d][1], marker=STYLE[d][2],
            markerfacecolor="white", markeredgecolor=STYLE[d][0], markersize=4,
            label=rf"$d_\sigma={d}$",
        )
        for d in sorted(floors)
    ]
    handles += [
        Line2D([], [], color=MUTED, linestyle=(0, (1, 2)), label="interpolation floor"),
        Line2D(
            [], [], color=MUTED, linestyle="none", marker="x", markersize=6,
            label=f"UNRESOLVED (>{UNRESOLVED_FACTOR:g}x floor)",
        ),
    ]
    # Five series is past the four the direct-label rule covers, and at column
    # width the end-of-curve labels collide with each other and with the legend.
    # The legend carries identity instead, and dash + marker keep it off hue
    # alone, which is also what makes the figure survive greyscale printing.
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=3,
        handlelength=2.4,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    # The budget ladder spans well under two decades, so decade-only ticks leave
    # a single labelled gridline and the abscissa stops being readable as a
    # scale. 1/2/5 per decade is the density that fits at column width.
    log_ticks_with_subs(axis, "x", subs=(1.0, 2.0, 5.0))
    axis.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{value / 1000:g}k")
    )
    axis.set_xlabel("realized oracle calls")
    axis.margins(x=0.08)
    return save(fig, out_dir, name)


# --------------------------------------------------------------------------
# figure 2 - rank profile
# --------------------------------------------------------------------------


def figure_rank_profile(rank_rows, result_rows, out_dir: Path, name: str):
    """Bond rank against core index, the two extreme d_sigma overlaid.

    The shared cores occupy the same indices in both chains by construction (the
    sigma block is appended, never inserted), so the curves are aligned on them
    without any re-indexing; the appended block is shaded.
    """
    resolved = {
        (int(row["d_sigma"]), int(row["budget_nominal"]), int(row["seed"]))
        for row in result_rows
        if row["resolution_status"] != "UNRESOLVED"
    }
    available = sorted({int(row["d_sigma"]) for row in rank_rows})
    low, high = min(available), max(available)

    budgets = sorted({int(row["budget_nominal"]) for row in rank_rows})
    seeds_at = {
        budget: {
            int(r["seed"]) for r in rank_rows if int(r["budget_nominal"]) == budget
        }
        for budget in budgets
    }

    def resolved_count(d, budget):
        return sum(1 for seed in seeds_at[budget] if (d, budget, seed) in resolved)

    # "The largest resolved budget" is only meaningful if a budget actually
    # resolves both chains. Prefer one that does; otherwise fall back to the
    # largest budget and say, on the figure, what fraction of seeds resolved -
    # a rank profile read off unresolved fits is not a statement about the
    # representation, and the reader has to be able to see that.
    chosen = next(
        (
            budget
            for budget in reversed(budgets)
            if all(resolved_count(d, budget) == len(seeds_at[budget])
                   for d in (low, high))
        ),
        budgets[-1],
    )
    status = " · ".join(
        rf"$d_\sigma$={d}: {resolved_count(d, chosen)}/{len(seeds_at[chosen])} resolved"
        for d in (low, high)
    )

    fig, axis = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
    style_axes(axis, log_y=True)

    sigma_indices = [
        int(row["bond_index"])
        for row in rank_rows
        if int(row["d_sigma"]) == high
        and row["bond_class"] in ("sigma_internal", "boundary")
    ]
    if sigma_indices:
        axis.axvspan(
            min(sigma_indices) - 0.5,
            max(sigma_indices) + 0.5,
            color=SIGMA_BAND,
            linewidth=0,
            zorder=0,
        )
        axis.annotate(
            r"$\sigma$ block",
            (0.5 * (min(sigma_indices) + max(sigma_indices)), 0.0),
            xycoords=("data", "axes fraction"),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            color=MUTED,
            fontsize=7.5,
            zorder=5,
        )

    for d_sigma in (low, high):
        color, dash, marker = STYLE[d_sigma]
        seeds = sorted(
            {
                int(row["seed"])
                for row in rank_rows
                if int(row["d_sigma"]) == d_sigma
                and int(row["budget_nominal"]) == chosen
            }
        )
        stacked = []
        for seed in seeds:
            profile = [
                (int(row["bond_index"]), int(row["rank"]))
                for row in rank_rows
                if int(row["d_sigma"]) == d_sigma
                and int(row["budget_nominal"]) == chosen
                and int(row["seed"]) == seed
            ]
            profile.sort()
            stacked.append([rank for _, rank in profile])
        array = np.asarray(stacked, dtype=float)
        index = np.arange(array.shape[1])
        mean = array.mean(axis=0)
        axis.fill_between(
            index, array.min(axis=0), array.max(axis=0),
            color=color, alpha=0.16, linewidth=0, zorder=1,
        )
        axis.plot(
            index, mean, color=color, linestyle=dash,
            marker=marker, markevery=6, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.0, zorder=3,
            label=rf"$d_\sigma={d_sigma}$",
        )

    axis.set_xlabel("QTT core index")
    axis.set_ylabel(r"bond rank $r_k$")
    handles = axis.get_legend_handles_labels()[0]
    handles.append(
        Line2D([], [], color=MUTED, alpha=0.3, linewidth=6, label="seed range")
    )
    # The curves fill the upper half and the appended block fills the right,
    # so the legend goes below the axes rather than hunting for a gap that the
    # real rank profile may not leave. The budget rides on the legend title,
    # which is the one place in the figure guaranteed not to overlap a mark.
    axis.legend(
        handles=handles,
        title=f"budget {chosen:,}".replace(",", " ") + f"  ·  {status}",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        handlelength=2.4,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    axis.get_legend().get_title().set_fontsize(7.5)
    axis.get_legend().get_title().set_color(MUTED)
    log_ticks_with_subs(axis, "y")
    axis.margins(x=0.02, y=0.12)
    return save(fig, out_dir, name)


# --------------------------------------------------------------------------
# figure 3 - dimensional slope
# --------------------------------------------------------------------------


def figure_dimensional_slope(aggregate_rows, out_dir: Path, name: str):
    """Two panels on a shared ``d_sigma`` abscissa.

    Left: the sigma-block bond rank and the global effective rank - both are
    ranks, so one y-scale. Right: TT parameters against logical tensor entries -
    both are counts, so again one y-scale, log because the second is
    exponential. No panel carries two scales.

    Colour here encodes the *measure*, not ``d_sigma`` - ``d_sigma`` is the
    abscissa. The mapping is held constant across the two panels: blue is the
    quantity the surrogate actually pays (global effective rank, TT parameters),
    violet the quantity the dimension count drives (sigma-block rank, logical
    entries). Both are drawn from the same validated pair, so the figure stays
    colourblind-safe and the reader is never asked to re-learn a hue mid-figure.
    """
    measure_paid = STYLE[0]
    measure_dimensional = STYLE[5]
    budgets = sorted({int(row["budget_nominal"]) for row in aggregate_rows})
    largest = budgets[-1]
    rows = [row for row in aggregate_rows if int(row["budget_nominal"]) == largest]
    rows.sort(key=lambda row: int(row["d_sigma"]))
    d_sigma = np.array([int(row["d_sigma"]) for row in rows])

    fig, axes = plt.subplots(
        1, 2, figsize=(2 * COLUMN_WIDTH, 2.5), sharex=True
    )

    def series(axis, key, label, slot, log=False):
        mean = np.array([number(row[f"{key}_mean"]) or np.nan for row in rows])
        std = np.array([number(row.get(f"{key}_std")) or 0.0 for row in rows])
        color, dash, marker = slot
        axis.errorbar(
            d_sigma, mean, yerr=std, color=color, linestyle=dash, marker=marker,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.1,
            capsize=2, elinewidth=0.8, label=label, zorder=3,
        )
        if log:
            axis.set_yscale("log")
        return mean

    style_axes(axes[0])
    series(
        axes[0], "sigma_block_internal_max_rank",
        r"max $r_k$ inside $\sigma$ block", measure_dimensional,
    )
    series(axes[0], "effective_rank", "global effective rank", measure_paid)
    axes[0].set_ylabel("rank")
    axes[0].set_xlabel(r"$d_\sigma$")
    # Both rank curves run along the top of this panel once the budget is large
    # enough to resolve every configuration, so the legend cannot live there.
    axes[0].legend(loc="lower right", handlelength=2.6)
    axes[0].margins(y=0.10)

    style_axes(axes[1], log_y=True)
    series(axes[1], "tt_parameters", "TT parameters", measure_paid, log=True)
    entries = np.array(
        [2.0 ** int(row["log2_logical_entries"]) for row in rows], dtype=float
    )
    color, dash, marker = measure_dimensional
    axes[1].plot(
        d_sigma, entries, color=color, linestyle=dash, marker=marker,
        markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.1,
        label="logical tensor entries", zorder=3,
    )
    axes[1].set_ylabel("count")
    axes[1].set_xlabel(r"$d_\sigma$")
    axes[1].legend(loc="center right", handlelength=2.6)

    for axis in axes:
        axis.set_xticks(d_sigma)
    return save(fig, out_dir, name)


# --------------------------------------------------------------------------


def figure_sigma_cell(cell_rows, floor_rows, sigma_ref, out_dir: Path, name: str,
                      axis_nodes=None, width=None):
    """NMAE against volatility, with the sigma-axis grid nodes marked.

    Over a single cell this answers the phase-1 node-placement question: is the
    value at ``sigma_ref`` a local minimum, or an ordinary point of the cell?
    Over several cells it shows the scallop directly - the interpolation floor
    dipping at every node and peaking between them - which is the same statement
    made visible rather than inferred from one arch.

    ``axis_nodes`` are drawn as thin verticals. Passing them explicitly matters
    once the range spans more than one cell: the endpoints of the sampled range
    are then no longer nodes, so they cannot stand in for them.
    """
    fig, axis = plt.subplots(figsize=(width or COLUMN_WIDTH, 2.3))
    style_axes(axis, log_y=True)

    by_sigma = defaultdict(list)
    for row in cell_rows:
        by_sigma[float(row["sigma"])].append(number(row["nmae"]))
    sigmas = np.array(sorted(by_sigma))
    mean = np.array([np.mean(by_sigma[s]) for s in sigmas])
    lo = np.array([np.min(by_sigma[s]) for s in sigmas])
    hi = np.array([np.max(by_sigma[s]) for s in sigmas])

    color, dash, marker = STYLE[5]
    axis.fill_between(sigmas, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=1)
    axis.plot(
        sigmas, mean, color=color, linestyle=dash, marker=marker, markersize=3.5,
        markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.0,
        label="TT surrogate NMAE", zorder=3,
    )

    if floor_rows:
        fcolor, fdash, fmarker = STYLE[0]
        fs = np.array([float(row["sigma"]) for row in floor_rows])
        order = np.argsort(fs)
        axis.plot(
            fs[order],
            np.array([number(row["floor_nmae"]) for row in floor_rows])[order],
            color=fcolor, linestyle=fdash, marker=fmarker, markersize=3.5,
            markerfacecolor="white", markeredgecolor=fcolor, markeredgewidth=1.0,
            label="interpolation floor", zorder=3,
        )

    axis.axvline(sigma_ref, color=INK, linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
    axis.annotate(
        r"$\sigma_{\mathrm{ref}}$",
        (sigma_ref, 1.0),
        xycoords=("data", "axes fraction"),
        textcoords="offset points",
        xytext=(3, -9),
        color=INK,
        fontsize=7.5,
        zorder=6,
    )
    nodes = (
        np.asarray(axis_nodes, dtype=float)
        if axis_nodes is not None
        else np.asarray([sigmas[0], sigmas[-1]], dtype=float)
    )
    for node in nodes:
        axis.axvline(node, color=MUTED, linewidth=0.6, zorder=2)

    axis.set_xlabel(
        r"$\sigma$ (all five assets); thin verticals are $\sigma$-axis nodes"
    )
    axis.set_ylabel("NMAE")
    # Twenty sample points across a cell of width ~0.018 would print twenty
    # four-decimal labels on top of one another; five is what fits.
    axis.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    axis.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{value:.3f}")
    )
    log_ticks_with_subs(axis, "y")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        handlelength=2.4,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    axis.margins(y=0.14)
    return save(fig, out_dir, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="phase4")
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "figures")
    parser.add_argument("--sigma-ref", type=float, default=0.20)
    parser.add_argument(
        "--figure1-only",
        type=int,
        nargs="+",
        default=None,
        metavar="D_SIGMA",
        help=(
            "restrict figure 1 to these d_sigma values and write it under a "
            "suffixed name, leaving the full-sweep figure untouched. Filters "
            "rows that were already measured; nothing is refitted, and the "
            "curves are identical to the corresponding ones in the full figure."
        ),
    )
    args = parser.parse_args()

    source, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = read_csv(source / f"{args.tag}_resultats.csv")
    ranks = read_csv(source / f"{args.tag}_profils_rangs.csv")
    aggregate = read_csv(source / f"{args.tag}_agrege.csv")

    written = []
    if args.figure1_only:
        kept = {int(value) for value in args.figure1_only}
        subset = [row for row in results if int(row["d_sigma"]) in kept]
        missing = kept - {int(row["d_sigma"]) for row in subset}
        if missing:
            raise SystemExit(f"no rows for d_sigma={sorted(missing)} in {args.tag}")
        suffix = "_".join(f"d{value}" for value in sorted(kept))
        written += figure_statistical_cost(
            subset, out_dir, f"figure1_statistical_cost_{suffix}"
        )
    else:
        written += figure_statistical_cost(
            results, out_dir, "figure1_statistical_cost"
        )
    written += figure_rank_profile(ranks, results, out_dir, "figure2_rank_profile")
    written += figure_dimensional_slope(
        aggregate, out_dir, "figure3_dimensional_slope"
    )

    cell_path = source / f"{args.tag}_sigma_cell_nmae.csv"
    if cell_path.exists():
        floor_path = source / f"{args.tag}_sigma_cell_floor.csv"
        written += figure_sigma_cell(
            read_csv(cell_path),
            read_csv(floor_path) if floor_path.exists() else [],
            float(args.sigma_ref),
            out_dir,
            "figureS1_sigma_cell",
        )

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
