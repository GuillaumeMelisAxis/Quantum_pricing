r"""Figure E3 - accuracy vs. cost per query: QTT surrogate against Monte Carlo.

Draws the query-level benchmark of table 4 as a log-log accuracy/cost plot: NMAE
on the abscissa, seconds per query on the ordinate, one polyline per method.
Nothing is interpreted on the figure - no guides, no callouts - the reading is
left to the manuscript.

Typography, palette and mark style are those of
``scripts/experiments/phase5_figures.py``, so this figure sits beside figures 1-3
in ``experiments/figures`` without a visible change of hand: DejaVu Sans at 8pt,
7.5pt legend with no frame, hollow markers on 0.6pt axes, and series identity
carried by dash and marker as well as hue so the plate survives greyscale.

The two hues are green and orange-red, both from the same reference palette
(slots 6 and 8). That pair sits in the CVD warn band - worst simulated Delta E
7.2 (protan), against a target of 8 and a hard floor of 6 - so the redundant
dash, marker and direct budget labels are not decoration here, they are what
makes the pair legal. Note that green against the palette's *orange* (#eb6834)
scores 3.2 and is unusable: for a protanope the two are one colour.

Writes PDF and SVG (vector, for \includegraphics) plus a PNG preview into
``experiments/figures``.

    .venv/Scripts/python.exe scripts/make_fig_e3_accuracy_vs_time.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.ticker as mticker

mpl.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Data - table 4 of the paper. Points are ordered by decreasing budget, which
# is also increasing NMAE, so each polyline reads left to right.
# --------------------------------------------------------------------------
QTT = [  # (label, NMAE(V), seconds per query)
    ("150k", 0.0052, 3.5e-4),
    ("60k", 0.0264, 3.5e-4),
    ("20k", 0.1386, 1.4e-4),
]

MC = [  # (label, NMAE(V), seconds per query)
    ("262,144", 0.0134, 1.4e-1),
    ("65,536", 0.0277, 3.7e-2),
    ("16,384", 0.0516, 6.6e-3),
    ("4,096", 0.1077, 8.0e-4),
]

# --------------------------------------------------------------------------
# Style - lifted from phase5_figures.py so the plates match.
# --------------------------------------------------------------------------
# (color, dash, marker), assigned once, following the method rather than its rank.
STYLE = {
    "qtt": ("#008300", (0, ()), "o"),
    "mc": ("#e34948", (0, (7, 2, 1.5, 2, 1.5, 2)), "v"),
}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

# The manuscript caption names the figure; phase5 draws no titles for that
# reason. Flip this to False to follow that convention exactly.
DRAW_TITLE = True

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

# Offsets in points, hand-placed so no label touches a marker, a line or another
# label. label -> (dx, dy, ha, va).
POINT_LABELS = {
    "150k": (0, -11, "center", "top"),
    "60k": (0, -11, "center", "top"),
    "20k": (6, 4, "left", "bottom"),
    "262,144": (7, 3, "left", "bottom"),
    "65,536": (7, 3, "left", "bottom"),
    "16,384": (7, 3, "left", "bottom"),
    "4,096": (7, 3, "left", "bottom"),
}


def series(axis, rows, slot, label):
    """One method: connected polyline, hollow markers, per-point budget labels."""
    color, dash, marker = STYLE[slot]
    xs = [nmae for _, nmae, _ in rows]
    ys = [seconds for _, _, seconds in rows]
    axis.plot(
        xs,
        ys,
        color=color,
        linestyle=dash,
        marker=marker,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.1,
        label=label,
        zorder=3,
    )
    for (text, x, y) in rows:
        dx, dy, ha, va = POINT_LABELS[text]
        axis.annotate(
            text,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            fontsize=7,
            color=MUTED,
            zorder=4,
        )


def main() -> int:
    fig, axis = plt.subplots(figsize=(6.0, 4.6))
    axis.set_xscale("log")
    axis.set_yscale("log")

    series(axis, QTT, "qtt", "QTT surrogate")
    series(axis, MC, "mc", "Monte Carlo")

    axis.set_xlim(0.0035, 0.26)
    axis.set_ylim(8e-5, 3.2e-1)

    # The NMAE range spans well under two decades, so decade-only ticks would
    # leave a single labelled gridline; 1/2/5 per decade is the density that
    # keeps the abscissa readable as a scale.
    axis.xaxis.set_major_locator(
        mticker.FixedLocator([0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    )
    axis.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{value * 100:g}%")
    )
    axis.xaxis.set_minor_formatter(mticker.NullFormatter())

    axis.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    axis.yaxis.set_minor_formatter(mticker.NullFormatter())

    axis.grid(True, which="major", axis="both")
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    axis.set_xlabel("NMAE(V)")
    axis.set_ylabel("time per query (s)")
    if DRAW_TITLE:
        axis.set_title("Accuracy vs. cost per query, QTT surrogate vs. Monte Carlo")

    # The curves leave the upper right empty, so the legend sits inside the axes
    # rather than below them as in the column-width plates.
    axis.legend(loc="upper right", handlelength=2.6, borderaxespad=0.6)

    out_dir = Path(__file__).resolve().parents[1] / "experiments" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, dpi in (("pdf", 300), ("svg", None), ("png", 200)):
        path = out_dir / f"e3_accuracy_vs_time.{suffix}"
        if dpi is None:
            fig.savefig(path)
        else:
            fig.savefig(path, dpi=dpi)
        written.append(path)
    plt.close(fig)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
