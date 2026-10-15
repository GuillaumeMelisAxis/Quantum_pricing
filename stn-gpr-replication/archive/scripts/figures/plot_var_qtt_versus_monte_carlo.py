r"""VaR 1 jour: l'ajustement converge, la VaR n'en herite pas.

Deux panneaux - QTT 13-D a gauche, Monte Carlo imbrique a droite - et deux
series par panneau, toutes deux des erreurs relatives, donc un seul axe:

  * l'erreur de prix du portefeuille, en pourcentage de la valeur du livre;
  * l'erreur de VaR 95%, en pourcentage de la VaR de reference, avec le nuage
    des trois graines independantes dont la ligne est la moyenne quadratique.

C'est la comparaison des deux series qui porte le resultat, et elle est en
miroir d'un panneau a l'autre. A gauche, l'erreur de prix du QTT descend
proprement, d'un facteur 180, mais son erreur de risque oscille: le budget le
plus genereux a la meilleure erreur de prix et une mesure de risque plusieurs
fois moins bonne qu'au budget precedent. A droite, c'est l'inverse: l'erreur de
prix du Monte Carlo n'est pas monotone - c'est la taille d'un biais commun tire
au hasard, que la difference ``V(s) - V(0)`` retranche - tandis que son erreur
de risque descend proprement. Dans aucun des deux cas l'erreur de prix ne
predit l'erreur de risque.

La raison n'est pas que l'ajustement se degrade - il ne se degrade jamais. Une
VaR est un quantile de ``V(s) - V(0)``, une difference prise sur un deplacement
d'environ 1% en spot et d'un jour en maturite; seule la part de l'erreur qui
*varie* sur cette distance survit, le reste se retranche. Or TT-cross est un
interpolant: il est exact aux pivots choisis et ondule entre eux, et
``maxvol_rect`` re-selectionne ces pivots a chaque balayage. L'amplitude de
l'erreur baisse donc de facon monotone, mais sa texture spatiale est rebattue a
chaque budget, et c'est elle qui decide de la fraction qui survit a la
difference.

L'abscisse est un compte d'evaluations, pas un temps: le temps mesure sur cette
machine porte une dispersion de +-2.5x pour un travail identique. La comparaison
en secondes est dans le rapport, avec ses reserves.

Palette et typographie suivent ``scripts/experiments/phase5_figures.py``. Dans
chaque panneau la teinte designe la methode et le trait designe la metrique, si
bien que la planche survit au niveau de gris.

    .venv/Scripts/python.exe scripts/figures/plot_var_qtt_versus_monte_carlo.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.ticker as mticker
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = Path("results/var_qtt_basket_book_bench_es.json")
FIGURES = Path("experiments/figures")

# metrique -> (cle dans le JSON, niveau, etiquette, nom court, nom de fichier)
METRICS = {
    "var": ("var_relative_error", "0.95", "erreur de VaR 95%", "VaR",
            "var_qtt_versus_monte_carlo"),
    "es": ("es_relative_error", "0.975", "erreur d'ES 97,5%", "ES",
           "es_qtt_versus_monte_carlo"),
}

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
QTT_COLOR = "#eda100"
MC_COLOR = "#2a78d6"
VAR_DASH = (0, (5, 1.5))
PRICE_DASH = (0, (1, 1.6))

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.0,
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

COLUMN_WIDTH = 3.4


def _decimal_k(value: float) -> str:
    """Cross budgets are round decimals: 100 000 reads as 100k."""
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def _binary_k(value: float) -> str:
    """Path counts are powers of two, so 16 384 reads as 16k and not 16.384k."""
    return f"{value / 1024:g}k" if value >= 1024 else f"{value:g}"


def _panel(ax, rows, key, color, marker, title, xlabel, formatter, book_value,
           metric):
    field, level, label, _, _ = METRICS[metric]
    x = np.asarray([row[key] for row in rows], dtype=float)
    order = np.argsort(x)
    x = x[order]
    ordered = [rows[j] for j in order]

    var_error = np.asarray(
        [100.0 * row[field][level] for row in ordered], dtype=float
    )
    # L'erreur de prix est ramenee a la valeur du livre pour qu'elle partage
    # l'axe avec l'erreur de VaR: deux erreurs relatives, un seul axe.
    price_error = np.asarray(
        [100.0 * row["price_mean_absolute_error"] / book_value for row in ordered],
        dtype=float,
    )

    for position, row in zip(x, ordered):
        seeds = [
            100.0 * entry[field][level]
            for entry in row["var_relative_error_by_seed"]
        ]
        ax.plot(
            [position] * len(seeds), seeds,
            linestyle="none", marker=marker, markersize=3.0,
            markerfacecolor="none", markeredgecolor=color,
            markeredgewidth=0.7, alpha=0.5, clip_on=False, zorder=2,
        )
    ax.plot(
        x, var_error,
        color=color, linestyle=VAR_DASH, marker=marker,
        markerfacecolor="white", markeredgecolor=color,
        markeredgewidth=1.2, clip_on=False, zorder=3,
    )
    ax.plot(
        x, price_error,
        color=color, linestyle=PRICE_DASH, linewidth=1.1, marker=marker,
        markersize=2.8, markerfacecolor=color, markeredgecolor=color,
        alpha=0.85, clip_on=False, zorder=3,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", axis="both")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(title, color=INK)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels([formatter(value) for value in x])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ax.legend(
        handles=[
            Line2D([], [], color=color, linestyle=VAR_DASH, marker=marker,
                   markerfacecolor="white", markeredgecolor=color,
                   label=label),
            Line2D([], [], color=color, linestyle=PRICE_DASH, linewidth=1.1,
                   marker=marker, markersize=2.8, markerfacecolor=color,
                   markeredgecolor=color,
                   label="erreur de prix (% du livre)"),
            Line2D([], [], color=color, linestyle="none", marker=marker,
                   markersize=3.0, markerfacecolor="none",
                   markeredgecolor=color, alpha=0.5,
                   label="graines individuelles"),
        ],
        loc="lower left",
    )
    return x, var_error, price_error


def draw(results: dict, metric: str = "var") -> plt.Figure:
    book_value = float(results["book"]["base_value_closed_form"])

    fig, axes = plt.subplots(1, 2, figsize=(2 * COLUMN_WIDTH, 3.1), sharey=True)
    x_qtt, var_qtt, price_qtt = _panel(
        axes[0], results["qtt_budget_sweep"], "budget",
        QTT_COLOR, "s",
        "QTT 13-D, grille de risque bornee",
        "budget TT-cross (evaluations de l'oracle)",
        _decimal_k, book_value, metric,
    )
    _panel(
        axes[1], results["monte_carlo_path_sweep"], "inner_paths",
        MC_COLOR, "o",
        "Monte Carlo imbrique",
        "chemins internes par revalorisation",
        _binary_k, book_value, metric,
    )

    axes[0].set_ylabel("erreur relative (%)")
    axes[0].set_ylim(0.013, 30.0)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    # Le seul point qui merite d'etre nomme: celui qui a la meilleure erreur
    # de prix du balayage, s'il n'a pas aussi la meilleure erreur de risque.
    # Calcule, pas ecrit en dur: la planche ne doit rien affirmer que les
    # donnees du run ne disent pas.
    best_price = int(np.argmin(price_qtt))
    best_risk = int(np.argmin(var_qtt))
    if best_price != best_risk:
        ratio = var_qtt[best_price] / var_qtt[best_risk]
        axes[0].annotate(
            f"meilleure erreur de prix,\n{METRICS[metric][3]} "
            f"{ratio:.1f}x moins bonne",
            xy=(x_qtt[best_price], var_qtt[best_price]),
            xytext=(-4, 26), textcoords="offset points",
            ha="right", color=MUTED, fontsize=6.5,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6},
        )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--metric", choices=tuple(METRICS), default="var")
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    for name in ("qtt_budget_sweep", "monte_carlo_path_sweep"):
        rows = results.get(name)
        if not rows:
            raise SystemExit(f"{RESULTS} carries no {name}")
        if "var_relative_error_by_seed" not in rows[0]:
            raise SystemExit(
                f"{name} in {args.results} predates the multi-seed sweep; "
                "rerun the benchmark"
            )
        if METRICS[args.metric][0] not in rows[0]:
            raise SystemExit(
                f"{name} in {args.results} carries no {METRICS[args.metric][0]}; "
                "rerun the benchmark"
            )
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure = draw(results, args.metric)
    stem = METRICS[args.metric][4]
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(FIGURES / f"{stem}.{suffix}", dpi=200)
    print(f"wrote {FIGURES / (stem + '.pdf')} (+ svg, png)")


if __name__ == "__main__":
    main()
