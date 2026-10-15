r"""Assemble ``rapport_experiences.md`` from the artefacts the phases wrote.

The report is generated, never hand-typed, so no number in it can differ from
the number in the CSV it came from. Sections that have no artefact yet say
``NOT RUN`` rather than being omitted, and the observations section is
descriptive by construction: it states what the tables contain and leaves the
reading of it to the manuscript.

    .venv/Scripts/python.exe experiments/05_volatility_cost/build_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    ANOVA_SAMPLES,
    D_SIGMA_SWEEP,
    MAX_SWEEPS,
    OUTPUT_DIR,
    PHASE1_TOLERANCE,
    PHASE2_GATE,
    PRICING_GRID,
    RANK_INCREMENT,
    RISK_GRID,
    SEEDS,
    SIGMA_BOUNDS,
    SIGMA_OFF_NODE,
    SIGMA_REF,
    TT_BUDGETS,
    UNRESOLVED_FACTOR,
)

NOT_RUN = "`NOT RUN`"


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value, spec=".4g"):
    if value is None or value == "":
        return "--"
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    return format(value, spec)


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def section_manifest(manifest, phase1, phase2) -> str:
    if manifest is None:
        return f"## 1. Manifeste de run\n\n{NOT_RUN}\n"
    frozen = manifest["frozen"]
    retained = phase1.get("retained_n_sigma") if phase1 else None
    lines = [
        "## 1. Manifeste de run",
        "",
        "### Environnement",
        "",
        table(
            ["item", "value"],
            [
                ["commit", f"`{manifest['commit']}`"],
                ["branch", f"`{manifest['branch']}`"],
                [
                    "working tree",
                    "clean" if not manifest["dirty_files"] else
                    f"dirty ({len(manifest['dirty_files'].splitlines())} files)",
                ],
                ["python", manifest["python"]],
                ["numpy", manifest["numpy"]],
                ["scipy", manifest["scipy"]],
                ["teneva", manifest["teneva"]],
                ["platform", manifest["platform"]],
                ["processor", manifest["processor"]],
                ["clock", "`time.perf_counter` (monotone)"],
            ],
        ),
        "",
        "### Paramètres gelés",
        "",
        table(
            ["parameter", "value"],
            [
                [
                    "grille principale",
                    (f"risk-hybrid `{RISK_GRID['mode']}`, "
                    f"`{RISK_GRID['spot_nodes']}^5 x {RISK_GRID['strike_nodes']} "
                    f"x {RISK_GRID['rate_nodes']} x {RISK_GRID['maturity_nodes']}` "
                    "(48 coeurs QTT)"),
                ],
                [
                    "grille de controle (6b)",
                    (f"`{PRICING_GRID['mode']}`, "
                    f"`{PRICING_GRID['spot_nodes']}^5 x {PRICING_GRID['strike_nodes']} "
                    f"x {PRICING_GRID['rate_nodes']} x {PRICING_GRID['maturity_nodes']}`"),
                ],
                ["oracle", "formule fermee du panier geometrique (deterministe)"],
                ["correlations", frozen["correlation"]],
                ["dividendes", frozen["dividends"]],
                ["cible de fit", frozen["fit_target"]],
                ["interpolant", frozen["interpolant"]],
                ["troncature SVD post-fit", "aucune (refusee par l'API)"],
                ["init ANOVA", f"`teneva.anova(r=2, order=2)`, {ANOVA_SAMPLES} tirages"],
                ["sweeps max", MAX_SWEEPS],
                ["increment de rang", f"`dr_min=1`, `dr_max={RANK_INCREMENT}`"],
                ["tolerance TT-cross", "aucune (`e` / `e_vld` non passes)"],
                ["sigma_ref", SIGMA_REF],
                ["domaine sigma", f"`{list(SIGMA_BOUNDS)}`"],
                ["sigma hors noeud", SIGMA_OFF_NODE],
                ["axes sigma", frozen["sigma_axis_scale"]],
                ["actifs promus", frozen["promoted_assets"]],
                ["d_sigma", f"`{list(D_SIGMA_SWEEP)}`"],
                ["budgets TT-cross", f"`{list(TT_BUDGETS)}`"],
                ["seeds", f"`{list(SEEDS)}`"],
                ["seed jeu de test (axes partages)", frozen["test_seed_shared"]],
                ["seed jeu de test (boite sigma)", frozen["test_seed_sigma_box"]],
                ["points de test", frozen["n_test_points"]],
                ["points de latence apparies", frozen["n_latency_points"]],
                ["seuil UNRESOLVED", f"NMAE > {UNRESOLVED_FACTOR:g} x plancher"],
                ["n_sigma retenu", retained if retained else NOT_RUN],
            ],
        ),
        "",
        "### Tests bloquants",
        "",
    ]
    if phase1:
        lines += [
            f"**Phase 1** — critere : plus petit `n_sigma` dont le plancher 13D "
            f"ne depasse pas le plancher 8D de plus de "
            f"{100 * (PHASE1_TOLERANCE - 1):.0f} %. "
            f"Retenu : **n_sigma = {retained}**."
            if retained
            else "**Phase 1** — aucun candidat ne satisfait le critere. ARRET.",
            "",
        ]
    else:
        lines += [f"**Phase 1** — {NOT_RUN}", ""]
    if phase2:
        lines += [f"**Phase 2** — {phase2['conclusion']}", ""]
    else:
        lines += [f"**Phase 2** — {NOT_RUN}", ""]
    return "\n".join(lines)


def section_phase1(rows, payload) -> str:
    if not rows:
        return f"### Phase 1 — choix de `n_sigma`\n\n{NOT_RUN}\n"
    body = table(
        ["n_sigma", "L_sigma", "coeurs QTT", "plancher 13D", "plancher 8D",
         "ratio", "critere", "retenu", "sigma_ref dans la maille"],
        [
            [
                row["n_sigma"], row["L_sigma"], row["qtt_cores"],
                fmt(row["floor_13d_nmae"], ".4e"),
                fmt(row["floor_8d_slice_nmae"], ".4e"),
                fmt(row["ratio_13d_over_8d"], ".4f"),
                "PASS" if row["satisfies_criterion"] == "True" else "FAIL",
                "**oui**" if row["retained"] == "True" else "non",
                fmt(row["cell_fraction_of_sigma_ref"], ".4f"),
            ]
            for row in rows
        ],
    )
    diag = payload.get("floor_8d_collapse_diagnostics", {}) if payload else {}
    note = (
        f"\nPlancher calcule par le chemin a stencil spot reduit : "
        f"{diag.get('effective_corners_per_query', '--')} coins effectifs "
        f"contre {diag.get('product_stencil_corners_per_query', '--')} pour le "
        f"stencil produit, residu de collapse "
        f"{fmt(diag.get('collapse_normalized_residual'), '.2e')} "
        f"(norme NMAE, seuil {fmt(diag.get('collapse_tolerance'), '.0e')}).\n"
    )

    # The three-sentence justification the protocol asks for, assembled from the
    # row that was retained so it cannot disagree with the table above it.
    kept = next((row for row in rows if row["retained"] == "True"), None)
    failed = [row for row in rows if row["satisfies_criterion"] != "True"]
    if kept:
        justification = (
            f"\n**Justification.** Les {len(failed)} candidats plus grossiers "
            f"echouent nettement — ratio "
            + ", ".join(fmt(row["ratio_13d_over_8d"], ".2f") for row in failed)
            + " pour n_sigma = "
            + ", ".join(str(row["n_sigma"]) for row in failed)
            + f" — donc les axes de volatilite relevent bel et bien le plancher "
            f"d'interpolation de la grille en dessous de {kept['n_sigma']} noeuds. "
            f"A n_sigma = {kept['n_sigma']} le plancher 13D vaut "
            f"{fmt(kept['floor_13d_nmae'], '.3e')} contre "
            f"{fmt(kept['floor_8d_slice_nmae'], '.3e')} pour la grille 8D sur la "
            f"tranche commune, soit un ratio de "
            f"{fmt(kept['ratio_13d_over_8d'], '.4f')}, dans la tolerance de "
            f"{100 * (PHASE1_TOLERANCE - 1):.0f} %. Le critere retenant le plus "
            f"petit candidat conforme, n_sigma = {kept['n_sigma']} est fige pour "
            f"toutes les phases suivantes, y compris son placement defavorable de "
            f"sigma_ref dans la maille "
            f"({fmt(kept['cell_fraction_of_sigma_ref'], '.4f')} de la maille, "
            f"soit a {fmt(float(kept['sigma_axis_step']) * float(kept['cell_fraction_of_sigma_ref']), '.2e')} "
            f"du noeud inferieur), que le balayage de maille de la phase 4 "
            f"quantifie.\n"
        )
    else:
        justification = (
            "\n**Justification.** Aucun candidat ne satisfait le critere : les "
            "axes de volatilite relevent le plancher d'interpolation de la "
            "grille a toutes les resolutions testees. L'experience s'arrete ici.\n"
        )
    return f"### Phase 1 — choix de `n_sigma`\n\n{body}\n{note}{justification}"


def section_phase2(rows, payload) -> str:
    if not rows or not payload:
        return f"### Phase 2 — coincidence des noeuds\n\n{NOT_RUN}\n"
    worst = payload["worst_by_d_sigma"]
    body = table(
        ["d_sigma", "max D_2", "max D_inf", "verdict"],
        [
            [
                key,
                fmt(value["max_D_2"], ".3e"),
                fmt(value["max_D_inf"], ".3e"),
                "PASS" if max(value["max_D_2"], value["max_D_inf"]) < PHASE2_GATE
                else "FAIL",
            ]
            for key, value in sorted(worst.items(), key=lambda kv: int(kv[0]))
        ],
    )
    return (
        "### Phase 2 — coincidence des noeuds\n\n"
        f"{len(payload['rates'])} taux x {len(payload['maturities'])} maturites, "
        f"sigma = {SIGMA_REF}, seuil `{PHASE2_GATE:.0e}`.\n\n{body}\n\n"
        f"{payload['conclusion']}\n"
    )


def section_phase3(payload) -> str:
    if not payload:
        return f"### Phase 3 — dry run\n\n{NOT_RUN}\n"
    body = table(
        ["d_sigma", "coeurs", "appels realises", "stop", "rang effectif",
         "parametres TT", "fit (s)"],
        [
            [
                fit["d_sigma"], fit["qtt_cores"], fit["oracle_calls_realized"],
                fit["cross_stop"], fmt(fit["effective_rank"], ".2f"),
                fit["tt_parameters"], fmt(fit["fit_seconds"], ".1f"),
            ]
            for fit in payload["fits"]
        ],
    )
    return (
        "### Phase 3 — dry run\n\n"
        "Chiffres jetes : ce run ne sert qu'a verifier que la chaine s'execute "
        "de bout en bout et que tous les compteurs sont collectes.\n\n"
        f"{body}\n\n"
        f"Tous les compteurs requis collectes : "
        f"**{'oui' if payload['all_counters_collected'] else 'non'}**.\n"
    )


PRE_REGISTERED = "pre-registered"


def _phase4_table(aggregate_rows) -> str:
    return table(
        ["d_sigma", "budget", "appels realises", "coeurs", "log2 entrees",
         "rang effectif", "rang max", "max r bloc sigma", "r frontiere",
         "parametres TT", "NMAE tranche", "plancher", "x plancher",
         "NMAE pave sigma", "NMAE hors noeud", "statut", "fit (s)", "s/requete"],
        [
            [
                row["d_sigma"], row["budget_nominal"],
                fmt(row["oracle_calls_realized_mean"], ".0f"),
                row["qtt_cores"], row["log2_logical_entries"],
                (f"{fmt(row['effective_rank_mean'], '.2f')} "
                f"± {fmt(row['effective_rank_std'], '.2f')}"),
                fmt(row["maximum_rank_mean"], ".1f"),
                fmt(row["sigma_block_internal_max_rank_mean"], ".1f"),
                fmt(row["sigma_boundary_max_rank_mean"], ".1f"),
                fmt(row["tt_parameters_mean"], ".0f"),
                (f"{fmt(row['nmae_common_slice_mean'], '.3e')} "
                f"± {fmt(row['nmae_common_slice_std'], '.1e')}"),
                fmt(row["floor_nmae_common_slice"], ".3e"),
                fmt(row["nmae_over_floor_common_slice_mean"], ".2f"),
                fmt(row["nmae_sigma_box_mean"], ".3e"),
                fmt(row["nmae_off_node_mean"], ".3e"),
                f"{row['n_unresolved']}/{row['n_seeds']} UNRESOLVED"
                if int(row["n_unresolved"]) else "resolved",
                fmt(row["fit_seconds_mean"], ".0f"),
                fmt(row["batch_seconds_per_query_mean"], ".2e"),
            ]
            for row in sorted(
                aggregate_rows,
                key=lambda r: (int(r["d_sigma"]), int(r["budget_nominal"])),
            )
        ],
    )


def section_phase4(rows, aggregate_rows, payload) -> str:
    if not rows or not aggregate_rows:
        return f"### Phase 4 — balayage en `d_sigma`\n\n{NOT_RUN}\n"

    # A budget added after the ladder had been inspected is not the same kind of
    # number as one fixed before the first fit. Both are reported; they are not
    # merged into one table, and the post-hoc one is named as such.
    tiers = {row.get("budget_tier", PRE_REGISTERED) for row in aggregate_rows}
    pre = [
        row for row in aggregate_rows
        if row.get("budget_tier", PRE_REGISTERED) == PRE_REGISTERED
    ]
    post = [
        row for row in aggregate_rows
        if row.get("budget_tier", PRE_REGISTERED) != PRE_REGISTERED
    ]
    main = _phase4_table(pre)
    extension = ""
    if post and len(tiers) > 1:
        post_budgets = sorted({int(row["budget_nominal"]) for row in post})
        extension = (
            "\n\n#### Palier post-hoc (hors pre-enregistrement)\n\n"
            f"Budget(s) `{post_budgets}`, ajoute(s) **apres** que l'echelle "
            "pre-enregistree ait ete executee et examinee, et applique(s) "
            "uniformement a tous les `d_sigma` — pas a une configuration "
            "isolee. Les planchers d'interpolation sont ceux de la phase 4, "
            "reutilises et non recalcules, donc les deux paliers se comparent "
            "au meme plancher exact. Ces lignes ne doivent pas etre presentees "
            "comme pre-enregistrees.\n\n"
            + _phase4_table(post)
        )

    checks = {
        "shared_core_order_preserved": all(
            row["shared_core_order_preserved"] == "True" for row in rows
        ),
        "sigma_cores_contiguous": all(
            row["sigma_cores_contiguous"] == "True" for row in rows
        ),
        "sigma_block_at_chain_tail": all(
            row["sigma_block_at_chain_tail"] == "True" for row in rows
        ),
    }
    verification = "\n".join(
        f"- `{key}` : **{'verifie sur les ' + str(len(rows)) + ' fits' if value else 'ECHEC'}**"
        for key, value in checks.items()
    )
    total = payload.get("total_wall_seconds") if payload else None
    return (
        "### Phase 4 — balayage en `d_sigma`\n\n"
        f"{len(rows)} fits au total. Moyennes ± ecarts-types sur les "
        f"{len({row['seed'] for row in rows})} seeds. Rang et NMAE sur la meme "
        "ligne ; un fit `UNRESOLVED` reste dans le tableau et est exclu de toute "
        "conclusion sur la compressibilite.\n\n"
        f"{main}{extension}\n\n"
        "**Verification de l'ordre des modes** (chaine `d_sigma=0` = chaine "
        "`d_sigma=k` privee des coeurs sigma) :\n\n"
        f"{verification}\n"
        + (f"\nTemps total : {total / 3600:.2f} h.\n" if total else "\n")
    )


FIGURE_NAMES = (
    ("figure1_statistical_cost", "Figure 1 — cout statistique"),
    ("figure2_rank_profile", "Figure 2 — profil de rang"),
    ("figure3_dimensional_slope", "Figure 3 — pente dimensionnelle"),
    ("figureS1_sigma_cell", "Figure S1 — NMAE sur une maille sigma"),
)


def _figure_table(figure_dir: Path) -> str:
    rows = []
    for stem, label in FIGURE_NAMES:
        pdf = figure_dir / f"{stem}.pdf"
        svg = figure_dir / f"{stem}.svg"
        rows.append([
            label,
            f"`{pdf.relative_to(pdf.parents[2])}`" if pdf.exists() else NOT_RUN,
            f"`{svg.relative_to(svg.parents[2])}`" if svg.exists() else NOT_RUN,
        ])
    return table(["figure", "PDF", "SVG"], rows)


def section_phase5(figure_sets) -> str:
    """One figure table per set, so a post-hoc set is never mistaken for the other.

    ``figure_sets`` is a sequence of ``(directory, caption)``. Sets whose
    directory does not exist are dropped rather than listed as missing, since a
    run without a post-hoc tier legitimately has only one set.
    """
    parts = ["## 3. Figures", "", "Format vectoriel, sans titre integre.", ""]
    present = [(path, caption) for path, caption in figure_sets if path.exists()]
    if not present:
        return "\n".join(parts + [NOT_RUN, ""])
    for path, caption in present:
        if len(present) > 1:
            parts += [f"**{caption}**", ""]
        parts += [_figure_table(path), ""]
    return "\n".join(parts)


def section_diagnostics(source: Path, specifications) -> str:
    """Runs that vary a frozen parameter, kept out of the main tables.

    Changing ``n_sigma`` changes the grid, so such a run is not another row of
    the ``d_sigma`` sweep - its interpolation floor and its core count are
    different. It answers a question about the sweep without being part of it,
    and is reported that way.
    """
    if not specifications:
        return ""
    parts = ["### Runs diagnostiques (parametre gele modifie)", ""]
    for specification in specifications:
        tag, _, label = specification.partition(":")
        rows = load_csv(source / f"{tag}_agrege.csv")
        payload = load_json(source / f"{tag}_run.json")
        parts += [f"**{label or tag}** (`{tag}_*`)", ""]
        if not rows:
            parts += [NOT_RUN, ""]
            continue
        if payload:
            parts += [
                (f"n_sigma = {payload['n_sigma']}, "
                 f"{payload['n_test_points']} points de test "
                 "(axes partages strictement identiques a ceux du balayage "
                 "principal)."),
                "",
            ]
        parts += [_phase4_table(rows), ""]
    return "\n".join(parts)


def section_phase6(rows_6a, rows_6b) -> str:
    if not rows_6a and not rows_6b:
        return f"### Phase 6 — controles\n\n{NOT_RUN}\n"
    out = ["### Phase 6 — controles", ""]
    if rows_6a:
        out += [
            "**6a — sensibilite a l'ordre des modes.**",
            "",
            table(
                ["ordre", "rang effectif", "parametres TT", "max r bloc sigma",
                 "NMAE tranche", "seeds", "UNRESOLVED"],
                [
                    [
                        row["sigma_block"],
                        (f"{fmt(row['effective_rank_mean'], '.2f')} "
                        f"± {fmt(row['effective_rank_std'], '.2f')}"),
                        fmt(row["tt_parameters_mean"], ".0f"),
                        fmt(row["sigma_block_internal_max_rank_mean"], ".1f"),
                        (f"{fmt(row['nmae_common_slice_mean'], '.3e')} "
                        f"± {fmt(row['nmae_common_slice_std'], '.1e')}"),
                        row["n_seeds"], row["n_unresolved"],
                    ]
                    for row in rows_6a
                ],
            ),
            "",
        ]
    else:
        out += [f"**6a** — {NOT_RUN}", ""]
    if rows_6b:
        out += [
            "**6b — controle sur la grille de pricing.**",
            "",
            table(
                ["d_sigma", "rang effectif", "parametres TT", "max r bloc sigma",
                 "NMAE tranche", "seeds", "UNRESOLVED"],
                [
                    [
                        row["d_sigma"],
                        (f"{fmt(row['effective_rank_mean'], '.2f')} "
                        f"± {fmt(row['effective_rank_std'], '.2f')}"),
                        fmt(row["tt_parameters_mean"], ".0f"),
                        fmt(row["sigma_block_internal_max_rank_mean"], ".1f"),
                        (f"{fmt(row['nmae_common_slice_mean'], '.3e')} "
                        f"± {fmt(row['nmae_common_slice_std'], '.1e')}"),
                        row["n_seeds"], row["n_unresolved"],
                    ]
                    for row in rows_6b
                ],
            ),
            "",
        ]
    else:
        out += [f"**6b** — {NOT_RUN}", ""]
    return "\n".join(out)


def section_observations(aggregate_rows, cell_sweeps, cell_floor_rows) -> str:
    """Strictly descriptive: what the numbers are, not what they mean."""
    if not aggregate_rows:
        return f"## 4. Observations\n\n{NOT_RUN}\n"
    lines = ["## 4. Observations", "",
             ("Section descriptive. Aucune interpretation, aucune redaction "
             "destinee au manuscrit."), ""]
    budgets = sorted({int(row["budget_nominal"]) for row in aggregate_rows})
    largest = budgets[-1]
    top = sorted(
        (row for row in aggregate_rows if int(row["budget_nominal"]) == largest),
        key=lambda row: int(row["d_sigma"]),
    )

    def series(key):
        return [
            (int(row["d_sigma"]), float(row[f"{key}_mean"]))
            for row in top
            if row[f"{key}_mean"] not in ("NOT APPLICABLE", "", None)
        ]

    rank = series("sigma_block_internal_max_rank")
    params = series("tt_parameters")
    nmae = series("nmae_common_slice")
    erank = series("effective_rank")

    if rank:
        lines.append(
            f"- Au plus grand budget ({largest}), le rang de lien maximal interne "
            f"au bloc sigma vaut "
            + ", ".join(f"{value:.1f} a d_sigma={d}" for d, value in rank)
            + "."
        )
    if erank:
        lines.append(
            "- Le rang effectif global vaut "
            + ", ".join(f"{value:.2f} a d_sigma={d}" for d, value in erank)
            + "."
        )
    if params:
        first, last = params[0], params[-1]
        ratio = last[1] / first[1] if first[1] else float("nan")
        lines.append(
            f"- Le nombre de parametres TT passe de {first[1]:.0f} "
            f"(d_sigma={first[0]}) a {last[1]:.0f} (d_sigma={last[0]}), "
            f"facteur {ratio:.2f}, tandis que le nombre d'entrees logiques passe "
            f"de 2^{top[0]['log2_logical_entries']} a "
            f"2^{top[-1]['log2_logical_entries']}."
        )
    if nmae:
        lines.append(
            "- La NMAE sur la tranche commune au plus grand budget vaut "
            + ", ".join(f"{value:.3e} a d_sigma={d}" for d, value in nmae)
            + "."
        )
    unresolved = [
        (int(row["d_sigma"]), int(row["budget_nominal"]), int(row["n_unresolved"]),
         int(row["n_seeds"]))
        for row in aggregate_rows
        if int(row["n_unresolved"]) > 0
    ]
    if unresolved:
        lines.append(
            "- Configurations comportant au moins un fit `UNRESOLVED` : "
            + "; ".join(
                f"d_sigma={d} budget={b} ({n}/{total})"
                for d, b, n, total in sorted(unresolved)
            )
            + "."
        )
    else:
        lines.append("- Aucun fit n'a ete marque `UNRESOLVED`.")

    for cell_rows, cell_label in cell_sweeps:
        if not cell_rows:
            continue
        values = {}
        for row in cell_rows:
            values.setdefault(float(row["sigma"]), []).append(float(row["nmae"]))
        sigmas = sorted(values)
        means = np.array([np.mean(values[s]) for s in sigmas])
        at_ref = int(np.argmin(np.abs(np.array(sigmas) - SIGMA_REF)))
        rank_of_ref = int(np.sum(means < means[at_ref])) + 1
        lines.append(f"- **{cell_label}**")
        lines.append(
            f"- Balayage de la maille sigma ({len(sigmas)} valeurs de "
            f"{sigmas[0]:.6f} a {sigmas[-1]:.6f}, bornes = noeuds de grille) : "
            f"la NMAE a sigma = {sigmas[at_ref]:.6f} vaut {means[at_ref]:.4e}, "
            f"soit le rang {rank_of_ref} sur {len(sigmas)} par valeur croissante ; "
            f"l'intervalle sur la maille est [{means.min():.4e}, {means.max():.4e}], "
            f"mediane {np.median(means):.4e}. "
            f"Le minimum de la maille est atteint a sigma = "
            f"{sigmas[int(np.argmin(means))]:.6f}. Amplitude relative sur la "
            f"maille : {means.max() / means.min():.3f}x."
        )
    if cell_floor_rows:
        fs = sorted((float(r["sigma"]), float(r["floor_nmae"])) for r in cell_floor_rows)
        fvals = np.array([v for _, v in fs])
        fat = int(np.argmin(np.abs(np.array([s for s, _ in fs]) - SIGMA_REF)))
        lines.append(
            f"- Plancher d'interpolation sur la meme maille : "
            f"{fvals[fat]:.4e} a sigma_ref, intervalle "
            f"[{fvals.min():.4e}, {fvals.max():.4e}], minimum a sigma = "
            f"{fs[int(np.argmin(fvals))][0]:.6f}."
        )
    return "\n".join(lines) + "\n"


def section_anomalies(rows, phase1, phase2, phase3, extra) -> str:
    lines = ["## 5. Anomalies et echecs", ""]
    found = []
    if rows:
        unresolved = [row for row in rows if row["resolution_status"] == "UNRESOLVED"]
        if unresolved:
            found.append(
                f"- {len(unresolved)} fits sur {len(rows)} depassent le plancher "
                f"d'interpolation de leur grille d'un facteur superieur a "
                f"{UNRESOLVED_FACTOR:g} et sont marques `UNRESOLVED`. Ils restent "
                "dans les tableaux et sont exclus de toute conclusion sur la "
                "compressibilite."
            )
        stops = {row["cross_stop"] for row in rows}
        if stops - {"m"}:
            for stop in sorted(stops - {"m"}):
                count = sum(1 for row in rows if row["cross_stop"] == stop)
                found.append(
                    f"- {count} fits se sont arretes sur `stop={stop}` et non sur "
                    "l'epuisement du budget."
                )
        broken = [row for row in rows if row["shared_core_order_preserved"] != "True"]
        if broken:
            found.append(
                f"- {len(broken)} fits ont echoue la verification d'additivite de "
                "la chaine. ECART AU PROTOCOLE."
            )
        # A seed whose NMAE is far from its siblings is reported, never dropped.
        by_config = {}
        for row in rows:
            by_config.setdefault(
                (row["d_sigma"], row["budget_nominal"]), []
            ).append(row)
        for (d_sigma, budget), group in sorted(by_config.items()):
            values = np.array([float(row["nmae_common_slice"]) for row in group])
            if values.size < 3:
                continue
            median = float(np.median(values))
            spread = np.abs(values - median) / max(median, 1e-300)
            if spread.max() > 1.0:
                worst = group[int(np.argmax(spread))]
                found.append(
                    f"- d_sigma={d_sigma}, budget={budget} : la seed "
                    f"{worst['seed']} s'ecarte de la mediane des seeds de "
                    f"{spread.max() * 100:.0f} % en NMAE "
                    f"({float(worst['nmae_common_slice']):.3e} contre "
                    f"{median:.3e}). Rapportee, non exclue."
                )
    if phase1 and phase1.get("retained_n_sigma") is None:
        found.append("- Phase 1 : aucun `n_sigma` <= 32 ne satisfait le critere.")
    if phase2 and not phase2.get("passed"):
        found.append("- Phase 2 : le test de coincidence des noeuds a echoue.")
    if phase3 and not phase3.get("all_counters_collected"):
        found.append("- Phase 3 : des compteurs sont absents du dry run.")
    found.extend(extra or [])

    if not found:
        lines.append(
            "Rien n'a ete observe : aucun run non converge, aucune seed "
            "divergente, aucun ecart au protocole."
        )
    else:
        lines.extend(found)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tag", default="phase4")
    parser.add_argument(
        "--output", type=Path,
        default=OUTPUT_DIR / "rapport_experiences.md",
    )
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument(
        "--cell-sweep",
        action="append",
        default=[],
        metavar="TAG:LABEL",
        help=(
            "an additional sigma-cell sweep to report alongside the "
            "pre-registered one. Repeatable."
        ),
    )
    parser.add_argument(
        "--diagnostic",
        action="append",
        default=[],
        metavar="TAG:LABEL",
        help=(
            "a run that varies a parameter the protocol froze, reported in its "
            "own section rather than alongside the main sweep. Repeatable."
        ),
    )
    args = parser.parse_args()

    source = Path(args.input_dir)
    phase1 = load_json(source / "phase1_choix_nsigma.json")
    phase1_rows = load_csv(source / "phase1_choix_nsigma.csv")
    phase2 = load_json(source / "phase2_coincidence_noeuds.json")
    phase2_rows = load_csv(source / "phase2_coincidence_noeuds.csv")
    phase3 = load_json(source / "phase3_dry_run.json")
    phase4 = load_json(source / f"{args.tag}_run.json")
    phase4_rows = load_csv(source / f"{args.tag}_resultats.csv")
    phase4_agg = load_csv(source / f"{args.tag}_agrege.csv")
    # The cell sweep can exist at more than one budget: the pre-registered one
    # and any later re-run. All of them are reported, each labelled with the
    # budget it was measured at, because a sweep taken on unresolved fits and one
    # taken on resolved fits are not the same measurement.
    cell_sweeps = []
    for tag, label in [
        (args.tag, f"balayage pre-enregistre ({args.tag})"),
        *[
            (spec.split(":", 1)[0], spec.split(":", 1)[1] if ":" in spec else spec)
            for spec in args.cell_sweep
        ],
    ]:
        rows = load_csv(source / f"{tag}_sigma_cell_nmae.csv")
        if rows:
            budgets = sorted({int(row["budget_nominal"]) for row in rows})
            cell_sweeps.append((rows, f"{label}, budget {budgets}"))
    cell_floor = load_csv(source / f"{args.tag}_sigma_cell_floor.csv")
    control_6a = load_csv(source / "phase6a_ordre_modes.csv")
    control_6b = load_csv(source / "phase6b_grille_pricing.csv")

    manifest = None
    for payload in (phase4, phase2, phase1, phase3):
        if payload and "manifest" in payload:
            manifest = payload["manifest"]
            break

    parts = [
        ("# Rapport d'experiences — cout tensoriel en fonction du nombre "
        "de facteurs de volatilite"),
        "",
        ("Genere par `experiments/05_volatility_cost/build_report.py` a partir des seuls "
        "artefacts de `results/05_volatility_cost/`. Aucun chiffre n'est saisi a la main. Une "
        f"mesure absente s'ecrit {NOT_RUN}."),
        "",
        section_manifest(manifest, phase1, phase2),
        "## 2. Tableaux",
        "",
        section_phase1(phase1_rows, phase1),
        section_phase2(phase2_rows, phase2),
        section_phase3(phase3),
        section_phase4(phase4_rows, phase4_agg, phase4),
        section_diagnostics(source, args.diagnostic),
        section_phase6(control_6a, control_6b),
        section_phase5(
            (
                (
                    source / "figures",
                    ("Echelle de budgets pre-enregistree, telle que produite "
                     "par la phase 4"),
                ),
                (
                    source / "figures_extended",
                    "Memes figures, palier post-hoc inclus",
                ),
            )
        ),
        section_observations(phase4_agg, cell_sweeps, cell_floor),
        section_anomalies(phase4_rows, phase1, phase2, phase3, args.note),
    ]
    Path(args.output).write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
