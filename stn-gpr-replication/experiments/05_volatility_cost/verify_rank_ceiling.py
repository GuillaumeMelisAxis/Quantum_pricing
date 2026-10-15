r"""Check whether the reported bond ranks are structural or schedule-limited.

The shared-axis bond ranks plateau near 50 at the largest budget. Three things
could produce that plateau, and they mean very different things:

1. an absolute rank cap inside TT-cross,
2. the QTT matricization bound ``r_k <= min(2**k, 2**(d-k))``,
3. the rank-increment schedule: ``maxvol_rect`` adds at most ``dr_max`` rows per
   call, each bond is visited twice per sweep, and the initial ANOVA rank is 2,
   so ``r_k <= 2 + 2 * dr_max * nswp``.

Only the third is a property of how the fit was run rather than of the problem.
This script measures which one binds, and writes the answer as a table.

    .venv/Scripts/python.exe experiments/05_volatility_cost/verify_rank_ceiling.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ANOVA_RANK_NOTE, OUTPUT_DIR, RANK_INCREMENT, write_csv, write_json


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def schedule_ceiling(n_sweeps: float, rank_increment: int = RANK_INCREMENT,
                     initial_rank: int = 2) -> float:
    """``2 + 2 * dr_max * nswp``: two half-sweeps per sweep, dr_max rows each."""
    return initial_rank + 2 * rank_increment * n_sweeps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="phase4_extended")
    parser.add_argument("--dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    source = Path(args.dir)
    fits = read_csv(source / f"{args.tag}_resultats.csv")
    profiles = defaultdict(dict)
    classes = defaultdict(dict)
    for row in read_csv(source / f"{args.tag}_profils_rangs.csv"):
        key = (int(row["d_sigma"]), int(row["budget_nominal"]), int(row["seed"]))
        profiles[key][int(row["bond_index"])] = int(row["rank"])
        classes[key][int(row["bond_index"])] = row["bond_class"]

    grouped = defaultdict(list)
    for row in fits:
        grouped[(int(row["d_sigma"]), int(row["budget_nominal"]))].append(row)

    rows = []
    for (d_sigma, budget), group in sorted(grouped.items()):
        sweeps = float(np.mean([int(r["cross_sweeps"]) for r in group]))
        ceiling = schedule_ceiling(sweeps)
        maximum = float(np.mean([int(r["maximum_rank"]) for r in group]))
        effective = float(np.mean([float(r["effective_rank"]) for r in group]))
        sigma_values = [
            float(r["sigma_block_internal_max_rank"])
            for r in group
            if r["sigma_block_internal_max_rank"] not in ("", "NOT APPLICABLE")
        ]
        sigma_max = float(np.mean(sigma_values)) if sigma_values else None

        # How many interior bonds sit at the QTT matricization bound? If none,
        # that bound is not what is limiting the plateau.
        key = (d_sigma, budget, int(group[0]["seed"]))
        profile = profiles[key]
        n_cores = max(profile)
        at_qtt_bound = sum(
            1
            for k, rank in profile.items()
            if 0 < k < n_cores and rank >= min(2.0**k, 2.0 ** (n_cores - k))
        )
        interior = [
            rank
            for k, rank in profile.items()
            if min(2.0**k, 2.0 ** (n_cores - k)) > ceiling
        ]

        rows.append(
            {
                "d_sigma": d_sigma,
                "budget_nominal": budget,
                "qtt_cores": int(group[0]["qtt_cores"]),
                "cross_stop": group[0]["cross_stop"],
                "sweeps_mean": sweeps,
                "schedule_ceiling": ceiling,
                "maximum_rank_mean": maximum,
                "maximum_over_ceiling": maximum / ceiling,
                "effective_rank_mean": effective,
                "effective_over_ceiling": effective / ceiling,
                "sigma_block_max_rank_mean": sigma_max,
                "sigma_over_ceiling": (
                    sigma_max / ceiling if sigma_max is not None else "NOT APPLICABLE"
                ),
                "interior_bonds_at_qtt_bound": at_qtt_bound,
                "interior_bond_rank_median": (
                    float(np.median(interior)) if interior else "NOT APPLICABLE"
                ),
            }
        )

    write_csv(source / "verification_plafond_rang.csv", rows)

    ratios = np.array([row["maximum_over_ceiling"] for row in rows], dtype=float)
    verdict = {
        "absolute_cap_in_tt_cross": (
            "none. teneva.maxvol_rect returns r + dr with dr <= dr_max, bounded "
            "only by the tall matrix row count (2 * r_left for a binary core). "
            "No ceiling parameter exists."
        ),
        "qtt_matricization_bound": (
            "binds only at the two chain ends, where min(2**k, 2**(d-k)) is "
            "small; in the interior it is astronomically loose and never binds."
        ),
        "schedule_ceiling_formula": f"2 + 2 * dr_max * nswp with dr_max = {RANK_INCREMENT}",
        "anova_initial_rank": ANOVA_RANK_NOTE,
        "maximum_over_ceiling_range": [float(ratios.min()), float(ratios.max())],
        "conclusion": (
            "The plateau is the rank-increment schedule ceiling, not a structural "
            "rank. The observed maximum rank sits at "
            f"{ratios.min():.2f}-{ratios.max():.2f} times 2 + 2*dr_max*nswp across "
            "every configuration, and nswp is set by the oracle budget because no "
            "tolerance is passed to teneva.cross (stop='m' everywhere). Because "
            "more cores cost more evaluations per sweep, nswp - and therefore the "
            "ceiling - falls as d_sigma grows, so a rank comparison across d_sigma "
            "at fixed budget compares chains with different ceilings. No rank in "
            "this experiment is converged, and the experiment as specified cannot "
            "test a prediction about structural rank."
        ),
    }
    write_json(source / "verification_plafond_rang.json", verdict)

    print(f"{'d':>2} {'budget':>7} {'cores':>5} {'stop':>5} {'nswp':>5} "
          f"{'ceiling':>8} {'max r':>6} {'max/ceil':>9} {'erank':>6} "
          f"{'erank/ceil':>10} {'sigma r':>8} {'sig/ceil':>9} {'@QTT bound':>11}")
    for row in rows:
        sigma = row["sigma_block_max_rank_mean"]
        sigma_ratio = row["sigma_over_ceiling"]
        print(
            f"{row['d_sigma']:>2} {row['budget_nominal']:>7} {row['qtt_cores']:>5} "
            f"{row['cross_stop']:>5} {row['sweeps_mean']:>5.1f} "
            f"{row['schedule_ceiling']:>8.1f} {row['maximum_rank_mean']:>6.1f} "
            f"{row['maximum_over_ceiling']:>9.2f} {row['effective_rank_mean']:>6.2f} "
            f"{row['effective_over_ceiling']:>10.3f} "
            f"{(f'{sigma:.1f}' if sigma is not None else '--'):>8} "
            f"{(f'{sigma_ratio:.2f}' if isinstance(sigma_ratio, float) else '--'):>9} "
            f"{row['interior_bonds_at_qtt_bound']:>11}"
        )
    print(f"\n{verdict['conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
