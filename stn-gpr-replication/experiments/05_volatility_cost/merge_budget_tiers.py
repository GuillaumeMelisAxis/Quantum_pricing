r"""Merge the pre-registered budget ladder with a post-hoc extension tier.

The four budgets of phase 4 were fixed before the first production fit. Any
budget added after the results were seen is a different kind of number, and
folding the two silently into one table would misrepresent the second as
pre-registered. So the merge keeps both, and stamps every row with the tier it
belongs to.

The merged artefacts are written under a separate tag. The pre-registered
``phase4_*`` files are never modified, so the original ladder stays readable
exactly as it was produced.

    .venv/Scripts/python.exe experiments/05_volatility_cost/merge_budget_tiers.py \
        --base phase4 --extension phase4b --tag phase4_extended
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUTPUT_DIR, TT_BUDGETS, write_csv, write_json
from phase4_dsigma_sweep import ROW_COLUMNS, aggregate

TIER_COLUMN = "budget_tier"
PRE_REGISTERED = "pre-registered"
POST_HOC = "post-hoc extension"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows):
    """Re-type the CSV round trip so the aggregation maths works again."""
    out = []
    for row in rows:
        record = dict(row)
        for key, value in row.items():
            if value in ("", "NOT APPLICABLE", "None"):
                record[key] = value or None
                continue
            if value in ("True", "False"):
                record[key] = value == "True"
                continue
            try:
                record[key] = int(value)
            except ValueError:
                try:
                    record[key] = float(value)
                except ValueError:
                    record[key] = value
        out.append(record)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="phase4")
    parser.add_argument("--extension", default="phase4b")
    parser.add_argument("--tag", default="phase4_extended")
    parser.add_argument("--dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    source = Path(args.dir)
    base = numeric(read_csv(source / f"{args.base}_resultats.csv"))
    extension = numeric(read_csv(source / f"{args.extension}_resultats.csv"))

    pre_registered = set(TT_BUDGETS)
    for row in base:
        row[TIER_COLUMN] = (
            PRE_REGISTERED if row["budget_nominal"] in pre_registered else POST_HOC
        )
    for row in extension:
        row[TIER_COLUMN] = POST_HOC

    rows = base + extension
    ranks = read_csv(source / f"{args.base}_profils_rangs.csv") + read_csv(
        source / f"{args.extension}_profils_rangs.csv"
    )

    aggregated = aggregate(rows)
    tier_of = {
        (row["d_sigma"], row["budget_nominal"]): row[TIER_COLUMN] for row in rows
    }
    for record in aggregated:
        record[TIER_COLUMN] = tier_of[(record["d_sigma"], record["budget_nominal"])]

    columns = (*ROW_COLUMNS, TIER_COLUMN)
    write_csv(source / f"{args.tag}_resultats.csv", rows, fieldnames=columns)
    write_csv(source / f"{args.tag}_agrege.csv", aggregated)
    write_csv(source / f"{args.tag}_profils_rangs.csv", ranks)

    # The sigma-cell sweep belongs to the pre-registered tier and was not re-run.
    # Carry it across unchanged so the merged tag is self-contained, rather than
    # silently dropping it from the merged report.
    for suffix in ("sigma_cell_nmae", "sigma_cell_floor"):
        origin = source / f"{args.base}_{suffix}.csv"
        if origin.exists():
            (source / f"{args.tag}_{suffix}.csv").write_bytes(origin.read_bytes())

    base_payload = json.loads(
        (source / f"{args.base}_run.json").read_text(encoding="utf-8")
    )
    extension_payload = json.loads(
        (source / f"{args.extension}_run.json").read_text(encoding="utf-8")
    )
    write_json(
        source / f"{args.tag}_run.json",
        {
            "manifest": extension_payload["manifest"],
            "n_sigma": base_payload["n_sigma"],
            "n_test_points": base_payload["n_test_points"],
            "sigma_node_placement": base_payload["sigma_node_placement"],
            "sigma_cell_sweep_values": base_payload["sigma_cell_sweep_values"],
            "floors_by_d_sigma": base_payload["floors_by_d_sigma"],
            "budget_tiers": {
                PRE_REGISTERED: sorted(pre_registered),
                POST_HOC: sorted(
                    {row["budget_nominal"] for row in extension}
                ),
            },
            "post_hoc_note": (
                "The extension tier was added after the pre-registered ladder had "
                "been run and inspected. It was applied uniformly to every "
                "d_sigma, not to a single configuration, and the interpolation "
                "floors were reused rather than recomputed, so the two tiers are "
                "measured against the identical floor. It is reported separately "
                "and must not be presented as pre-registered."
            ),
            "total_wall_seconds": (
                base_payload["total_wall_seconds"]
                + extension_payload["total_wall_seconds"]
            ),
            "fits": rows,
        },
    )
    print(
        f"merged {len(base)} pre-registered + {len(extension)} post-hoc fits "
        f"-> {args.tag}_* in {source}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
