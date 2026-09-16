r"""Phase 6 - supplementary controls.

6a, mode-ordering sensitivity. At budget 150 000, three seeds, ``d_sigma = 5``,
the reference order (sigma block appended at the chain tail) is compared to the
sigma block at the head and to sigma interleaved with the spots. The point is to
show the conclusion does not depend on an order that happened to be favourable.
Note that only the tail order is additive - the other two move the shared cores
by construction - so their rank profiles are not comparable core by core to
``d_sigma = 0``; the aggregate quantities are.

6b, pricing-grid control. The same ``d_sigma in {0, 5}`` contrast at budget
150 000 over five seeds, on the price-adaptive pricing grid
``32^5 x 64 x 8 x 8`` augmented with the sigma axes, to show that the dimensional
cost is a property of the representation rather than of the risk-hybrid geometry.

Run only after phases 4 and 5 have been validated.

    .venv/Scripts/python.exe scripts/experiments/phase6_controls.py --n-sigma 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    N_TEST_POINTS,
    OUTPUT_DIR,
    PRICING_GRID,
    RISK_GRID,
    SEEDS,
    SIGMA_BOUNDS,
    SIGMA_REF,
    TEST_SEED_SHARED,
    TEST_SEED_SIGMA_BOX,
    run_manifest,
    shared_axis_points,
    sigma_box_draws,
    write_csv,
    write_json,
)
from measure import build_test_sets, interpolation_floors, run_one_fit

from stngpr.sigma_axes import SigmaAugmentedConfig
from stngpr.sigma_surrogate import SigmaAugmentedSurrogate

CONTROL_BUDGET = 150_000
ORDER_SEEDS = SEEDS[:3]
ORDERS = ("tail", "head", "interleaved")

COLUMNS = (
    "control",
    "grid",
    "mode",
    "sigma_block",
    "d_sigma",
    "n_sigma",
    "budget_nominal",
    "oracle_calls_realized",
    "seed",
    "resolution_status",
    "qtt_cores",
    "log2_logical_entries",
    "effective_rank",
    "maximum_rank",
    "sigma_block_internal_max_rank",
    "sigma_boundary_max_rank",
    "tt_parameters",
    "nmae_common_slice",
    "floor_nmae_common_slice",
    "nmae_over_floor_common_slice",
    "nmae_sigma_box",
    "nmae_off_node",
    "fit_seconds",
    "batch_seconds_per_query",
    "cross_stop",
    "shared_core_order_preserved",
    "sigma_cores_contiguous",
    "sigma_block_at_chain_tail",
)


def build_config(d_sigma, n_sigma, grid, sigma_block="tail"):
    return SigmaAugmentedConfig(
        n_sigma_axes=d_sigma,
        sigma_block=sigma_block,
        sigma_reference=SIGMA_REF,
        volatility_bounds=SIGMA_BOUNDS,
        volatility_nodes=n_sigma,
        spot_nodes=grid["spot_nodes"],
        strike_nodes=grid["strike_nodes"],
        rate_nodes=grid["rate_nodes"],
        maturity_nodes=grid["maturity_nodes"],
    )


def run_block(label, grid_name, grid, d_sigma_values, orders, seeds, n_sigma, shared,
              sigma_box, rows, floors_payload):
    for sigma_block in orders:
        for d_sigma in d_sigma_values:
            block = "tail" if d_sigma == 0 else sigma_block
            config = build_config(d_sigma, n_sigma, grid, block)
            reference = build_config(0, n_sigma, grid, "tail")
            test_sets = build_test_sets(config, shared, sigma_box)

            floor_model = SigmaAugmentedSurrogate(config, grid["mode"])
            start = perf_counter()
            floors = interpolation_floors(floor_model, test_sets)
            print(
                f"  {label} {grid_name} block={block} d_sigma={d_sigma} floors "
                + " ".join(
                    f"{name}={value['normalized_mae']:.4e}"
                    for name, value in floors.items()
                )
                + f"  [{perf_counter() - start:.1f}s]",
                flush=True,
            )
            floors_payload[f"{label}|{grid_name}|{block}|{d_sigma}"] = floors
            del floor_model

            for seed in seeds:
                record, model = run_one_fit(
                    config, grid["mode"], CONTROL_BUDGET, int(seed),
                    test_sets, floors, reference,
                )
                record.update({"control": label, "grid": grid_name})
                rows.append(record)
                del model
                print(
                    f"    {label} block={block:12s} d_sigma={d_sigma} seed={seed} "
                    f"calls={record['oracle_calls_realized']:6d} "
                    f"erank={record['effective_rank']:6.2f} "
                    f"sig_int={record['sigma_block_internal_max_rank']} "
                    f"par={record['tt_parameters']:8d} "
                    f"nmae={record['nmae_common_slice']:.4e} "
                    f"{record['resolution_status']} "
                    f"fit={record['fit_seconds']:.1f}s",
                    flush=True,
                )
            if d_sigma == 0:
                break  # the 8-D chain has no sigma block to reorder


def summarize(rows, keys):
    out = []
    groups = sorted({tuple(row[key] for key in keys) for row in rows})
    for group in groups:
        subset = [
            row for row in rows
            if tuple(row[key] for key in keys) == group
        ]
        record = dict(zip(keys, group))
        record["n_seeds"] = len(subset)
        record["n_unresolved"] = sum(
            1 for row in subset if row["resolution_status"] == "UNRESOLVED"
        )
        for column in (
            "effective_rank", "maximum_rank", "sigma_block_internal_max_rank",
            "tt_parameters", "nmae_common_slice", "nmae_sigma_box",
            "oracle_calls_realized", "fit_seconds",
        ):
            values = [row[column] for row in subset]
            if any(not isinstance(value, (int, float)) or value is None
                   for value in values):
                record[f"{column}_mean"] = "NOT APPLICABLE"
                record[f"{column}_std"] = "NOT APPLICABLE"
                continue
            array = np.asarray(values, dtype=float)
            record[f"{column}_mean"] = float(np.mean(array))
            record[f"{column}_std"] = (
                float(np.std(array, ddof=1)) if array.size > 1 else 0.0
            )
        out.append(record)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sigma", type=int, required=True)
    parser.add_argument("--points", type=int, default=N_TEST_POINTS)
    parser.add_argument("--skip-6a", action="store_true")
    parser.add_argument("--skip-6b", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    n_sigma = int(args.n_sigma)
    reference = build_config(0, n_sigma, RISK_GRID)
    shared = shared_axis_points(int(args.points), reference, TEST_SEED_SHARED)
    sigma_box = sigma_box_draws(
        int(args.points), reference.n_assets, TEST_SEED_SIGMA_BOX
    )

    rows, floors_payload = [], {}
    wall = perf_counter()

    if not args.skip_6a:
        print(
            f"6a mode-ordering sensitivity: budget {CONTROL_BUDGET}, "
            f"seeds {list(ORDER_SEEDS)}, d_sigma=5, orders {list(ORDERS)}",
            flush=True,
        )
        run_block(
            "6a", "risk_hybrid", RISK_GRID, (5,), ORDERS, ORDER_SEEDS,
            n_sigma, shared, sigma_box, rows, floors_payload,
        )

    if not args.skip_6b:
        print(
            f"\n6b pricing-grid control: budget {CONTROL_BUDGET}, "
            f"seeds {list(SEEDS)}, d_sigma in (0, 5)",
            flush=True,
        )
        run_block(
            "6b", "pricing", PRICING_GRID, (0, 5), ("tail",), SEEDS,
            n_sigma, shared, sigma_box, rows, floors_payload,
        )

    out = Path(args.output_dir)
    write_csv(out / "phase6_controles.csv", rows, fieldnames=COLUMNS)
    write_csv(
        out / "phase6a_ordre_modes.csv",
        summarize([r for r in rows if r["control"] == "6a"],
                  ("control", "grid", "sigma_block", "d_sigma")),
    )
    write_csv(
        out / "phase6b_grille_pricing.csv",
        summarize([r for r in rows if r["control"] == "6b"],
                  ("control", "grid", "sigma_block", "d_sigma")),
    )
    write_json(
        out / "phase6_run.json",
        {
            "manifest": run_manifest(),
            "n_sigma": n_sigma,
            "control_budget": CONTROL_BUDGET,
            "order_seeds": list(ORDER_SEEDS),
            "orders": list(ORDERS),
            "floors": floors_payload,
            "total_wall_seconds": perf_counter() - wall,
            "fits": rows,
        },
    )
    print(f"\nwrote phase6_* into {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
