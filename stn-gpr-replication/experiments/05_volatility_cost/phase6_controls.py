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

    .venv/Scripts/python.exe experiments/05_volatility_cost/phase6_controls.py --n-sigma 16
"""

from __future__ import annotations

import argparse
import json
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


FLOOR_INVARIANCE_POINTS = 32
FLOOR_INVARIANCE_TOLERANCE = 1e-12


def _subsample_floor(config, grid, test_sets, n_points):
    """Pointwise interpolation floor on the first ``n_points`` of the slice."""
    model = SigmaAugmentedSurrogate(config, grid["mode"])
    subset = test_sets["common_slice"][:n_points]
    values = model.interpolation_floor(subset, rows_per_call=1 << 20)
    del model
    return np.asarray(values, dtype=float)


def _check_floor_is_order_invariant(reference_values, candidate_values):
    """Compare the same points' floors under two different mode orders.

    Reordering the modes permutes the grid axes and the point columns together,
    so the interpolated value cannot change - which is why one floor may be
    reused across the three orderings of control 6a. That is an argument, not a
    measurement, so it is measured: the floor is recomputed on a subsample under
    the permuted order and compared pointwise against the order it was cached
    under. Normalized by the mean magnitude, the same norm the reported NMAE
    uses, so the tolerance means the same thing here as it does there.
    """
    scale = float(np.mean(np.abs(reference_values)))
    absolute = float(np.max(np.abs(candidate_values - reference_values)))
    residual = absolute / scale if scale > 0.0 else absolute
    return {
        "subsample_points": int(reference_values.size),
        "max_absolute_difference": absolute,
        "normalized_difference": residual,
        "tolerance": FLOOR_INVARIANCE_TOLERANCE,
        "order_invariant": bool(residual <= FLOOR_INVARIANCE_TOLERANCE),
    }


def _run_seeds(label, grid_name, grid, config, reference, test_sets, floors,
               seeds, budget, block, d_sigma, rows):
    """Fit every seed for one (grid, order, d_sigma) and record the results."""
    for seed in seeds:
        record, model = run_one_fit(
            config, grid["mode"], int(budget), int(seed),
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


def run_block(label, grid_name, grid, d_sigma_values, orders, seeds, n_sigma, shared,
              sigma_box, rows, floors_payload, floor_cache=None,
              budget=CONTROL_BUDGET):
    floor_cache = {} if floor_cache is None else floor_cache
    for sigma_block in orders:
        for d_sigma in d_sigma_values:
            # The 8-D chain has no sigma block to reorder, so it is run once,
            # under the first ordering, rather than once per ordering. Skipping
            # it has to be a `continue` on this iteration: a `break` would end
            # the d_sigma loop and silently drop every larger d_sigma with it.
            if d_sigma == 0 and sigma_block != orders[0]:
                continue
            block = "tail" if d_sigma == 0 else sigma_block
            config = build_config(d_sigma, n_sigma, grid, block)
            reference = build_config(0, n_sigma, grid, "tail")
            test_sets = build_test_sets(config, shared, sigma_box)

            start = perf_counter()
            cache_key = (grid_name, d_sigma)
            if cache_key in floor_cache:
                floors, first_block, reference_values = floor_cache[cache_key]
                subsample = _subsample_floor(
                    config, grid, test_sets, FLOOR_INVARIANCE_POINTS
                )
                if reference_values is None:
                    # Floors loaded from a previous run carry no subsample, so
                    # there is nothing to compare this order against yet. Record
                    # the baseline here and say plainly that no check ran, rather
                    # than printing a check that did not happen.
                    floor_cache[cache_key] = (floors, block, subsample)
                    print(
                        f"  {label} {grid_name} block={block} d_sigma={d_sigma} "
                        f"floors reused from {first_block}; no order-invariance "
                        "check available (previous run stored no subsample), "
                        "baseline established for this run",
                        flush=True,
                    )
                    floors_payload[f"{label}|{grid_name}|{block}|{d_sigma}"] = floors
                    _run_seeds(
                        label, grid_name, grid, config, reference, test_sets,
                        floors, seeds, budget, block, d_sigma, rows,
                    )
                    continue
                check = _check_floor_is_order_invariant(
                    reference_values, subsample
                )
                origin = (
                    f"reused from block={first_block}; order-invariance "
                    f"{'OK' if check['order_invariant'] else 'FAILED'} at "
                    f"{check['normalized_difference']:.2e} on "
                    f"{check['subsample_points']} points"
                )
                floors_payload[f"{label}|{grid_name}|{block}|{d_sigma}|check"] = check
                if not check["order_invariant"]:
                    raise RuntimeError(
                        f"the interpolation floor is not invariant under the mode "
                        f"order (block={block} vs {first_block}): "
                        f"{check['normalized_difference']:.3e} > "
                        f"{FLOOR_INVARIANCE_TOLERANCE:.0e}. Reusing it would "
                        "compare the orderings against different floors."
                    )
            else:
                floor_model = SigmaAugmentedSurrogate(config, grid["mode"])
                floors = interpolation_floors(floor_model, test_sets)
                del floor_model
                floor_cache[cache_key] = (
                    floors,
                    block,
                    _subsample_floor(
                        config, grid, test_sets, FLOOR_INVARIANCE_POINTS
                    ),
                )
                origin = f"{perf_counter() - start:.1f}s"
            print(
                f"  {label} {grid_name} block={block} d_sigma={d_sigma} floors "
                + " ".join(
                    f"{name}={value['normalized_mae']:.4e}"
                    for name, value in floors.items()
                )
                + f"  [{origin}]",
                flush=True,
            )
            floors_payload[f"{label}|{grid_name}|{block}|{d_sigma}"] = floors

            _run_seeds(
                label, grid_name, grid, config, reference, test_sets, floors,
                seeds, budget, block, d_sigma, rows,
            )


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
    parser.add_argument("--budget", type=int, default=CONTROL_BUDGET)
    parser.add_argument("--orders", nargs="+", default=list(ORDERS))
    parser.add_argument(
        "--floors-from",
        type=Path,
        default=None,
        help=(
            "reuse the interpolation floors recorded in a previous phase 6 run. "
            "The floor depends only on the grid and the test set, neither of "
            "which the budget touches, so reusing is exact."
        ),
    )
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
    floor_cache: dict = {}
    if args.floors_from:
        previous = json.loads(Path(args.floors_from).read_text(encoding="utf-8"))
        if int(previous["n_sigma"]) != n_sigma:
            parser.error(
                f"--floors-from was produced at n_sigma={previous['n_sigma']}"
            )
        for key, value in previous["floors"].items():
            parts = key.split("|")
            if len(parts) == 4:
                _, grid_name, block, d_sigma = parts
                floor_cache.setdefault(
                    (grid_name, int(d_sigma)), (value, f"{block} (previous run)", None)
                )
    wall = perf_counter()

    if not args.skip_6a:
        print(
            f"6a mode-ordering sensitivity: budget {args.budget}, "
            f"seeds {list(ORDER_SEEDS)}, d_sigma=5, orders {list(args.orders)}",
            flush=True,
        )
        run_block(
            "6a", "risk_hybrid", RISK_GRID, (5,), tuple(args.orders), ORDER_SEEDS,
            n_sigma, shared, sigma_box, rows, floors_payload,
            floor_cache=floor_cache, budget=int(args.budget),
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
            floor_cache=floor_cache, budget=int(args.budget),
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
            "control_budget": int(args.budget),
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
