r"""Phase 4 - the main experiment: tensor cost as a function of ``d_sigma``.

Five configurations ``d_sigma in {0, 1, 2, 3, 5}`` on the risk-hybrid grid, four
TT-cross budgets, five seeds: one hundred fits. Rank and NMAE are always written
on the same row, because a low rank on an unresolved fit is not a result; a fit
whose NMAE exceeds its grid's own interpolation floor by more than a factor of
three is marked ``UNRESOLVED`` and stays in the table while being excluded from
any statement about compressibility.

Also here, pre-registered before any fit was inspected, is the volatility-cell
sweep that phase 1's node-placement decision requires: at the retained
``n_sigma`` the reference volatility does not sit at a cell midpoint, so the NMAE
is swept across one whole cell from node to node to show where ``sigma_ref``
sits in the cell's own range rather than asserting it is unexceptional.

    .venv/Scripts/python.exe experiments/05_volatility_cost/phase4_dsigma_sweep.py --n-sigma 32
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
    D_SIGMA_SWEEP,
    N_TEST_POINTS,
    OUTPUT_DIR,
    RISK_GRID,
    SEEDS,
    SIGMA_BOUNDS,
    SIGMA_REF,
    TEST_SEED_SHARED,
    TEST_SEED_SIGMA_BOX,
    TT_BUDGETS,
    assemble_market_points,
    run_manifest,
    shared_axis_points,
    sigma_box_draws,
    sigma_node_placement,
    write_csv,
    write_json,
)
from measure import (
    bond_classification,
    build_test_sets,
    interpolation_floors,
    run_one_fit,
)

from stngpr.sigma_axes import SigmaAugmentedConfig
from stngpr.sigma_surrogate import SigmaAugmentedSurrogate
from stngpr.validation import error_metrics

N_CELL_SWEEP = 20
CELL_FLOOR_POINTS = 400

AGGREGATE_COLUMNS = (
    "oracle_calls_realized",
    "effective_rank",
    "maximum_rank",
    "sigma_block_internal_max_rank",
    "sigma_boundary_max_rank",
    "tt_parameters",
    "fit_seconds",
    "batch_seconds_per_query",
    "single_seconds_per_query",
    "nmae_common_slice",
    "nmae_sigma_box",
    "nmae_off_node",
    "nmae_over_floor_common_slice",
)

ROW_COLUMNS = (
    "d_sigma",
    "n_sigma",
    "sigma_block",
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
    "floor_nmae_sigma_box",
    "nmae_over_floor_sigma_box",
    "nmae_off_node",
    "floor_nmae_off_node",
    "nmae_over_floor_off_node",
    "fit_seconds",
    "batch_seconds_per_query",
    "single_seconds_per_query",
    "cross_stop",
    "cross_sweeps",
    "mae_common_slice",
    "bias_common_slice",
    "mae_sigma_box",
    "bias_sigma_box",
    "mae_off_node",
    "bias_off_node",
    "shared_core_order_preserved",
    "sigma_cores_contiguous",
    "sigma_block_at_chain_tail",
    "shared_core_count",
    "sigma_core_count",
    "post_fit_svd_truncation",
    "anova_samples",
    "mode",
)


def build_config(d_sigma: int, n_sigma: int, grid=None) -> SigmaAugmentedConfig:
    grid = grid or RISK_GRID
    return SigmaAugmentedConfig(
        n_sigma_axes=d_sigma,
        sigma_block="tail",
        sigma_reference=SIGMA_REF,
        volatility_bounds=SIGMA_BOUNDS,
        volatility_nodes=n_sigma,
        spot_nodes=grid["spot_nodes"],
        strike_nodes=grid["strike_nodes"],
        rate_nodes=grid["rate_nodes"],
        maturity_nodes=grid["maturity_nodes"],
    )


def cell_sigma_values(n_sigma: int) -> np.ndarray:
    """``N_CELL_SWEEP`` volatilities spanning the cell that brackets sigma_ref.

    Node to node inclusive, so the endpoints are the two grid nodes themselves
    and the interior shows how the interpolation error varies inside one cell.
    """
    placement = sigma_node_placement(n_sigma)
    return np.linspace(
        placement["bracketing_node_low"],
        placement["bracketing_node_high"],
        N_CELL_SWEEP,
    )


def sweep_sigma_cell(model, config, shared, sigma_values, floor_points):
    """NMAE of this fit at every volatility across one sigma cell."""
    rows = []
    for value in sigma_values:
        market = assemble_market_points(
            config, shared, np.full((shared.shape[0], config.n_assets), value)
        )
        exact = model.reference_price(market)
        metrics = error_metrics(exact, model.price(market))
        rows.append(
            {
                "sigma": float(value),
                "nmae": metrics["normalized_mae"],
                "mae": metrics["mae"],
                "is_sigma_ref": bool(abs(value - SIGMA_REF) < 1e-12),
                "floor_points": int(floor_points),
            }
        )
    return rows


def cell_floor_sweep(model, config, shared, sigma_values, n_points):
    """Seed-free companion to the cell sweep: the interpolation floor itself."""
    subset = shared[:n_points]
    rows = []
    for value in sigma_values:
        market = assemble_market_points(
            config, subset, np.full((subset.shape[0], config.n_assets), value)
        )
        exact = model.reference_price(market)
        floor = model.interpolation_floor(market, rows_per_call=1 << 20)
        metrics = error_metrics(exact, floor)
        rows.append(
            {
                "sigma": float(value),
                "floor_nmae": metrics["normalized_mae"],
                "floor_mae": metrics["mae"],
                "n_points": len(subset),
            }
        )
    return rows


def aggregate(rows):
    out = []
    keys = sorted({(row["d_sigma"], row["budget_nominal"]) for row in rows})
    for d_sigma, budget in keys:
        group = [
            row
            for row in rows
            if row["d_sigma"] == d_sigma and row["budget_nominal"] == budget
        ]
        record = {
            "d_sigma": d_sigma,
            "budget_nominal": budget,
            "n_seeds": len(group),
            "n_unresolved": sum(
                1 for row in group if row["resolution_status"] == "UNRESOLVED"
            ),
            "qtt_cores": group[0]["qtt_cores"],
            "log2_logical_entries": group[0]["log2_logical_entries"],
            "floor_nmae_common_slice": group[0]["floor_nmae_common_slice"],
        }
        for column in AGGREGATE_COLUMNS:
            values = [row[column] for row in group]
            if any(not isinstance(value, (int, float)) for value in values):
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
    parser.add_argument("--d-sigma", type=int, nargs="+", default=list(D_SIGMA_SWEEP))
    parser.add_argument("--budgets", type=int, nargs="+", default=list(TT_BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--skip-cell-sweep", action="store_true")
    parser.add_argument(
        "--floors-from",
        type=Path,
        default=None,
        help=(
            "reuse the interpolation floors recorded in a previous run's JSON "
            "instead of recomputing them. The floor depends only on the grid and "
            "the test set, both of which are frozen, so recomputation is "
            "deterministic; reusing guarantees the two runs are compared against "
            "the exact same floor rather than merely the same recipe."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tag", type=str, default="phase4")
    args = parser.parse_args()

    reused_floors = None
    if args.floors_from:
        source_payload = json.loads(
            Path(args.floors_from).read_text(encoding="utf-8")
        )
        reused_floors = source_payload["floors_by_d_sigma"]
        if int(source_payload["n_sigma"]) != int(args.n_sigma):
            parser.error(
                f"--floors-from was produced at n_sigma="
                f"{source_payload['n_sigma']}, not {args.n_sigma}"
            )
        if int(source_payload["n_test_points"]) != int(args.points):
            parser.error(
                f"--floors-from was produced on "
                f"{source_payload['n_test_points']} test points, not {args.points}"
            )

    n_sigma = int(args.n_sigma)
    reference_config = build_config(0, n_sigma)
    shared = shared_axis_points(int(args.points), reference_config, TEST_SEED_SHARED)
    sigma_box = sigma_box_draws(
        int(args.points), reference_config.n_assets, TEST_SEED_SIGMA_BOX
    )
    sigma_values = cell_sigma_values(n_sigma)
    largest_budget = max(args.budgets)

    total = len(args.d_sigma) * len(args.budgets) * len(args.seeds)
    print(
        f"{args.tag}: n_sigma={n_sigma} d_sigma={list(args.d_sigma)} "
        f"budgets={list(args.budgets)} seeds={len(args.seeds)} -> {total} fits",
        flush=True,
    )

    rows, rank_rows, cell_rows, cell_floor_rows, floor_payload = [], [], [], [], {}
    wall_start = perf_counter()
    done = 0

    for d_sigma in args.d_sigma:
        config = build_config(d_sigma, n_sigma)
        test_sets = build_test_sets(config, shared, sigma_box)
        bonds = bond_classification(config)

        floor_model = SigmaAugmentedSurrogate(config, RISK_GRID["mode"])
        start = perf_counter()
        if reused_floors is not None:
            floors = reused_floors[str(d_sigma)]
            origin = f"reused from {args.floors_from.name}"
        else:
            floors = interpolation_floors(floor_model, test_sets)
            origin = f"{perf_counter() - start:.1f}s"
        print(
            f"  d_sigma={d_sigma} floors "
            + " ".join(
                f"{name}={value['normalized_mae']:.4e}"
                for name, value in floors.items()
            )
            + f"  [{origin}]",
            flush=True,
        )
        floor_payload[str(d_sigma)] = floors

        if not args.skip_cell_sweep and d_sigma == max(args.d_sigma) and d_sigma:
            start = perf_counter()
            cell_floor_rows.extend(
                {
                    "d_sigma": d_sigma,
                    "n_sigma": n_sigma,
                    **row,
                }
                for row in cell_floor_sweep(
                    floor_model, config, shared, sigma_values, CELL_FLOOR_POINTS
                )
            )
            print(
                f"  d_sigma={d_sigma} sigma-cell floor sweep over "
                f"{len(sigma_values)} volatilities "
                f"[{perf_counter() - start:.1f}s]",
                flush=True,
            )
        del floor_model

        for budget in args.budgets:
            for seed in args.seeds:
                record, model = run_one_fit(
                    config,
                    RISK_GRID["mode"],
                    int(budget),
                    int(seed),
                    test_sets,
                    floors,
                    reference_config,
                )
                rows.append(record)
                for index, rank in enumerate(record["bond_ranks_full"]):
                    rank_rows.append(
                        {
                            "d_sigma": d_sigma,
                            "n_sigma": n_sigma,
                            "budget_nominal": int(budget),
                            "seed": int(seed),
                            "bond_index": index,
                            "rank": int(rank),
                            "bond_class": (
                                "sigma_internal"
                                if index in bonds["internal_sigma_bonds"]
                                else "boundary"
                                if index in bonds["boundary_bonds"]
                                else "shared"
                            ),
                        }
                    )
                if (
                    not args.skip_cell_sweep
                    and d_sigma == max(args.d_sigma)
                    and d_sigma
                    and budget == largest_budget
                ):
                    cell_rows.extend(
                        {
                            "d_sigma": d_sigma,
                            "n_sigma": n_sigma,
                            "budget_nominal": int(budget),
                            "seed": int(seed),
                            **row,
                        }
                        for row in sweep_sigma_cell(
                            model, config, shared, sigma_values, CELL_FLOOR_POINTS
                        )
                    )
                del model
                done += 1
                elapsed = perf_counter() - wall_start
                print(
                    f"  [{done:3d}/{total}] d_sigma={d_sigma} budget={budget:6d} "
                    f"seed={seed} calls={record['oracle_calls_realized']:6d} "
                    f"erank={record['effective_rank']:6.2f} "
                    f"sig_int={record['sigma_block_internal_max_rank']} "
                    f"bnd={record['sigma_boundary_max_rank']} "
                    f"par={record['tt_parameters']:8d} "
                    f"nmae={record['nmae_common_slice']:.4e} "
                    f"x_floor={record['nmae_over_floor_common_slice']:7.2f} "
                    f"{record['resolution_status']:10s} "
                    f"fit={record['fit_seconds']:6.1f}s "
                    f"eta={(elapsed / done) * (total - done) / 60:.0f}min",
                    flush=True,
                )

    out = Path(args.output_dir)
    write_csv(out / f"{args.tag}_resultats.csv", rows, fieldnames=ROW_COLUMNS)
    write_csv(out / f"{args.tag}_agrege.csv", aggregate(rows))
    write_csv(out / f"{args.tag}_profils_rangs.csv", rank_rows)
    if cell_rows:
        write_csv(out / f"{args.tag}_sigma_cell_nmae.csv", cell_rows)
    if cell_floor_rows:
        write_csv(out / f"{args.tag}_sigma_cell_floor.csv", cell_floor_rows)
    write_json(
        out / f"{args.tag}_run.json",
        {
            "manifest": run_manifest(),
            "n_sigma": n_sigma,
            "n_test_points": int(args.points),
            "sigma_node_placement": sigma_node_placement(n_sigma),
            "sigma_cell_sweep_values": sigma_values.tolist(),
            "floors_by_d_sigma": floor_payload,
            "total_wall_seconds": perf_counter() - wall_start,
            "fits": rows,
        },
    )
    print(f"\nwrote {args.tag}_* into {out}", flush=True)

    unresolved = [row for row in rows if row["resolution_status"] == "UNRESOLVED"]
    print(f"{len(unresolved)} of {len(rows)} fits flagged UNRESOLVED")
    broken = [row for row in rows if not row["shared_core_order_preserved"]]
    if broken:
        print(f"WARNING: {len(broken)} fits failed the additive-chain check")
        return 1
    print("additive-chain check passed on every fit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
