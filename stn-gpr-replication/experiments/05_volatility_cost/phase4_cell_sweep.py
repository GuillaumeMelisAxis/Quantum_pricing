r"""Re-run the sigma-cell sweep at a chosen budget.

The pre-registered sweep ran at the top of the pre-registered ladder, 300 000
calls, where ``d_sigma = 5`` was still ``UNRESOLVED`` at 4.15 times its own
interpolation floor. At that budget the NMAE across a volatility cell is
dominated by unconverged TT error, which is flat in sigma, so the sweep mostly
measures something other than the node placement it was designed to probe. Re-run
at a budget that resolves the fit, the sigma-directional interpolation structure
is what is left.

The floor half of the sweep does not depend on the budget at all - it has no TT
in it - so it is reused rather than recomputed, which also guarantees the two
budgets are compared against the identical floor curve.

    .venv/Scripts/python.exe experiments/05_volatility_cost/phase4_cell_sweep.py \
        --n-sigma 16 --budget 600000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    ANOVA_SAMPLES,
    MAX_SWEEPS,
    N_TEST_POINTS,
    OUTPUT_DIR,
    RANK_INCREMENT,
    RISK_GRID,
    SEEDS,
    SIGMA_BOUNDS,
    SIGMA_REF,
    TEST_SEED_SHARED,
    assemble_market_points,
    run_manifest,
    shared_axis_points,
    sigma_node_placement,
    write_csv,
    write_json,
)
from phase4_dsigma_sweep import (
    CELL_FLOOR_POINTS,
    build_config,
    cell_floor_sweep,
    cell_sigma_values,
    sweep_sigma_cell,
)

from stngpr.sigma_surrogate import SigmaAugmentedSurrogate
from stngpr.validation import error_metrics


def node_aligned_sigma_values(n_sigma, lower, upper, sub_per_cell):
    """Sample ``[lower, upper]`` on a grid that lands exactly on the axis nodes.

    The interpolation floor is a scalloped curve: zero-ish at every grid node,
    maximal near each cell midpoint. Sampling it on an arbitrary uniform grid
    slides in and out of phase with the nodes and smears those extrema away.
    Stepping at ``h / sub_per_cell`` from the axis origin instead puts a sample
    on every node and, for even ``sub_per_cell``, on every midpoint too - so the
    scallop is resolved with the fewest points.
    """
    axis = np.linspace(*SIGMA_BOUNDS, int(n_sigma))
    step = float(axis[1] - axis[0]) / int(sub_per_cell)
    index = np.arange(0, int(np.ceil((SIGMA_BOUNDS[1] - SIGMA_BOUNDS[0]) / step)) + 1)
    values = float(SIGMA_BOUNDS[0]) + index * step
    keep = values[(values >= float(lower) - 1e-12) & (values <= float(upper) + 1e-12)]
    if keep.size < 2:
        raise ValueError("the requested sigma range contains fewer than two samples")
    return keep


def read_floor_csv(path: Path):
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sigma", type=int, required=True)
    parser.add_argument("--d-sigma", type=int, default=5)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--points", type=int, default=N_TEST_POINTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--floor-from",
        type=Path,
        default=None,
        help=(
            "reuse a previously computed cell floor sweep. The floor contains no "
            "TT and is budget-independent, so reusing it is exact and makes the "
            "two budgets share one floor curve."
        ),
    )
    parser.add_argument(
        "--sigma-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help=(
            "sweep this volatility range instead of the single cell bracketing "
            "sigma_ref. Sampled on a node-aligned grid so every node and every "
            "cell midpoint in range is hit exactly."
        ),
    )
    parser.add_argument("--sub-per-cell", type=int, default=4)
    parser.add_argument("--floor-points", type=int, default=CELL_FLOOR_POINTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    config = build_config(int(args.d_sigma), int(args.n_sigma))
    shared = shared_axis_points(int(args.points), config, TEST_SEED_SHARED)
    if args.sigma_range:
        sigma_values = node_aligned_sigma_values(
            int(args.n_sigma), *args.sigma_range, int(args.sub_per_cell)
        )
    else:
        sigma_values = cell_sigma_values(int(args.n_sigma))
    placement = sigma_node_placement(int(args.n_sigma))
    axis_nodes = np.linspace(*SIGMA_BOUNDS, int(args.n_sigma))
    nodes_in_range = axis_nodes[
        (axis_nodes >= sigma_values[0] - 1e-12)
        & (axis_nodes <= sigma_values[-1] + 1e-12)
    ]

    print(
        f"{args.tag}: d_sigma={args.d_sigma} n_sigma={args.n_sigma} "
        f"budget={args.budget} seeds={len(args.seeds)}; sigma in "
        f"[{sigma_values[0]:.6f}, {sigma_values[-1]:.6f}] at "
        f"{len(sigma_values)} volatilities, spanning {nodes_in_range.size} grid "
        f"nodes ({max(nodes_in_range.size - 1, 0)} full cells); "
        f"floor on {args.floor_points} test points",
        flush=True,
    )

    rows = []
    for seed in args.seeds:
        model = SigmaAugmentedSurrogate(config, RISK_GRID["mode"], seed=int(seed))
        start = perf_counter()
        diagnostics = model.fit(
            int(args.budget),
            anova_samples=ANOVA_SAMPLES,
            max_sweeps=MAX_SWEEPS,
            rank_increment=RANK_INCREMENT,
            log=False,
        )
        rows.extend(
            {
                "d_sigma": int(args.d_sigma),
                "n_sigma": int(args.n_sigma),
                "budget_nominal": int(args.budget),
                "seed": int(seed),
                **row,
            }
            for row in sweep_sigma_cell(
                model, config, shared, sigma_values, int(args.floor_points)
            )
        )
        reference = model.reference_price(
            assemble_market_points(
                config, shared, np.full((shared.shape[0], config.n_assets), SIGMA_REF)
            )
        )
        at_reference = error_metrics(
            reference,
            model.price(
                assemble_market_points(
                    config,
                    shared,
                    np.full((shared.shape[0], config.n_assets), SIGMA_REF),
                )
            ),
        )
        print(
            f"  seed={seed} calls={diagnostics.function_evaluations} "
            f"erank={diagnostics.effective_rank:.2f} "
            f"nmae@sigma_ref={at_reference['normalized_mae']:.4e} "
            f"[{perf_counter() - start:.1f}s]",
            flush=True,
        )
        del model

    if args.floor_from and Path(args.floor_from).exists():
        floor_rows = read_floor_csv(Path(args.floor_from))
        floor_origin = f"reused from {Path(args.floor_from).name}"
    else:
        floor_model = SigmaAugmentedSurrogate(config, RISK_GRID["mode"])
        start = perf_counter()
        floor_rows = [
            {"d_sigma": int(args.d_sigma), "n_sigma": int(args.n_sigma), **row}
            for row in cell_floor_sweep(
                floor_model, config, shared, sigma_values, int(args.floor_points)
            )
        ]
        floor_origin = f"computed in {perf_counter() - start:.1f}s"
        del floor_model
    print(f"  cell floor: {floor_origin}", flush=True)

    out = Path(args.output_dir)
    write_csv(out / f"{args.tag}_sigma_cell_nmae.csv", rows)
    write_csv(out / f"{args.tag}_sigma_cell_floor.csv", floor_rows)

    by_sigma: dict[float, list[float]] = {}
    for row in rows:
        by_sigma.setdefault(float(row["sigma"]), []).append(float(row["nmae"]))
    sigmas = sorted(by_sigma)
    means = np.array([np.mean(by_sigma[value]) for value in sigmas])
    nearest = int(np.argmin(np.abs(np.asarray(sigmas) - SIGMA_REF)))

    summary = {
        "manifest": run_manifest(),
        "d_sigma": int(args.d_sigma),
        "n_sigma": int(args.n_sigma),
        "budget": int(args.budget),
        "seeds": [int(seed) for seed in args.seeds],
        "n_test_points": int(args.points),
        "sigma_bounds": list(SIGMA_BOUNDS),
        "sigma_reference": SIGMA_REF,
        "sigma_node_placement": placement,
        "sigma_range": [float(sigma_values[0]), float(sigma_values[-1])],
        "sub_per_cell": int(args.sub_per_cell),
        "axis_nodes_in_range": nodes_in_range.tolist(),
        "floor_points": int(args.floor_points),
        "cell_values": sigmas,
        "cell_floor_origin": floor_origin,
        "tt_cell_range": [float(means.min()), float(means.max())],
        "tt_cell_median": float(np.median(means)),
        "tt_cell_relative_spread": float(means.max() / means.min()),
        "nearest_sample_to_sigma_ref": sigmas[nearest],
        "tt_at_nearest_sample": float(means[nearest]),
        "tt_rank_of_nearest_sample": int(np.sum(means < means[nearest])) + 1,
        "tt_argmin_sigma": sigmas[int(np.argmin(means))],
    }
    write_json(out / f"{args.tag}_sigma_cell.json", summary)

    print(
        f"\nTT NMAE across the cell: "
        f"[{means.min():.4e}, {means.max():.4e}], "
        f"spread {summary['tt_cell_relative_spread']:.3f}x, "
        f"median {summary['tt_cell_median']:.4e}; "
        f"minimum at sigma={summary['tt_argmin_sigma']:.6f}; "
        f"sample nearest sigma_ref ({summary['nearest_sample_to_sigma_ref']:.6f}) "
        f"ranks {summary['tt_rank_of_nearest_sample']}/{len(sigmas)}"
    )
    print(f"wrote {args.tag}_* into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
