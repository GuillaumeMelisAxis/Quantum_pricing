r"""Phase 1 - pre-registered choice of the number of volatility nodes.

``n_sigma`` is not picked for convenience. For each candidate the *production*
interpolation floor of the corresponding 13-D grid is measured and compared to
the floor of the 8-D grid on the common slice: the smallest candidate whose
floor does not exceed the 8-D one by more than 10 % relative is retained. The
volatility axes are not allowed to raise the grid's own interpolation floor.

No TT fit happens here and no dense tensor is assembled. The floor is the
production interpolant - cubic on every axis - fed the exact closed form at the
stencil nodes, computed through the spot-collapsed path of
``stngpr.sigma_surrogate``, which is that same quantity evaluated exactly rather
than an approximation of it (see that module, and the equivalence test in
``tests/test_sigma_axes.py``).

If no candidate at or below 32 satisfies the criterion the script stops and says
so. That outcome is a result, not an obstacle.

    .venv/Scripts/python.exe experiments/05_volatility_cost/phase1_select_nsigma.py
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
    PHASE1_TOLERANCE,
    RISK_GRID,
    SIGMA_BOUNDS,
    SIGMA_REF,
    TEST_SEED_SHARED,
    TEST_SEED_SIGMA_BOX,
    assemble_market_points,
    run_manifest,
    shared_axis_points,
    sigma_box_draws,
    sigma_node_placement,
    write_csv,
    write_json,
)

from stngpr.sigma_axes import SigmaAugmentedConfig
from stngpr.sigma_surrogate import SigmaAugmentedSurrogate
from stngpr.validation import error_metrics

CANDIDATES = (4, 8, 16, 32)


def build_config(n_sigma_axes: int, n_sigma_nodes: int) -> SigmaAugmentedConfig:
    return SigmaAugmentedConfig(
        n_sigma_axes=n_sigma_axes,
        sigma_block="tail",
        sigma_reference=SIGMA_REF,
        volatility_bounds=SIGMA_BOUNDS,
        volatility_nodes=n_sigma_nodes,
        spot_nodes=RISK_GRID["spot_nodes"],
        strike_nodes=RISK_GRID["strike_nodes"],
        rate_nodes=RISK_GRID["rate_nodes"],
        maturity_nodes=RISK_GRID["maturity_nodes"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=N_TEST_POINTS)
    parser.add_argument("--rows-per-call", type=int, default=1 << 20)
    parser.add_argument(
        "--candidates", type=int, nargs="+", default=list(CANDIDATES)
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    n_points = int(args.points)
    reference_config = build_config(0, 16)
    shared = shared_axis_points(n_points, reference_config, TEST_SEED_SHARED)
    sigma = sigma_box_draws(
        n_points, reference_config.n_assets, TEST_SEED_SIGMA_BOX
    )

    print(
        f"phase 1: {n_points} off-grid test points, sigma uniform and "
        f"independent per asset on {SIGMA_BOUNDS}"
    )

    # --- the 8-D floor on the common slice. Independent of n_sigma, so once. --
    slice_model = SigmaAugmentedSurrogate(reference_config, RISK_GRID["mode"])
    slice_market = assemble_market_points(reference_config, shared, sigma)
    start = perf_counter()
    slice_floor = slice_model.interpolation_floor(
        slice_market, rows_per_call=int(args.rows_per_call)
    )
    slice_exact = slice_model.reference_price(slice_market)
    slice_metrics = error_metrics(slice_exact, slice_floor)
    slice_seconds = perf_counter() - start
    print(
        f"  8D floor (d_sigma=0, sigma={SIGMA_REF}): "
        f"nmae={slice_metrics['normalized_mae']:.6e} "
        f"mae={slice_metrics['mae']:.6e}  [{slice_seconds:.1f}s]"
    )

    rows, retained = [], None
    for n_sigma_nodes in args.candidates:
        # Every asset promoted: the 13-D end of the sweep, which is where the
        # volatility axes could raise the floor the most.
        config = build_config(reference_config.n_assets, n_sigma_nodes)
        model = SigmaAugmentedSurrogate(config, RISK_GRID["mode"])
        market = assemble_market_points(config, shared, sigma)
        start = perf_counter()
        floor = model.interpolation_floor(
            market, rows_per_call=int(args.rows_per_call)
        )
        exact = model.reference_price(market)
        metrics = error_metrics(exact, floor)
        seconds = perf_counter() - start

        ratio = metrics["normalized_mae"] / slice_metrics["normalized_mae"]
        passes = bool(ratio <= PHASE1_TOLERANCE)
        placement = sigma_node_placement(n_sigma_nodes)
        rows.append(
            {
                "n_sigma": int(n_sigma_nodes),
                "L_sigma": int(np.log2(n_sigma_nodes)),
                "d_sigma": int(config.n_sigma_axes),
                "qtt_cores": int(config.n_qtt_cores),
                "floor_13d_nmae": metrics["normalized_mae"],
                "floor_8d_slice_nmae": slice_metrics["normalized_mae"],
                "ratio_13d_over_8d": ratio,
                "satisfies_criterion": passes,
                "retained": False,
                "floor_13d_mae": metrics["mae"],
                "floor_13d_max_abs_error": metrics["max_absolute_error"],
                "floor_13d_bias": metrics["mean_error_bias"],
                "floor_8d_slice_mae": slice_metrics["mae"],
                "sigma_axis_step": placement["sigma_axis_step"],
                "bracketing_node_low": placement["bracketing_node_low"],
                "bracketing_node_high": placement["bracketing_node_high"],
                "cell_fraction_of_sigma_ref": placement["cell_fraction_of_sigma_ref"],
                "sigma_ref_on_node": placement["sigma_ref_on_node"],
                "effective_corners_per_query": model.floor_diagnostics[
                    "effective_corners_per_query"
                ],
                "product_stencil_corners_per_query": model.floor_diagnostics[
                    "product_stencil_corners_per_query"
                ],
                "collapse_normalized_residual": model.floor_diagnostics[
                    "collapse_normalized_residual"
                ],
                "seconds": seconds,
            }
        )
        print(
            f"  n_sigma={n_sigma_nodes:3d} (L={int(np.log2(n_sigma_nodes))}) "
            f"cores={config.n_qtt_cores:3d} "
            f"floor_13D={metrics['normalized_mae']:.6e} "
            f"ratio={ratio:7.4f} {'PASS' if passes else 'FAIL'}  [{seconds:.1f}s]"
        )
        if passes and retained is None:
            retained = int(n_sigma_nodes)

    for row in rows:
        row["retained"] = bool(row["n_sigma"] == retained)

    csv_path = write_csv(Path(args.output_dir) / "phase1_choix_nsigma.csv", rows)
    write_json(
        Path(args.output_dir) / "phase1_choix_nsigma.json",
        {
            "manifest": run_manifest(),
            "n_test_points": n_points,
            "criterion": (
                "smallest n_sigma whose 13-D production interpolation floor "
                f"exceeds the 8-D floor on the common slice by at most "
                f"{100 * (PHASE1_TOLERANCE - 1):.0f}% relative"
            ),
            "floor_8d_slice": slice_metrics,
            "floor_8d_collapse_diagnostics": slice_model.floor_diagnostics,
            "retained_n_sigma": retained,
            "rows": rows,
        },
    )
    print(f"\nwrote {csv_path}")

    if retained is None:
        print(
            "\nSTOP: no candidate n_sigma <= 32 satisfies the criterion. The "
            "volatility axes raise the grid's interpolation floor by more than "
            f"{100 * (PHASE1_TOLERANCE - 1):.0f}% at every tested resolution. "
            "This is itself a result; phases 2 onward are not run."
        )
        return 1

    print(f"\nRETAINED n_sigma = {retained}. Frozen for every later phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
