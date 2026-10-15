r"""Phase 3 - dry run of the whole chain at a tiny budget.

One purpose only: confirm the pipeline runs end to end and that every counter
the protocol asks for is actually collected - realized oracle calls, the bond
rank profile core by core, the sigma-block and boundary ranks, TT parameters,
timings, the three test sets, the interpolation floors and the additivity check.

The numbers this produces are discarded. They are not results and are not
reported as such. What is reported is the per-fit wall time, which is what sizes
the main sweep.

    .venv/Scripts/python.exe experiments/05_volatility_cost/phase3_dry_run.py --n-sigma 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    OUTPUT_DIR,
    RISK_GRID,
    SEEDS,
    SIGMA_BOUNDS,
    SIGMA_REF,
    TEST_SEED_SHARED,
    TEST_SEED_SIGMA_BOX,
    run_manifest,
    shared_axis_points,
    sigma_box_draws,
    write_json,
)
from measure import build_test_sets, interpolation_floors, run_one_fit

from stngpr.sigma_axes import SigmaAugmentedConfig
from stngpr.sigma_surrogate import SigmaAugmentedSurrogate

# Counters every fit must carry, whatever d_sigma is.
REQUIRED_KEYS = (
    "oracle_calls_realized",
    "bond_ranks_full",
    "tt_parameters",
    "effective_rank",
    "maximum_rank",
    "qtt_cores",
    "log2_logical_entries",
    "fit_seconds",
    "batch_seconds_per_query",
    "single_seconds_per_query",
    "nmae_common_slice",
    "floor_nmae_common_slice",
    "resolution_status",
    "shared_core_order_preserved",
    "cross_stop",
)

# Counters that only exist once there is a sigma block to measure. At
# d_sigma = 0 they are legitimately absent, and demanding them would turn a
# correct run into a spurious failure.
SIGMA_ONLY_KEYS = (
    "sigma_block_internal_max_rank",
    "sigma_boundary_max_rank",
    "nmae_sigma_box",
    "nmae_off_node",
)


def build_config(d_sigma: int, n_sigma: int, **overrides) -> SigmaAugmentedConfig:
    return SigmaAugmentedConfig(
        n_sigma_axes=d_sigma,
        sigma_block="tail",
        sigma_reference=SIGMA_REF,
        volatility_bounds=SIGMA_BOUNDS,
        volatility_nodes=n_sigma,
        spot_nodes=RISK_GRID["spot_nodes"],
        strike_nodes=RISK_GRID["strike_nodes"],
        rate_nodes=RISK_GRID["rate_nodes"],
        maturity_nodes=RISK_GRID["maturity_nodes"],
        **overrides,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sigma", type=int, required=True)
    parser.add_argument("--budget", type=int, default=2_000)
    parser.add_argument("--anova-samples", type=int, default=500)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    reference_config = build_config(0, int(args.n_sigma))
    shared = shared_axis_points(int(args.points), reference_config, TEST_SEED_SHARED)
    sigma_box = sigma_box_draws(
        int(args.points), reference_config.n_assets, TEST_SEED_SIGMA_BOX
    )

    print(
        f"phase 3 dry run: budget={args.budget} anova={args.anova_samples} "
        f"points={args.points} n_sigma={args.n_sigma}  (numbers discarded)"
    )

    payload, missing = {"manifest": run_manifest(), "fits": []}, {}
    for d_sigma in (0, 5):
        config = build_config(d_sigma, int(args.n_sigma))
        model = SigmaAugmentedSurrogate(config, RISK_GRID["mode"])
        test_sets = build_test_sets(config, shared, sigma_box)

        start = perf_counter()
        floors = interpolation_floors(model, test_sets)
        floor_seconds = perf_counter() - start

        record, fitted = run_one_fit(
            config,
            RISK_GRID["mode"],
            int(args.budget),
            SEEDS[0],
            test_sets,
            floors,
            reference_config,
            anova_samples=int(args.anova_samples),
            latency_repeats=2,
        )
        expected = REQUIRED_KEYS + (SIGMA_ONLY_KEYS if d_sigma else ())
        missing[d_sigma] = [key for key in expected if record.get(key) is None]
        record["floor_seconds_all_sets"] = floor_seconds
        payload["fits"].append(record)

        print(
            f"  d_sigma={d_sigma}: cores={record['qtt_cores']} "
            f"calls={record['oracle_calls_realized']} stop={record['cross_stop']} "
            f"erank={record['effective_rank']:.2f} "
            f"max_rank={record['maximum_rank']} "
            f"params={record['tt_parameters']} "
            f"fit={record['fit_seconds']:.1f}s floors={floor_seconds:.1f}s"
        )
        print(
            f"    sigma_internal_max_rank={record['sigma_block_internal_max_rank']} "
            f"sigma_boundary_max_rank={record['sigma_boundary_max_rank']} "
            f"bond_profile_length={len(record['bond_ranks_full'])} "
            f"additive_chain_ok={record['shared_core_order_preserved']}"
        )
        print(
            f"    nmae_slice={record['nmae_common_slice']:.4e} "
            f"floor_slice={record['floor_nmae_common_slice']:.4e} "
            f"query={record['batch_seconds_per_query']:.2e}s "
            f"status={record['resolution_status']}"
        )
        del fitted, model

    payload["missing_counters"] = missing
    all_present = not any(missing.values())
    payload["all_counters_collected"] = all_present
    write_json(Path(args.output_dir) / "phase3_dry_run.json", payload)

    scale = np.array([fit["fit_seconds"] for fit in payload["fits"]])
    print(
        f"\nfit wall time at budget {args.budget}: "
        f"d_sigma=0 {scale[0]:.1f}s, d_sigma=5 {scale[1]:.1f}s"
    )
    if not all_present:
        print(f"\nSTOP: counters missing: {missing}")
        return 1
    print("all required counters collected; the chain runs end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
