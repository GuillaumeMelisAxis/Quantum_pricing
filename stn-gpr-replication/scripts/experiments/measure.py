r"""Per-fit measurement used by the dry run, the main sweep and the controls.

One place defines what is recorded for a fit, so phases 3, 4 and 6 cannot drift
apart in what they report or in how they compute it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    ANOVA_SAMPLES,
    MAX_SWEEPS,
    N_LATENCY_POINTS,
    RANK_INCREMENT,
    SIGMA_OFF_NODE,
    SIGMA_REF,
    UNRESOLVED_FACTOR,
    assemble_market_points,
)

from stngpr.sigma_surrogate import SigmaAugmentedSurrogate
from stngpr.validation import error_metrics


def bond_classification(config) -> dict:
    r"""Split the bonds ``r_0 ... r_d`` by their relation to the sigma block.

    Bond ``r_k`` joins cores ``k-1`` and ``k``, with ``r_0 = r_d = 1``. A bond is
    *internal* to the sigma block when both of its cores are sigma cores, and a
    *boundary* bond when exactly one is. Stated this way the split is
    layout-agnostic: the tail order has a single boundary bond, the interleaved
    control order has many, and neither needs a special case.

    The two terminal bonds ``r_0`` and ``r_d`` are excluded from both classes:
    they are one by construction, carry no information, and tagging the last one
    as a boundary bond would put a structural 1 in the same column as the rank
    that actually crosses from the shared axes into the sigma block.
    """
    sigma = set(config.sigma_core_indices)
    n_cores = config.n_qtt_cores
    internal, boundary = [], []
    for k in range(1, n_cores):
        left, right = (k - 1) in sigma, k in sigma
        if left and right:
            internal.append(k)
        elif left or right:
            boundary.append(k)
    return {"internal_sigma_bonds": internal, "boundary_bonds": boundary}


def verify_additive_chain(config, reference_config) -> dict:
    """Check that this chain is the shared chain with sigma cores inserted.

    Reports rather than merely asserts, so the verification appears in the
    results instead of only in a traceback that nobody reads.
    """
    sigma = set(config.sigma_core_indices)
    core_axis = []
    for column, bits in enumerate(config.qtt_bits):
        core_axis.extend([column] * bits)
    kept = [
        config.shared_columns.index(column)
        for core, column in enumerate(core_axis)
        if core not in sigma
    ]
    expected = []
    for column, bits in enumerate(reference_config.qtt_bits):
        expected.extend([column] * bits)
    return {
        "shared_core_order_preserved": bool(kept == expected),
        "shared_core_count": len(kept),
        "reference_core_count": len(expected),
        "sigma_core_count": len(sigma),
        "sigma_cores_contiguous": bool(
            not sigma or sorted(sigma) == list(range(min(sigma), max(sigma) + 1))
        ),
        "sigma_block_at_chain_tail": bool(
            not sigma or max(sigma) == config.n_qtt_cores - 1
        ),
    }


def build_test_sets(config, shared, sigma_box):
    """The three test sets, all sharing one set of shared-axis values.

    * ``common_slice`` - every volatility at ``sigma_ref``. Defined for both the
      8-D and the augmented configurations, so this is the only set on which an
      error comparison across ``d_sigma`` means anything.
    * ``sigma_box`` - volatilities drawn from the full box. Undefined at
      ``d_sigma = 0``.
    * ``off_node`` - every volatility at ``sigma_off_node``, a value deliberately
      away from any sigma axis node. Undefined at ``d_sigma = 0``.
    """
    n = shared.shape[0]
    n_assets = config.n_assets
    sets = {
        "common_slice": assemble_market_points(
            config, shared, np.full((n, n_assets), SIGMA_REF)
        )
    }
    if config.n_sigma_axes:
        sets["sigma_box"] = assemble_market_points(config, shared, sigma_box)
        sets["off_node"] = assemble_market_points(
            config, shared, np.full((n, n_assets), SIGMA_OFF_NODE)
        )
    return sets


def interpolation_floors(model, test_sets, rows_per_call: int = 1 << 20) -> dict:
    """Production interpolation floor of this grid on each test set."""
    out = {}
    for name, market in test_sets.items():
        start = perf_counter()
        floor = model.interpolation_floor(market, rows_per_call=rows_per_call)
        exact = model.reference_price(market)
        out[name] = {
            **error_metrics(exact, floor),
            "collapse_diagnostics": dict(model.floor_diagnostics),
            "seconds": perf_counter() - start,
        }
    return out


def query_latency(model, market_points, repeats: int = 5) -> dict:
    """Per-query cost on a fixed paired subset, monotonic clock.

    The same parameter vectors are used for every configuration, so the numbers
    compare; batch and single-query costs are both kept because they answer
    different questions and the batch one is the fairer headline.
    """
    points = np.atleast_2d(market_points)[:N_LATENCY_POINTS]
    model.price(points)  # warm up caches and any lazy import
    batch = []
    for _ in range(int(repeats)):
        start = perf_counter()
        model.price(points)
        batch.append(perf_counter() - start)
    single = []
    for _ in range(int(repeats)):
        start = perf_counter()
        for row in range(len(points)):
            model.price(points[row : row + 1])
        single.append(perf_counter() - start)
    return {
        "latency_points": len(points),
        "batch_seconds_per_query": float(np.median(batch)) / len(points),
        "single_seconds_per_query": float(np.median(single)) / len(points),
    }


def run_one_fit(
    config,
    mode: str,
    budget: int,
    seed: int,
    test_sets: dict,
    floors: dict,
    reference_config,
    anova_samples: int = ANOVA_SAMPLES,
    latency_repeats: int = 5,
    cross_log: bool = False,
):
    """Fit once and record everything the protocol asks for about that fit."""
    model = SigmaAugmentedSurrogate(config, mode, seed=seed)
    diagnostics = model.fit(
        budget,
        anova_samples=anova_samples,
        max_sweeps=MAX_SWEEPS,
        rank_increment=RANK_INCREMENT,
        log=cross_log,
    )
    profile = model.bond_rank_profile()
    bonds = bond_classification(config)
    cores = model.core_diagnostics()

    internal = [profile[k] for k in bonds["internal_sigma_bonds"]]
    boundary = [profile[k] for k in bonds["boundary_bonds"]]

    record = {
        "d_sigma": int(config.n_sigma_axes),
        "n_sigma": int(config.volatility_nodes),
        "sigma_block": config.sigma_block,
        "mode": mode,
        "budget_nominal": int(budget),
        "oracle_calls_realized": int(diagnostics.function_evaluations),
        "anova_samples": int(anova_samples),
        "cross_stop": diagnostics.stop,
        "cross_sweeps": int(diagnostics.sweeps),
        "seed": int(seed),
        "qtt_cores": int(config.n_qtt_cores),
        "log2_logical_entries": int(config.n_qtt_cores),
        "effective_rank": float(diagnostics.effective_rank),
        "maximum_rank": int(diagnostics.maximum_rank),
        "sigma_block_internal_max_rank": int(max(internal)) if internal else None,
        "sigma_boundary_max_rank": int(max(boundary)) if boundary else None,
        "tt_parameters": int(cores.parameter_count),
        "post_fit_svd_truncation": diagnostics.truncation,
        "fit_seconds": float(model.fit_seconds),
        "bond_ranks_full": profile,
        **verify_additive_chain(config, reference_config),
    }

    record.update(query_latency(model, test_sets["common_slice"], latency_repeats))

    for name, market in test_sets.items():
        exact = model.reference_price(market)
        metrics = error_metrics(exact, model.price(market))
        record[f"nmae_{name}"] = metrics["normalized_mae"]
        record[f"mae_{name}"] = metrics["mae"]
        record[f"bias_{name}"] = metrics["mean_error_bias"]
        record[f"floor_nmae_{name}"] = floors[name]["normalized_mae"]
        record[f"nmae_over_floor_{name}"] = (
            metrics["normalized_mae"] / floors[name]["normalized_mae"]
        )
    for name in ("sigma_box", "off_node"):
        if name not in test_sets:
            for key in ("nmae", "mae", "bias", "floor_nmae", "nmae_over_floor"):
                record[f"{key}_{name}"] = "NOT APPLICABLE"

    ratio = record["nmae_over_floor_common_slice"]
    record["resolution_status"] = (
        "UNRESOLVED" if ratio > UNRESOLVED_FACTOR else "resolved"
    )
    return record, model
