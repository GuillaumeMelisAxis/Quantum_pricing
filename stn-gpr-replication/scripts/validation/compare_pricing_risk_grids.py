from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from audit_price_greek_interpolant_consistency import (
    LOCAL_THRESHOLDS,
    _compare,
    _direct_oracle_interpolator,
    _evaluate_mode,
    _record,
)
from validate_dense_greek_profiles import (
    SPOT_PANELS,
    analytical_components,
    dense_market_panel,
)
from validate_full_spot_hessian import _component_labels, _diagnostics
from validate_tt_greek_convergence import (
    ACCEPTANCE_THRESHOLDS,
    _power_of_two,
    _scalar_summary,
)

from stngpr.config import PaperConfig
from stngpr.convergence import components_to_json, finite_difference_component_arrays
from stngpr.coordinates import CachedGridInterpolator, TransformedPricer
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk_grids import build_greek_coordinate_grid
from stngpr.tt_surrogate import TTPriceSurrogate

PROFILES = {
    "smoke": {
        "budget": 5_000,
        "anova_samples": 300,
        "tt_seeds": [20260401],
        "moneyness": [-0.05, 0.0, 0.05],
        "maturity_days": [30],
        "matched_shape": (16, 16, 16, 16, 16, 64, 8, 16),
        "dense_points": 21,
    },
    "intermediate": {
        "budget": 100_000,
        "anova_samples": 1_000,
        "tt_seeds": [20260401, 20260402],
        "moneyness": [-0.1, -0.05, 0.0, 0.05, 0.1],
        "maturity_days": [7, 30],
        "matched_shape": (32, 32, 32, 32, 32, 256, 8, 32),
        "dense_points": 51,
    },
    "paper": {
        "budget": 200_000,
        "anova_samples": 2_000,
        "tt_seeds": [20260401, 20260402, 20260403, 20260404, 20260405],
        "moneyness": [-0.1, -0.05, 0.0, 0.05, 0.1],
        "maturity_days": [7, 30, 90],
        "matched_shape": (64, 64, 64, 64, 64, 512, 8, 64),
        "dense_points": 101,
    },
}

PRODUCTION_GRIDS = {
    "pricing_grid": {
        "coordinate_mode": "price_adaptive",
        "physical_shape": (32, 32, 32, 32, 32, 64, 8, 8),
        "role": "paper-I price-adaptive log-moneyness grid",
    },
    "risk_hybrid_grid": {
        "coordinate_mode": "bounded_standardized_risk",
        "physical_shape": (64, 64, 64, 64, 64, 512, 8, 64),
        "role": "final Greek-oriented standardized-risk grid",
    },
}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _config(shape) -> PaperConfig:
    return replace(PaperConfig(), physical_shape=tuple(int(value) for value in shape))


def _validate_shape(parser, name, shape) -> None:
    if len(shape) != PaperConfig().n_assets + 3:
        parser.error(f"{name} must contain eight physical mode sizes")
    if any(not _power_of_two(value) for value in shape):
        parser.error(f"every entry of {name} must be a power of two greater than one")


def _fit_or_load(
    grid,
    oracle,
    seed,
    budget,
    anova_samples,
    max_sweeps,
    rank_increment,
    cross_log,
    cache_directory,
    force_refit,
):
    cache_directory.mkdir(parents=True, exist_ok=True)
    core_path = cache_directory / f"seed_{seed}__budget_{budget}.npz"
    metadata_path = cache_directory / f"seed_{seed}__budget_{budget}.json"
    model = TTPriceSurrogate(grid, oracle, seed=seed)
    if core_path.exists() and metadata_path.exists() and not force_refit:
        core_diagnostics = model.load_untruncated_cores(core_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return model, {
            "source": "raw_core_cache",
            "fit": metadata,
            "core_diagnostics": asdict(core_diagnostics),
        }

    start = perf_counter()
    fit = model.fit(
        budget,
        anova_samples=anova_samples,
        max_sweeps=max_sweeps,
        rank_increment=rank_increment,
        truncation=None,
        log=cross_log,
    )
    metadata = {**asdict(fit), "total_time_seconds": perf_counter() - start}
    model.save_untruncated_cores(core_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, {
        "source": "new_fit",
        "fit": metadata,
        "core_diagnostics": asdict(model.apply_truncation(None)),
    }


def _interpolator(model, batch_size):
    def predict(model_points, cubic_columns=()):
        return model.predict_hybrid_cubic(
            model_points,
            cubic_columns=cubic_columns,
            batch_size=batch_size,
        )

    return predict


def _evaluate_record(
    mode,
    interpolator,
    transform,
    points,
    risk_columns,
    relative_bump,
    analytical,
    component_labels,
    m_values,
    slices,
):
    start = perf_counter()
    estimate = _evaluate_mode(
        mode,
        interpolator,
        transform,
        points,
        risk_columns,
        relative_bump,
    )
    return estimate, _record(
        analytical,
        estimate,
        component_labels,
        m_values,
        slices,
        perf_counter() - start,
        mode,
    )


def _summary_values(record, fit_record=None):
    global_error = record["global_error_against_analytical"]
    hessian = record["full_hessian_error_against_analytical"]
    local = record["local_error_audit"]
    values = {
        "price_normalized_mae": global_error["price"]["normalized_mae"],
        "delta_normalized_mae": global_error["delta"]["normalized_mae"],
        "gamma_diagonal_normalized_mae": global_error["gamma_diagonal"][
            "normalized_mae"
        ],
        "cross_gamma_normalized_mae": global_error["cross_gamma"][
            "normalized_mae"
        ],
        "hessian_relative_frobenius_error": hessian[
            "aggregate_relative_frobenius_error"
        ],
        "negative_part_relative_frobenius_norm": hessian[
            "negative_part_relative_frobenius_norm"
        ],
        "cross_gamma_sign_agreement": record["cross_gamma_sign_fidelity"][
            "sign_agreement_fraction"
        ],
        **local,
        "evaluation_time_seconds": record["wall_time_seconds"],
    }
    if fit_record is not None:
        core = fit_record["core_diagnostics"]
        fit = fit_record["fit"]
        values.update(
            {
                "effective_rank": core["effective_rank"],
                "maximum_rank": core["maximum_rank"],
                "parameter_count": core["parameter_count"],
                "function_evaluations": fit["function_evaluations"],
                "fit_time_seconds": fit["total_time_seconds"],
            }
        )
    return values


def _summarize_production(payload, grid_names, seeds):
    summaries = {}
    for grid_name in grid_names:
        grid_record = payload["production_comparison"]["grids"][grid_name]
        runs = grid_record["tt_runs"]
        complete = all(str(seed) in runs for seed in seeds)
        rows = [
            _summary_values(runs[str(seed)]["record"], runs[str(seed)]["fit"])
            for seed in seeds
            if str(seed) in runs
        ]
        metrics = (
            {
                name: _scalar_summary([row[name] for row in rows])
                for name in rows[0]
            }
            if rows
            else {}
        )
        summaries[grid_name] = {
            "run_count": len(rows),
            "expected_run_count": len(seeds),
            "complete": complete,
            "robustly_accepted": complete
            and all(
                runs[str(seed)]["record"]["greek_aware_local_acceptance"][
                    "passed"
                ]
                for seed in seeds
            ),
            "metrics": metrics,
        }

    paired = {}
    left, right = grid_names
    for seed in seeds:
        left_run = payload["production_comparison"]["grids"][left]["tt_runs"].get(
            str(seed)
        )
        right_run = payload["production_comparison"]["grids"][right]["tt_runs"].get(
            str(seed)
        )
        if left_run is None or right_run is None:
            continue
        left_values = _summary_values(left_run["record"], left_run["fit"])
        right_values = _summary_values(right_run["record"], right_run["fit"])
        paired[str(seed)] = {
            name: {
                "pricing_grid": left_values[name],
                "risk_hybrid_grid": right_values[name],
                "risk_over_pricing_ratio": (
                    right_values[name] / left_values[name]
                    if abs(left_values[name]) > np.finfo(float).tiny
                    else None
                ),
            }
            for name in left_values
        }

    risk = summaries["risk_hybrid_grid"]
    pricing = summaries["pricing_grid"]
    role_recommendation = {
        "pricing_surface": (
            "pricing_grid"
            if pricing["complete"]
            and pricing["metrics"]["price_normalized_mae"]["maximum"]
            <= ACCEPTANCE_THRESHOLDS["price_normalized_mae"]
            else "undetermined"
        ),
        "risk_surface": (
            "risk_hybrid_grid" if risk["robustly_accepted"] else "undetermined"
        ),
        "single_surface_constraint": (
            "price, Delta and Hessian must be taken from the same selected surface"
        ),
    }
    return {
        "grids": summaries,
        "paired_seed_comparison": paired,
        "role_recommendation": role_recommendation,
    }


def _cache_signature(path, signature, overwrite):
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / "manifest.json"
    if manifest.exists() and not overwrite:
        current = json.loads(manifest.read_text(encoding="utf-8"))
        if current != signature:
            raise RuntimeError(
                "the core cache is incompatible; select another cache directory "
                "or pass --overwrite"
            )
    else:
        manifest.write_text(json.dumps(signature, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Final v9.8 comparison of the paper-I pricing grid and the "
            "Greek-oriented risk-hybrid grid on the analytical European "
            "geometric-basket benchmark."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--tt-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--anova-samples", type=int, default=None)
    parser.add_argument("--relative-bump", type=float, default=0.002)
    parser.add_argument("--matched-shape", nargs=8, type=int, default=None)
    parser.add_argument("--grid-batch-size", type=int, default=256)
    parser.add_argument("--tt-batch-size", type=int, default=1)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--rank-increment", type=int, default=2)
    parser.add_argument("--dense-profile-points", type=int, default=None)
    parser.add_argument("--dense-maturity-days", type=float, default=30.0)
    parser.add_argument("--cross-log", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-controlled-grid", action="store_true")
    parser.add_argument("--skip-production-grid-oracle", action="store_true")
    parser.add_argument("--skip-dense-profile", action="store_true")
    parser.add_argument(
        "--core-cache-dir",
        type=Path,
        default=Path("results/greeks_v98_raw_cores"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v98_pricing_vs_risk_grid.json"),
    )
    args = parser.parse_args()

    design = PROFILES[args.profile]
    budget = int(args.budget or design["budget"])
    anova_samples = int(args.anova_samples or design["anova_samples"])
    seeds = list(dict.fromkeys(args.tt_seeds or design["tt_seeds"]))
    matched_shape = tuple(args.matched_shape or design["matched_shape"])
    dense_points = int(args.dense_profile_points or design["dense_points"])
    if budget <= 0 or anova_samples <= 0:
        parser.error("budget and anova-samples must be positive")
    if not seeds:
        parser.error("at least one TT seed is required")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative-bump must lie in (0,1)")
    if dense_points < 5:
        parser.error("dense-profile-points must be at least five")
    _validate_shape(parser, "matched-shape", matched_shape)
    for grid_name, specification in PRODUCTION_GRIDS.items():
        _validate_shape(parser, f"{grid_name} shape", specification["physical_shape"])

    signature = json.loads(json.dumps({
        "profile": args.profile,
        "budget": budget,
        "anova_samples": anova_samples,
        "tt_seeds": seeds,
        "relative_bump": float(args.relative_bump),
        "moneyness": list(design["moneyness"]),
        "maturity_days": list(design["maturity_days"]),
        "matched_shape": list(matched_shape),
        "production_grids": PRODUCTION_GRIDS,
        "dense_profile_points": dense_points,
        "dense_maturity_days": float(args.dense_maturity_days),
        "max_sweeps": int(args.max_sweeps),
        "rank_increment": int(args.rank_increment),
    }))
    _cache_signature(args.core_cache_dir, signature, args.overwrite)

    reference_config = PaperConfig()
    risk_columns = tuple(range(reference_config.n_assets))
    component_labels = _component_labels(risk_columns)
    points, m_values, t_values, panel_labels, slices, rate = dense_market_panel(
        reference_config,
        ["oos_mid"],
        design["maturity_days"],
        np.asarray(design["moneyness"], dtype=float),
        args.relative_bump,
    )
    market_pricer = EuropeanGeometricBasketPricer(reference_config)
    analytical = analytical_components(
        reference_config,
        market_pricer,
        points,
        risk_columns,
    )
    exact_fd = finite_difference_component_arrays(
        market_pricer,
        points,
        risk_columns,
        args.relative_bump,
    )

    if args.output.exists() and not args.overwrite:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "greeks-pricing-risk-grid-v9.8"
            or payload.get("experiment_signature") != signature
        ):
            parser.error(
                "the output checkpoint is incompatible; choose another path "
                "or pass --overwrite"
            )
        print(f"Resuming checkpoint {args.output}")
    else:
        payload = {
            "schema_version": "greeks-pricing-risk-grid-v9.8",
            "profile": args.profile,
            "benchmark": "European geometric-basket put with analytical Greeks",
            "scope": {
                "included": ["European geometric-basket put"],
                "excluded": [
                    "European arithmetic-basket options",
                    "American options",
                    "American arithmetic-basket options",
                ],
                "interpretation": (
                    "excluded products are separate research directions and are "
                    "not part of the claims of this experiment"
                ),
            },
            "hypothesis": (
                "the paper-I price-adaptive grid remains efficient for prices, "
                "whereas the standardized risk-hybrid grid is required for "
                "stable fixed-strike Delta and Hessian recovery from one scalar surface"
            ),
            "experiment_signature": signature,
            "settings": {
                "budget": budget,
                "anova_samples": anova_samples,
                "tt_seeds": seeds,
                "relative_bump": float(args.relative_bump),
                "matched_shape": list(matched_shape),
                "interpolation_floor_mode": "component_hybrid",
                "production_interpolation_mode": "unified_risk_cubic",
                "post_fit_truncation": None,
                "monte_carlo": False,
            },
            "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
            "local_acceptance_thresholds": LOCAL_THRESHOLDS,
            "test_design": {
                "market_parameters": points.tolist(),
                "log_moneyness": m_values.tolist(),
                "maturity_days": t_values.tolist(),
                "spot_panel_labels": panel_labels.tolist(),
                "curve_slices": slices,
                "spot_panels": {"oos_mid": SPOT_PANELS["oos_mid"]},
                "component_labels": component_labels,
                "rate": rate,
            },
            "analytical_reference": {"components": components_to_json(analytical)},
            "exact_price_finite_difference_control": {
                "components": components_to_json(exact_fd),
                **_diagnostics(
                    analytical,
                    exact_fd,
                    component_labels,
                    m_values,
                    slices,
                ),
            },
            "controlled_coordinate_ablation": {
                "purpose": (
                    "isolate coordinate placement at identical physical mode sizes, "
                    "spot/rate/maturity axes and exact grid-node prices"
                ),
                "physical_shape": list(matched_shape),
                "interpolation_mode": "component_hybrid",
                "grids": {},
            },
            "production_comparison": {
                "purpose": (
                    "compare the two native end-to-end QTT architectures using "
                    "raw TT-cross cores and one unified scalar cubic surface"
                ),
                "grids": {},
                "summary": {},
            },
            "dense_profiles": {
                "anchor_seed": int(seeds[0]),
                "maturity_days": float(args.dense_maturity_days),
                "moneyness_bounds": [-0.35, 0.35],
                "point_count": dense_points,
                "risk_columns": [0, 1],
                "grids": {},
            },
        }
        _write(args.output, payload)

    if not args.skip_controlled_grid:
        controlled_config = _config(matched_shape)
        for grid_name, specification in PRODUCTION_GRIDS.items():
            controlled = payload["controlled_coordinate_ablation"]["grids"]
            if grid_name in controlled:
                continue
            mode = specification["coordinate_mode"]
            print(f"controlled exact-grid evaluation: {grid_name}")
            grid, transform, description = build_greek_coordinate_grid(
                controlled_config,
                mode,
            )
            oracle = TransformedPricer(market_pricer, transform)
            interpolator = CachedGridInterpolator(
                grid,
                oracle,
                evaluation_batch_size=args.grid_batch_size,
            )
            estimate, record = _evaluate_record(
                "component_hybrid",
                interpolator,
                transform,
                points,
                risk_columns,
                args.relative_bump,
                analytical,
                component_labels,
                m_values,
                slices,
            )
            controlled[grid_name] = {
                "coordinate_mode": mode,
                "coordinate_description": description,
                "cache": interpolator.diagnostics(),
                "record": record,
            }
            _write(args.output, payload)

    models = {}
    for grid_name, specification in PRODUCTION_GRIDS.items():
        production_grids = payload["production_comparison"]["grids"]
        production = production_grids.setdefault(
            grid_name,
            {
                "role": specification["role"],
                "coordinate_mode": specification["coordinate_mode"],
                "physical_shape": list(specification["physical_shape"]),
                "qtt_core_count": int(
                    sum(_config(specification["physical_shape"]).qtt_bits)
                ),
                "logical_tensor_entries": int(np.prod(specification["physical_shape"])),
                "coordinate_description": {},
                "grid_oracle": {},
                "tt_runs": {},
            },
        )
        config = _config(specification["physical_shape"])
        grid, transform, description = build_greek_coordinate_grid(
            config,
            specification["coordinate_mode"],
        )
        production["coordinate_description"] = description
        oracle = TransformedPricer(market_pricer, transform)

        if not args.skip_production_grid_oracle and not production["grid_oracle"]:
            print(f"production exact-grid scalar audit: {grid_name}")
            estimate, record = _evaluate_record(
                "unified_risk_cubic",
                _direct_oracle_interpolator(grid, oracle),
                transform,
                points,
                risk_columns,
                args.relative_bump,
                analytical,
                component_labels,
                m_values,
                slices,
            )
            production["grid_oracle"] = {
                "evaluation_strategy": (
                    "direct vectorized exact-node evaluation without a persistent "
                    "multi-million-node cache"
                ),
                "record": record,
            }
            _write(args.output, payload)

        for seed in seeds:
            if str(seed) in production["tt_runs"]:
                continue
            print(f"raw TT audit: grid={grid_name} seed={seed}")
            model, fit_record = _fit_or_load(
                grid,
                oracle,
                seed,
                budget,
                anova_samples,
                args.max_sweeps,
                args.rank_increment,
                args.cross_log,
                args.core_cache_dir / grid_name,
                args.overwrite,
            )
            models[(grid_name, seed)] = model
            estimate, record = _evaluate_record(
                "unified_risk_cubic",
                _interpolator(model, args.tt_batch_size),
                transform,
                points,
                risk_columns,
                args.relative_bump,
                analytical,
                component_labels,
                m_values,
                slices,
            )
            if production["grid_oracle"]:
                record["error_against_exact_grid_interpolant"] = _compare(
                    production["grid_oracle"]["record"]["components"],
                    estimate,
                )
            production["tt_runs"][str(seed)] = {
                "tt_seed": int(seed),
                "fit": fit_record,
                "record": record,
            }
            _write(args.output, payload)

    payload["production_comparison"]["summary"] = _summarize_production(
        payload,
        list(PRODUCTION_GRIDS),
        seeds,
    )
    _write(args.output, payload)

    if not args.skip_dense_profile:
        dense_m = np.linspace(-0.35, 0.35, dense_points)
        dense_columns = (0, 1)
        dense_labels = _component_labels(dense_columns)
        dense_points_market, dense_m_values, _, _, dense_slices, _ = dense_market_panel(
            reference_config,
            ["oos_mid"],
            [args.dense_maturity_days],
            dense_m,
            args.relative_bump,
        )
        dense_analytical = analytical_components(
            reference_config,
            market_pricer,
            dense_points_market,
            dense_columns,
        )
        anchor = seeds[0]
        for grid_name, specification in PRODUCTION_GRIDS.items():
            if grid_name in payload["dense_profiles"]["grids"]:
                continue
            config = _config(specification["physical_shape"])
            grid, transform, _ = build_greek_coordinate_grid(
                config,
                specification["coordinate_mode"],
            )
            oracle = TransformedPricer(market_pricer, transform)
            model = models.get((grid_name, anchor))
            if model is None:
                model, _ = _fit_or_load(
                    grid,
                    oracle,
                    anchor,
                    budget,
                    anova_samples,
                    args.max_sweeps,
                    args.rank_increment,
                    args.cross_log,
                    args.core_cache_dir / grid_name,
                    False,
                )
            print(f"dense scalar profile: grid={grid_name} seed={anchor}")
            estimate, record = _evaluate_record(
                "unified_risk_cubic",
                _interpolator(model, args.tt_batch_size),
                transform,
                dense_points_market,
                dense_columns,
                args.relative_bump,
                dense_analytical,
                dense_labels,
                dense_m_values,
                dense_slices,
            )
            payload["dense_profiles"]["grids"][grid_name] = {
                "coordinate_mode": specification["coordinate_mode"],
                "market_parameters": dense_points_market.tolist(),
                "log_moneyness": dense_m_values.tolist(),
                "analytical_components": components_to_json(dense_analytical),
                "record": record,
            }
            _write(args.output, payload)

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
