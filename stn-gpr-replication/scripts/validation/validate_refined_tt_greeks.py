from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import PaperConfig
from stngpr.coordinates import (
    TransformedPricer,
    build_coordinate_grid,
    oracle_hybrid_cubic_predict,
)
from stngpr.greeks import project_symmetric_matrix_psd
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.tt_surrogate import TTPriceSurrogate

from validate_european_greeks import (
    analytical_references,
    component_metrics,
    finite_difference_hybrid_arrays,
)
from validate_short_maturity_greeks import (
    conditional_metrics,
    gamma_matrices,
    grid_layer_diagnostics,
    hessian_shape_diagnostics,
    pointwise_metrics,
    projection_error_diagnostics,
    replace_gamma_components,
    short_atm_panel,
)


DEFAULT_BUDGETS = (20_000, 50_000, 100_000)


def _power_of_two(value: int) -> bool:
    return value > 1 and value & (value - 1) == 0


def matrix_comparison(estimate, reference) -> dict:
    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    errors = np.linalg.norm(estimate - reference, axis=(-2, -1))
    scale = np.linalg.norm(reference, axis=(-2, -1))
    mean_scale = float(np.mean(scale))
    return {
        "mean_frobenius_error": float(np.mean(errors)),
        "rmse_frobenius_error": float(np.sqrt(np.mean(errors**2))),
        "maximum_frobenius_error": float(np.max(errors)),
        "mean_reference_frobenius_norm": mean_scale,
        "normalized_mean_frobenius_error": float(
            np.mean(errors) / mean_scale
        ),
    }


def summarize_estimates(
    references,
    estimates,
    log_moneyness,
    maturity_days,
    replicate_ids,
    n_assets,
):
    raw_hessian = gamma_matrices(estimates, n_assets)
    reference_hessian = gamma_matrices(references, n_assets)
    projected_hessian = project_symmetric_matrix_psd(raw_hessian)
    projected_estimates = replace_gamma_components(
        estimates,
        projected_hessian,
    )
    payload = {
        "global": component_metrics(references, estimates),
        "global_psd_projected": component_metrics(
            references,
            projected_estimates,
        ),
        "conditional": conditional_metrics(
            references,
            estimates,
            log_moneyness,
            maturity_days,
        ),
        "conditional_psd_projected": conditional_metrics(
            references,
            projected_estimates,
            log_moneyness,
            maturity_days,
        ),
        "hessian_shape": {
            "reference": hessian_shape_diagnostics(reference_hessian),
            "raw": hessian_shape_diagnostics(raw_hessian),
            "psd_projected": hessian_shape_diagnostics(
                projected_hessian,
            ),
            "projection_error": projection_error_diagnostics(
                raw_hessian,
                projected_hessian,
                reference_hessian,
            ),
        },
        "pointwise": pointwise_metrics(
            references,
            estimates,
            log_moneyness,
            maturity_days,
            replicate_ids,
        ),
        "pointwise_psd_projected": pointwise_metrics(
            references,
            projected_estimates,
            log_moneyness,
            maturity_days,
            replicate_ids,
        ),
    }
    return payload, projected_estimates, raw_hessian, projected_hessian


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moneyness-nodes", type=int, default=512)
    parser.add_argument("--maturity-nodes", type=int, default=64)
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument("--anova-samples", type=int, default=2_000)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--relative-bump", type=float, default=0.002)
    parser.add_argument("--design-seed", type=int, default=None)
    parser.add_argument("--tt-seed", type=int, default=None)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--rank-increment", type=int, default=2)
    parser.add_argument("--truncation", type=float, default=1e-8)
    parser.add_argument("--cross-log", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/refined_tt_greeks_budget_convergence.json"),
    )
    args = parser.parse_args()

    if not _power_of_two(args.moneyness_nodes):
        parser.error("moneyness node count must be a power of two")
    if not _power_of_two(args.maturity_nodes):
        parser.error("maturity node count must be a power of two")
    if any(budget <= 0 for budget in args.budgets):
        parser.error("budgets must be positive")
    if args.anova_samples <= 0 or args.replicates <= 0:
        parser.error("ANOVA samples and replicates must be positive")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative bump must lie in (0, 1)")
    if args.max_sweeps <= 0 or args.rank_increment <= 0:
        parser.error("cross sweep settings must be positive")
    if args.truncation <= 0.0:
        parser.error("truncation must be positive")

    base_config = PaperConfig()
    design_seed = (
        base_config.seed + 202
        if args.design_seed is None
        else args.design_seed
    )
    tt_seed = base_config.seed if args.tt_seed is None else args.tt_seed
    shape = list(base_config.physical_shape)
    shape[base_config.n_assets] = args.moneyness_nodes
    shape[-1] = args.maturity_nodes
    config = replace(base_config, physical_shape=tuple(shape))

    points, log_moneyness, maturity_days, replicate_ids = short_atm_panel(
        config,
        args.replicates,
        (-0.05, -0.025, 0.0, 0.025, 0.05),
        (3.0, 7.0, 14.0, 30.0, 90.0),
        np.random.default_rng(design_seed),
    )
    grid, transform, grid_description = build_coordinate_grid(
        config,
        "moneyness_adaptive",
        basket_kind="geometric",
    )
    market_pricer = EuropeanGeometricBasketPricer(config)
    model_oracle = TransformedPricer(market_pricer, transform)
    references = analytical_references(config, market_pricer, points)

    results = {
        "experiment": "refined-grid TT convergence for short-maturity Greeks",
        "design_seed": design_seed,
        "tt_seed": tt_seed,
        "relative_bump": args.relative_bump,
        "n_points": int(len(points)),
        "replicates": args.replicates,
        "budgets": list(args.budgets),
        "anova_samples": args.anova_samples,
        "cross_settings": {
            "max_sweeps": args.max_sweeps,
            "rank_increment": args.rank_increment,
            "truncation": args.truncation,
        },
        "grid": {
            "physical_shape": list(config.physical_shape),
            "qtt_order": int(sum(config.qtt_bits)),
            "logical_tensor_entries": int(np.prod(config.physical_shape)),
            "description": grid_description,
            "resolution": grid_layer_diagnostics(
                grid,
                config,
                np.array([3.0, 7.0, 14.0, 30.0, 90.0]),
            ),
        },
        "test_design": {
            "market_parameters": points.tolist(),
            "log_moneyness": log_moneyness.tolist(),
            "maturity_days": maturity_days.tolist(),
            "replicate_ids": replicate_ids.tolist(),
        },
        "reference": "analytical geometric-basket put spot Greeks",
        "oracle": None,
        "tt": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def oracle_interpolator(model_parameters, cubic_columns):
        return oracle_hybrid_cubic_predict(
            grid,
            model_oracle,
            model_parameters,
            cubic_columns=cubic_columns,
        )

    oracle_start = perf_counter()
    oracle_estimates = finite_difference_hybrid_arrays(
        oracle_interpolator,
        transform,
        points,
        config.n_assets,
        args.relative_bump,
    )
    oracle_time = perf_counter() - oracle_start
    (
        oracle_summary,
        oracle_projected,
        oracle_hessian,
        oracle_projected_hessian,
    ) = summarize_estimates(
        references,
        oracle_estimates,
        log_moneyness,
        maturity_days,
        replicate_ids,
        config.n_assets,
    )
    results["oracle"] = {
        "wall_time_seconds": oracle_time,
        **oracle_summary,
    }
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"oracle complete in {oracle_time:.2f}s")

    for budget in args.budgets:
        model = TTPriceSurrogate(grid, model_oracle, seed=tt_seed)
        fit_start = perf_counter()
        diagnostics = model.fit(
            budget,
            anova_samples=args.anova_samples,
            max_sweeps=args.max_sweeps,
            rank_increment=args.rank_increment,
            truncation=args.truncation,
            log=args.cross_log,
        )
        fit_time = perf_counter() - fit_start

        evaluation_start = perf_counter()
        tt_estimates = finite_difference_hybrid_arrays(
            model.predict_hybrid_cubic,
            transform,
            points,
            config.n_assets,
            args.relative_bump,
        )
        evaluation_time = perf_counter() - evaluation_start
        (
            tt_summary,
            tt_projected,
            tt_hessian,
            tt_projected_hessian,
        ) = summarize_estimates(
            references,
            tt_estimates,
            log_moneyness,
            maturity_days,
            replicate_ids,
            config.n_assets,
        )

        key = str(int(budget))
        results["tt"][key] = {
            "budget": int(budget),
            "fit": {
                "total_time_seconds": fit_time,
                "cross_time_seconds": diagnostics.wall_time,
                "function_evaluations": diagnostics.function_evaluations,
                "sweeps": diagnostics.sweeps,
                "stop": diagnostics.stop,
                "effective_rank": diagnostics.effective_rank,
            },
            "greek_evaluation_time_seconds": evaluation_time,
            **tt_summary,
            "reconstruction_error_against_oracle": {
                "raw_components": component_metrics(
                    oracle_estimates,
                    tt_estimates,
                ),
                "psd_projected_components": component_metrics(
                    oracle_projected,
                    tt_projected,
                ),
                "raw_hessian": matrix_comparison(
                    tt_hessian,
                    oracle_hessian,
                ),
                "psd_projected_hessian": matrix_comparison(
                    tt_projected_hessian,
                    oracle_projected_hessian,
                ),
            },
        }
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

        raw = tt_summary["global"]
        projected = tt_summary["global_psd_projected"]
        shape_diagnostics = tt_summary["hessian_shape"]["raw"]
        print(
            f"budget={budget:7d} rank={diagnostics.effective_rank:6.2f} "
            f"price={100 * raw['price']['normalized_mae']:6.2f}% "
            f"delta={100 * raw['delta']['normalized_mae']:6.2f}% "
            f"gamma={100 * raw['gamma_diagonal']['normalized_mae']:6.2f}%"
            f"->{100 * projected['gamma_diagonal']['normalized_mae']:6.2f}% "
            f"cross={100 * raw['cross_gamma']['normalized_mae']:6.2f}%"
            f"->{100 * projected['cross_gamma']['normalized_mae']:6.2f}% "
            f"non_psd={shape_diagnostics['matrices_outside_psd_cone']:3d}->0 "
            f"fit={fit_time:6.2f}s eval={evaluation_time:6.2f}s"
        )

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
