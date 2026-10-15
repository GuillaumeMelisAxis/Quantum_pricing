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
from stngpr.pricers import EuropeanArithmeticBasketQMC
from stngpr.tt_surrogate import TTPriceSurrogate

from validate_european_greeks import (
    component_metrics,
    finite_difference_arrays,
    finite_difference_hybrid_arrays,
)
from validate_refined_tt_greeks import summarize_estimates
from validate_short_maturity_greeks import (
    gamma_matrices,
    hessian_shape_diagnostics,
    replace_gamma_components,
    short_atm_panel,
)


PROFILES = {
    "smoke": {
        "moneyness_nodes": 64,
        "maturity_nodes": 8,
        "budget": 2_000,
        "anova": 300,
        "training_paths": 512,
        "reference_paths": 4_096,
        "reference_seeds": [8101, 8102, 8103],
        "replicates": 1,
        "moneyness": [-0.05, 0.0, 0.05],
        "maturity_days": [7.0, 30.0],
    },
    "intermediate": {
        "moneyness_nodes": 256,
        "maturity_nodes": 32,
        "budget": 20_000,
        "anova": 1_000,
        "training_paths": 1_024,
        "reference_paths": 32_768,
        "reference_seeds": [8101, 8102, 8103, 8104, 8105],
        "replicates": 2,
        "moneyness": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "maturity_days": [7.0, 30.0, 90.0],
    },
    "paper": {
        "moneyness_nodes": 512,
        "maturity_nodes": 64,
        "budget": 150_000,
        "anova": 2_000,
        "training_paths": 2_048,
        "reference_paths": 131_072,
        "reference_seeds": [8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108],
        "replicates": 3,
        "moneyness": [-0.05, -0.025, 0.0, 0.025, 0.05],
        "maturity_days": [3.0, 7.0, 14.0, 30.0, 90.0],
    },
}

COMPONENTS = ("price", "delta", "gamma_diagonal", "cross_gamma")


def _power_of_two(value: int) -> bool:
    return value > 1 and value & (value - 1) == 0


def _components_to_json(components) -> dict:
    return {
        name: np.asarray(components[name], dtype=float).tolist()
        for name in COMPONENTS
    }


def _mean_and_standard_error(replications):
    output_mean, output_se = {}, {}
    for name in COMPONENTS:
        values = np.stack([run[name] for run in replications], axis=0)
        output_mean[name] = np.mean(values, axis=0)
        if values.shape[0] > 1:
            output_se[name] = np.std(values, axis=0, ddof=1) / np.sqrt(
                values.shape[0]
            )
        else:
            output_se[name] = np.zeros_like(output_mean[name])
    return output_mean, output_se


def _uncertainty_summary(reference, standard_error) -> dict:
    result = {}
    for name in COMPONENTS:
        scale = float(np.mean(np.abs(reference[name])))
        mean_se = float(np.mean(np.asarray(standard_error[name])))
        result[name] = {
            "mean_standard_error": mean_se,
            "mean_absolute_reference": scale,
            "normalized_mean_standard_error": (
                mean_se / scale if scale > 1e-14 else None
            ),
        }
    return result


def _project_components(components, n_assets):
    matrices = gamma_matrices(components, n_assets)
    projected = project_symmetric_matrix_psd(matrices)
    return replace_gamma_components(components, projected), matrices, projected


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate fixed-strike spot Greeks for a European arithmetic-basket "
            "put using independent scrambled Sobol-QMC references."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--stage",
        choices=("reference", "grid", "tt", "all"),
        default="all",
    )
    parser.add_argument("--moneyness-nodes", type=int, default=None)
    parser.add_argument("--maturity-nodes", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--anova-samples", type=int, default=None)
    parser.add_argument("--training-paths", type=int, default=None)
    parser.add_argument("--reference-paths", type=int, default=None)
    parser.add_argument("--reference-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=None)
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
        default=Path("results/european_arithmetic_greeks_smoke.json"),
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    moneyness_nodes = args.moneyness_nodes or profile["moneyness_nodes"]
    maturity_nodes = args.maturity_nodes or profile["maturity_nodes"]
    budget = args.budget or profile["budget"]
    anova = args.anova_samples or profile["anova"]
    training_paths = args.training_paths or profile["training_paths"]
    reference_paths = args.reference_paths or profile["reference_paths"]
    reference_seeds = args.reference_seeds or profile["reference_seeds"]
    replicates = args.replicates or profile["replicates"]
    for value, name in (
        (moneyness_nodes, "moneyness nodes"),
        (maturity_nodes, "maturity nodes"),
        (training_paths, "training paths"),
        (reference_paths, "reference paths"),
    ):
        if not _power_of_two(int(value)):
            parser.error(f"{name} must be a power of two greater than one")
    if budget <= 0 or anova <= 0 or replicates <= 0:
        parser.error("budget, ANOVA samples and replicates must be positive")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative bump must lie in (0, 1)")

    base = PaperConfig()
    shape = list(base.physical_shape)
    shape[base.n_assets] = moneyness_nodes
    shape[-1] = maturity_nodes
    config = replace(base, physical_shape=tuple(shape))
    design_seed = base.seed + 801 if args.design_seed is None else args.design_seed
    tt_seed = base.seed if args.tt_seed is None else args.tt_seed
    points, log_moneyness, maturity_days, replicate_ids = short_atm_panel(
        config,
        replicates,
        profile["moneyness"],
        profile["maturity_days"],
        np.random.default_rng(design_seed),
        basket_kind="arithmetic",
    )
    grid, transform, grid_description = build_coordinate_grid(
        config,
        "moneyness_adaptive",
        basket_kind="arithmetic",
    )

    reference_runs = []
    reference_start = perf_counter()
    for seed in reference_seeds:
        pricer = EuropeanArithmeticBasketQMC(
            config,
            n_paths=reference_paths,
            seed=seed,
            control_variate=True,
            batch_size=16,
        )
        reference_runs.append(
            finite_difference_arrays(
                pricer,
                points,
                config.n_assets,
                args.relative_bump,
            )
        )
        print(f"reference scramble {seed} complete")
    reference_time = perf_counter() - reference_start
    references, reference_se = _mean_and_standard_error(reference_runs)
    reference_hessian = gamma_matrices(references, config.n_assets)

    training_pricer = EuropeanArithmeticBasketQMC(
        config,
        n_paths=training_paths,
        seed=base.seed,
        control_variate=True,
    )
    direct_training = finite_difference_arrays(
        training_pricer,
        points,
        config.n_assets,
        args.relative_bump,
    )
    model_oracle = TransformedPricer(training_pricer, transform)

    results = {
        "schema_version": 1,
        "product": "european_arithmetic_basket_put",
        "profile": args.profile,
        "stage": args.stage,
        "fixed_strike_relative_bump": args.relative_bump,
        "design_seed": design_seed,
        "tt_seed": tt_seed,
        "test_design": {
            "market_parameters": points.tolist(),
            "log_moneyness": log_moneyness.tolist(),
            "maturity_days": maturity_days.tolist(),
            "replicate_ids": replicate_ids.tolist(),
        },
        "grid": {
            "physical_shape": list(config.physical_shape),
            "qtt_order": int(sum(config.qtt_bits)),
            "description": grid_description,
        },
        "reference": {
            "method": "scrambled Sobol-QMC with exact geometric control variate",
            "paths_per_scramble": reference_paths,
            "seeds": list(reference_seeds),
            "wall_time_seconds": reference_time,
            "uncertainty": _uncertainty_summary(references, reference_se),
            "hessian_shape": hessian_shape_diagnostics(reference_hessian),
            "components": _components_to_json(references),
            "component_standard_errors": _components_to_json(reference_se),
        },
        "training_oracle": {
            "paths": training_paths,
            "seed": base.seed,
            "error_against_independent_reference": component_metrics(
                references,
                direct_training,
            ),
            "components": _components_to_json(direct_training),
        },
        "runs": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.stage in ("grid", "all"):
        def oracle_interpolator(model_parameters, cubic_columns):
            return oracle_hybrid_cubic_predict(
                grid,
                model_oracle,
                model_parameters,
                cubic_columns=cubic_columns,
            )

        start = perf_counter()
        estimate = finite_difference_hybrid_arrays(
            oracle_interpolator,
            transform,
            points,
            config.n_assets,
            args.relative_bump,
        )
        projected, raw_hessian, projected_hessian = _project_components(
            estimate,
            config.n_assets,
        )
        summary, _, _, _ = summarize_estimates(
            references,
            estimate,
            log_moneyness,
            maturity_days,
            replicate_ids,
            config.n_assets,
        )
        results["runs"]["exact_grid"] = {
            "wall_time_seconds": perf_counter() - start,
            **summary,
            "components": _components_to_json(estimate),
            "components_psd_projected": _components_to_json(projected),
        }
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("exact-grid layer complete")

    if args.stage in ("tt", "all"):
        model = TTPriceSurrogate(grid, model_oracle, seed=tt_seed)
        start = perf_counter()
        diagnostics = model.fit(
            budget,
            anova_samples=anova,
            max_sweeps=args.max_sweeps,
            rank_increment=args.rank_increment,
            truncation=args.truncation,
            log=args.cross_log,
        )
        fit_time = perf_counter() - start
        start = perf_counter()
        estimate = finite_difference_hybrid_arrays(
            model.predict_hybrid_cubic,
            transform,
            points,
            config.n_assets,
            args.relative_bump,
        )
        evaluation_time = perf_counter() - start
        projected, _, _ = _project_components(estimate, config.n_assets)
        summary, _, _, _ = summarize_estimates(
            references,
            estimate,
            log_moneyness,
            maturity_days,
            replicate_ids,
            config.n_assets,
        )
        results["runs"]["tt"] = {
            "fit": {
                "budget": budget,
                "anova_samples": anova,
                "function_evaluations": diagnostics.function_evaluations,
                "sweeps": diagnostics.sweeps,
                "stop": diagnostics.stop,
                "effective_rank": diagnostics.effective_rank,
                "wall_time_seconds": fit_time,
            },
            "greek_evaluation_time_seconds": evaluation_time,
            **summary,
            "components": _components_to_json(estimate),
            "components_psd_projected": _components_to_json(projected),
        }
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"TT complete: rank={diagnostics.effective_rank:.2f}, "
            f"evaluations={diagnostics.function_evaluations}"
        )

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
