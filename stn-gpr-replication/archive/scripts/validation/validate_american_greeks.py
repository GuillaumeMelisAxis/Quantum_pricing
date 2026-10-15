from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import PaperConfig
from stngpr.coordinates import TransformedPricer, build_coordinate_grid
from stngpr.greeks import project_symmetric_matrix_psd
from stngpr.pricers import AmericanArithmeticBasketLSMC
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
        "budgets": [2_000],
        "anova": 300,
        "training_paths": 500,
        "training_steps": 8,
        "reference_paths": 2_000,
        "reference_steps": 20,
        "reference_seeds": [9101, 9102, 9103],
        "replicates": 1,
        "moneyness": [-0.05, 0.0, 0.05],
        "maturity_days": [30.0, 90.0],
    },
    "intermediate": {
        "moneyness_nodes": 256,
        "maturity_nodes": 32,
        "budgets": [5_000, 9_000],
        "anova": 1_000,
        "training_paths": 2_000,
        "training_steps": 15,
        "reference_paths": 10_000,
        "reference_steps": 40,
        "reference_seeds": [9101, 9102, 9103, 9104, 9105],
        "replicates": 1,
        "moneyness": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "maturity_days": [30.0, 90.0, 365.0],
    },
    "paper": {
        "moneyness_nodes": 512,
        "maturity_nodes": 64,
        "budgets": [9_000, 20_000],
        "anova": 2_000,
        "training_paths": 10_000,
        "training_steps": 30,
        "reference_paths": 50_000,
        "reference_steps": 60,
        "reference_seeds": [9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108],
        "replicates": 1,
        "moneyness": [-0.15, -0.075, 0.0, 0.075, 0.15],
        "maturity_days": [14.0, 30.0, 90.0, 365.0],
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
    mean, standard_error = {}, {}
    for name in COMPONENTS:
        values = np.stack([run[name] for run in replications], axis=0)
        mean[name] = np.mean(values, axis=0)
        standard_error[name] = (
            np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
            if values.shape[0] > 1
            else np.zeros_like(mean[name])
        )
    return mean, standard_error


def _project_components(components, n_assets):
    raw = gamma_matrices(components, n_assets)
    projected = project_symmetric_matrix_psd(raw)
    return replace_gamma_components(components, projected)


def _uncertainty_summary(reference, standard_error) -> dict:
    output = {}
    for name in COMPONENTS:
        scale = float(np.mean(np.abs(reference[name])))
        mean_se = float(np.mean(standard_error[name]))
        output[name] = {
            "mean_standard_error": mean_se,
            "normalized_mean_standard_error": (
                mean_se / scale if scale > 1e-14 else None
            ),
        }
    return output


def _coverage(estimate, reference, standard_error) -> dict:
    output = {}
    for name in COMPONENTS:
        error = np.abs(np.asarray(estimate[name]) - reference[name])
        band = 1.96 * np.asarray(standard_error[name])
        output[name] = {
            "fraction_inside_reference_ci95": float(np.mean(error <= band)),
            "mean_ci95_half_width": float(np.mean(band)),
        }
    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate American arithmetic-basket spot Greeks while separating "
            "TT error from independent multi-seed LSMC uncertainty."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--stage",
        choices=("reference", "tt"),
        default="tt",
        help="reference stops after the independent LSMC and training-label diagnostics",
    )
    parser.add_argument("--moneyness-nodes", type=int, default=None)
    parser.add_argument("--maturity-nodes", type=int, default=None)
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--anova-samples", type=int, default=None)
    parser.add_argument("--training-paths", type=int, default=None)
    parser.add_argument("--training-steps", type=int, default=None)
    parser.add_argument("--reference-paths", type=int, default=None)
    parser.add_argument("--reference-steps", type=int, default=None)
    parser.add_argument("--reference-seeds", nargs="+", type=int, default=None)
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
        default=Path("results/american_arithmetic_greeks_smoke.json"),
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    moneyness_nodes = args.moneyness_nodes or profile["moneyness_nodes"]
    maturity_nodes = args.maturity_nodes or profile["maturity_nodes"]
    budgets = args.budgets or profile["budgets"]
    anova = args.anova_samples or profile["anova"]
    training_paths = args.training_paths or profile["training_paths"]
    training_steps = args.training_steps or profile["training_steps"]
    reference_paths = args.reference_paths or profile["reference_paths"]
    reference_steps = args.reference_steps or profile["reference_steps"]
    reference_seeds = args.reference_seeds or profile["reference_seeds"]
    for value, name in (
        (moneyness_nodes, "moneyness nodes"),
        (maturity_nodes, "maturity nodes"),
    ):
        if not _power_of_two(int(value)):
            parser.error(f"{name} must be a power of two greater than one")
    if any(value <= 0 for value in budgets):
        parser.error("budgets must be positive")
    if min(
        anova,
        training_paths,
        training_steps,
        reference_paths,
        reference_steps,
    ) <= 0:
        parser.error("sampling and LSMC settings must be positive")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative bump must lie in (0, 1)")

    base = PaperConfig()
    shape = list(base.physical_shape)
    shape[base.n_assets] = moneyness_nodes
    shape[-1] = maturity_nodes
    config = replace(base, physical_shape=tuple(shape))
    design_seed = base.seed + 901 if args.design_seed is None else args.design_seed
    tt_seed = base.seed if args.tt_seed is None else args.tt_seed
    points, log_moneyness, maturity_days, replicate_ids = short_atm_panel(
        config,
        profile["replicates"],
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
    start = perf_counter()
    for seed in reference_seeds:
        pricer = AmericanArithmeticBasketLSMC(
            config,
            n_paths=reference_paths,
            n_steps=reference_steps,
            seed=seed,
        )
        reference_runs.append(
            finite_difference_arrays(
                pricer,
                points,
                config.n_assets,
                args.relative_bump,
            )
        )
        print(f"independent LSMC seed {seed} complete")
    reference_time = perf_counter() - start
    references, reference_se = _mean_and_standard_error(reference_runs)
    reference_hessian = gamma_matrices(references, config.n_assets)

    training_pricer = AmericanArithmeticBasketLSMC(
        config,
        n_paths=training_paths,
        n_steps=training_steps,
        seed=base.seed,
    )
    start = perf_counter()
    same_seed_components = finite_difference_arrays(
        training_pricer,
        points,
        config.n_assets,
        args.relative_bump,
    )
    same_seed_time = perf_counter() - start
    model_oracle = TransformedPricer(training_pricer, transform)

    results = {
        "schema_version": 1,
        "product": "american_arithmetic_basket_put",
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
        "independent_reference": {
            "method": "multi-seed high-resolution LSMC with common random numbers within each seed",
            "paths": reference_paths,
            "exercise_steps": reference_steps,
            "seeds": list(reference_seeds),
            "wall_time_seconds": reference_time,
            "uncertainty": _uncertainty_summary(references, reference_se),
            "hessian_shape": hessian_shape_diagnostics(reference_hessian),
            "components": _components_to_json(references),
            "component_standard_errors": _components_to_json(reference_se),
        },
        "training_lsmc": {
            "paths": training_paths,
            "exercise_steps": training_steps,
            "seed": base.seed,
            "greek_evaluation_time_seconds": same_seed_time,
            "error_against_independent_reference": component_metrics(
                references,
                same_seed_components,
            ),
            "components": _components_to_json(same_seed_components),
        },
        "runs": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.stage == "reference":
        print(f"Results written to {args.output}")
        return

    for budget in budgets:
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
        projected = _project_components(estimate, config.n_assets)
        summary, _, _, _ = summarize_estimates(
            references,
            estimate,
            log_moneyness,
            maturity_days,
            replicate_ids,
            config.n_assets,
        )
        key = str(int(budget))
        results["runs"][key] = {
            "fit": {
                "budget": int(budget),
                "anova_samples": anova,
                "function_evaluations": diagnostics.function_evaluations,
                "sweeps": diagnostics.sweeps,
                "stop": diagnostics.stop,
                "effective_rank": diagnostics.effective_rank,
                "wall_time_seconds": fit_time,
            },
            "greek_evaluation_time_seconds": evaluation_time,
            **summary,
            "error_against_same_seed_lsmc": component_metrics(
                same_seed_components,
                estimate,
            ),
            "coverage_against_independent_reference": _coverage(
                estimate,
                references,
                reference_se,
            ),
            "components": _components_to_json(estimate),
            "components_psd_projected": _components_to_json(projected),
        }
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"budget={budget} rank={diagnostics.effective_rank:.2f} "
            f"evaluations={diagnostics.function_evaluations}"
        )

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
