from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import PaperConfig
from stngpr.convergence import (
    COMPONENTS,
    component_metrics,
    components_to_json,
    deterministic_moneyness_panel,
    finite_difference_component_arrays,
    finite_difference_hybrid_component_arrays,
    gamma_matrices,
    hessian_shape_diagnostics,
    hessian_uncertainty_summary,
    mean_and_standard_error,
    uncertainty_summary,
)
from stngpr.coordinates import (
    CachedGridInterpolator,
    TransformedPricer,
    build_coordinate_grid,
)
from stngpr.greeks import project_symmetric_matrix_psd
from stngpr.pricers import AmericanArithmeticBasketLSMC
from stngpr.tt_surrogate import TTPriceSurrogate


PROFILES = {
    "smoke": {
        "moneyness_nodes": 64,
        "maturity_nodes": 8,
        "budgets": [2_000],
        "anova_samples": 300,
        "training_paths": 500,
        "training_steps": 8,
        "reference_paths": 2_000,
        "reference_steps": 20,
        "reference_seeds": [9101, 9102, 9103],
        "moneyness": [-0.05, 0.0, 0.05],
        "maturity_days": [30.0, 90.0],
    },
    "intermediate": {
        "moneyness_nodes": 256,
        "maturity_nodes": 32,
        "budgets": [5_000, 10_000],
        "anova_samples": 1_000,
        "training_paths": 2_000,
        "training_steps": 15,
        "reference_paths": 50_000,
        "reference_steps": 80,
        "reference_seeds": [9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108],
        "moneyness": [-0.05, 0.0, 0.05],
        "maturity_days": [30.0, 90.0],
    },
    "paper": {
        "moneyness_nodes": 512,
        "maturity_nodes": 64,
        "budgets": [10_000, 20_000, 50_000],
        "anova_samples": 2_000,
        "training_paths": 10_000,
        "training_steps": 30,
        "reference_paths": 50_000,
        "reference_steps": 80,
        "reference_seeds": [9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108],
        "moneyness": [-0.10, -0.05, 0.0, 0.05, 0.10],
        "maturity_days": [14.0, 30.0, 90.0, 365.0],
    },
}


def _positive_power_of_two(value):
    value = int(value)
    return value > 1 and value & (value - 1) == 0


def _project_components(components):
    raw = gamma_matrices(components)
    projected = project_symmetric_matrix_psd(raw)
    output = {
        name: np.asarray(components[name], dtype=float).copy()
        for name in COMPONENTS
    }
    output["gamma_diagonal"] = np.diagonal(
        projected,
        axis1=-2,
        axis2=-1,
    ).copy()
    pairs = [
        (left, right)
        for left in range(projected.shape[-1])
        for right in range(left + 1, projected.shape[-1])
    ]
    output["cross_gamma"] = np.column_stack([
        projected[:, left, right] for left, right in pairs
    ])
    return output, raw, projected


def _matrix_metrics(reference, estimate):
    reference = gamma_matrices(reference)
    estimate = gamma_matrices(estimate)
    error = np.linalg.norm(estimate - reference, axis=(1, 2))
    scale = np.linalg.norm(reference, axis=(1, 2))
    mean_scale = float(np.mean(scale))
    return {
        "mean_frobenius_error": float(np.mean(error)),
        "rmse_frobenius_error": float(np.sqrt(np.mean(error**2))),
        "maximum_frobenius_error": float(np.max(error)),
        "mean_reference_frobenius_norm": mean_scale,
        "normalized_mean_frobenius_error": (
            float(np.mean(error) / mean_scale) if mean_scale > 1e-14 else None
        ),
    }


def _layer_payload(components, reference):
    projected, raw_hessian, projected_hessian = _project_components(components)
    raw_norm = np.linalg.norm(raw_hessian, axis=(1, 2))
    projection_norm = np.linalg.norm(
        projected_hessian - raw_hessian,
        axis=(1, 2),
    )
    mean_raw_norm = float(np.mean(raw_norm))
    return {
        "components": components_to_json(components),
        "components_psd_projected": components_to_json(projected),
        "error_against_independent_reference": component_metrics(
            reference,
            components,
        ),
        "error_against_independent_reference_psd_projected": component_metrics(
            reference,
            projected,
        ),
        "hessian_error_against_independent_reference": _matrix_metrics(
            reference,
            components,
        ),
        "hessian_error_against_independent_reference_psd_projected": (
            _matrix_metrics(reference, projected)
        ),
        "hessian_shape": hessian_shape_diagnostics(raw_hessian),
        "hessian_shape_psd_projected": hessian_shape_diagnostics(
            projected_hessian
        ),
        "psd_projection": {
            "mean_frobenius_adjustment": float(np.mean(projection_norm)),
            "maximum_frobenius_adjustment": float(np.max(projection_norm)),
            "normalized_mean_frobenius_adjustment": (
                float(np.mean(projection_norm) / mean_raw_norm)
                if mean_raw_norm > 1e-14 else None
            ),
        },
    }


def _coverage(estimate, reference, standard_error):
    payload = {}
    for name in COMPONENTS:
        error = np.abs(
            np.asarray(estimate[name], dtype=float)
            - np.asarray(reference[name], dtype=float)
        )
        band = 1.96 * np.asarray(standard_error[name], dtype=float)
        payload[name] = {
            "fraction_inside_reference_ci95": float(np.mean(error <= band)),
            "mean_ci95_half_width": float(np.mean(band)),
        }
    return payload


def _additive_decomposition(reference, training, grid, tt):
    payload = {}
    for name in COMPONENTS:
        truth = np.asarray(reference[name], dtype=float)
        train = np.asarray(training[name], dtype=float)
        interpolated = np.asarray(grid[name], dtype=float)
        compressed = np.asarray(tt[name], dtype=float)
        contributions = {
            "training_label": train - truth,
            "grid_interpolation": interpolated - train,
            "tt_reconstruction": compressed - interpolated,
        }
        total = compressed - truth
        closure = total - sum(contributions.values())
        scale = float(np.mean(np.abs(truth)))
        payload[name] = {
            "signed_mean_contribution_normalized": {
                key: float(np.mean(value) / scale) if scale > 1e-14 else None
                for key, value in contributions.items()
            },
            "signed_mean_total_error_normalized": (
                float(np.mean(total) / scale) if scale > 1e-14 else None
            ),
            "maximum_absolute_closure_error": float(np.max(np.abs(closure))),
            "normalized_maximum_closure_error": (
                float(np.max(np.abs(closure)) / scale)
                if scale > 1e-14 else None
            ),
        }
    return payload


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Separate American-basket LSMC label, adaptive-grid interpolation "
            "and TT reconstruction errors for fixed-strike spot Greeks."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--stage",
        choices=("reference", "grid", "tt"),
        default="tt",
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
    parser.add_argument("--risk-columns", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--relative-bump", type=float, default=0.03)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--rank-increment", type=int, default=2)
    parser.add_argument("--truncation", type=float, default=1e-8)
    parser.add_argument("--grid-batch-size", type=int, default=64)
    parser.add_argument("--cross-log", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v82_american_error_decomposition.json"),
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]

    def setting(name):
        value = getattr(args, name)
        return profile[name] if value is None else value

    moneyness_nodes = setting("moneyness_nodes")
    maturity_nodes = setting("maturity_nodes")
    budgets = setting("budgets")
    anova_samples = setting("anova_samples")
    training_paths = setting("training_paths")
    training_steps = setting("training_steps")
    reference_paths = setting("reference_paths")
    reference_steps = setting("reference_steps")
    reference_seeds = setting("reference_seeds")
    risk_columns = tuple(args.risk_columns)

    if not _positive_power_of_two(moneyness_nodes):
        parser.error("moneyness-nodes must be a power of two greater than one")
    if not _positive_power_of_two(maturity_nodes):
        parser.error("maturity-nodes must be a power of two greater than one")
    if any(value <= 0 for value in budgets):
        parser.error("budgets must be positive")
    if min(
        anova_samples,
        training_paths,
        training_steps,
        reference_paths,
        reference_steps,
        args.grid_batch_size,
    ) <= 0:
        parser.error("sampling, LSMC and batching settings must be positive")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative-bump must lie in (0, 1)")

    base = PaperConfig()
    if (
        not risk_columns
        or len(set(risk_columns)) != len(risk_columns)
        or any(column < 0 or column >= base.n_assets for column in risk_columns)
    ):
        parser.error("risk-columns must be unique valid spot columns")

    shape = list(base.physical_shape)
    shape[base.n_assets] = int(moneyness_nodes)
    shape[-1] = int(maturity_nodes)
    config = replace(base, physical_shape=tuple(shape))
    points, log_moneyness, maturity_days = deterministic_moneyness_panel(
        config,
        profile["moneyness"],
        profile["maturity_days"],
    )
    grid, transform, grid_description = build_coordinate_grid(
        config,
        "moneyness_adaptive",
        basket_kind="arithmetic",
    )
    logical_entries = math.prod(config.physical_shape)

    output = {
        "schema_version": 1,
        "experiment": "american_arithmetic_greek_error_decomposition",
        "profile": args.profile,
        "stage": args.stage,
        "identity": (
            "TT-reference = training-reference + grid-training + TT-grid"
        ),
        "fixed_strike_relative_bump": args.relative_bump,
        "risk_columns": list(risk_columns),
        "test_design": {
            "market_parameters": points.tolist(),
            "log_moneyness": log_moneyness.tolist(),
            "maturity_days": maturity_days.tolist(),
            "point_count": int(len(points)),
        },
        "grid": {
            "physical_shape": list(config.physical_shape),
            "qtt_order": int(sum(config.qtt_bits)),
            "logical_tensor_entries": int(logical_entries),
            "description": grid_description,
        },
        "settings": {
            "training_paths": training_paths,
            "training_steps": training_steps,
            "training_seed": base.seed,
            "reference_paths": reference_paths,
            "reference_steps": reference_steps,
            "reference_seeds": list(reference_seeds),
            "budgets": list(budgets),
            "anova_samples": anova_samples,
            "max_sweeps": args.max_sweeps,
            "rank_increment": args.rank_increment,
            "truncation": args.truncation,
        },
        "independent_reference": None,
        "training_labels": None,
        "grid_only": None,
        "tt": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    reference_runs = []
    start = perf_counter()
    for seed in reference_seeds:
        pricer = AmericanArithmeticBasketLSMC(
            config,
            n_paths=reference_paths,
            n_steps=reference_steps,
            seed=seed,
            policy_mode="refit",
        )
        reference_runs.append(finite_difference_component_arrays(
            pricer,
            points,
            risk_columns,
            args.relative_bump,
        ))
        print(f"reference seed {seed} complete")
        del pricer
    reference_time = perf_counter() - start
    reference, reference_se = mean_and_standard_error(reference_runs)
    output["independent_reference"] = {
        "method": "multi-seed 50k/80 LSMC oracle when using intermediate or paper profile",
        "wall_time_seconds": reference_time,
        "components": components_to_json(reference),
        "component_standard_errors": components_to_json(reference_se),
        "uncertainty": uncertainty_summary(reference, reference_se),
        "hessian_uncertainty": hessian_uncertainty_summary(
            reference,
            reference_se,
        ),
        "hessian_shape": hessian_shape_diagnostics(gamma_matrices(reference)),
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.stage == "reference":
        print(f"Results written to {args.output}")
        return

    training_pricer = AmericanArithmeticBasketLSMC(
        config,
        n_paths=training_paths,
        n_steps=training_steps,
        seed=base.seed,
        policy_mode="refit",
    )
    start = perf_counter()
    training_components = finite_difference_component_arrays(
        training_pricer,
        points,
        risk_columns,
        args.relative_bump,
    )
    training_time = perf_counter() - start
    output["training_labels"] = {
        "greek_evaluation_time_seconds": training_time,
        **_layer_payload(training_components, reference),
        "coverage_against_reference_ci95": _coverage(
            training_components,
            reference,
            reference_se,
        ),
    }

    model_oracle = TransformedPricer(training_pricer, transform)
    cached_grid = CachedGridInterpolator(
        grid,
        model_oracle,
        evaluation_batch_size=args.grid_batch_size,
    )
    start = perf_counter()
    grid_components = finite_difference_hybrid_component_arrays(
        cached_grid,
        transform,
        points,
        risk_columns,
        args.relative_bump,
    )
    grid_time = perf_counter() - start
    output["grid_only"] = {
        "method": "exact cached grid-node labels with hybrid cubic interpolation",
        "greek_evaluation_time_seconds": grid_time,
        "oracle_cache": cached_grid.diagnostics(),
        "error_against_direct_training_labels": component_metrics(
            training_components,
            grid_components,
        ),
        "hessian_error_against_direct_training_labels": _matrix_metrics(
            training_components,
            grid_components,
        ),
        **_layer_payload(grid_components, reference),
        "coverage_against_reference_ci95": _coverage(
            grid_components,
            reference,
            reference_se,
        ),
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.stage == "grid":
        print(f"Results written to {args.output}")
        return

    for budget in budgets:
        model = TTPriceSurrogate(grid, model_oracle, seed=base.seed)
        start = perf_counter()
        diagnostics = model.fit(
            budget,
            anova_samples=anova_samples,
            max_sweeps=args.max_sweeps,
            rank_increment=args.rank_increment,
            truncation=args.truncation,
            log=args.cross_log,
        )
        fit_time = perf_counter() - start
        start = perf_counter()
        tt_components = finite_difference_hybrid_component_arrays(
            model.predict_hybrid_cubic,
            transform,
            points,
            risk_columns,
            args.relative_bump,
        )
        evaluation_time = perf_counter() - start
        key = str(int(budget))
        output["tt"][key] = {
            "fit": {
                "requested_budget": int(budget),
                "anova_samples": int(anova_samples),
                "function_evaluations": diagnostics.function_evaluations,
                "sweeps": diagnostics.sweeps,
                "stop": diagnostics.stop,
                "effective_rank": diagnostics.effective_rank,
                "maximum_rank": diagnostics.maximum_rank,
                "parameter_count": diagnostics.parameter_count,
                "logical_entries_per_tt_parameter": (
                    logical_entries / diagnostics.parameter_count
                ),
                "logical_entries_per_function_evaluation": (
                    logical_entries / diagnostics.function_evaluations
                ),
                "wall_time_seconds": fit_time,
            },
            "greek_evaluation_time_seconds": evaluation_time,
            "error_against_grid_only": component_metrics(
                grid_components,
                tt_components,
            ),
            "hessian_error_against_grid_only": _matrix_metrics(
                grid_components,
                tt_components,
            ),
            **_layer_payload(tt_components, reference),
            "coverage_against_reference_ci95": _coverage(
                tt_components,
                reference,
                reference_se,
            ),
            "additive_error_decomposition": _additive_decomposition(
                reference,
                training_components,
                grid_components,
                tt_components,
            ),
        }
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        total = output["tt"][key]["error_against_independent_reference"]
        reconstruction = output["tt"][key]["error_against_grid_only"]
        print(
            f"budget={budget} rank={diagnostics.effective_rank:.2f} "
            f"evals={diagnostics.function_evaluations} "
            f"total_gamma={100 * total['gamma_diagonal']['normalized_mae']:.2f}% "
            f"tt_gamma={100 * reconstruction['gamma_diagonal']['normalized_mae']:.2f}%"
        )

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
