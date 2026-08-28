from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from validate_dense_greek_profiles import (
    SPOT_PANELS,
    analytical_components,
    dense_market_panel,
)
from validate_full_spot_hessian import _component_labels, _diagnostics

from stngpr.config import PaperConfig
from stngpr.convergence import (
    COMPONENTS,
    component_metrics,
    components_to_json,
    finite_difference_component_arrays,
    finite_difference_hybrid_component_arrays,
    full_hessian_error_metrics,
    gamma_matrices,
    hessian_shape_diagnostics,
    tt_component_error_decomposition,
)
from stngpr.coordinates import CachedGridInterpolator, TransformedPricer
from stngpr.greeks import project_symmetric_matrix_psd
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk_grids import build_greek_coordinate_grid
from stngpr.tt_surrogate import TTPriceSurrogate

PROFILES = {
    "smoke": {
        "moneyness": [-0.05, 0.0, 0.05],
        "maturity_days": [7.0, 30.0],
        "panels": ["oos_mid"],
        "budgets": [2_000, 5_000],
        "anova_samples": 300,
        "retained_budget": 5_000,
        "tt_seeds": [20260401, 20260402],
    },
    "intermediate": {
        "moneyness": [-0.15, -0.05, 0.0, 0.05, 0.15],
        "maturity_days": [3.0, 7.0, 30.0, 90.0],
        "panels": ["oos_low", "oos_mid", "oos_dispersed"],
        "budgets": [20_000, 50_000, 100_000],
        "anova_samples": 1_000,
        "retained_budget": 100_000,
        "tt_seeds": [20260401, 20260402, 20260403],
    },
    "paper": {
        "moneyness": [-0.15, -0.05, 0.0, 0.05, 0.15],
        "maturity_days": [3.0, 7.0, 30.0, 90.0],
        "panels": ["oos_low", "oos_mid", "oos_dispersed"],
        "budgets": [20_000, 50_000, 100_000, 150_000, 200_000],
        "anova_samples": 2_000,
        "retained_budget": 150_000,
        "tt_seeds": [20260401, 20260402, 20260403, 20260404, 20260405],
    },
}

ACCEPTANCE_THRESHOLDS = {
    "price_normalized_mae": 0.005,
    "delta_normalized_mae": 0.01,
    "gamma_diagonal_normalized_mae": 0.025,
    "cross_gamma_normalized_mae": 0.025,
    "hessian_relative_frobenius_error": 0.03,
    "cross_gamma_sign_agreement": 0.99,
    "negative_part_relative_frobenius_norm": 0.005,
}


def _power_of_two(value):
    value = int(value)
    return value > 1 and value & (value - 1) == 0


def _project_components(components):
    matrices = project_symmetric_matrix_psd(gamma_matrices(components))
    n_points, n_assets, _ = matrices.shape
    cross = []
    for left in range(n_assets):
        for right in range(left + 1, n_assets):
            cross.append(matrices[:, left, right])
    projected = {
        "price": np.asarray(components["price"], dtype=float).copy(),
        "delta": np.asarray(components["delta"], dtype=float).copy(),
        "gamma_diagonal": np.diagonal(
            matrices,
            axis1=1,
            axis2=2,
        ).copy(),
        "cross_gamma": np.column_stack(cross),
        "risk_columns": tuple(range(n_assets)),
        "cross_gamma_pairs": [
            (left, right)
            for left in range(n_assets)
            for right in range(left + 1, n_assets)
        ],
    }
    if projected["cross_gamma"].shape[0] != n_points:
        raise RuntimeError("invalid projected Hessian shape")
    return projected


def _decomposition_metrics(analytical, exact_fd, grid_fd, tt_fd):
    decomposition = tt_component_error_decomposition(
        analytical,
        exact_fd,
        grid_fd,
        tt_fd,
    )
    output = {
        layer: record["metrics"]
        for layer, record in decomposition.items()
    }
    output["closure_max_absolute_residual"] = {
        name: float(
            np.max(np.abs(decomposition["closure"]["residuals"][name]))
        )
        for name in COMPONENTS
    }
    return output


def _acceptance(diagnostics):
    global_error = diagnostics["global_error_against_analytical"]
    hessian = diagnostics["full_hessian_error_against_analytical"]
    signs = diagnostics["cross_gamma_sign_fidelity"]
    checks = {
        "price_normalized_mae": (
            global_error["price"]["normalized_mae"]
            <= ACCEPTANCE_THRESHOLDS["price_normalized_mae"]
        ),
        "delta_normalized_mae": (
            global_error["delta"]["normalized_mae"]
            <= ACCEPTANCE_THRESHOLDS["delta_normalized_mae"]
        ),
        "gamma_diagonal_normalized_mae": (
            global_error["gamma_diagonal"]["normalized_mae"]
            <= ACCEPTANCE_THRESHOLDS["gamma_diagonal_normalized_mae"]
        ),
        "cross_gamma_normalized_mae": (
            global_error["cross_gamma"]["normalized_mae"]
            <= ACCEPTANCE_THRESHOLDS["cross_gamma_normalized_mae"]
        ),
        "hessian_relative_frobenius_error": (
            hessian["aggregate_relative_frobenius_error"]
            <= ACCEPTANCE_THRESHOLDS["hessian_relative_frobenius_error"]
        ),
        "cross_gamma_sign_agreement": (
            signs["sign_agreement_fraction"]
            >= ACCEPTANCE_THRESHOLDS["cross_gamma_sign_agreement"]
        ),
        "negative_part_relative_frobenius_norm": (
            hessian["negative_part_relative_frobenius_norm"]
            <= ACCEPTANCE_THRESHOLDS[
                "negative_part_relative_frobenius_norm"
            ]
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _scalar_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1))
        if values.size > 1
        else 0.0,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _run_metrics(record):
    global_error = record["global_error_against_analytical"]
    hessian = record["full_hessian_error_against_analytical"]
    return {
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
        "cross_gamma_sign_agreement": record["cross_gamma_sign_fidelity"][
            "sign_agreement_fraction"
        ],
        "negative_part_relative_frobenius_norm": hessian[
            "negative_part_relative_frobenius_norm"
        ],
        "effective_rank": record["fit"]["effective_rank"],
        "maximum_rank": record["fit"]["maximum_rank"],
        "parameter_count": record["fit"]["parameter_count"],
        "function_evaluations": record["fit"]["function_evaluations"],
        "fit_time_seconds": record["fit"]["total_time_seconds"],
    }


def _summarize_runs(runs):
    if not runs:
        return {}
    output = {}
    by_budget = {}
    for record in runs.values():
        by_budget.setdefault(str(record["budget"]), []).append(record)
    for budget, group in sorted(by_budget.items(), key=lambda item: int(item[0])):
        metrics = [_run_metrics(record) for record in group]
        output[budget] = {
            name: _scalar_summary([record[name] for record in metrics])
            for name in metrics[0]
        }
        output[budget]["acceptance_pass_count"] = int(
            sum(record["acceptance"]["passed"] for record in group)
        )
        output[budget]["run_count"] = len(group)
    return output


def _reference_convergence(runs, reference_budget):
    by_seed = {}
    for record in runs.values():
        by_seed.setdefault(int(record["tt_seed"]), {})[
            int(record["budget"])
        ] = record
    output = {}
    for seed, records in by_seed.items():
        if int(reference_budget) not in records:
            continue
        reference = records[int(reference_budget)]["components"]
        output[str(seed)] = {}
        for budget, record in sorted(records.items()):
            output[str(seed)][str(budget)] = {
                "component_difference_from_reference_budget": component_metrics(
                    reference,
                    record["components"],
                ),
                "hessian_difference_from_reference_budget": (
                    full_hessian_error_metrics(
                        reference,
                        record["components"],
                    )
                ),
            }
    return output


def _write_checkpoint(path, payload):
    payload["summary_by_budget"] = _summarize_runs(payload["runs"])
    payload["convergence_to_reference_budget"] = _reference_convergence(
        payload["runs"],
        payload["settings"]["reference_budget"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure QTT-cross budget and seed convergence above the "
            "v9.4 tensor-compatible full-Hessian interpolation floor."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--campaign",
        choices=("budget", "seeds", "all"),
        default="budget",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--retained-budget", type=int, default=None)
    parser.add_argument("--tt-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--anova-samples", type=int, default=None)
    parser.add_argument("--relative-bump", type=float, default=0.002)
    parser.add_argument("--spot-nodes", type=int, default=64)
    parser.add_argument("--coordinate-nodes", type=int, default=512)
    parser.add_argument("--maturity-nodes", type=int, default=64)
    parser.add_argument("--grid-batch-size", type=int, default=256)
    parser.add_argument("--tt-batch-size", type=int, default=16)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--rank-increment", type=int, default=2)
    parser.add_argument("--truncation", type=float, default=1e-8)
    parser.add_argument("--cross-log", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v94_tt_convergence.json"),
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    budgets = sorted(set(args.budgets or profile["budgets"]))
    retained_budget = int(
        args.retained_budget or profile["retained_budget"]
    )
    tt_seeds = list(dict.fromkeys(args.tt_seeds or profile["tt_seeds"]))
    anova_samples = int(args.anova_samples or profile["anova_samples"])
    if any(budget <= 0 for budget in budgets) or retained_budget <= 0:
        parser.error("budgets must be positive")
    if not tt_seeds:
        parser.error("at least one TT seed is required")
    if anova_samples <= 0:
        parser.error("anova-samples must be positive")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative-bump must lie in (0,1)")
    if args.max_sweeps <= 0 or args.rank_increment <= 0:
        parser.error("TT-cross sweep settings must be positive")
    if args.truncation <= 0.0:
        parser.error("truncation must be positive")
    for name, value in (
        ("spot-nodes", args.spot_nodes),
        ("coordinate-nodes", args.coordinate_nodes),
        ("maturity-nodes", args.maturity_nodes),
    ):
        if not _power_of_two(value):
            parser.error(f"{name} must be a power of two greater than one")

    if args.campaign == "budget":
        requested_runs = [(budget, tt_seeds[0]) for budget in budgets]
    elif args.campaign == "seeds":
        requested_runs = [(retained_budget, seed) for seed in tt_seeds]
    else:
        requested_runs = [(budget, tt_seeds[0]) for budget in budgets]
        requested_runs.extend((retained_budget, seed) for seed in tt_seeds)
        requested_runs = list(dict.fromkeys(requested_runs))

    base = PaperConfig()
    shape = list(base.physical_shape)
    shape[: base.n_assets] = [int(args.spot_nodes)] * base.n_assets
    shape[base.n_assets] = int(args.coordinate_nodes)
    shape[-1] = int(args.maturity_nodes)
    config = replace(base, physical_shape=tuple(shape))
    risk_columns = tuple(range(config.n_assets))
    component_labels = _component_labels(risk_columns)
    moneyness = np.asarray(profile["moneyness"], dtype=float)
    maturity_days = list(profile["maturity_days"])
    panel_names = list(profile["panels"])
    points, m_values, t_values, panel_labels, slices, rate = dense_market_panel(
        config,
        panel_names,
        maturity_days,
        moneyness,
        args.relative_bump,
    )
    market_pricer = EuropeanGeometricBasketPricer(config)
    analytical = analytical_components(
        config,
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
    grid, transform, coordinate_description = build_greek_coordinate_grid(
        config,
        "bounded_standardized_risk",
    )
    audit_rng = np.random.default_rng(9401)
    audit_indices = grid.random_physical_indices(4_096, audit_rng)
    audit_market = transform.to_market(grid.indices_to_points(audit_indices))
    audit_basket = np.exp(
        np.mean(np.log(audit_market[:, : config.n_assets]), axis=1)
    )
    audit_moneyness = np.log(
        audit_market[:, config.n_assets] / audit_basket
    )
    expected_moneyness_bounds = (
        np.log(config.strike_bounds[0] / config.spot_bounds[1]),
        np.log(config.strike_bounds[1] / config.spot_bounds[0]),
    )
    tensor_domain_audit = {
        "sample_size": len(audit_indices),
        "log_moneyness_minimum": float(np.min(audit_moneyness)),
        "log_moneyness_maximum": float(np.max(audit_moneyness)),
        "expected_log_moneyness_bounds": [
            float(value) for value in expected_moneyness_bounds
        ],
        "strike_minimum": float(np.min(audit_market[:, config.n_assets])),
        "strike_maximum": float(np.max(audit_market[:, config.n_assets])),
        "all_market_parameters_finite": bool(np.all(np.isfinite(audit_market))),
        "all_samples_within_moneyness_bounds": bool(
            np.all(audit_moneyness >= expected_moneyness_bounds[0] - 1e-12)
            and np.all(audit_moneyness <= expected_moneyness_bounds[1] + 1e-12)
        ),
    }
    model_oracle = TransformedPricer(market_pricer, transform)
    grid_interpolator = CachedGridInterpolator(
        grid,
        model_oracle,
        evaluation_batch_size=args.grid_batch_size,
    )
    grid_start = perf_counter()
    grid_fd = finite_difference_hybrid_component_arrays(
        grid_interpolator,
        transform,
        points,
        risk_columns,
        args.relative_bump,
    )
    grid_wall_time = perf_counter() - grid_start

    signature = {
        "profile": args.profile,
        "coordinate_mode": "bounded_standardized_risk",
        "physical_shape": list(config.physical_shape),
        "relative_bump": float(args.relative_bump),
        "moneyness": moneyness.tolist(),
        "maturity_days": maturity_days,
        "panels": panel_names,
        "anova_samples": anova_samples,
        "max_sweeps": int(args.max_sweeps),
        "rank_increment": int(args.rank_increment),
        "truncation": float(args.truncation),
    }
    if args.output.exists() and not args.overwrite:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "greeks-tt-convergence-v9.4"
            or payload.get("experiment_signature") != signature
        ):
            parser.error(
                "the output contains an incompatible experiment; choose a "
                "different path or pass --overwrite"
            )
        print(f"Resuming {len(payload['runs'])} completed runs from {args.output}")
        payload["settings"].update({
            "budgets": budgets,
            "reference_budget": int(max(budgets)),
            "retained_budget": retained_budget,
            "tt_seeds": tt_seeds,
            "grid_batch_size": int(args.grid_batch_size),
            "tt_batch_size": int(args.tt_batch_size),
        })
    else:
        payload = {
            "schema_version": "greeks-tt-convergence-v9.4",
            "profile": args.profile,
            "campaign": args.campaign,
            "hypothesis": (
                "QTT-cross error decreases with oracle budget and remains a "
                "controlled additive layer above the tensor-compatible "
                "bounded standardized-risk grid floor"
            ),
            "experiment_signature": signature,
            "settings": {
                "physical_shape": list(config.physical_shape),
                "qtt_core_count": int(sum(config.qtt_bits)),
                "logical_tensor_entries": int(np.prod(config.physical_shape)),
                "coordinate_mode": "bounded_standardized_risk",
                "interpolation_mode": "risk_hybrid_cubic",
                "relative_bump": float(args.relative_bump),
                "budgets": budgets,
                "reference_budget": int(max(budgets)),
                "retained_budget": retained_budget,
                "tt_seeds": tt_seeds,
                "anova_samples": anova_samples,
                "max_sweeps": int(args.max_sweeps),
                "rank_increment": int(args.rank_increment),
                "truncation": float(args.truncation),
                "grid_batch_size": int(args.grid_batch_size),
                "tt_batch_size": int(args.tt_batch_size),
                "rate": rate,
                "finite_difference_prices_per_point": 1 + 2 * base.n_assets**2,
                "monte_carlo": False,
            },
            "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
            "test_design": {
                "market_parameters": points.tolist(),
                "log_moneyness": m_values.tolist(),
                "maturity_days": t_values.tolist(),
                "spot_panel_labels": panel_labels.tolist(),
                "curve_slices": slices,
                "spot_panels": {
                    name: SPOT_PANELS[name] for name in panel_names
                },
                "component_labels": component_labels,
            },
            "coordinate_description": coordinate_description,
            "tensor_domain_audit": tensor_domain_audit,
            "analytical_reference": {
                "components": components_to_json(analytical),
                "hessian_shape": hessian_shape_diagnostics(
                    gamma_matrices(analytical)
                ),
            },
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
            "grid_oracle": {
                "wall_time_seconds": grid_wall_time,
                "cache": grid_interpolator.diagnostics(),
                "components": components_to_json(grid_fd),
                **_diagnostics(
                    analytical,
                    grid_fd,
                    component_labels,
                    m_values,
                    slices,
                ),
            },
            "requested_runs": [
                {"budget": int(budget), "tt_seed": int(seed)}
                for budget, seed in requested_runs
            ],
            "runs": {},
            "summary_by_budget": {},
            "convergence_to_reference_budget": {},
        }
        _write_checkpoint(args.output, payload)

    for budget, tt_seed in requested_runs:
        run_key = f"seed_{int(tt_seed)}__budget_{int(budget)}"
        if run_key in payload["runs"]:
            print(f"Skipping completed {run_key}")
            continue
        print(f"Fitting {run_key}")
        model = TTPriceSurrogate(grid, model_oracle, seed=tt_seed)
        fit_start = perf_counter()
        fit = model.fit(
            budget,
            anova_samples=anova_samples,
            max_sweeps=args.max_sweeps,
            rank_increment=args.rank_increment,
            truncation=args.truncation,
            log=args.cross_log,
        )
        fit_total_time = perf_counter() - fit_start

        evaluation_start = perf_counter()

        def tt_interpolator(model_points, cubic_columns=(), fitted=model):
            return fitted.predict_hybrid_cubic(
                model_points,
                cubic_columns=cubic_columns,
                batch_size=args.tt_batch_size,
            )

        tt_fd = finite_difference_hybrid_component_arrays(
            tt_interpolator,
            transform,
            points,
            risk_columns,
            args.relative_bump,
        )
        evaluation_time = perf_counter() - evaluation_start
        diagnostics = _diagnostics(
            analytical,
            tt_fd,
            component_labels,
            m_values,
            slices,
        )
        projected = _project_components(tt_fd)
        projected_diagnostics = {
            "global_error_against_analytical": component_metrics(
                analytical,
                projected,
            ),
            "full_hessian_error_against_analytical": (
                full_hessian_error_metrics(analytical, projected)
            ),
            "hessian_shape": hessian_shape_diagnostics(
                gamma_matrices(projected)
            ),
        }
        record = {
            "budget": int(budget),
            "tt_seed": int(tt_seed),
            "fit": {
                **asdict(fit),
                "total_time_seconds": fit_total_time,
            },
            "greek_evaluation_time_seconds": evaluation_time,
            "components": components_to_json(tt_fd),
            **diagnostics,
            "error_against_grid_oracle": {
                "components": component_metrics(grid_fd, tt_fd),
                "full_hessian": full_hessian_error_metrics(grid_fd, tt_fd),
            },
            "deterministic_error_decomposition": _decomposition_metrics(
                analytical,
                exact_fd,
                grid_fd,
                tt_fd,
            ),
            "psd_projection_diagnostic": projected_diagnostics,
        }
        record["acceptance"] = _acceptance(record)
        payload["runs"][run_key] = record
        _write_checkpoint(args.output, payload)

        metrics = record["global_error_against_analytical"]
        hessian = record["full_hessian_error_against_analytical"]
        print(
            f"budget={budget:7d} seed={tt_seed} "
            f"rank={fit.effective_rank:6.2f} "
            f"delta={100 * metrics['delta']['normalized_mae']:6.3f}% "
            f"gamma={100 * metrics['gamma_diagonal']['normalized_mae']:6.3f}% "
            f"cross={100 * metrics['cross_gamma']['normalized_mae']:6.3f}% "
            f"H_F={100 * hessian['aggregate_relative_frobenius_error']:6.3f}% "
            f"pass={record['acceptance']['passed']}"
        )

    payload["campaign"] = args.campaign
    payload["requested_runs"] = [
        {"budget": int(budget), "tt_seed": int(seed)}
        for budget, seed in requested_runs
    ]
    _write_checkpoint(args.output, payload)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
