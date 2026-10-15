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
from validate_tt_greek_convergence import (
    ACCEPTANCE_THRESHOLDS,
    PROFILES,
    _acceptance,
    _decomposition_metrics,
    _power_of_two,
    _project_components,
    _scalar_summary,
    _sign_sensitivity,
)

from stngpr.config import PaperConfig
from stngpr.convergence import (
    component_metrics,
    components_to_json,
    finite_difference_component_arrays,
    finite_difference_hybrid_component_arrays,
    full_hessian_error_metrics,
    gamma_matrices,
    hessian_shape_diagnostics,
)
from stngpr.coordinates import CachedGridInterpolator, TransformedPricer
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk_grids import build_greek_coordinate_grid
from stngpr.tt_surrogate import TTPriceSurrogate

ABLATION_DEFAULTS = {
    "smoke": {
        "budgets": [5_000],
        "tt_seeds": [20260401],
        "truncations": [0.0, 1e-8, 1e-6],
        "anova_samples": 300,
    },
    "intermediate": {
        "budgets": [100_000, 150_000],
        "tt_seeds": [20260401],
        "truncations": [0.0, 1e-10, 1e-8, 1e-6],
        "anova_samples": 1_000,
    },
    "paper": {
        "budgets": [200_000],
        "tt_seeds": [20260401, 20260402, 20260403, 20260404, 20260405],
        "truncations": [0.0, 1e-14, 1e-12, 1e-10, 1e-8],
        "anova_samples": 2_000,
    },
}

GREEK_AWARE_THRESHOLDS = {
    **ACCEPTANCE_THRESHOLDS,
    "cross_gamma_sign_zero_tolerance": 5e-4,
}


def _truncation_label(value):
    return "untruncated" if float(value) == 0.0 else f"{float(value):.0e}"


def _variant_metrics(variant):
    global_error = variant["global_error_against_analytical"]
    hessian = variant["full_hessian_error_against_analytical"]
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
        "negative_part_relative_frobenius_norm": hessian[
            "negative_part_relative_frobenius_norm"
        ],
        "cross_gamma_sign_agreement": variant["cross_gamma_sign_fidelity"][
            "sign_agreement_fraction"
        ],
        "material_cross_gamma_sign_agreement": variant[
            "cross_gamma_sign_sensitivity"
        ]["5e-04"]["sign_agreement_fraction"],
        "effective_rank": variant["core_diagnostics"]["effective_rank"],
        "parameter_count": variant["core_diagnostics"]["parameter_count"],
        "rank_retention_fraction": variant["rank_retention_fraction"],
        "parameter_retention_fraction": variant[
            "parameter_retention_fraction"
        ],
        "greek_evaluation_time_seconds": variant[
            "greek_evaluation_time_seconds"
        ],
    }


def _greek_aware_acceptance(variant):
    metrics = _variant_metrics(variant)
    checks = {
        name: metrics[name] <= GREEK_AWARE_THRESHOLDS[name]
        for name in (
            "price_normalized_mae",
            "delta_normalized_mae",
            "gamma_diagonal_normalized_mae",
            "cross_gamma_normalized_mae",
            "hessian_relative_frobenius_error",
            "negative_part_relative_frobenius_norm",
        )
    }
    checks["material_cross_gamma_sign_agreement"] = (
        metrics["material_cross_gamma_sign_agreement"]
        >= GREEK_AWARE_THRESHOLDS["cross_gamma_sign_agreement"]
    )
    return {"checks": checks, "passed": bool(all(checks.values()))}


def _summarize(runs):
    grouped = {}
    for run in runs.values():
        for label, variant in run.get("variants", {}).items():
            grouped.setdefault((int(run["budget"]), label), []).append(variant)
    output = {}
    for (budget, label), variants in sorted(grouped.items()):
        metrics = [_variant_metrics(variant) for variant in variants]
        output.setdefault(str(budget), {})[label] = {
            name: _scalar_summary([record[name] for record in metrics])
            for name in metrics[0]
        }
        output[str(budget)][label]["run_count"] = len(variants)
        output[str(budget)][label]["acceptance_pass_count"] = sum(
            variant["acceptance"]["passed"] for variant in variants
        )
        output[str(budget)][label]["greek_aware_pass_count"] = sum(
            variant["greek_aware_acceptance"]["passed"]
            for variant in variants
        )
    return output


def _selection_summary(summary, requested_labels, expected_seeds):
    output = {}
    for budget, variants in summary.items():
        candidates = []
        for label in requested_labels:
            record = variants.get(label)
            if record is None:
                continue
            complete = record["run_count"] == expected_seeds
            robust = complete and record["greek_aware_pass_count"] == expected_seeds
            candidates.append({
                "label": label,
                "complete": complete,
                "robustly_accepted": robust,
                "run_count": record["run_count"],
                "mean_parameter_retention_fraction": record[
                    "parameter_retention_fraction"
                ]["mean"],
                "worst_hessian_relative_frobenius_error": record[
                    "hessian_relative_frobenius_error"
                ]["maximum"],
                "worst_gamma_diagonal_normalized_mae": record[
                    "gamma_diagonal_normalized_mae"
                ]["maximum"],
                "worst_cross_gamma_normalized_mae": record[
                    "cross_gamma_normalized_mae"
                ]["maximum"],
            })
        feasible = [item for item in candidates if item["robustly_accepted"]]
        selected = min(
            feasible,
            key=lambda item: (
                item["mean_parameter_retention_fraction"],
                item["worst_hessian_relative_frobenius_error"],
            ),
            default=None,
        )
        output[budget] = {
            "selection_rule": (
                "most compressed complete candidate passing every Greek-aware "
                "criterion for every requested seed"
            ),
            "expected_seed_count": expected_seeds,
            "candidates": candidates,
            "selected_truncation": None if selected is None else selected["label"],
        }
    return output


def _write_checkpoint(path, payload, requested_labels, expected_seeds):
    payload["summary"] = _summarize(payload["runs"])
    payload["selection"] = _selection_summary(
        payload["summary"],
        requested_labels,
        expected_seeds,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the v9.5 multi-seed QTT truncation campaign on identical raw "
            "TT-cross cores and select a Greek-aware compression tolerance."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--tt-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--truncations", nargs="+", type=float, default=None)
    parser.add_argument("--anova-samples", type=int, default=None)
    parser.add_argument("--relative-bump", type=float, default=0.002)
    parser.add_argument("--spot-nodes", type=int, default=64)
    parser.add_argument("--coordinate-nodes", type=int, default=512)
    parser.add_argument("--maturity-nodes", type=int, default=64)
    parser.add_argument("--grid-batch-size", type=int, default=256)
    parser.add_argument("--tt-batch-size", type=int, default=16)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--rank-increment", type=int, default=2)
    parser.add_argument("--cross-log", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v95_tt_truncation_multiseed.json"),
    )
    args = parser.parse_args()

    design = PROFILES[args.profile]
    defaults = ABLATION_DEFAULTS[args.profile]
    budgets = sorted(set(args.budgets or defaults["budgets"]))
    seeds = list(dict.fromkeys(args.tt_seeds or defaults["tt_seeds"]))
    truncations = list(dict.fromkeys(
        args.truncations or defaults["truncations"]
    ))
    anova_samples = int(args.anova_samples or defaults["anova_samples"])
    if any(budget <= 0 for budget in budgets):
        parser.error("budgets must be positive")
    if not seeds:
        parser.error("at least one TT seed is required")
    if any(value < 0.0 for value in truncations):
        parser.error("truncations must be non-negative; use zero for raw cores")
    if 0.0 not in truncations:
        parser.error("truncations must include zero as the untruncated control")
    if anova_samples <= 0:
        parser.error("anova-samples must be positive")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative-bump must lie in (0,1)")
    if args.max_sweeps <= 0 or args.rank_increment <= 0:
        parser.error("TT-cross sweep settings must be positive")
    for name, value in (
        ("spot-nodes", args.spot_nodes),
        ("coordinate-nodes", args.coordinate_nodes),
        ("maturity-nodes", args.maturity_nodes),
    ):
        if not _power_of_two(value):
            parser.error(f"{name} must be a power of two greater than one")

    base = PaperConfig()
    shape = list(base.physical_shape)
    shape[: base.n_assets] = [int(args.spot_nodes)] * base.n_assets
    shape[base.n_assets] = int(args.coordinate_nodes)
    shape[-1] = int(args.maturity_nodes)
    config = replace(base, physical_shape=tuple(shape))
    risk_columns = tuple(range(config.n_assets))
    component_labels = _component_labels(risk_columns)
    moneyness = np.asarray(design["moneyness"], dtype=float)
    maturity_days = list(design["maturity_days"])
    panel_names = list(design["panels"])
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
    grid_time = perf_counter() - grid_start

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
    }
    if args.output.exists() and not args.overwrite:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "greeks-tt-truncation-v9.5"
            or payload.get("experiment_signature") != signature
        ):
            parser.error(
                "the output contains an incompatible experiment; choose a "
                "different path or pass --overwrite"
            )
        print(f"Resuming {len(payload['runs'])} fitted pairs from {args.output}")
    else:
        payload = {
            "schema_version": "greeks-tt-truncation-v9.5",
            "profile": args.profile,
            "hypothesis": (
                "the raw 200k TT-cross surrogate is seed-robust, while a fixed "
                "price-norm truncation may remove modes that remain material "
                "for first and second spot derivatives"
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
                "tt_seeds": seeds,
                "truncations": truncations,
                "anova_samples": anova_samples,
                "max_sweeps": int(args.max_sweeps),
                "rank_increment": int(args.rank_increment),
                "grid_batch_size": int(args.grid_batch_size),
                "tt_batch_size": int(args.tt_batch_size),
                "rate": rate,
                "finite_difference_prices_per_point": 1 + 2 * base.n_assets**2,
                "monte_carlo": False,
            },
            "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
            "greek_aware_thresholds": GREEK_AWARE_THRESHOLDS,
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
                "wall_time_seconds": grid_time,
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
            "runs": {},
            "summary": {},
            "selection": {},
        }
        requested_labels = [_truncation_label(value) for value in truncations]
        _write_checkpoint(args.output, payload, requested_labels, len(seeds))

    payload["settings"].update({
        "budgets": budgets,
        "tt_seeds": seeds,
        "truncations": truncations,
    })
    requested_labels = [_truncation_label(value) for value in truncations]
    for budget in budgets:
        for seed in seeds:
            run_key = f"seed_{int(seed)}__budget_{int(budget)}"
            existing = payload["runs"].get(run_key, {})
            if all(
                label in existing.get("variants", {})
                for label in requested_labels
            ):
                print(f"Skipping completed {run_key}")
                continue

            print(f"Fitting raw TT-cross cores for {run_key}")
            model = TTPriceSurrogate(grid, model_oracle, seed=seed)
            fit_start = perf_counter()
            fit = model.fit(
                budget,
                anova_samples=anova_samples,
                max_sweeps=args.max_sweeps,
                rank_increment=args.rank_increment,
                truncation=None,
                log=args.cross_log,
            )
            fit_total_time = perf_counter() - fit_start
            raw_core_diagnostics = model.apply_truncation(None)
            run = {
                "budget": int(budget),
                "tt_seed": int(seed),
                "raw_fit": {
                    **asdict(fit),
                    "total_time_seconds": fit_total_time,
                },
                "raw_core_diagnostics": asdict(raw_core_diagnostics),
                "variants": {},
            }
            payload["runs"][run_key] = run

            for truncation in truncations:
                label = _truncation_label(truncation)
                core_diagnostics = model.apply_truncation(truncation)
                evaluation_start = perf_counter()

                def tt_interpolator(
                    model_points,
                    cubic_columns=(),
                    fitted=model,
                ):
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
                variant = {
                    "truncation": None if truncation == 0.0 else truncation,
                    "core_diagnostics": asdict(core_diagnostics),
                    "rank_retention_fraction": (
                        core_diagnostics.effective_rank
                        / fit.pre_truncation_effective_rank
                    ),
                    "parameter_retention_fraction": (
                        core_diagnostics.parameter_count
                        / fit.pre_truncation_parameter_count
                    ),
                    "greek_evaluation_time_seconds": evaluation_time,
                    "components": components_to_json(tt_fd),
                    **diagnostics,
                    "cross_gamma_sign_sensitivity": _sign_sensitivity(
                        analytical,
                        tt_fd,
                        component_labels,
                    ),
                    "error_against_grid_oracle": {
                        "components": component_metrics(grid_fd, tt_fd),
                        "full_hessian": full_hessian_error_metrics(
                            grid_fd,
                            tt_fd,
                        ),
                    },
                    "deterministic_error_decomposition": (
                        _decomposition_metrics(
                            analytical,
                            exact_fd,
                            grid_fd,
                            tt_fd,
                        )
                    ),
                    "psd_projection_diagnostic": {
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
                    },
                }
                variant["acceptance"] = _acceptance(variant)
                variant["greek_aware_acceptance"] = _greek_aware_acceptance(
                    variant
                )
                run["variants"][label] = variant
                _write_checkpoint(
                    args.output,
                    payload,
                    requested_labels,
                    len(seeds),
                )
                metrics = variant["global_error_against_analytical"]
                hessian = variant["full_hessian_error_against_analytical"]
                print(
                    f"budget={budget:7d} seed={seed} trunc={label:>11s} "
                    f"rank={core_diagnostics.effective_rank:6.2f} "
                    f"gamma={100 * metrics['gamma_diagonal']['normalized_mae']:6.3f}% "
                    f"cross={100 * metrics['cross_gamma']['normalized_mae']:6.3f}% "
                    f"H_F={100 * hessian['aggregate_relative_frobenius_error']:6.3f}%"
                )

            raw_components = run["variants"]["untruncated"]["components"]
            for variant in run["variants"].values():
                variant["error_against_untruncated_tt"] = {
                    "components": component_metrics(
                        raw_components,
                        variant["components"],
                    ),
                    "full_hessian": full_hessian_error_metrics(
                        raw_components,
                        variant["components"],
                    ),
                }
            _write_checkpoint(
                args.output,
                payload,
                requested_labels,
                len(seeds),
            )

    _write_checkpoint(args.output, payload, requested_labels, len(seeds))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
