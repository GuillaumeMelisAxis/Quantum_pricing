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
    _acceptance,
    _power_of_two,
    _sign_sensitivity,
)

from stngpr.config import PaperConfig
from stngpr.convergence import (
    component_metrics,
    components_to_json,
    finite_difference_component_arrays,
    finite_difference_fixed_hybrid_component_arrays,
    finite_difference_hybrid_component_arrays,
    full_hessian_error_metrics,
)
from stngpr.coordinates import TransformedPricer, oracle_hybrid_cubic_predict
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk_grids import build_greek_coordinate_grid
from stngpr.tt_surrogate import TTPriceSurrogate

MODES = (
    "multilinear",
    "risk_coordinate_cubic",
    "component_hybrid",
    "unified_risk_cubic",
)

PROFILES = {
    "smoke": {
        "budget": 5_000,
        "anova_samples": 300,
        "panels": ["oos_mid"],
        "maturity_days": [30],
        "moneyness": [-0.05, 0.0, 0.05],
    },
    "intermediate": {
        "budget": 100_000,
        "anova_samples": 1_000,
        "panels": ["oos_mid"],
        "maturity_days": [7, 30],
        "moneyness": [-0.1, -0.05, 0.0, 0.05, 0.1],
    },
    "paper": {
        "budget": 200_000,
        "anova_samples": 2_000,
        "panels": ["oos_mid"],
        "maturity_days": [7, 30, 90],
        "moneyness": [-0.1, -0.05, 0.0, 0.05, 0.1],
    },
}

LOCAL_THRESHOLDS = {
    "gamma_curve_normalized_mae": 0.025,
    "cross_gamma_curve_normalized_mae": 0.025,
    "gamma_point_normalized_max_error": 0.10,
    "cross_gamma_point_normalized_max_error": 0.10,
}


def _fixed_columns(mode, n_assets):
    if mode == "multilinear":
        return ()
    if mode == "risk_coordinate_cubic":
        return (n_assets,)
    if mode == "unified_risk_cubic":
        return tuple(range(n_assets + 1))
    return None


def _is_scalar_surface(mode):
    return mode != "component_hybrid"


def _local_metrics(diagnostics):
    curves = diagnostics["curve_diagnostics"].values()
    gamma = [curve["components"]["gamma_diagonal"] for curve in curves]
    curves = diagnostics["curve_diagnostics"].values()
    cross = [curve["components"]["cross_gamma"] for curve in curves]
    return {
        "gamma_curve_normalized_mae": max(item["normalized_mae"] for item in gamma),
        "cross_gamma_curve_normalized_mae": max(
            item["normalized_mae"] for item in cross
        ),
        "gamma_point_normalized_max_error": max(
            item["normalized_max_absolute_error"] for item in gamma
        ),
        "cross_gamma_point_normalized_max_error": max(
            item["normalized_max_absolute_error"] for item in cross
        ),
    }


def _greek_aware_acceptance(record):
    global_error = record["global_error_against_analytical"]
    hessian = record["full_hessian_error_against_analytical"]
    material_sign = record["cross_gamma_sign_sensitivity"]["5e-04"][
        "sign_agreement_fraction"
    ]
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
        "negative_part_relative_frobenius_norm": (
            hessian["negative_part_relative_frobenius_norm"]
            <= ACCEPTANCE_THRESHOLDS["negative_part_relative_frobenius_norm"]
        ),
        "material_cross_gamma_sign_agreement": (
            material_sign >= ACCEPTANCE_THRESHOLDS["cross_gamma_sign_agreement"]
        ),
    }
    local = record["local_error_audit"]
    checks.update(
        {name: local[name] <= threshold for name, threshold in LOCAL_THRESHOLDS.items()}
    )
    return {"checks": checks, "passed": bool(all(checks.values()))}


def _direct_oracle_interpolator(grid, oracle):
    def interpolate(points, cubic_columns=()):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return np.asarray(
            [
                oracle_hybrid_cubic_predict(
                    grid,
                    oracle,
                    point[None, :],
                    cubic_columns=cubic_columns,
                )[0]
                for point in points
            ]
        )

    return interpolate


def _evaluate_mode(
    mode,
    interpolator,
    transform,
    points,
    risk_columns,
    relative_bump,
):
    if mode == "component_hybrid":
        return finite_difference_hybrid_component_arrays(
            interpolator,
            transform,
            points,
            risk_columns,
            relative_bump,
        )
    return finite_difference_fixed_hybrid_component_arrays(
        interpolator,
        transform,
        points,
        risk_columns,
        relative_bump,
        cubic_columns=_fixed_columns(mode, transform.n_assets),
    )


def _record(
    analytical,
    estimate,
    component_labels,
    m_values,
    slices,
    wall_time,
    mode,
):
    diagnostics = _diagnostics(
        analytical,
        estimate,
        component_labels,
        m_values,
        slices,
    )
    record = {
        "interpolation_mode": mode,
        "single_scalar_surface": _is_scalar_surface(mode),
        "cubic_columns": (
            None
            if mode == "component_hybrid"
            else list(_fixed_columns(mode, len(estimate["risk_columns"])))
        ),
        "wall_time_seconds": wall_time,
        "components": components_to_json(estimate),
        **diagnostics,
        "cross_gamma_sign_sensitivity": _sign_sensitivity(
            analytical,
            estimate,
            component_labels,
        ),
    }
    record["local_error_audit"] = _local_metrics(record)
    record["legacy_acceptance"] = _acceptance(record)
    record["greek_aware_local_acceptance"] = _greek_aware_acceptance(record)
    return record


def _compare(left, right):
    return {
        "components": component_metrics(left, right),
        "full_hessian": full_hessian_error_metrics(left, right),
    }


def _select_mode(records):
    candidates = [
        {
            "mode": mode,
            "wall_time_seconds": record["wall_time_seconds"],
            "accepted": record["greek_aware_local_acceptance"]["passed"],
        }
        for mode, record in records.items()
        if record["single_scalar_surface"]
    ]
    feasible = [candidate for candidate in candidates if candidate["accepted"]]
    selected = min(
        feasible,
        key=lambda candidate: candidate["wall_time_seconds"],
        default=None,
    )
    return {
        "rule": (
            "fastest single-surface interpolant passing global and local "
            "Greek-aware criteria"
        ),
        "candidates": candidates,
        "selected_mode": None if selected is None else selected["mode"],
    }


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether price, Delta and the five-asset Hessian can be "
            "derived from one fixed interpolated TT price surface."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--tt-seed", type=int, default=20260401)
    parser.add_argument("--truncation", type=float, default=1e-8)
    parser.add_argument("--anova-samples", type=int, default=None)
    parser.add_argument("--relative-bump", type=float, default=0.002)
    parser.add_argument("--spot-nodes", type=int, default=64)
    parser.add_argument("--coordinate-nodes", type=int, default=512)
    parser.add_argument("--maturity-nodes", type=int, default=64)
    parser.add_argument("--tt-batch-size", type=int, default=16)
    parser.add_argument("--full-cubic-batch-size", type=int, default=1)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--rank-increment", type=int, default=2)
    parser.add_argument("--cross-log", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=list(MODES),
    )
    parser.add_argument("--skip-grid-oracle", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v96_interpolant_consistency.json"),
    )
    args = parser.parse_args()

    design = PROFILES[args.profile]
    budget = int(args.budget or design["budget"])
    anova_samples = int(args.anova_samples or design["anova_samples"])
    modes = list(dict.fromkeys(args.modes))
    if budget <= 0 or anova_samples <= 0:
        parser.error("budget and anova-samples must be positive")
    if args.truncation <= 0.0:
        parser.error("truncation must be positive")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative-bump must lie in (0,1)")
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
    points, m_values, t_values, panel_labels, slices, rate = dense_market_panel(
        config,
        design["panels"],
        design["maturity_days"],
        np.asarray(design["moneyness"], dtype=float),
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
    payload = {
        "schema_version": "greeks-interpolant-consistency-v9.6",
        "profile": args.profile,
        "hypothesis": (
            "a fixed smooth tensor-product interpolant can retain the Greek "
            "accuracy of component-wise hybrid interpolation while making "
            "price, gradient and Hessian derivatives of one scalar surface"
        ),
        "settings": {
            "physical_shape": list(config.physical_shape),
            "qtt_core_count": int(sum(config.qtt_bits)),
            "logical_tensor_entries": int(np.prod(config.physical_shape)),
            "coordinate_mode": "bounded_standardized_risk",
            "budget": budget,
            "tt_seed": int(args.tt_seed),
            "truncation": float(args.truncation),
            "anova_samples": anova_samples,
            "relative_bump": float(args.relative_bump),
            "modes": modes,
            "rate": rate,
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
            "spot_panels": {name: SPOT_PANELS[name] for name in design["panels"]},
            "component_labels": component_labels,
        },
        "coordinate_description": coordinate_description,
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
        "grid_oracle": {},
        "fit": {},
        "variants": {},
        "selection": {},
    }
    _write(args.output, payload)

    if not args.skip_grid_oracle:
        oracle_interpolator = _direct_oracle_interpolator(grid, model_oracle)
        for mode in modes:
            print(f"Evaluating exact-grid {mode}")
            start = perf_counter()
            estimate = _evaluate_mode(
                mode,
                oracle_interpolator,
                transform,
                points,
                risk_columns,
                args.relative_bump,
            )
            payload["grid_oracle"][mode] = _record(
                analytical,
                estimate,
                component_labels,
                m_values,
                slices,
                perf_counter() - start,
                mode,
            )
            _write(args.output, payload)

    model = TTPriceSurrogate(grid, model_oracle, seed=args.tt_seed)
    print("Fitting raw TT-cross cores")
    start = perf_counter()
    fit = model.fit(
        budget,
        anova_samples=anova_samples,
        max_sweeps=args.max_sweeps,
        rank_increment=args.rank_increment,
        truncation=None,
        log=args.cross_log,
    )
    payload["fit"] = {**asdict(fit), "total_time_seconds": perf_counter() - start}
    _write(args.output, payload)

    for variant_label, truncation in (
        ("untruncated", None),
        (f"truncated_{args.truncation:.0e}", args.truncation),
    ):
        core_diagnostics = model.apply_truncation(truncation)
        variant = {
            "truncation": truncation,
            "core_diagnostics": asdict(core_diagnostics),
            "modes": {},
        }
        payload["variants"][variant_label] = variant
        for mode in modes:
            batch_size = (
                args.full_cubic_batch_size
                if mode == "unified_risk_cubic"
                else args.tt_batch_size
            )

            def tt_interpolator(
                model_points,
                cubic_columns=(),
                fitted=model,
                evaluation_batch_size=batch_size,
            ):
                return fitted.predict_hybrid_cubic(
                    model_points,
                    cubic_columns=cubic_columns,
                    batch_size=evaluation_batch_size,
                )

            print(f"Evaluating {variant_label} {mode}")
            start = perf_counter()
            estimate = _evaluate_mode(
                mode,
                tt_interpolator,
                transform,
                points,
                risk_columns,
                args.relative_bump,
            )
            record = _record(
                analytical,
                estimate,
                component_labels,
                m_values,
                slices,
                perf_counter() - start,
                mode,
            )
            if mode in payload["grid_oracle"]:
                record["error_against_grid_oracle"] = _compare(
                    payload["grid_oracle"][mode]["components"],
                    record["components"],
                )
            variant["modes"][mode] = record
            _write(args.output, payload)

        if "unified_risk_cubic" in variant["modes"]:
            unified = variant["modes"]["unified_risk_cubic"]["components"]
            for mode, record in variant["modes"].items():
                record["difference_from_unified_risk_cubic"] = _compare(
                    unified,
                    record["components"],
                )
        variant["selection"] = _select_mode(variant["modes"])
        payload["selection"][variant_label] = variant["selection"]
        _write(args.output, payload)

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
