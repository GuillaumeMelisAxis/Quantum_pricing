from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
from validate_dense_greek_profiles import (
    SPOT_PANELS,
    analytical_components,
    curve_diagnostics,
    dense_market_panel,
)

from stngpr.config import PaperConfig
from stngpr.convergence import (
    columnwise_error_metrics,
    component_metrics,
    components_to_json,
    cross_gamma_sign_diagnostics,
    finite_difference_component_arrays,
    finite_difference_hybrid_component_arrays,
    full_hessian_error_metrics,
    gamma_matrices,
    hessian_shape_diagnostics,
)
from stngpr.coordinates import CachedGridInterpolator, TransformedPricer
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk_grids import build_greek_coordinate_grid

PROFILES = {
    "smoke": {
        "curve_moneyness_nodes": 31,
        "maturity_days": [7.0, 30.0],
        "panels": ["oos_mid"],
    },
    "intermediate": {
        "curve_moneyness_nodes": 51,
        "maturity_days": [3.0, 7.0, 30.0, 90.0],
        "panels": ["oos_mid", "oos_dispersed"],
    },
    "paper": {
        "curve_moneyness_nodes": 101,
        "maturity_days": [3.0, 7.0, 30.0, 90.0],
        "panels": ["oos_low", "oos_mid", "oos_dispersed"],
    },
}


def _positive_power_of_two(value):
    value = int(value)
    return value > 1 and value & (value - 1) == 0


def _component_labels(risk_columns):
    columns = tuple(int(column) for column in risk_columns)
    pairs = list(itertools.combinations(columns, 2))
    return {
        "delta": [f"Delta_{column + 1}" for column in columns],
        "gamma_diagonal": [
            f"Gamma_{column + 1},{column + 1}" for column in columns
        ],
        "cross_gamma": [
            f"Gamma_{left + 1},{right + 1}" for left, right in pairs
        ],
    }


def _component_breakdown(reference, estimate, labels):
    return {
        name: columnwise_error_metrics(
            reference[name],
            estimate[name],
            labels[name],
        )
        for name in ("delta", "gamma_diagonal", "cross_gamma")
    }


def _diagnostics(reference, estimate, labels, moneyness, slices):
    matrices = gamma_matrices(estimate)
    return {
        "global_error_against_analytical": component_metrics(
            reference,
            estimate,
        ),
        "componentwise_error_against_analytical": _component_breakdown(
            reference,
            estimate,
            labels,
        ),
        "full_hessian_error_against_analytical": full_hessian_error_metrics(
            reference,
            estimate,
        ),
        "cross_gamma_sign_fidelity": cross_gamma_sign_diagnostics(
            reference["cross_gamma"],
            estimate["cross_gamma"],
            labels=labels["cross_gamma"],
        ),
        "hessian_shape": hessian_shape_diagnostics(matrices),
        "curve_diagnostics": curve_diagnostics(
            reference,
            estimate,
            moneyness,
            slices,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate all five spot Deltas and the complete 5x5 fixed-strike "
            "Hessian of the selected risk-hybrid grid."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--curve-moneyness-nodes", type=int, default=None)
    parser.add_argument("--moneyness-min", type=float, default=-0.35)
    parser.add_argument("--moneyness-max", type=float, default=0.35)
    parser.add_argument("--maturity-days", nargs="+", type=float, default=None)
    parser.add_argument("--panels", nargs="+", choices=SPOT_PANELS, default=None)
    parser.add_argument("--relative-bump", type=float, default=0.002)
    parser.add_argument("--spot-nodes", type=int, default=64)
    parser.add_argument("--coordinate-nodes", type=int, default=512)
    parser.add_argument("--maturity-nodes", type=int, default=64)
    parser.add_argument("--grid-batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v93_full_hessian.json"),
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    n_curve = int(args.curve_moneyness_nodes or profile["curve_moneyness_nodes"])
    maturity_days = list(args.maturity_days or profile["maturity_days"])
    panel_names = list(args.panels or profile["panels"])
    if n_curve < 21:
        parser.error("curve-moneyness-nodes must be at least 21")
    if not args.moneyness_min < args.moneyness_max:
        parser.error("moneyness-min must be smaller than moneyness-max")
    if any(days <= 0.0 for days in maturity_days):
        parser.error("maturity-days must be positive")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative-bump must lie in (0,1)")
    for name, value in (
        ("spot-nodes", args.spot_nodes),
        ("coordinate-nodes", args.coordinate_nodes),
        ("maturity-nodes", args.maturity_nodes),
    ):
        if not _positive_power_of_two(value):
            parser.error(f"{name} must be a power of two greater than one")

    base = PaperConfig()
    risk_columns = tuple(range(base.n_assets))
    labels = _component_labels(risk_columns)
    shape = list(base.physical_shape)
    shape[: base.n_assets] = [int(args.spot_nodes)] * base.n_assets
    shape[base.n_assets] = int(args.coordinate_nodes)
    shape[-1] = int(args.maturity_nodes)
    config = replace(base, physical_shape=tuple(shape))
    dense_moneyness = np.linspace(
        float(args.moneyness_min),
        float(args.moneyness_max),
        n_curve,
    )
    points, m_values, t_values, panel_labels, slices, rate = dense_market_panel(
        config,
        panel_names,
        maturity_days,
        dense_moneyness,
        args.relative_bump,
    )
    market_pricer = EuropeanGeometricBasketPricer(config)
    analytical = analytical_components(
        config,
        market_pricer,
        points,
        risk_columns,
    )
    grid, transform, coordinate_description = build_greek_coordinate_grid(
        config,
        "standardized_risk",
    )
    model_oracle = TransformedPricer(market_pricer, transform)

    exact_start = perf_counter()
    exact_fd = finite_difference_component_arrays(
        market_pricer,
        points,
        risk_columns,
        args.relative_bump,
    )
    exact_wall_time = perf_counter() - exact_start

    interpolator = CachedGridInterpolator(
        grid,
        model_oracle,
        evaluation_batch_size=args.grid_batch_size,
    )
    hybrid_start = perf_counter()
    hybrid = finite_difference_hybrid_component_arrays(
        interpolator,
        transform,
        points,
        risk_columns,
        args.relative_bump,
    )
    hybrid_wall_time = perf_counter() - hybrid_start

    analytical_matrices = gamma_matrices(analytical)
    payload = {
        "schema_version": "greeks-full-spot-hessian-v9.3",
        "profile": args.profile,
        "hypothesis": (
            "the v9.2 selected standardized-risk geometry and risk-hybrid "
            "interpolation remain accurate for all five spot Deltas and all "
            "fifteen independent entries of the 5x5 spot Hessian"
        ),
        "settings": {
            "physical_shape": list(config.physical_shape),
            "qtt_core_count": int(sum(config.qtt_bits)),
            "spot_nodes": int(args.spot_nodes),
            "coordinate_nodes": int(args.coordinate_nodes),
            "maturity_nodes": int(args.maturity_nodes),
            "curve_moneyness_nodes": n_curve,
            "moneyness_bounds": [float(args.moneyness_min), float(args.moneyness_max)],
            "maturity_days": maturity_days,
            "panels": panel_names,
            "relative_bump": float(args.relative_bump),
            "risk_columns": list(risk_columns),
            "delta_count_per_point": base.n_assets,
            "gamma_diagonal_count_per_point": base.n_assets,
            "cross_gamma_count_per_point": base.n_assets * (base.n_assets - 1) // 2,
            "finite_difference_prices_per_point": 1 + 2 * base.n_assets**2,
            "coordinate_mode": "standardized_risk",
            "interpolation_mode": "risk_hybrid_cubic",
            "maturity_placement": "exact requested market maturity, generally off-grid",
            "rate": rate,
            "grid_batch_size": int(args.grid_batch_size),
            "tt_compression": False,
            "monte_carlo": False,
        },
        "test_design": {
            "market_parameters": points.tolist(),
            "log_moneyness": m_values.tolist(),
            "maturity_days": t_values.tolist(),
            "spot_panel_labels": panel_labels.tolist(),
            "curve_slices": slices,
            "spot_panels": {name: SPOT_PANELS[name] for name in panel_names},
            "component_labels": labels,
        },
        "coordinate_description": coordinate_description,
        "analytical_reference": {
            "components": components_to_json(analytical),
            "hessian_shape": hessian_shape_diagnostics(analytical_matrices),
            "cross_gamma_sign_structure": cross_gamma_sign_diagnostics(
                analytical["cross_gamma"],
                analytical["cross_gamma"],
                labels=labels["cross_gamma"],
            ),
        },
        "exact_price_finite_difference_control": {
            "wall_time_seconds": exact_wall_time,
            "components": components_to_json(exact_fd),
            **_diagnostics(
                analytical,
                exact_fd,
                labels,
                m_values,
                slices,
            ),
        },
        "risk_hybrid_cubic": {
            "wall_time_seconds": hybrid_wall_time,
            "cache": interpolator.diagnostics(),
            "components": components_to_json(hybrid),
            **_diagnostics(
                analytical,
                hybrid,
                labels,
                m_values,
                slices,
            ),
        },
    }

    metrics = payload["risk_hybrid_cubic"]["global_error_against_analytical"]
    hessian = payload["risk_hybrid_cubic"][
        "full_hessian_error_against_analytical"
    ]
    signs = payload["risk_hybrid_cubic"]["cross_gamma_sign_fidelity"]
    print(
        f"points={len(points)} prices/point={1 + 2 * base.n_assets**2} "
        f"delta={100.0 * metrics['delta']['normalized_mae']:.4g}% "
        f"gamma={100.0 * metrics['gamma_diagonal']['normalized_mae']:.4g}% "
        f"cross={100.0 * metrics['cross_gamma']['normalized_mae']:.4g}% "
        f"H_F={100.0 * hessian['aggregate_relative_frobenius_error']:.4g}% "
        f"sign={100.0 * signs['sign_agreement_fraction']:.4g}% "
        f"material_psd={hessian['material_psd_violation_count']}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
