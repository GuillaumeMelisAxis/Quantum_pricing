from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import PaperConfig
from stngpr.convergence import (
    COMPONENTS,
    component_metrics,
    components_to_json,
    curve_error_diagnostics,
    finite_difference_component_arrays,
    finite_difference_hybrid_component_arrays,
    gamma_matrices,
    hessian_shape_diagnostics,
)
from stngpr.coordinates import (
    CachedGridInterpolator,
    MarketCoordinatePricer,
    TransformedPricer,
)
from stngpr.greeks import geometric_basket_put_spot_greeks
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk_grids import build_greek_coordinate_grid

INTERPOLATION_MODES = ("multilinear", "risk_hybrid_cubic")
SPOT_PANELS = {
    "oos_low": [47.3, 63.8, 82.1, 101.6, 127.9],
    "oos_mid": [61.7, 78.9, 97.2, 118.6, 143.1],
    "oos_dispersed": [34.7, 57.9, 86.2, 121.3, 146.4],
}
PROFILES = {
    "smoke": {
        "curve_moneyness_nodes": 41,
        "maturity_days": [7.0, 30.0],
        "panels": ["oos_mid"],
        "relative_bumps": [0.001, 0.002],
    },
    "intermediate": {
        "curve_moneyness_nodes": 101,
        "maturity_days": [3.0, 7.0, 30.0, 90.0],
        "panels": ["oos_mid"],
        "relative_bumps": [0.001, 0.002],
    },
    "paper": {
        "curve_moneyness_nodes": 201,
        "maturity_days": [3.0, 7.0, 30.0, 90.0],
        "panels": ["oos_mid"],
        "relative_bumps": [0.001, 0.002],
    },
}


def _positive_power_of_two(value):
    value = int(value)
    return value > 1 and value & (value - 1) == 0


def _bump_key(value):
    return f"{float(value):.10g}"


def _curve_key(panel, maturity_days):
    return f"{panel}__T_{float(maturity_days):.10g}d"


def dense_market_panel(
    config,
    panel_names,
    maturity_days,
    moneyness,
    maximum_relative_bump,
):
    rate_axis = np.linspace(*config.rate_bounds, config.physical_shape[-2])
    rate = float(rate_axis[np.argmin(np.abs(rate_axis - 0.03))])
    points, m_values, t_values, labels, slices = [], [], [], [], []
    for panel_name in panel_names:
        spots = np.asarray(SPOT_PANELS[panel_name], dtype=float)
        basket = float(np.exp(np.mean(np.log(spots))))
        for days in maturity_days:
            start = len(points)
            maturity = float(days) / 365.0
            for m in moneyness:
                strike = basket * np.exp(float(m))
                point = np.concatenate((spots, [strike, rate, maturity]))
                for column, (lower, upper) in enumerate(config.bounds):
                    if not lower < point[column] < upper:
                        raise RuntimeError("the dense profile left the market domain")
                for column in range(config.n_assets):
                    lower, upper = config.bounds[column]
                    bump = maximum_relative_bump * point[column]
                    if not lower < point[column] - bump < point[column] + bump < upper:
                        raise RuntimeError("a dense-profile bump left the market domain")
                points.append(point)
                m_values.append(float(m))
                t_values.append(float(days))
                labels.append(panel_name)
            slices.append({
                "key": _curve_key(panel_name, days),
                "panel": panel_name,
                "maturity_days": float(days),
                "start_index": int(start),
                "stop_index": len(points),
            })
    return (
        np.asarray(points, dtype=float),
        np.asarray(m_values, dtype=float),
        np.asarray(t_values, dtype=float),
        np.asarray(labels),
        slices,
        rate,
    )


def analytical_components(config, pricer, points, risk_columns):
    columns = tuple(int(column) for column in risk_columns)
    delta, gamma = geometric_basket_put_spot_greeks(
        points[:, : config.n_assets],
        points[:, config.n_assets],
        points[:, config.n_assets + 1],
        points[:, config.n_assets + 2],
        config.volatilities,
        config.correlation,
        config.dividends,
    )
    pairs = [
        (left, right)
        for position, left in enumerate(columns)
        for right in columns[position + 1 :]
    ]
    return {
        "price": np.asarray(pricer(points), dtype=float),
        "delta": delta[:, columns],
        "gamma_diagonal": np.column_stack(
            [gamma[:, column, column] for column in columns]
        ),
        "cross_gamma": np.column_stack(
            [gamma[:, left, right] for left, right in pairs]
        ),
        "risk_columns": columns,
        "cross_gamma_pairs": pairs,
    }


def evaluate_grid_components(
    mode,
    interpolator,
    transform,
    points,
    risk_columns,
    relative_bump,
):
    if mode == "multilinear":
        return finite_difference_component_arrays(
            MarketCoordinatePricer(interpolator, transform),
            points,
            risk_columns,
            relative_bump,
        )
    if mode == "risk_hybrid_cubic":
        components = finite_difference_hybrid_component_arrays(
            interpolator,
            transform,
            points,
            risk_columns,
            relative_bump,
        )
        components["price"] = np.asarray(
            interpolator(
                transform.to_model(points),
                cubic_columns=(transform.n_assets,),
            ),
            dtype=float,
        )
        return components
    raise ValueError(f"unknown interpolation mode: {mode}")


def curve_diagnostics(reference, estimate, moneyness, slices):
    output = {}
    for record in slices:
        start, stop = record["start_index"], record["stop_index"]
        curve_m = moneyness[start:stop]
        component_diagnostics = {
            name: curve_error_diagnostics(
                curve_m,
                np.asarray(reference[name], dtype=float)[start:stop],
                np.asarray(estimate[name], dtype=float)[start:stop],
            )
            for name in COMPONENTS
        }
        subset = {
            name: np.asarray(estimate[name], dtype=float)[start:stop]
            for name in COMPONENTS
        }
        output[record["key"]] = {
            "panel": record["panel"],
            "maturity_days": record["maturity_days"],
            "components": component_diagnostics,
            "hessian_shape": hessian_shape_diagnostics(gamma_matrices(subset)),
        }
    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Densely audit Delta and Hessian curves between the sparse v9.2 "
            "validation points."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--curve-moneyness-nodes", type=int, default=None)
    parser.add_argument("--moneyness-min", type=float, default=-0.35)
    parser.add_argument("--moneyness-max", type=float, default=0.35)
    parser.add_argument("--maturity-days", nargs="+", type=float, default=None)
    parser.add_argument("--panels", nargs="+", choices=SPOT_PANELS, default=None)
    parser.add_argument("--relative-bumps", nargs="+", type=float, default=None)
    parser.add_argument("--risk-columns", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--spot-nodes", type=int, default=64)
    parser.add_argument("--coordinate-nodes", type=int, default=512)
    parser.add_argument("--maturity-nodes", type=int, default=64)
    parser.add_argument("--grid-batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v921_dense_profiles.json"),
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    n_curve = int(args.curve_moneyness_nodes or profile["curve_moneyness_nodes"])
    maturity_days = list(args.maturity_days or profile["maturity_days"])
    panel_names = list(args.panels or profile["panels"])
    bumps = sorted(set(args.relative_bumps or profile["relative_bumps"]))
    if n_curve < 21:
        parser.error("curve-moneyness-nodes must be at least 21")
    if not args.moneyness_min < args.moneyness_max:
        parser.error("moneyness-min must be smaller than moneyness-max")
    if any(days <= 0.0 for days in maturity_days):
        parser.error("maturity-days must be positive")
    if not bumps or any(not 0.0 < bump < 1.0 for bump in bumps):
        parser.error("relative-bumps must lie in (0,1)")
    for name, value in (
        ("spot-nodes", args.spot_nodes),
        ("coordinate-nodes", args.coordinate_nodes),
        ("maturity-nodes", args.maturity_nodes),
    ):
        if not _positive_power_of_two(value):
            parser.error(f"{name} must be a power of two greater than one")

    base = PaperConfig()
    risk_columns = tuple(args.risk_columns)
    if (
        len(risk_columns) < 2
        or len(set(risk_columns)) != len(risk_columns)
        or any(column < 0 or column >= base.n_assets for column in risk_columns)
    ):
        parser.error("risk-columns must contain at least two unique spot columns")

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
    points, m_values, t_values, labels, slices, rate = dense_market_panel(
        config,
        panel_names,
        maturity_days,
        dense_moneyness,
        max(bumps),
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

    payload = {
        "schema_version": "greeks-dense-profile-audit-v9.2.1",
        "profile": args.profile,
        "hypothesis": (
            "low sparse-panel MAE must persist on densely sampled moneyness "
            "curves without hidden cubic oscillations"
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
            "relative_bumps": bumps,
            "risk_columns": list(risk_columns),
            "interpolation_modes": list(INTERPOLATION_MODES),
            "coordinate_mode": "standardized_risk",
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
            "spot_panel_labels": labels.tolist(),
            "curve_slices": slices,
            "spot_panels": {name: SPOT_PANELS[name] for name in panel_names},
        },
        "coordinate_description": coordinate_description,
        "analytical_reference": {
            "components": components_to_json(analytical),
            "hessian_shape": hessian_shape_diagnostics(gamma_matrices(analytical)),
        },
        "exact_price_finite_difference_controls": {},
        "interpolation_results": {},
    }

    exact_by_bump = {}
    for bump in bumps:
        key = _bump_key(bump)
        start = perf_counter()
        exact_fd = finite_difference_component_arrays(
            market_pricer,
            points,
            risk_columns,
            bump,
        )
        exact_by_bump[key] = exact_fd
        payload["exact_price_finite_difference_controls"][key] = {
            "relative_bump": float(bump),
            "wall_time_seconds": perf_counter() - start,
            "components": components_to_json(exact_fd),
            "global_error_against_analytical": component_metrics(
                analytical,
                exact_fd,
            ),
            "curve_diagnostics": curve_diagnostics(
                analytical,
                exact_fd,
                m_values,
                slices,
            ),
        }

    for mode in INTERPOLATION_MODES:
        interpolator = CachedGridInterpolator(
            grid,
            model_oracle,
            evaluation_batch_size=args.grid_batch_size,
        )
        method_start = perf_counter()
        bump_results = {}
        for bump in bumps:
            key = _bump_key(bump)
            start = perf_counter()
            estimate = evaluate_grid_components(
                mode,
                interpolator,
                transform,
                points,
                risk_columns,
                bump,
            )
            metrics = component_metrics(analytical, estimate)
            curves = curve_diagnostics(
                analytical,
                estimate,
                m_values,
                slices,
            )
            bump_results[key] = {
                "relative_bump": float(bump),
                "incremental_wall_time_seconds": perf_counter() - start,
                "components": components_to_json(estimate),
                "global_error_against_analytical": metrics,
                "global_error_against_exact_price_finite_difference": (
                    component_metrics(exact_by_bump[key], estimate)
                ),
                "curve_diagnostics": curves,
                "hessian_shape": hessian_shape_diagnostics(
                    gamma_matrices(estimate)
                ),
            }
            maximum_gamma_error = max(
                curve["components"]["gamma_diagonal"][
                    "normalized_max_absolute_error"
                ]
                for curve in curves.values()
            )
            print(
                f"method={mode:19s} bump={100.0 * bump:5.2f}% "
                f"delta={100.0 * metrics['delta']['normalized_mae']:.4g}% "
                f"gamma={100.0 * metrics['gamma_diagonal']['normalized_mae']:.4g}% "
                f"cross={100.0 * metrics['cross_gamma']['normalized_mae']:.4g}% "
                f"max_curve_gamma={100.0 * maximum_gamma_error:.4g}%"
            )
        payload["interpolation_results"][mode] = {
            "wall_time_seconds": perf_counter() - method_start,
            "cache": interpolator.diagnostics(),
            "bump_results": bump_results,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
