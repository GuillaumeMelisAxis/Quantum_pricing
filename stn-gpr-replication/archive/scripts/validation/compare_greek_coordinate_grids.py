from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import PaperConfig
from stngpr.convergence import (
    COMPONENTS,
    component_metrics,
    components_to_json,
    finite_difference_component_arrays,
    finite_difference_hybrid_component_arrays,
    gamma_matrices,
    hessian_shape_diagnostics,
)
from stngpr.coordinates import CachedGridInterpolator, TransformedPricer
from stngpr.greeks import geometric_basket_put_spot_greeks
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk_grids import GREEK_GRID_MODES, build_greek_coordinate_grid


PROFILES = {
    "smoke": {
        "moneyness_nodes": 64,
        "maturity_nodes": 16,
        "moneyness": [-0.15, 0.0, 0.15],
        "maturity_days": [14.0, 90.0],
    },
    "intermediate": {
        "moneyness_nodes": 256,
        "maturity_nodes": 32,
        "moneyness": [-0.35, -0.15, -0.05, 0.0, 0.05, 0.15, 0.35],
        "maturity_days": [7.0, 14.0, 30.0, 90.0, 365.0],
    },
    "paper": {
        "moneyness_nodes": 512,
        "maturity_nodes": 64,
        "moneyness": [-0.50, -0.35, -0.15, -0.05, 0.0, 0.05, 0.15, 0.35, 0.50],
        "maturity_days": [3.0, 7.0, 14.0, 30.0, 90.0, 365.0, 730.0],
    },
}

MONITOR_MATURITY_DAYS = [3.0, 10.0, 21.0, 60.0, 180.0, 540.0, 1095.0]


def _positive_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 1 and value & (value - 1) == 0


def _nearest_unique(axis: np.ndarray, targets: np.ndarray) -> np.ndarray:
    indices = []
    for target in np.asarray(targets, dtype=float):
        order = np.argsort(np.abs(axis - target))
        index = next(int(item) for item in order if int(item) not in indices)
        indices.append(index)
    return axis[np.sort(indices)]


def validation_panel(config, moneyness, maturity_days, rate_target=0.03):
    """Use common physical nodes outside the ablated coordinate axis.

    This removes spot, rate and maturity placement as confounders.  Only the
    sixth physical coordinate differs between candidates.
    """
    spot_axis = np.linspace(*config.spot_bounds, config.physical_shape[0])
    central = np.linspace(
        config.physical_shape[0] // 2 - 4,
        config.physical_shape[0] // 2 + 4,
        config.n_assets,
        dtype=int,
    )
    spots = spot_axis[central]
    basket = float(np.exp(np.mean(np.log(spots))))

    rate_axis = np.linspace(*config.rate_bounds, config.physical_shape[-2])
    rate = float(rate_axis[np.argmin(np.abs(rate_axis - float(rate_target)))])
    maturity_axis = (
        config.maturity_bounds[0]
        + (config.maturity_bounds[1] - config.maturity_bounds[0])
        * np.linspace(0.0, 1.0, config.physical_shape[-1]) ** 2
    )
    maturities = _nearest_unique(
        maturity_axis,
        np.asarray(maturity_days, dtype=float) / 365.0,
    )

    points, m_values, t_days = [], [], []
    for maturity in maturities:
        for m in moneyness:
            strike = basket * np.exp(float(m))
            points.append(np.concatenate((spots, [strike, rate, maturity])))
            m_values.append(float(m))
            t_days.append(float(maturity * 365.0))
    return (
        np.asarray(points, dtype=float),
        np.asarray(m_values, dtype=float),
        np.asarray(t_days, dtype=float),
        {
            "spots": spots.tolist(),
            "spot_axis_indices": central.tolist(),
            "rate": rate,
            "maturity_days": (maturities * 365.0).tolist(),
        },
    )


def analytical_components(config, pricer, points, risk_columns):
    columns = tuple(int(value) for value in risk_columns)
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


def matrix_metrics(reference, estimate):
    truth = gamma_matrices(reference)
    value = gamma_matrices(estimate)
    errors = np.linalg.norm(value - truth, axis=(1, 2))
    scales = np.linalg.norm(truth, axis=(1, 2))
    mean_scale = float(np.mean(scales))
    return {
        "mean_frobenius_error": float(np.mean(errors)),
        "rmse_frobenius_error": float(np.sqrt(np.mean(errors**2))),
        "maximum_frobenius_error": float(np.max(errors)),
        "mean_reference_frobenius_norm": mean_scale,
        "normalized_mean_frobenius_error": (
            float(np.mean(errors) / mean_scale) if mean_scale > 1e-14 else None
        ),
    }


def conditional_metrics(reference, estimate, m_values, maturity_days):
    def masked(mask):
        reduced_reference = {
            name: np.asarray(reference[name], dtype=float)[mask]
            for name in COMPONENTS
        }
        reduced_estimate = {
            name: np.asarray(estimate[name], dtype=float)[mask]
            for name in COMPONENTS
        }
        return component_metrics(reduced_reference, reduced_estimate)

    output = {"by_moneyness": {}, "by_maturity_days": {}}
    for m in np.unique(m_values):
        mask = np.isclose(m_values, m)
        output["by_moneyness"][f"{m:+.10g}"] = masked(mask)
    for days in np.unique(maturity_days):
        mask = np.isclose(maturity_days, days)
        output["by_maturity_days"][f"{days:.10g}"] = masked(mask)
    return output


def pointwise_errors(reference, estimate):
    return {
        name: (
            np.asarray(estimate[name], dtype=float)
            - np.asarray(reference[name], dtype=float)
        ).tolist()
        for name in COMPONENTS
    }


def local_resolution(grid, transform, points, config):
    model = transform.to_model(points)
    coordinate_column = config.n_assets
    axis = grid.axes[coordinate_column]
    coordinate_width, m_width, standardized_m_width = [], [], []
    weights = np.full(config.n_assets, 1.0 / config.n_assets)
    covariance = (
        np.outer(config.volatilities, config.volatilities)
        * config.correlation
    )
    basket_sigma = float(np.sqrt(weights @ covariance @ weights))

    for market, model_point in zip(points, model):
        upper = int(np.searchsorted(axis, model_point[coordinate_column], side="right"))
        upper = int(np.clip(upper, 1, axis.size - 1))
        lower = upper - 1
        coordinate_width.append(float(axis[upper] - axis[lower]))
        endpoints = np.vstack((model_point, model_point))
        endpoints[:, coordinate_column] = axis[[lower, upper]]
        market_endpoints = transform.to_market(endpoints)
        baskets = np.exp(
            np.mean(np.log(market_endpoints[:, : config.n_assets]), axis=1)
        )
        m_endpoints = np.log(
            market_endpoints[:, config.n_assets] / baskets
        )
        width = float(abs(m_endpoints[1] - m_endpoints[0]))
        m_width.append(width)
        maturity = float(market[config.n_assets + 2])
        standardized_m_width.append(width / (basket_sigma * np.sqrt(maturity)))

    def summary(values):
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    return {
        "coordinate_cell_width": summary(coordinate_width),
        "implied_log_moneyness_cell_width": summary(m_width),
        "implied_standardized_cell_width": summary(standardized_m_width),
        "pointwise_implied_log_moneyness_cell_width": m_width,
    }


def implied_moneyness_axes(grid, transform, config, node_design):
    spots = np.asarray(node_design["spots"], dtype=float)
    basket = float(np.exp(np.mean(np.log(spots))))
    rate = float(node_design["rate"])
    coordinate_axis = grid.axes[config.n_assets]
    output = {}
    for days in node_design["maturity_days"]:
        maturity = float(days) / 365.0
        market = np.concatenate((spots, [basket, rate, maturity]))
        base = transform.to_model(market)[0]
        model = np.repeat(base[None, :], coordinate_axis.size, axis=0)
        model[:, config.n_assets] = coordinate_axis
        mapped = transform.to_market(model)
        mapped_basket = np.exp(
            np.mean(np.log(mapped[:, : config.n_assets]), axis=1)
        )
        m = np.log(mapped[:, config.n_assets] / mapped_basket)
        output[f"{days:.10g}"] = m.tolist()
    return output


def selection_score(metrics):
    values = []
    for component in ("delta", "gamma_diagonal", "cross_gamma"):
        value = metrics[component]["normalized_mae"]
        if value is not None:
            values.append(float(value))
    return float(sum(values))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare equal-size price- and Greek-oriented coordinate grids "
            "against exact European geometric-basket Greeks."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=GREEK_GRID_MODES,
        default=list(GREEK_GRID_MODES),
    )
    parser.add_argument("--moneyness-nodes", type=int, default=None)
    parser.add_argument("--maturity-nodes", type=int, default=None)
    parser.add_argument("--risk-columns", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--relative-bump", type=float, default=0.002)
    parser.add_argument("--grid-batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v9_coordinate_ablation.json"),
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    moneyness_nodes = args.moneyness_nodes or profile["moneyness_nodes"]
    maturity_nodes = args.maturity_nodes or profile["maturity_nodes"]
    if not _positive_power_of_two(moneyness_nodes):
        parser.error("moneyness-nodes must be a power of two greater than one")
    if not _positive_power_of_two(maturity_nodes):
        parser.error("maturity-nodes must be a power of two greater than one")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative-bump must lie in (0,1)")

    base = PaperConfig()
    risk_columns = tuple(args.risk_columns)
    if (
        len(risk_columns) < 2
        or len(set(risk_columns)) != len(risk_columns)
        or any(column < 0 or column >= base.n_assets for column in risk_columns)
    ):
        parser.error("risk-columns must contain at least two unique valid spot columns")
    shape = list(base.physical_shape)
    shape[base.n_assets] = int(moneyness_nodes)
    shape[-1] = int(maturity_nodes)
    config = replace(base, physical_shape=tuple(shape))

    points, m_values, maturity_days, node_design = validation_panel(
        config,
        profile["moneyness"],
        profile["maturity_days"],
    )
    market_pricer = EuropeanGeometricBasketPricer(config)
    analytical = analytical_components(
        config,
        market_pricer,
        points,
        risk_columns,
    )
    exact_fd_start = perf_counter()
    exact_fd = finite_difference_component_arrays(
        market_pricer,
        points,
        risk_columns,
        args.relative_bump,
    )
    exact_fd_time = perf_counter() - exact_fd_start
    exact_fd_metrics = component_metrics(analytical, exact_fd)

    payload = {
        "schema_version": "greeks-coordinate-ablation-v9",
        "profile": args.profile,
        "experiment": (
            "grid-only coordinate ablation before TT compression; exact "
            "European geometric-basket oracle"
        ),
        "design_principle": (
            "all candidates have identical physical mode sizes, spot/rate/"
            "maturity axes, validation points, interpolation order and bump"
        ),
        "settings": {
            "physical_shape": list(config.physical_shape),
            "qtt_core_count": int(sum(config.qtt_bits)),
            "modes": list(args.modes),
            "risk_columns": list(risk_columns),
            "relative_bump": float(args.relative_bump),
            "grid_batch_size": int(args.grid_batch_size),
            "gamma_monitor_design_maturity_days": MONITOR_MATURITY_DAYS,
            "gamma_monitor_design_rate": 0.03,
            "volatilities": config.volatilities.tolist(),
            "correlation": config.correlation.tolist(),
            "dividends": config.dividends.tolist(),
        },
        "test_design": {
            "market_parameters": points.tolist(),
            "log_moneyness": m_values.tolist(),
            "maturity_days": maturity_days.tolist(),
            "common_grid_nodes": node_design,
        },
        "analytical_reference": {
            "definition": "closed-form fixed-strike spot Greeks",
            "components": components_to_json(analytical),
            "hessian_shape": hessian_shape_diagnostics(
                gamma_matrices(analytical)
            ),
        },
        "exact_price_finite_difference_control": {
            "definition": (
                "same bump stencil applied directly to exact prices; its "
                "error is finite-difference truncation, not interpolation"
            ),
            "wall_time_seconds": exact_fd_time,
            "components": components_to_json(exact_fd),
            "error_against_analytical": exact_fd_metrics,
            "hessian_error_against_analytical": matrix_metrics(
                analytical,
                exact_fd,
            ),
            "hessian_shape": hessian_shape_diagnostics(
                gamma_matrices(exact_fd)
            ),
        },
        "grids": {},
    }

    for mode in args.modes:
        print(f"building {mode}")
        grid, transform, description = build_greek_coordinate_grid(
            config,
            mode,
            monitor_maturities=np.asarray(MONITOR_MATURITY_DAYS) / 365.0,
            monitor_rate=0.03,
        )
        model_oracle = TransformedPricer(market_pricer, transform)
        interpolator = CachedGridInterpolator(
            grid,
            model_oracle,
            evaluation_batch_size=args.grid_batch_size,
        )
        start = perf_counter()
        components = finite_difference_hybrid_component_arrays(
            interpolator,
            transform,
            points,
            risk_columns,
            args.relative_bump,
        )
        coordinate_column = config.n_assets
        components["price"] = np.asarray(
            interpolator(
                transform.to_model(points),
                cubic_columns=(coordinate_column,),
            ),
            dtype=float,
        )
        elapsed = perf_counter() - start
        against_analytical = component_metrics(analytical, components)
        against_exact_fd = component_metrics(exact_fd, components)
        payload["grids"][mode] = {
            "description": description,
            "implied_moneyness_axes_at_test_maturities": implied_moneyness_axes(
                grid,
                transform,
                config,
                node_design,
            ),
            "wall_time_seconds": elapsed,
            "cache": interpolator.diagnostics(),
            "local_resolution": local_resolution(
                grid,
                transform,
                points,
                config,
            ),
            "components": components_to_json(components),
            "pointwise_error_against_analytical": pointwise_errors(
                analytical,
                components,
            ),
            "error_against_analytical": against_analytical,
            "error_against_exact_price_finite_difference": against_exact_fd,
            "conditional_error_against_analytical": conditional_metrics(
                analytical,
                components,
                m_values,
                maturity_days,
            ),
            "hessian_error_against_analytical": matrix_metrics(
                analytical,
                components,
            ),
            "hessian_shape": hessian_shape_diagnostics(
                gamma_matrices(components)
            ),
            "selection_score_normalized_greek_mae_sum": selection_score(
                against_analytical
            ),
        }
        print(
            f"{mode:>18s}  evals={interpolator.function_evaluations:7d}  "
            f"price={against_analytical['price']['normalized_mae']:.3e}  "
            f"delta={against_analytical['delta']['normalized_mae']:.3e}  "
            f"gamma={against_analytical['gamma_diagonal']['normalized_mae']:.3e}  "
            f"cross={against_analytical['cross_gamma']['normalized_mae']:.3e}"
        )

    rankings = {}
    for component in COMPONENTS:
        rankings[component] = sorted(
            args.modes,
            key=lambda mode: (
                payload["grids"][mode]["error_against_analytical"][component][
                    "normalized_mae"
                ]
                if payload["grids"][mode]["error_against_analytical"][component][
                    "normalized_mae"
                ]
                is not None
                else np.inf
            ),
        )
    rankings["joint_greek_score"] = sorted(
        args.modes,
        key=lambda mode: payload["grids"][mode][
            "selection_score_normalized_greek_mae_sum"
        ],
    )
    payload["rankings_best_to_worst"] = rankings

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
