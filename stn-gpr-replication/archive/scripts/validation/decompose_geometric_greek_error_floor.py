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
    component_error_decomposition,
    component_metrics,
    component_residual_metrics,
    components_to_json,
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
from stngpr.grids import short_maturity_axis
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk_grids import build_greek_coordinate_grid

INTERPOLATION_MODES = ("multilinear", "risk_hybrid_cubic")

# These panels are intentionally different from the three panels used to
# select N_S in v9.1.  Their values are also off every candidate grid by
# construction, so the experiment cannot reward a resolution through exact
# node coincidence.
OUT_OF_SAMPLE_SPOT_PANELS = {
    "oos_low": [47.3, 63.8, 82.1, 101.6, 127.9],
    "oos_mid": [61.7, 78.9, 97.2, 118.6, 143.1],
    "oos_dispersed": [34.7, 57.9, 86.2, 121.3, 146.4],
}

PROFILES = {
    "smoke": {
        "panel_names": ["oos_mid"],
        "moneyness": [-0.05, 0.0, 0.05],
        "maturity_days": [7.0, 30.0],
        "relative_bumps": [0.001, 0.002, 0.005],
    },
    "intermediate": {
        "panel_names": ["oos_low", "oos_mid"],
        "moneyness": [-0.15, -0.05, 0.0, 0.05, 0.15],
        "maturity_days": [3.0, 7.0, 14.0, 30.0, 90.0],
        "relative_bumps": [0.0005, 0.001, 0.002, 0.005],
    },
    "paper": {
        "panel_names": list(OUT_OF_SAMPLE_SPOT_PANELS),
        "moneyness": [-0.35, -0.15, -0.05, 0.0, 0.05, 0.15, 0.35],
        "maturity_days": [3.0, 7.0, 14.0, 30.0, 90.0, 365.0],
        "relative_bumps": [0.0005, 0.001, 0.002, 0.005, 0.01],
    },
}


def _positive_power_of_two(value):
    value = int(value)
    return value > 1 and value & (value - 1) == 0


def _bump_key(value):
    return f"{float(value):.10g}"


def _nearest_unique(axis, targets):
    selected = []
    for target in np.asarray(targets, dtype=float):
        order = np.argsort(np.abs(axis - target))
        selected.append(next(int(index) for index in order if int(index) not in selected))
    return axis[np.sort(selected)]


def fixed_out_of_sample_panel(config, profile, maximum_relative_bump):
    rate_axis = np.linspace(*config.rate_bounds, config.physical_shape[-2])
    rate = float(rate_axis[np.argmin(np.abs(rate_axis - 0.03))])
    maturity_axis = short_maturity_axis(
        *config.maturity_bounds,
        config.physical_shape[-1],
        power=2.0,
    )
    maturities = _nearest_unique(
        maturity_axis,
        np.asarray(profile["maturity_days"], dtype=float) / 365.0,
    )

    points, m_values, t_days, panel_labels = [], [], [], []
    for panel_name in profile["panel_names"]:
        spots = np.asarray(OUT_OF_SAMPLE_SPOT_PANELS[panel_name], dtype=float)
        basket = float(np.exp(np.mean(np.log(spots))))
        for maturity in maturities:
            for moneyness in profile["moneyness"]:
                strike = basket * np.exp(float(moneyness))
                point = np.concatenate((spots, [strike, rate, maturity]))
                for column, (lower, upper) in enumerate(config.bounds):
                    if not lower < point[column] < upper:
                        raise RuntimeError("the validation panel left the market domain")
                for column in range(config.n_assets):
                    lower, upper = config.bounds[column]
                    bump = maximum_relative_bump * point[column]
                    if not lower < point[column] - bump < point[column] + bump < upper:
                        raise RuntimeError("a spot bump left the market domain")
                points.append(point)
                m_values.append(float(moneyness))
                t_days.append(float(maturity * 365.0))
                panel_labels.append(panel_name)

    return (
        np.asarray(points, dtype=float),
        np.asarray(m_values, dtype=float),
        np.asarray(t_days, dtype=float),
        np.asarray(panel_labels),
        {
            "spot_panels": {
                name: OUT_OF_SAMPLE_SPOT_PANELS[name]
                for name in profile["panel_names"]
            },
            "rate": rate,
            "maturity_days_requested": list(profile["maturity_days"]),
            "maturity_days_realized": (maturities * 365.0).tolist(),
        },
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


def _masked_components(components, mask):
    return {
        name: np.asarray(components[name], dtype=float)[mask]
        for name in COMPONENTS
    }


def conditional_layer_metrics(
    analytical,
    decomposition,
    m_values,
    maturity_days,
    panel_labels,
):
    masks = {
        "atm_abs_m_le_0p05": np.abs(m_values) <= 0.05 + 1e-14,
        "risk_core_abs_m_le_0p15": np.abs(m_values) <= 0.15 + 1e-14,
        "short_T_le_30d": maturity_days <= 30.0 + 1e-12,
        "short_risk_core": (
            (maturity_days <= 30.0 + 1e-12)
            & (np.abs(m_values) <= 0.15 + 1e-14)
        ),
    }
    for days in np.unique(maturity_days):
        masks[f"maturity_{days:.6g}d"] = np.isclose(maturity_days, days)
    for panel in np.unique(panel_labels):
        masks[f"panel_{panel}"] = panel_labels == panel

    output = {}
    for region, mask in masks.items():
        if not np.any(mask):
            continue
        reference = _masked_components(analytical, mask)
        output[region] = {
            layer: component_residual_metrics(
                reference,
                _masked_components(record["residuals"], mask),
            )
            for layer, record in decomposition.items()
        }
    return output


def _json_decomposition(decomposition):
    return {
        layer: {
            "metrics": record["metrics"],
            "residuals": components_to_json(record["residuals"]),
        }
        for layer, record in decomposition.items()
    }


def _evaluate_grid_components(
    mode,
    interpolator,
    transform,
    points,
    risk_columns,
    relative_bump,
):
    if mode == "multilinear":
        market_pricer = MarketCoordinatePricer(interpolator, transform)
        return finite_difference_component_arrays(
            market_pricer,
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


def _best_bumps(interpolation_results):
    output = {}
    for component in COMPONENTS:
        ranking = sorted(
            interpolation_results,
            key=lambda key: interpolation_results[key]["decomposition"]["total"][
                "metrics"
            ][component]["normalized_mae"],
        )
        output[component] = [float(value) for value in ranking]
    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the v9.1 standardized-risk Greek error floor into "
            "finite-difference and grid-interpolation layers."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--spot-nodes", type=int, default=64)
    parser.add_argument("--moneyness-nodes", type=int, default=512)
    parser.add_argument("--maturity-nodes", type=int, default=64)
    parser.add_argument("--risk-columns", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--relative-bumps", nargs="+", type=float, default=None)
    parser.add_argument(
        "--interpolation-modes",
        nargs="+",
        choices=INTERPOLATION_MODES,
        default=list(INTERPOLATION_MODES),
    )
    parser.add_argument("--grid-batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v92_error_floor_decomposition.json"),
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    bumps = sorted(set(args.relative_bumps or profile["relative_bumps"]))
    modes = list(dict.fromkeys(args.interpolation_modes))
    for name, value in (
        ("spot-nodes", args.spot_nodes),
        ("moneyness-nodes", args.moneyness_nodes),
        ("maturity-nodes", args.maturity_nodes),
    ):
        if not _positive_power_of_two(value):
            parser.error(f"{name} must be a power of two greater than one")
    if not bumps or any(not 0.0 < bump < 1.0 for bump in bumps):
        parser.error("relative-bumps must lie in (0,1)")
    if args.grid_batch_size <= 0:
        parser.error("grid-batch-size must be positive")

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
    shape[base.n_assets] = int(args.moneyness_nodes)
    shape[-1] = int(args.maturity_nodes)
    config = replace(base, physical_shape=tuple(shape))
    points, m_values, maturity_days, panel_labels, design = (
        fixed_out_of_sample_panel(config, profile, max(bumps))
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
        "schema_version": "greeks-error-floor-decomposition-v9.2",
        "profile": args.profile,
        "hypothesis": (
            "the v9.1 Hessian plateau is a coupled finite-difference and "
            "grid-interpolation floor, not a TT-cross limitation"
        ),
        "controlled_design": (
            "N_S=64 and standardized-risk coordinates are frozen; bump size "
            "and interpolation order are ablated on spot panels not used in v9.1"
        ),
        "additive_identity": (
            "grid_FD - analytical = (exact_price_FD - analytical) + "
            "(grid_FD - exact_price_FD)"
        ),
        "settings": {
            "physical_shape": list(config.physical_shape),
            "qtt_core_count": int(sum(config.qtt_bits)),
            "spot_nodes": int(args.spot_nodes),
            "moneyness_nodes": int(args.moneyness_nodes),
            "rate_nodes": int(config.physical_shape[-2]),
            "maturity_nodes": int(args.maturity_nodes),
            "risk_columns": list(risk_columns),
            "relative_bumps": bumps,
            "baseline_relative_bump": min(bumps, key=lambda value: abs(value - 0.002)),
            "interpolation_modes": modes,
            "coordinate_mode": "standardized_risk",
            "grid_batch_size": int(args.grid_batch_size),
            "tt_compression": False,
            "monte_carlo": False,
        },
        "test_design": {
            "market_parameters": points.tolist(),
            "log_moneyness": m_values.tolist(),
            "maturity_days": maturity_days.tolist(),
            "spot_panel_labels": panel_labels.tolist(),
            "independent_of_v91_spot_panels": True,
            **design,
        },
        "coordinate_description": coordinate_description,
        "analytical_reference": {
            "definition": "closed-form fixed-strike geometric-basket spot Greeks",
            "components": components_to_json(analytical),
            "hessian_shape": hessian_shape_diagnostics(gamma_matrices(analytical)),
        },
        "exact_price_finite_difference_controls": {},
        "interpolation_results": {},
    }

    exact_components = {}
    for bump in bumps:
        start = perf_counter()
        exact_fd = finite_difference_component_arrays(
            market_pricer,
            points,
            risk_columns,
            bump,
        )
        elapsed = perf_counter() - start
        exact_components[_bump_key(bump)] = exact_fd
        residual = {
            name: (
                np.asarray(exact_fd[name], dtype=float)
                - np.asarray(analytical[name], dtype=float)
            )
            for name in COMPONENTS
        }
        payload["exact_price_finite_difference_controls"][_bump_key(bump)] = {
            "relative_bump": float(bump),
            "wall_time_seconds": elapsed,
            "components": components_to_json(exact_fd),
            "error_against_analytical": component_metrics(analytical, exact_fd),
            "residual_metrics_on_analytical_scale": component_residual_metrics(
                analytical,
                residual,
            ),
            "hessian_shape": hessian_shape_diagnostics(gamma_matrices(exact_fd)),
        }

    for mode in modes:
        interpolator = CachedGridInterpolator(
            grid,
            model_oracle,
            evaluation_batch_size=args.grid_batch_size,
        )
        method_results = {}
        method_start = perf_counter()
        for bump in bumps:
            key = _bump_key(bump)
            start = perf_counter()
            grid_components = _evaluate_grid_components(
                mode,
                interpolator,
                transform,
                points,
                risk_columns,
                bump,
            )
            elapsed = perf_counter() - start
            decomposition = component_error_decomposition(
                analytical,
                exact_components[key],
                grid_components,
            )
            closure_max = max(
                float(np.max(np.abs(decomposition["closure"]["residuals"][name])))
                for name in COMPONENTS
            )
            method_results[key] = {
                "relative_bump": float(bump),
                "incremental_wall_time_seconds": elapsed,
                "components": components_to_json(grid_components),
                "error_against_analytical": component_metrics(
                    analytical,
                    grid_components,
                ),
                "error_against_exact_price_finite_difference": component_metrics(
                    exact_components[key],
                    grid_components,
                ),
                "decomposition": _json_decomposition(decomposition),
                "conditional_decomposition_metrics": conditional_layer_metrics(
                    analytical,
                    decomposition,
                    m_values,
                    maturity_days,
                    panel_labels,
                ),
                "maximum_absolute_closure_residual": closure_max,
                "hessian_shape": hessian_shape_diagnostics(
                    gamma_matrices(grid_components)
                ),
            }
            metrics = method_results[key]["decomposition"]["total"]["metrics"]
            print(
                f"method={mode:19s} bump={100.0 * bump:6.3f}% "
                f"delta={100.0 * metrics['delta']['normalized_mae']:.4g}% "
                f"gamma={100.0 * metrics['gamma_diagonal']['normalized_mae']:.4g}% "
                f"cross={100.0 * metrics['cross_gamma']['normalized_mae']:.4g}% "
                f"closure={closure_max:.2e}"
            )

        payload["interpolation_results"][mode] = {
            "wall_time_seconds": perf_counter() - method_start,
            "cache": interpolator.diagnostics(),
            "bump_results": method_results,
            "ranking_by_component": _best_bumps(method_results),
        }

    payload["paper_gate_summary"] = {
        mode: {
            component: {
                "best_relative_bump": records["ranking_by_component"][component][0],
                "best_normalized_mae": records["bump_results"][
                    _bump_key(records["ranking_by_component"][component][0])
                ]["decomposition"]["total"]["metrics"][component]["normalized_mae"],
            }
            for component in COMPONENTS
        }
        for mode, records in payload["interpolation_results"].items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
