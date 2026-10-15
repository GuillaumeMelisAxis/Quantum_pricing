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
from stngpr.grids import short_maturity_axis
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk_grids import build_greek_coordinate_grid


SPOT_PANELS = {
    "balanced_low": [56.3, 67.7, 79.1, 91.4, 104.8],
    "balanced_mid": [72.6, 84.1, 96.8, 109.7, 123.4],
    "dispersed": [42.7, 64.9, 88.6, 116.3, 141.2],
}

PROFILES = {
    "smoke": {
        "spot_nodes": [16, 32],
        "panel_names": ["balanced_mid"],
        "moneyness": [-0.15, 0.0, 0.15],
        "maturity_days": [14.0, 90.0],
    },
    "intermediate": {
        "spot_nodes": [16, 32, 64],
        "panel_names": ["balanced_low", "balanced_mid"],
        "moneyness": [-0.35, -0.15, -0.05, 0.0, 0.05, 0.15, 0.35],
        "maturity_days": [7.0, 14.0, 30.0, 90.0, 365.0],
    },
    "paper": {
        "spot_nodes": [16, 32, 64, 128],
        "panel_names": list(SPOT_PANELS),
        "moneyness": [-0.50, -0.35, -0.15, -0.05, 0.0, 0.05, 0.15, 0.35, 0.50],
        "maturity_days": [3.0, 7.0, 14.0, 30.0, 90.0, 365.0, 730.0],
    },
}


def _positive_power_of_two(value):
    value = int(value)
    return value > 1 and value & (value - 1) == 0


def _nearest_unique(axis, targets):
    selected = []
    for target in np.asarray(targets, dtype=float):
        order = np.argsort(np.abs(axis - target))
        selected.append(next(int(index) for index in order if int(index) not in selected))
    return axis[np.sort(selected)]


def fixed_validation_panel(config, profile):
    """Return market points independent of every candidate spot grid.

    None of the declared spot values is selected from a candidate axis. This
    avoids giving the 32-node baseline an exact-node advantage.
    """
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
        spots = np.asarray(SPOT_PANELS[panel_name], dtype=float)
        basket = float(np.exp(np.mean(np.log(spots))))
        for maturity in maturities:
            for m in profile["moneyness"]:
                strike = basket * np.exp(float(m))
                point = np.concatenate((spots, [strike, rate, maturity]))
                for column, (lower, upper) in enumerate(config.bounds):
                    if not lower < point[column] < upper:
                        raise RuntimeError("the fixed validation panel left the market domain")
                points.append(point)
                m_values.append(float(m))
                t_days.append(float(maturity * 365.0))
                panel_labels.append(panel_name)
    return (
        np.asarray(points, dtype=float),
        np.asarray(m_values, dtype=float),
        np.asarray(t_days, dtype=float),
        np.asarray(panel_labels),
        {
            "spot_panels": {
                name: SPOT_PANELS[name] for name in profile["panel_names"]
            },
            "rate": rate,
            "maturity_days": (maturities * 365.0).tolist(),
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


def matrix_metrics(reference, estimate):
    truth = gamma_matrices(reference)
    value = gamma_matrices(estimate)
    error = np.linalg.norm(value - truth, axis=(1, 2))
    scale = np.linalg.norm(truth, axis=(1, 2))
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


def masked_metrics(reference, estimate, mask):
    reduced_reference = {
        name: np.asarray(reference[name], dtype=float)[mask] for name in COMPONENTS
    }
    reduced_estimate = {
        name: np.asarray(estimate[name], dtype=float)[mask] for name in COMPONENTS
    }
    return component_metrics(reduced_reference, reduced_estimate)


def regional_metrics(reference, estimate, m_values, maturity_days, panel_labels):
    masks = {
        "risk_core_abs_m_le_0p15": np.abs(m_values) <= 0.15 + 1e-14,
        "short_T_le_30d": maturity_days <= 30.0,
        "short_risk_core": (
            (maturity_days <= 30.0) & (np.abs(m_values) <= 0.15 + 1e-14)
        ),
        "put_deep_itm_m_ge_0p35": m_values >= 0.35 - 1e-14,
        "put_deep_otm_m_le_minus_0p35": m_values <= -0.35 + 1e-14,
    }
    output = {
        name: masked_metrics(reference, estimate, mask)
        for name, mask in masks.items()
        if np.any(mask)
    }
    output["by_spot_panel"] = {
        str(panel): masked_metrics(
            reference,
            estimate,
            panel_labels == panel,
        )
        for panel in np.unique(panel_labels)
    }
    return output


def spot_resolution_diagnostics(grid, points, n_assets):
    widths, relative_widths = [], []
    per_axis = {}
    for column in range(n_assets):
        axis = grid.axes[column]
        values = points[:, column]
        upper = np.searchsorted(axis, values, side="right")
        upper = np.clip(upper, 1, axis.size - 1)
        lower = upper - 1
        local = axis[upper] - axis[lower]
        relative = local / values
        widths.extend(local.tolist())
        relative_widths.extend(relative.tolist())
        per_axis[str(column)] = {
            "mean_absolute_cell_width": float(np.mean(local)),
            "mean_relative_cell_width": float(np.mean(relative)),
        }

    def summary(values):
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    return {
        "absolute_spot_cell_width": summary(widths),
        "relative_spot_cell_width": summary(relative_widths),
        "by_spot_axis": per_axis,
    }


def pointwise_errors(reference, estimate):
    return {
        name: (
            np.asarray(estimate[name], dtype=float)
            - np.asarray(reference[name], dtype=float)
        ).tolist()
        for name in COMPONENTS
    }


def convergence_payload(results, spot_nodes):
    output = {}
    for component in COMPONENTS:
        records = []
        for left, right in zip(spot_nodes[:-1], spot_nodes[1:]):
            left_error = results[str(left)]["error_against_analytical"][component][
                "normalized_mae"
            ]
            right_error = results[str(right)]["error_against_analytical"][component][
                "normalized_mae"
            ]
            ratio = float(left_error / right_error) if right_error > 0.0 else None
            order = (
                float(np.log(left_error / right_error) / np.log(right / left))
                if left_error > 0.0 and right_error > 0.0
                else None
            )
            records.append({
                "from_spot_nodes": int(left),
                "to_spot_nodes": int(right),
                "error_reduction_factor": ratio,
                "observed_order": order,
            })
        output[component] = records
    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the remaining standardized-risk diagonal-Gamma "
            "error is a spot-grid resolution floor."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--spot-nodes", nargs="+", type=int, default=None)
    parser.add_argument("--moneyness-nodes", type=int, default=512)
    parser.add_argument("--maturity-nodes", type=int, default=64)
    parser.add_argument("--risk-columns", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--relative-bump", type=float, default=0.002)
    parser.add_argument("--grid-batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v91_spot_resolution_ablation.json"),
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    spot_nodes = sorted(set(args.spot_nodes or profile["spot_nodes"]))
    if len(spot_nodes) < 2 or any(not _positive_power_of_two(n) for n in spot_nodes):
        parser.error("provide at least two distinct power-of-two spot resolutions")
    if not _positive_power_of_two(args.moneyness_nodes):
        parser.error("moneyness-nodes must be a power of two")
    if not _positive_power_of_two(args.maturity_nodes):
        parser.error("maturity-nodes must be a power of two")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative-bump must lie in (0,1)")

    base = PaperConfig()
    risk_columns = tuple(args.risk_columns)
    if (
        len(risk_columns) < 2
        or len(set(risk_columns)) != len(risk_columns)
        or any(column < 0 or column >= base.n_assets for column in risk_columns)
    ):
        parser.error("risk-columns must contain at least two unique spot columns")

    reference_shape = list(base.physical_shape)
    reference_shape[base.n_assets] = int(args.moneyness_nodes)
    reference_shape[-1] = int(args.maturity_nodes)
    reference_config = replace(base, physical_shape=tuple(reference_shape))
    points, m_values, maturity_days, panel_labels, design = fixed_validation_panel(
        reference_config,
        profile,
    )
    market_pricer = EuropeanGeometricBasketPricer(reference_config)
    analytical = analytical_components(
        reference_config,
        market_pricer,
        points,
        risk_columns,
    )
    exact_start = perf_counter()
    exact_fd = finite_difference_component_arrays(
        market_pricer,
        points,
        risk_columns,
        args.relative_bump,
    )
    exact_time = perf_counter() - exact_start

    payload = {
        "schema_version": "greeks-standardized-spot-ablation-v9.1",
        "profile": args.profile,
        "hypothesis": (
            "after standardizing the moneyness coordinate, the remaining "
            "diagonal-Gamma error is limited by the five spot axes"
        ),
        "controlled_design": (
            "only the five equal-size spot axes change; validation market "
            "points are fixed and off-grid by construction"
        ),
        "settings": {
            "spot_nodes": spot_nodes,
            "moneyness_nodes": int(args.moneyness_nodes),
            "maturity_nodes": int(args.maturity_nodes),
            "rate_nodes": int(base.physical_shape[-2]),
            "risk_columns": list(risk_columns),
            "relative_bump": float(args.relative_bump),
            "grid_batch_size": int(args.grid_batch_size),
            "coordinate_mode": "standardized_risk",
        },
        "test_design": {
            "market_parameters": points.tolist(),
            "log_moneyness": m_values.tolist(),
            "maturity_days": maturity_days.tolist(),
            "spot_panel_labels": panel_labels.tolist(),
            **design,
        },
        "analytical_reference": {
            "definition": "closed-form fixed-strike geometric-basket spot Greeks",
            "components": components_to_json(analytical),
            "hessian_shape": hessian_shape_diagnostics(gamma_matrices(analytical)),
        },
        "exact_price_finite_difference_control": {
            "wall_time_seconds": exact_time,
            "components": components_to_json(exact_fd),
            "error_against_analytical": component_metrics(analytical, exact_fd),
            "hessian_error_against_analytical": matrix_metrics(analytical, exact_fd),
            "hessian_shape": hessian_shape_diagnostics(gamma_matrices(exact_fd)),
        },
        "spot_grids": {},
    }

    for n_spot in spot_nodes:
        shape = list(reference_config.physical_shape)
        shape[: base.n_assets] = [int(n_spot)] * base.n_assets
        config = replace(reference_config, physical_shape=tuple(shape))
        grid, transform, description = build_greek_coordinate_grid(
            config,
            "standardized_risk",
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
        components["price"] = np.asarray(
            interpolator(
                transform.to_model(points),
                cubic_columns=(config.n_assets,),
            ),
            dtype=float,
        )
        elapsed = perf_counter() - start
        metrics = component_metrics(analytical, components)
        physical_entries = int(np.prod(np.asarray(config.physical_shape, dtype=object)))
        payload["spot_grids"][str(n_spot)] = {
            "physical_shape": list(config.physical_shape),
            "physical_tensor_entries": str(physical_entries),
            "qtt_core_count": int(sum(config.qtt_bits)),
            "coordinate_description": description,
            "wall_time_seconds": elapsed,
            "cache": interpolator.diagnostics(),
            "spot_resolution": spot_resolution_diagnostics(
                grid,
                points,
                config.n_assets,
            ),
            "components": components_to_json(components),
            "pointwise_error_against_analytical": pointwise_errors(
                analytical,
                components,
            ),
            "error_against_analytical": metrics,
            "error_against_exact_price_finite_difference": component_metrics(
                exact_fd,
                components,
            ),
            "regional_error_against_analytical": regional_metrics(
                analytical,
                components,
                m_values,
                maturity_days,
                panel_labels,
            ),
            "hessian_error_against_analytical": matrix_metrics(
                analytical,
                components,
            ),
            "hessian_shape": hessian_shape_diagnostics(
                gamma_matrices(components)
            ),
        }
        print(
            f"spots={n_spot:3d} cores={sum(config.qtt_bits):2d} "
            f"evals={interpolator.function_evaluations:7d} "
            f"price={100.0 * metrics['price']['normalized_mae']:.4g}% "
            f"delta={100.0 * metrics['delta']['normalized_mae']:.4g}% "
            f"gamma={100.0 * metrics['gamma_diagonal']['normalized_mae']:.4g}% "
            f"cross={100.0 * metrics['cross_gamma']['normalized_mae']:.4g}%"
        )

    payload["convergence"] = convergence_payload(
        payload["spot_grids"],
        spot_nodes,
    )
    payload["ranking_by_diagonal_gamma"] = sorted(
        spot_nodes,
        key=lambda n: payload["spot_grids"][str(n)]["error_against_analytical"][
            "gamma_diagonal"
        ]["normalized_mae"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
