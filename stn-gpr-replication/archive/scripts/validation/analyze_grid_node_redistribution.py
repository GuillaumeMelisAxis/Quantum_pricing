from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.special import ndtr
from scipy.stats import spearmanr

from stngpr.config import PaperConfig
from stngpr.diagnostics import (
    geometric_basket_convexity_ridge,
    geometric_basket_effective_parameters,
    geometric_basket_log_moneyness_convexity,
    geometric_basket_normalized_put,
)
from stngpr.grid_geometry import (
    interval_fill_distance,
    local_coverage_metrics,
    local_cubic_lagrange,
    normalized_node_displacement,
    physical_log_moneyness_nodes,
    weighted_fill_proxy,
)
from stngpr.risk_grids import build_greek_coordinate_grid


PROFILES = {
    "smoke": {
        "shape": (16, 16, 16, 16, 16, 64, 8, 16),
        "rates": [0.03],
        "maturity_days": [7, 30, 90],
        "dense_points": 4097,
        "profile_points": 401,
    },
    "intermediate": {
        "shape": (32, 32, 32, 32, 32, 256, 8, 32),
        "rates": [0.005, 0.03, 0.08],
        "maturity_days": [3, 7, 14, 30, 90, 365],
        "dense_points": 8193,
        "profile_points": 801,
    },
    "paper": {
        "shape": (64, 64, 64, 64, 64, 512, 8, 64),
        "rates": np.linspace(0.005, 0.08, 8).tolist(),
        "maturity_days": [1, 3, 7, 14, 30, 60, 90, 180, 365, 730, 1095],
        "dense_points": 32769,
        "profile_points": 1201,
    },
}

GRID_MODES = {
    "pricing_grid": "price_adaptive",
    "risk_hybrid_grid": "bounded_standardized_risk",
}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summary(values) -> dict:
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return {}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "standard_deviation": float(np.std(x)),
        "minimum": float(np.min(x)),
        "q05": float(np.quantile(x, 0.05)),
        "median": float(np.median(x)),
        "q95": float(np.quantile(x, 0.95)),
        "maximum": float(np.max(x)),
    }


def _error_summary(reference, estimate, mask=None) -> dict:
    truth = np.asarray(reference, dtype=float)
    value = np.asarray(estimate, dtype=float)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        truth = truth[selected]
        value = value[selected]
    error = value - truth
    absolute = np.abs(error)
    scale = max(float(np.mean(np.abs(truth))), np.finfo(float).tiny)
    return {
        "point_count": int(truth.size),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "maximum_absolute_error": float(np.max(absolute)),
        "normalized_mae": float(np.mean(absolute) / scale),
        "normalization": "mean absolute analytical component",
    }


def _analytical_curve(m, maturity, rate, basket_sigma, basket_carry):
    m = np.asarray(m, dtype=float)
    value = geometric_basket_normalized_put(
        m,
        maturity,
        rate,
        basket_sigma,
        basket_carry,
    )
    std = basket_sigma * np.sqrt(float(maturity))
    d1 = (-m + (basket_carry + 0.5 * basket_sigma**2) * maturity) / std
    d2 = d1 - std
    first = np.exp(m - rate * maturity) * ndtr(-d2)
    convexity = geometric_basket_log_moneyness_convexity(
        m,
        maturity,
        rate,
        basket_sigma,
        basket_carry,
    )
    second = first + convexity
    return value, first, second, convexity


def _fourth_derivative_monitor(values, spacing):
    n = len(values)
    window = min(41, n - (1 - n % 2))
    if window < 9:
        raise ValueError("the dense audit needs at least nine points")
    fourth = savgol_filter(
        values,
        window_length=window,
        polyorder=6,
        deriv=4,
        delta=float(spacing),
        mode="interp",
    )
    absolute = np.abs(fourth)
    scale = max(float(np.quantile(absolute, 0.995)), np.finfo(float).tiny)
    return fourth, np.minimum(absolute / scale, 1.0), scale, window


def _scenario(
    config,
    grids,
    rate,
    maturity_days,
    kappas,
    dense_points,
):
    maturity = float(maturity_days) / 365.0
    basket_sigma, basket_carry = geometric_basket_effective_parameters(
        float(rate),
        config.volatilities,
        config.correlation,
        config.dividends,
    )
    ridge = float(
        geometric_basket_convexity_ridge(
            maturity,
            basket_sigma,
            basket_carry,
        )
    )
    scale = float(basket_sigma * np.sqrt(maturity))
    nodes = {
        name: physical_log_moneyness_nodes(
            grid,
            transform,
            config.n_assets,
            rate,
            maturity,
        )
        for name, (grid, transform, _) in grids.items()
    }
    domain = (float(nodes["pricing_grid"][0]), float(nodes["pricing_grid"][-1]))
    if not np.allclose(
        nodes["risk_hybrid_grid"][[0, -1]],
        np.asarray(domain),
        rtol=0.0,
        atol=5e-13,
    ):
        raise RuntimeError("the compared grids do not share the same physical endpoints")
    nodes["risk_hybrid_grid"] = nodes["risk_hybrid_grid"].copy()
    nodes["risk_hybrid_grid"][[0, -1]] = np.asarray(domain)

    dense = np.linspace(*domain, int(dense_points))
    value, first, second, convexity = _analytical_curve(
        dense,
        maturity,
        float(rate),
        basket_sigma,
        basket_carry,
    )
    fourth, fourth_monitor, fourth_scale, smoothing_window = (
        _fourth_derivative_monitor(value, dense[1] - dense[0])
    )
    convexity_scale = max(float(np.max(convexity)), np.finfo(float).tiny)
    convexity_monitor = convexity / convexity_scale

    grid_records = {}
    interpolants = {}
    for name, x in nodes.items():
        node_value, _, _, _ = _analytical_curve(
            x,
            maturity,
            float(rate),
            basket_sigma,
            basket_carry,
        )
        estimate = local_cubic_lagrange(x, node_value, dense)
        interpolants[name] = estimate
        grid_records[name] = {
            "physical_nodes": x.tolist(),
            "global_fill": interval_fill_distance(x, *domain),
            "curvature_weighted_price_proxy": weighted_fill_proxy(
                x,
                dense,
                convexity_monitor,
                distance_power=2,
            ),
            "fourth_derivative_gamma_proxy": weighted_fill_proxy(
                x,
                dense,
                fourth_monitor,
                distance_power=2,
            ),
            "full_domain_interpolation": {
                "u": _error_summary(value, estimate[0]),
                "u_m": _error_summary(first, estimate[1]),
                "u_mm": _error_summary(second, estimate[2]),
                "chi": _error_summary(convexity, estimate[2] - estimate[1]),
            },
            "bands": {},
        }

    for kappa in kappas:
        lower = max(domain[0], ridge - float(kappa) * scale)
        upper = min(domain[1], ridge + float(kappa) * scale)
        mask = (dense >= lower) & (dense <= upper)
        key = f"kappa_{float(kappa):g}"
        for name, x in nodes.items():
            estimate = interpolants[name]
            grid_records[name]["bands"][key] = {
                "kappa": float(kappa),
                "bounds": [float(lower), float(upper)],
                **local_coverage_metrics(x, lower, upper),
                "curvature_weighted_proxy": weighted_fill_proxy(
                    x,
                    dense[mask],
                    convexity_monitor[mask],
                    distance_power=2,
                ),
                "fourth_derivative_gamma_proxy": weighted_fill_proxy(
                    x,
                    dense[mask],
                    fourth_monitor[mask],
                    distance_power=2,
                ),
                "interpolation": {
                    "u": _error_summary(value, estimate[0], mask),
                    "u_m": _error_summary(first, estimate[1], mask),
                    "u_mm": _error_summary(second, estimate[2], mask),
                    "chi": _error_summary(
                        convexity,
                        estimate[2] - estimate[1],
                        mask,
                    ),
                },
            }

    displacement = normalized_node_displacement(
        nodes["pricing_grid"],
        nodes["risk_hybrid_grid"],
        domain,
    )
    return {
        "rate": float(rate),
        "maturity": maturity,
        "maturity_days": float(maturity_days),
        "basket_volatility": basket_sigma,
        "basket_carry": basket_carry,
        "convexity_ridge": ridge,
        "risk_scale": scale,
        "moneyness_domain": list(domain),
        "node_displacement": displacement,
        "analytical_monitor": {
            "convexity_maximum": convexity_scale,
            "fourth_derivative_q995": fourth_scale,
            "savgol_window": int(smoothing_window),
            "fourth_derivative_at_ridge": float(
                fourth[int(np.argmin(np.abs(dense - ridge)))]
            ),
        },
        "grids": grid_records,
    }


def _aggregate(scenarios, kappas):
    grids = list(GRID_MODES)
    summary = {
        "scenario_count": len(scenarios),
        "node_displacement": {
            key: _summary(row["node_displacement"][key] for row in scenarios)
            for key in ("normalized_l1", "normalized_l2", "normalized_linf")
        },
        "grids": {},
        "paired_risk_over_pricing": {},
        "paired_gains": {},
        "evidence_gates": {},
    }
    for name in grids:
        summary["grids"][name] = {
            "global_fill_distance": _summary(
                row["grids"][name]["global_fill"]["fill_distance"]
                for row in scenarios
            ),
            "curvature_weighted_price_proxy": _summary(
                row["grids"][name]["curvature_weighted_price_proxy"]["score"]
                for row in scenarios
            ),
            "fourth_derivative_gamma_proxy": _summary(
                row["grids"][name]["fourth_derivative_gamma_proxy"]["score"]
                for row in scenarios
            ),
            "bands": {},
        }
        for kappa in kappas:
            key = f"kappa_{float(kappa):g}"
            band = {}
            for metric in ("fill_distance", "node_fraction", "maximum_intersecting_gap"):
                band[metric] = _summary(
                    row["grids"][name]["bands"][key][metric]
                    for row in scenarios
                )
            for metric in ("curvature_weighted_proxy", "fourth_derivative_gamma_proxy"):
                band[metric] = _summary(
                    row["grids"][name]["bands"][key][metric]["score"]
                    for row in scenarios
                )
            for component in ("u", "u_m", "u_mm", "chi"):
                band[f"{component}_normalized_mae"] = _summary(
                    row["grids"][name]["bands"][key]["interpolation"][component][
                        "normalized_mae"
                    ]
                    for row in scenarios
                )
            summary["grids"][name]["bands"][key] = band

    for kappa in kappas:
        key = f"kappa_{float(kappa):g}"
        ratios = {}
        gains = {}
        for metric in ("fill_distance", "node_fraction", "maximum_intersecting_gap"):
            ratios[metric] = _summary(
                row["grids"]["risk_hybrid_grid"]["bands"][key][metric]
                / max(
                    row["grids"]["pricing_grid"]["bands"][key][metric],
                    np.finfo(float).tiny,
                )
                for row in scenarios
            )
        gains["coverage_gain"] = _summary(
            row["grids"]["pricing_grid"]["bands"][key]["fill_distance"]
            / max(
                row["grids"]["risk_hybrid_grid"]["bands"][key]["fill_distance"],
                np.finfo(float).tiny,
            )
            for row in scenarios
        )
        gains["node_concentration_gain"] = _summary(
            row["grids"]["risk_hybrid_grid"]["bands"][key]["node_fraction"]
            / max(
                row["grids"]["pricing_grid"]["bands"][key]["node_fraction"],
                np.finfo(float).tiny,
            )
            for row in scenarios
        )
        gains["maximum_gap_reduction_factor"] = _summary(
            row["grids"]["pricing_grid"]["bands"][key][
                "maximum_intersecting_gap"
            ]
            / max(
                row["grids"]["risk_hybrid_grid"]["bands"][key][
                    "maximum_intersecting_gap"
                ],
                np.finfo(float).tiny,
            )
            for row in scenarios
        )
        for metric in ("curvature_weighted_proxy", "fourth_derivative_gamma_proxy"):
            ratios[metric] = _summary(
                row["grids"]["risk_hybrid_grid"]["bands"][key][metric]["score"]
                / max(
                    row["grids"]["pricing_grid"]["bands"][key][metric]["score"],
                    np.finfo(float).tiny,
                )
                for row in scenarios
            )
            gains[f"{metric}_reduction_factor"] = _summary(
                row["grids"]["pricing_grid"]["bands"][key][metric]["score"]
                / max(
                    row["grids"]["risk_hybrid_grid"]["bands"][key][metric][
                        "score"
                    ],
                    np.finfo(float).tiny,
                )
                for row in scenarios
            )
        for component in ("u", "u_m", "u_mm", "chi"):
            ratios[f"{component}_normalized_mae"] = _summary(
                row["grids"]["risk_hybrid_grid"]["bands"][key]["interpolation"][
                    component
                ]["normalized_mae"]
                / max(
                    row["grids"]["pricing_grid"]["bands"][key]["interpolation"][
                        component
                    ]["normalized_mae"],
                    np.finfo(float).tiny,
                )
                for row in scenarios
            )
            gains[f"{component}_error_reduction_factor"] = _summary(
                row["grids"]["pricing_grid"]["bands"][key]["interpolation"][
                    component
                ]["normalized_mae"]
                / max(
                    row["grids"]["risk_hybrid_grid"]["bands"][key][
                        "interpolation"
                    ][component]["normalized_mae"],
                    np.finfo(float).tiny,
                )
                for row in scenarios
            )
        summary["paired_risk_over_pricing"][key] = ratios
        summary["paired_gains"][key] = gains

    observations = []
    for row in scenarios:
        pricing = row["grids"]["pricing_grid"]["bands"]["kappa_2"]
        risk = row["grids"]["risk_hybrid_grid"]["bands"]["kappa_2"]
        observations.append(
            (
                pricing["fill_distance"]
                / max(risk["fill_distance"], np.finfo(float).tiny),
                pricing["interpolation"]["chi"]["normalized_mae"]
                / max(
                    risk["interpolation"]["chi"]["normalized_mae"],
                    np.finfo(float).tiny,
                ),
            )
        )
    coverage_gain, curvature_error_reduction = np.asarray(
        observations, dtype=float
    ).T
    correlation = spearmanr(coverage_gain, curvature_error_reduction)
    summary["paired_gain_association"] = {
        "band": "kappa_2",
        "spearman_correlation": float(correlation.statistic),
        "p_value": float(correlation.pvalue),
        "observation_count": int(len(observations)),
        "interpretation": (
            "descriptive association between the within-scenario coverage gain "
            "h_pricing/h_risk and curvature-error reduction "
            "E_chi_pricing/E_chi_risk; both exceed one when risk-hybrid improves"
        ),
    }
    gains = summary["paired_gains"]["kappa_2"]
    summary["evidence_gates"] = {
        "physical_nodes_differ": bool(
            summary["node_displacement"]["normalized_l2"]["minimum"] > 0.0
        ),
        "risk_grid_reduces_median_local_fill": bool(
            gains["coverage_gain"]["median"] > 1.0
        ),
        "risk_grid_increases_median_local_node_share": bool(
            gains["node_concentration_gain"]["median"] > 1.0
        ),
        "risk_grid_reduces_median_chi_error": bool(
            gains["chi_error_reduction_factor"]["median"] > 1.0
        ),
    }
    return summary


def _representative_profiles(scenarios, target_rate=0.03, target_days=(7, 30, 90)):
    selected = []
    for maturity_days in target_days:
        row = min(
            scenarios,
            key=lambda item: (
                abs(item["maturity_days"] - maturity_days),
                abs(item["rate"] - target_rate),
            ),
        )
        lower = max(-0.35, row["moneyness_domain"][0])
        upper = min(0.35, row["moneyness_domain"][1])
        selected.append(
            {
                "rate": row["rate"],
                "maturity_days": row["maturity_days"],
                "zoom_bounds": [lower, upper],
                "basket_volatility": row["basket_volatility"],
                "basket_carry": row["basket_carry"],
                "convexity_ridge": row["convexity_ridge"],
                "risk_scale": row["risk_scale"],
                "grids": {
                    name: {
                        "physical_nodes": row["grids"][name]["physical_nodes"],
                        "zoom_node_count": int(
                            sum(
                                lower <= value <= upper
                                for value in row["grids"][name]["physical_nodes"]
                            )
                        ),
                    }
                    for name in GRID_MODES
                },
            }
        )
    return selected


def main():
    parser = argparse.ArgumentParser(
        description=(
            "v9.9.1 matched-budget audit of the physical node redistribution, "
            "local covering radius and cubic derivative interpolation proxies."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--rates", nargs="+", type=float, default=None)
    parser.add_argument("--maturity-days", nargs="+", type=float, default=None)
    parser.add_argument("--kappas", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--dense-points", type=int, default=None)
    parser.add_argument("--moneyness-nodes", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v99_grid_geometry.json"),
    )
    args = parser.parse_args()

    design = PROFILES[args.profile]
    rates = [float(value) for value in (args.rates or design["rates"])]
    maturity_days = [
        float(value) for value in (args.maturity_days or design["maturity_days"])
    ]
    kappas = sorted(set(float(value) for value in args.kappas))
    dense_points = int(args.dense_points or design["dense_points"])
    shape = list(design["shape"])
    if args.moneyness_nodes is not None:
        shape[5] = int(args.moneyness_nodes)
    if any(value < PaperConfig().rate_bounds[0] or value > PaperConfig().rate_bounds[1] for value in rates):
        parser.error("rates must lie inside the paper domain")
    if any(value <= 0.0 or value > 365.0 * PaperConfig().maturity_bounds[1] for value in maturity_days):
        parser.error("maturity-days must lie inside the paper domain")
    if not kappas or any(value <= 0.0 for value in kappas):
        parser.error("kappas must be positive")
    if dense_points < 1025 or dense_points % 2 == 0:
        parser.error("dense-points must be an odd integer at least 1025")
    if any(value <= 1 or value & (value - 1) for value in shape):
        parser.error("every physical mode size must be a power of two")

    config = replace(PaperConfig(), physical_shape=tuple(shape))
    grids = {
        name: build_greek_coordinate_grid(config, mode)
        for name, mode in GRID_MODES.items()
    }
    scenarios = []
    for rate in rates:
        for days in maturity_days:
            print(f"rate={rate:.6f}, T={days:g}d")
            scenarios.append(
                _scenario(
                    config,
                    grids,
                    rate,
                    days,
                    kappas,
                    dense_points,
                )
            )

    payload = {
        "schema_version": "greeks-grid-geometry-v9.9.1",
        "profile": args.profile,
        "benchmark": "European geometric-basket put with analytical m-derivatives",
        "scientific_scope": {
            "purpose": (
                "isolate the physical redistribution induced by the pricing and "
                "risk-hybrid coordinates at identical node count"
            ),
            "included": [
                "paired node displacement",
                "global and ridge-restricted fill distance",
                "node concentration in kappa sigma_G sqrt(T) bands",
                "curvature- and fourth-derivative-weighted coverage proxies",
                "one-dimensional local-cubic errors for u, u_m and u_mm",
            ],
            "excluded": [
                "TT-cross reconstruction error",
                "full fixed-strike five-asset spot Greeks",
                "arithmetic and American products",
            ],
            "interpretation": (
                "this is a deterministic coordinate audit, not a replacement "
                "for the full v9.8 end-to-end Greek benchmark"
            ),
        },
        "settings": {
            "physical_shape": list(config.physical_shape),
            "coordinate_node_count": int(config.physical_shape[config.n_assets]),
            "rates": rates,
            "maturity_days": maturity_days,
            "kappas": kappas,
            "dense_points": dense_points,
            "volatilities": config.volatilities.tolist(),
            "correlation": config.correlation.tolist(),
            "dividends": config.dividends.tolist(),
            "moneyness_bounds": scenarios[0]["moneyness_domain"],
        },
        "ratio_conventions": {
            "raw_audit_ratios": (
                "paired_risk_over_pricing stores risk-hybrid divided by pricing "
                "for each raw metric and error"
            ),
            "gain_contract": (
                "paired_gains is oriented so that a value greater than one "
                "always denotes an improvement of the risk-hybrid grid"
            ),
            "coverage_gain": "G_fill = h_pricing / h_risk",
            "node_concentration_gain": "G_nodes = P_risk / P_pricing",
            "error_reduction_factor": "G_error = E_pricing / E_risk",
        },
        "grid_definitions": {
            name: {
                "coordinate_mode": GRID_MODES[name],
                "description": description,
            }
            for name, (_, _, description) in grids.items()
        },
        "scenarios": scenarios,
        "summary": _aggregate(scenarios, kappas),
        "representative_profiles": _representative_profiles(scenarios),
    }
    _write(args.output, payload)
    print(json.dumps(payload["summary"]["evidence_gates"], indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
