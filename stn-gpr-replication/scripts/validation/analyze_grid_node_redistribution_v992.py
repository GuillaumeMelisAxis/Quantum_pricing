from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from stngpr.config import PaperConfig
from stngpr.diagnostics import (
    geometric_basket_convexity_ridge,
    geometric_basket_effective_parameters,
)
from stngpr.grid_geometry import (
    interval_fill_distance,
    normalized_node_displacement,
    physical_log_moneyness_nodes,
)
from stngpr.risk_grids import build_greek_coordinate_grid


PROFILES = {
    "smoke": {
        "shape": (16, 16, 16, 16, 16, 64, 8, 16),
        "rates": [0.03],
        "maturity_days": [7, 30, 90],
    },
    "intermediate": {
        "shape": (32, 32, 32, 32, 32, 256, 8, 32),
        "rates": [0.005, 0.03, 0.08],
        "maturity_days": [3, 7, 14, 30, 90, 365],
    },
    "paper": {
        "shape": (64, 64, 64, 64, 64, 512, 8, 64),
        "rates": np.linspace(0.005, 0.08, 8).tolist(),
        "maturity_days": [1, 3, 7, 14, 30, 60, 90, 180, 365, 730, 1095],
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


def _band_metrics(nodes: np.ndarray, lower: float, upper: float) -> dict:
    """Return exactly the two band metrics defined in the paper."""
    x = np.asarray(nodes, dtype=float)
    h_fill = interval_fill_distance(x, lower, upper)["fill_distance"]
    p_nodes = np.count_nonzero((x >= lower) & (x <= upper)) / x.size
    return {"h_fill": float(h_fill), "P_nodes": float(p_nodes)}


def _scenario(config, grids, rate, maturity_days, kappas):
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
    risk_scale = float(basket_sigma * np.sqrt(maturity))

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

    displacement = normalized_node_displacement(
        nodes["pricing_grid"],
        nodes["risk_hybrid_grid"],
        domain,
    )
    bands = {}
    for kappa in kappas:
        lower = max(domain[0], ridge - float(kappa) * risk_scale)
        upper = min(domain[1], ridge + float(kappa) * risk_scale)
        key = f"kappa_{float(kappa):g}"
        pricing = _band_metrics(nodes["pricing_grid"], lower, upper)
        risk = _band_metrics(nodes["risk_hybrid_grid"], lower, upper)
        bands[key] = {
            "kappa": float(kappa),
            "bounds": [float(lower), float(upper)],
            "pricing_grid": pricing,
            "risk_hybrid_grid": risk,
            "coverage_gain": float(
                pricing["h_fill"]
                / max(risk["h_fill"], np.finfo(float).tiny)
            ),
        }

    return {
        "theta": {
            "rate": float(rate),
            "maturity": maturity,
            "maturity_days": float(maturity_days),
            "volatilities": config.volatilities.tolist(),
        },
        "basket_volatility": float(basket_sigma),
        "ridge_center": ridge,
        "risk_scale": risk_scale,
        "moneyness_domain": list(domain),
        "node_displacement": {
            "D2": displacement["normalized_l2"],
            "D_infinity": displacement["normalized_linf"],
        },
        "physical_nodes": {
            name: values.tolist() for name, values in nodes.items()
        },
        "bands": bands,
    }


def _aggregate(scenarios, kappas):
    summary = {
        "scenario_count": len(scenarios),
        "node_displacement": {
            "D2": _summary(
                row["node_displacement"]["D2"] for row in scenarios
            ),
            "D_infinity": _summary(
                row["node_displacement"]["D_infinity"] for row in scenarios
            ),
        },
        "bands": {},
    }
    for kappa in kappas:
        key = f"kappa_{float(kappa):g}"
        summary["bands"][key] = {
            "h_fill": {
                name: _summary(
                    row["bands"][key][name]["h_fill"] for row in scenarios
                )
                for name in GRID_MODES
            },
            "P_nodes": {
                name: _summary(
                    row["bands"][key][name]["P_nodes"] for row in scenarios
                )
                for name in GRID_MODES
            },
            "coverage_gain": _summary(
                row["bands"][key]["coverage_gain"] for row in scenarios
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "v9.9.2 audit of D2, D_infinity, local coverage gain and node "
            "concentration for matched pricing and risk-hybrid grids."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--rates", nargs="+", type=float, default=None)
    parser.add_argument("--maturity-days", nargs="+", type=float, default=None)
    parser.add_argument("--kappas", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    parser.add_argument("--moneyness-nodes", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v992_grid_geometry.json"),
    )
    args = parser.parse_args()

    design = PROFILES[args.profile]
    rates = [float(value) for value in (args.rates or design["rates"])]
    maturity_days = [
        float(value) for value in (args.maturity_days or design["maturity_days"])
    ]
    kappas = sorted(set(float(value) for value in args.kappas))
    shape = list(design["shape"])
    if args.moneyness_nodes is not None:
        shape[5] = int(args.moneyness_nodes)
    base = PaperConfig()
    if any(value < base.rate_bounds[0] or value > base.rate_bounds[1] for value in rates):
        parser.error("rates must lie inside the paper domain")
    if any(
        value <= 0.0 or value > 365.0 * base.maturity_bounds[1]
        for value in maturity_days
    ):
        parser.error("maturity-days must lie inside the paper domain")
    if not kappas or any(value <= 0.0 for value in kappas):
        parser.error("kappas must be positive")
    if any(value <= 1 or value & (value - 1) for value in shape):
        parser.error("every physical mode size must be a power of two")

    config = replace(base, physical_shape=tuple(shape))
    grids = {
        name: build_greek_coordinate_grid(config, mode)
        for name, mode in GRID_MODES.items()
    }
    scenarios = []
    for rate in rates:
        for days in maturity_days:
            print(f"rate={rate:.6f}, T={days:g}d")
            scenarios.append(_scenario(config, grids, rate, days, kappas))

    payload = {
        "schema_version": "greeks-grid-geometry-v9.9.2",
        "profile": args.profile,
        "purpose": (
            "matched-cardinality geometric comparison of the pricing and "
            "risk-hybrid physical log-moneyness nodes"
        ),
        "settings": {
            "physical_shape": list(config.physical_shape),
            "coordinate_node_count": int(config.physical_shape[config.n_assets]),
            "rates": rates,
            "maturity_days": maturity_days,
            "kappas": kappas,
            "fixed_volatilities": config.volatilities.tolist(),
            "correlation": config.correlation.tolist(),
            "moneyness_bounds": scenarios[0]["moneyness_domain"],
        },
        "metric_definitions": {
            "D2": (
                "sqrt((1/n) sum_j ((m_j^risk-m_j^pricing)/(m_plus-m_minus))^2)"
            ),
            "D_infinity": (
                "max_j abs(m_j^risk-m_j^pricing)/(m_plus-m_minus)"
            ),
            "h_fill": (
                "sup over m in R_kappa of min_j abs(m-m_j^(grid))"
            ),
            "coverage_gain": "h_fill_pricing/h_fill_risk_hybrid",
            "P_nodes": (
                "(1/n) sum_j indicator{m_j^(grid) belongs to R_kappa}"
            ),
        },
        "excluded_quantities": [
            "pricing or Greek interpolation errors",
            "curvature-weighted proxies",
            "TT-cross or TT reconstruction diagnostics",
        ],
        "grid_definitions": {
            name: {
                "coordinate_mode": GRID_MODES[name],
                "description": description,
            }
            for name, (_, _, description) in grids.items()
        },
        "scenarios": scenarios,
        "summary": _aggregate(scenarios, kappas),
    }
    _write(args.output, payload)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
