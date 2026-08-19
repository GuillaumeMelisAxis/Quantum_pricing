from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import PaperConfig
from stngpr.coordinates import (
    TransformedPricer,
    build_coordinate_grid,
    oracle_hybrid_cubic_predict,
)
from stngpr.pricers import EuropeanGeometricBasketPricer

from validate_european_greeks import (
    analytical_references,
    component_metrics,
    error_metrics,
    finite_difference_hybrid_arrays,
)


DEFAULT_MONEYNESS = (-0.05, -0.025, 0.0, 0.025, 0.05)
DEFAULT_MATURITY_DAYS = (3.0, 7.0, 14.0, 30.0, 90.0)
DEFAULT_MONEYNESS_NODES = (64, 128, 256)
DEFAULT_MATURITY_NODES = (8, 16, 32)


def _power_of_two(value: int) -> bool:
    return value > 1 and value & (value - 1) == 0


def short_atm_panel(
    config: PaperConfig,
    replicates: int,
    moneyness_levels,
    maturity_days,
    rng,
):
    """Construct a paired design concentrated on the short-dated ATM layer."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    moneyness_levels = np.asarray(moneyness_levels, dtype=float)
    maturity_days = np.asarray(maturity_days, dtype=float)
    if np.any(maturity_days <= 1.0):
        raise ValueError("maturity levels must lie above the one-day grid bound")

    points = []
    log_moneyness = []
    maturity_labels = []
    replicate_ids = []
    for replicate in range(replicates):
        spots = rng.uniform(45.0, 115.0, size=config.n_assets)
        basket = float(np.exp(np.mean(np.log(spots))))
        rate = float(rng.uniform(0.01, 0.06))
        for days in maturity_days:
            for moneyness in moneyness_levels:
                strike = basket * np.exp(moneyness)
                points.append([*spots, strike, rate, days / 365.0])
                log_moneyness.append(moneyness)
                maturity_labels.append(days)
                replicate_ids.append(replicate)

    points = np.asarray(points, dtype=float)
    for column, (lower, upper) in enumerate(config.bounds):
        outside = (
            np.any(points[:, column] <= lower)
            or np.any(points[:, column] >= upper)
        )
        if outside:
            raise RuntimeError("short-ATM validation design lies outside grid bounds")
    return (
        points,
        np.asarray(log_moneyness),
        np.asarray(maturity_labels),
        np.asarray(replicate_ids, dtype=int),
    )


def _cell(axis: np.ndarray, value: float) -> dict:
    upper = int(np.searchsorted(axis, value, side="right"))
    upper = int(np.clip(upper, 1, axis.size - 1))
    lower = upper - 1
    return {
        "lower": float(axis[lower]),
        "upper": float(axis[upper]),
        "width": float(axis[upper] - axis[lower]),
    }


def grid_layer_diagnostics(grid, config, maturity_days) -> dict:
    covariance = np.outer(config.volatilities, config.volatilities)
    covariance *= config.correlation
    weights = np.full(config.n_assets, 1.0 / config.n_assets)
    basket_sigma = float(np.sqrt(weights @ covariance @ weights))
    m_axis = grid.axes[config.n_assets]
    maturity_axis = grid.axes[-1]
    atm_cell = _cell(m_axis, 0.0)

    maturity_cells = {}
    resolution_ratio = {}
    for days in maturity_days:
        maturity = float(days / 365.0)
        maturity_cells[f"{days:g}d"] = _cell(maturity_axis, maturity)
        resolution_ratio[f"{days:g}d"] = float(
            atm_cell["width"] / (basket_sigma * np.sqrt(maturity))
        )
    return {
        "basket_volatility": basket_sigma,
        "atm_moneyness_cell": atm_cell,
        "maturity_cells": maturity_cells,
        "atm_cell_width_over_sigma_sqrt_T": resolution_ratio,
    }


def conditional_metrics(
    references,
    estimates,
    log_moneyness,
    maturity_days,
):
    by_moneyness = {}
    for value in np.unique(log_moneyness):
        by_moneyness[f"m={value:+.6g}"] = component_metrics(
            references,
            estimates,
            np.isclose(log_moneyness, value),
        )

    by_maturity = {}
    for value in np.unique(maturity_days):
        by_maturity[f"T={value:g}d"] = component_metrics(
            references,
            estimates,
            np.isclose(maturity_days, value),
        )

    focus = {
        "atm_band": np.abs(log_moneyness) <= 0.025 + 1e-14,
        "seven_day_atm": (
            np.isclose(maturity_days, 7.0)
            & (np.abs(log_moneyness) <= 0.025 + 1e-14)
        ),
        "short_atm": (
            (maturity_days <= 14.0)
            & (np.abs(log_moneyness) <= 0.025 + 1e-14)
        ),
    }
    return {
        "by_moneyness": by_moneyness,
        "by_maturity": by_maturity,
        "focus": {
            name: component_metrics(references, estimates, mask)
            for name, mask in focus.items()
        },
    }


def pointwise_metrics(
    references,
    estimates,
    log_moneyness,
    maturity_days,
    replicate_ids,
):
    records = []
    for index in range(len(log_moneyness)):
        records.append({
            "index": index,
            "replicate": int(replicate_ids[index]),
            "log_moneyness": float(log_moneyness[index]),
            "maturity_days": float(maturity_days[index]),
            "errors": {
                name: error_metrics(
                    references[name][index],
                    estimates[name][index],
                )
                for name in (
                    "price",
                    "delta",
                    "gamma_diagonal",
                    "cross_gamma",
                )
            },
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moneyness-nodes",
        nargs="+",
        type=int,
        default=DEFAULT_MONEYNESS_NODES,
    )
    parser.add_argument(
        "--maturity-nodes",
        nargs="+",
        type=int,
        default=DEFAULT_MATURITY_NODES,
    )
    parser.add_argument(
        "--moneyness-levels",
        nargs="+",
        type=float,
        default=DEFAULT_MONEYNESS,
    )
    parser.add_argument(
        "--maturity-days",
        nargs="+",
        type=float,
        default=DEFAULT_MATURITY_DAYS,
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--relative-bump", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/short_maturity_greeks_grid_ablation.json"),
    )
    args = parser.parse_args()

    for name, values in (
        ("moneyness", args.moneyness_nodes),
        ("maturity", args.maturity_nodes),
    ):
        if any(not _power_of_two(value) for value in values):
            parser.error(f"all {name} node counts must be powers of two")
    if args.replicates <= 0:
        parser.error("replicates must be positive")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative bump must lie in (0, 1)")

    base_config = PaperConfig()
    seed = base_config.seed + 202 if args.seed is None else args.seed
    rng = np.random.default_rng(seed)
    points, log_moneyness, maturity_days, replicate_ids = short_atm_panel(
        base_config,
        args.replicates,
        args.moneyness_levels,
        args.maturity_days,
        rng,
    )
    market_pricer = EuropeanGeometricBasketPricer(base_config)
    references = analytical_references(base_config, market_pricer, points)

    results = {
        "experiment": "short-maturity ATM grid ablation for spot Greeks",
        "seed": seed,
        "relative_bump": args.relative_bump,
        "n_points": int(len(points)),
        "replicates": args.replicates,
        "moneyness_nodes": list(args.moneyness_nodes),
        "maturity_nodes": list(args.maturity_nodes),
        "moneyness_levels": [float(value) for value in args.moneyness_levels],
        "maturity_days": [float(value) for value in args.maturity_days],
        "interpolation": {
            "spot_and_moneyness": "local four-node Lagrange cubic",
            "maturity_and_other_axes": "multilinear",
        },
        "test_design": {
            "market_parameters": points.tolist(),
            "log_moneyness": log_moneyness.tolist(),
            "maturity_days": maturity_days.tolist(),
            "replicate_ids": replicate_ids.tolist(),
        },
        "reference": "analytical geometric-basket put spot Greeks",
        "configurations": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for n_moneyness in args.moneyness_nodes:
        for n_maturity in args.maturity_nodes:
            shape = list(base_config.physical_shape)
            shape[base_config.n_assets] = n_moneyness
            shape[-1] = n_maturity
            config = replace(base_config, physical_shape=tuple(shape))
            grid, transform, description = build_coordinate_grid(
                config,
                "moneyness_adaptive",
                basket_kind="geometric",
            )
            model_oracle = TransformedPricer(market_pricer, transform)

            def interpolator(model_parameters, cubic_columns):
                return oracle_hybrid_cubic_predict(
                    grid,
                    model_oracle,
                    model_parameters,
                    cubic_columns=cubic_columns,
                )

            start = perf_counter()
            estimates = finite_difference_hybrid_arrays(
                interpolator,
                transform,
                points,
                config.n_assets,
                args.relative_bump,
            )
            elapsed = perf_counter() - start
            key = f"moneyness_{n_moneyness}_maturity_{n_maturity}"
            global_metrics = component_metrics(references, estimates)
            results["configurations"][key] = {
                "n_moneyness": n_moneyness,
                "n_maturity": n_maturity,
                "physical_shape": list(config.physical_shape),
                "qtt_order": int(sum(config.qtt_bits)),
                "logical_tensor_entries": int(np.prod(config.physical_shape)),
                "grid": description,
                "resolution": grid_layer_diagnostics(
                    grid,
                    config,
                    np.asarray(args.maturity_days),
                ),
                "wall_time_seconds": elapsed,
                "global": global_metrics,
                "conditional": conditional_metrics(
                    references,
                    estimates,
                    log_moneyness,
                    maturity_days,
                ),
                "pointwise": pointwise_metrics(
                    references,
                    estimates,
                    log_moneyness,
                    maturity_days,
                    replicate_ids,
                ),
            }
            print(
                f"n_m={n_moneyness:3d} n_T={n_maturity:2d} "
                f"price={100 * global_metrics['price']['normalized_mae']:7.3f}% "
                f"delta={100 * global_metrics['delta']['normalized_mae']:7.3f}% "
                f"gamma={100 * global_metrics['gamma_diagonal']['normalized_mae']:7.3f}% "
                f"cross={100 * global_metrics['cross_gamma']['normalized_mae']:7.3f}% "
                f"time={elapsed:7.2f}s"
            )
            args.output.write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
