"""European geometric basket with one volatility dimension per underlying.

The paper's European experiment fixes the volatility vector and learns a price
surface over ``(S_1, ..., S_5, K, r, T)``. This script adds sigma_i as a
sampled coordinate, so the surrogate covers
``(S_1, sigma_1, ..., S_5, sigma_5, K, r, T)``: 13 physical modes and 57 QTT
cores instead of 8 modes and 37 cores.

Volatilities live on ``u_i = log(sigma_i)`` over ``[log 0.05, log 0.80]``, and
each spot is interleaved with its own volatility so the strongest coupling sits
between adjacent tensor-train modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.baselines import ExactLaplacianGPR
from stngpr.config import VolatilityExtendedConfig
from stngpr.coordinates import (
    GRID_MODES,
    TransformedPricer,
    oracle_multilinear_predict,
)
from stngpr.pricers import EuropeanGeometricBasketVolPricer
from stngpr.tt_surrogate import TTPriceSurrogate
from stngpr.vol_extension import (
    CUBIC_AXIS_NAMES,
    BasketScaledModelPricer,
    build_vol_extended_grid,
    resolve_cubic_columns,
    sample_market_points,
)

# Budgets are sized for the 13-dimensional domain: 5,000 evaluations spread
# over 57 cores leave TT-cross one sweep from its ANOVA initialization, so the
# meaningful range starts an order of magnitude above the 8-dimensional runs.
PROFILES = {
    "smoke": {
        "budgets": [2_000, 5_000],
        "anova": 500,
        "gpr": [200, 500],
        "test": 500,
    },
    "intermediate": {
        "budgets": [20_000, 50_000, 100_000, 200_000],
        "anova": 2_000,
        "gpr": [1_000, 2_000, 5_000],
        "test": 2_000,
    },
    "paper": {
        "budgets": [50_000, 100_000, 200_000, 500_000, 1_000_000],
        "anova": 2_000,
        "gpr": [1_000, 2_000, 5_000, 10_000],
        "test": 2_000,
    },
}


MONEYNESS_BUCKETS = (
    ("deep_otm", -np.inf, 0.80),
    ("otm", 0.80, 0.95),
    ("atm", 0.95, 1.05),
    ("itm", 1.05, 1.20),
    ("deep_itm", 1.20, np.inf),
)

MATURITY_BUCKETS = (
    ("very_short", -np.inf, 0.25),
    ("short", 0.25, 1.00),
    ("medium", 1.00, 2.00),
    ("long", 2.00, np.inf),
)

BASKET_VOLATILITY_BUCKETS = (
    ("low", -np.inf, 0.08),
    ("moderate", 0.08, 0.15),
    ("elevated", 0.15, 0.25),
    ("high", 0.25, np.inf),
)


def error_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    absolute = np.abs(errors)
    mae = float(np.mean(absolute))
    mean_absolute_price = float(np.mean(np.abs(y_true)))
    return {
        "count": int(y_true.size),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error": float(np.median(absolute)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "max_absolute_error": float(np.max(absolute)),
        "mean_absolute_price": mean_absolute_price,
        "normalized_mae": (
            mae / mean_absolute_price if mean_absolute_price > 1e-14 else None
        ),
        "mean_error_bias": float(np.mean(errors)),
    }


def bucketed_metrics(values, buckets, y_true, y_pred):
    values = np.asarray(values, dtype=float)
    output = {}
    for name, lower, upper in buckets:
        mask = (values >= lower) & (values < upper)
        output[name] = (
            error_metrics(y_true[mask], y_pred[mask])
            if np.any(mask)
            else {"count": 0}
        )
    return output


def basket_volatility(config, market_parameters):
    """sqrt(w' (sigma sigma' o C) w) for each row of market parameters."""
    sigma = market_parameters[:, list(config.volatility_columns)]
    weights = np.full(config.n_assets, 1.0 / config.n_assets)
    scaled = sigma * weights
    return np.sqrt(np.sum((scaled @ config.correlation) * scaled, axis=1))


def evaluate_predictions(config, x_test, y_test, predictions):
    spots = x_test[:, list(config.spot_columns)]
    geometric_spot = np.exp(np.mean(np.log(spots), axis=1))
    strike_over_basket = x_test[:, config.strike_column] / geometric_spot
    maturity = x_test[:, config.maturity_column]
    return {
        "global": error_metrics(y_test, predictions),
        "by_moneyness_k_over_geometric_spot": bucketed_metrics(
            strike_over_basket, MONEYNESS_BUCKETS, y_test, predictions
        ),
        "by_maturity_years": bucketed_metrics(
            maturity, MATURITY_BUCKETS, y_test, predictions
        ),
        "by_basket_volatility": bucketed_metrics(
            basket_volatility(config, x_test),
            BASKET_VOLATILITY_BUCKETS,
            y_test,
            predictions,
        ),
    }


def time_stencil_prediction(model, model_x_test, n_points, factorized):
    """Per-query cost of the 2^d corner stencil, with an agreement check."""
    if n_points <= 0:
        return None
    subset = model_x_test[:n_points]
    start = perf_counter()
    predictions = model.predict(subset, batch_size=8)
    elapsed = perf_counter() - start
    scale = max(float(np.max(np.abs(predictions))), 1e-12)
    difference = float(np.max(np.abs(predictions - factorized[: len(subset)])))
    return {
        "points": len(subset),
        "corners_per_query": int(2 ** len(model.grid.shape)),
        "total_time": elapsed,
        "seconds_per_query": elapsed / len(subset),
        "max_relative_difference_vs_factorized": difference / scale,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    # With log(K / basket_spot) the grid corners reach strikes far outside
    # [K_min, K_max]; fitting the raw price there plateaus once the volatility
    # modes are present, so moneyness grids default to the P / basket_spot
    # target (see --target), which restores convergence to the grid floor.
    parser.add_argument(
        "--grid-mode", choices=GRID_MODES, default="moneyness_adaptive"
    )
    parser.add_argument(
        "--target",
        choices=("auto", "price", "price_over_basket"),
        default="auto",
        help=(
            "Fit target for TT, oracle and GPR; 'auto' fits P / basket_spot on "
            "moneyness grids, whose corners otherwise reach strikes far outside "
            "the market box, and the raw price elsewhere"
        ),
    )
    parser.add_argument(
        "--vol-scale",
        choices=("log", "linear"),
        default="log",
        help="Volatility grid coordinate: u = log(sigma) or sigma itself",
    )
    parser.add_argument(
        "--vol-nodes",
        type=int,
        default=16,
        help="Grid nodes on each log-volatility axis (power of two)",
    )
    parser.add_argument(
        "--vol-sampling",
        choices=("log_uniform", "uniform"),
        default="log_uniform",
        help="Test-point law for sigma; log_uniform matches the grid coordinate",
    )
    parser.add_argument(
        "--stencil-timing-points",
        type=int,
        default=100,
        help="Test points used to time the 2^d corner stencil (0 disables it)",
    )
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Measure the grid interpolation floor without fitting TT or GPR",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="Override the profile TT-cross budgets",
    )
    parser.add_argument(
        "--cubic-axes", nargs="+", default=["all"], choices=CUBIC_AXIS_NAMES,
        help="Axes interpolated cubically rather than linearly",
    )
    parser.add_argument("--skip-gpr", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    profile = dict(PROFILES[args.profile])
    if args.budgets:
        profile["budgets"] = list(args.budgets)
    output = args.output or Path(
        f"results/european_vol_{args.profile}_{args.grid_mode}.json"
    )

    config = VolatilityExtendedConfig(volatility_nodes=args.vol_nodes)
    rng = np.random.default_rng(config.seed)
    grid, transform, grid_description = build_vol_extended_grid(
        config,
        args.grid_mode,
        basket_kind="geometric",
        volatility_scale=args.vol_scale,
    )
    market_pricer = EuropeanGeometricBasketVolPricer(config)
    model_pricer = TransformedPricer(market_pricer, transform)
    cubic_columns = resolve_cubic_columns(config, args.cubic_axes)
    target = args.target
    if target == "auto":
        target = (
            "price_over_basket"
            if args.grid_mode.startswith("moneyness")
            else "price"
        )
    if target == "price_over_basket":
        fit_pricer = BasketScaledModelPricer(model_pricer, config.spot_columns)
        prediction_scale = fit_pricer.basket
    else:
        fit_pricer = model_pricer
        prediction_scale = lambda z: 1.0

    x_test = sample_market_points(
        config, profile["test"], rng, vol_sampling=args.vol_sampling
    )
    model_x_test = transform.to_model(x_test)
    start = perf_counter()
    y_test = market_pricer(x_test)
    test_label_generation_time = perf_counter() - start

    # 2^13 corners per query, so the oracle runs on small batches.
    oracle_start = perf_counter()
    oracle_predictions = oracle_multilinear_predict(
        grid, fit_pricer, model_x_test, batch_size=8
    ) * prediction_scale(model_x_test)
    oracle_inference_time = perf_counter() - oracle_start
    oracle_metrics = evaluate_predictions(
        config, x_test, y_test, oracle_predictions
    )

    results = {
        "profile": args.profile,
        "budgets": list(profile["budgets"]),
        "experiment": "european geometric basket with per-underlying volatility",
        "assumptions": {
            "n_assets": config.n_assets,
            "market_coordinates": "(S_1, sigma_1, ..., S_5, sigma_5, K, r, T)",
            "model_coordinates": (
                "(S_1, log sigma_1, ..., S_5, log sigma_5, strike coordinate, r, T)"
            ),
            "volatility_bounds": list(config.volatility_bounds),
            "volatility_scale": args.vol_scale,
            "cubic_axes": args.cubic_axes,
            "fit_target": target,
            "volatility_nodes": config.volatility_nodes,
            "volatility_test_sampling": args.vol_sampling,
            "correlation": config.correlation.tolist(),
            "grid": list(config.physical_shape),
            "physical_dimensions": config.n_dimensions,
            "qtt_cores": len(config.qtt_shape),
            "test_size": int(profile["test"]),
            "coordinate_grid": grid_description,
        },
        "test_label_generation_time": test_label_generation_time,
        "oracle_interpolation": {
            "description": "Exact grid-corner interpolation without TT-cross error",
            "inference_total_time": oracle_inference_time,
            "inference_seconds_per_query": oracle_inference_time / len(x_test),
            "metrics": oracle_metrics,
        },
        "tt": [],
        "gpr": [],
    }

    if args.oracle_only:
        results["headline"] = headline(results)
        write_results(results, output)
        return

    for budget in profile["budgets"]:
        model = TTPriceSurrogate(grid, fit_pricer, seed=config.seed)
        total_start = perf_counter()
        diag = model.fit(budget, anova_samples=profile["anova"], log=True)
        training_total_time = perf_counter() - total_start
        initialization_time = max(training_total_time - diag.wall_time, 0.0)

        inference_start = perf_counter()
        raw_predictions = model.predict_factorized(
            model_x_test, cubic_columns=cubic_columns
        )
        predictions = raw_predictions * prediction_scale(model_x_test)
        inference_total_time = perf_counter() - inference_start
        metrics = evaluate_predictions(config, x_test, y_test, predictions)
        results["tt"].append({
            "budget": budget,
            "function_evaluations": diag.function_evaluations,
            "training_total_time": training_total_time,
            "initialization_time_including_anova_labels": initialization_time,
            "cross_time_including_adaptive_labels": diag.wall_time,
            "inference_total_time": inference_total_time,
            "inference_seconds_per_query": inference_total_time / len(x_test),
            "stencil_inference": time_stencil_prediction(
                model, model_x_test, args.stencil_timing_points, raw_predictions
            ),
            "sweeps": diag.sweeps,
            "stop": diag.stop,
            "effective_rank": diag.effective_rank,
            "metrics": metrics,
        })

    if not args.skip_gpr:
        for n_train in profile["gpr"]:
            x_train = sample_market_points(
                config, n_train, rng, vol_sampling=args.vol_sampling
            )
            model_x_train = transform.to_model(x_train)
            total_start = perf_counter()
            label_start = perf_counter()
            y_train = market_pricer(x_train)
            label_generation_time = perf_counter() - label_start
            fit_start = perf_counter()
            model = ExactLaplacianGPR(optimize=True, seed=config.seed).fit(
                model_x_train, y_train / prediction_scale(model_x_train)
            )
            model_fit_time = perf_counter() - fit_start
            training_total_time = perf_counter() - total_start

            inference_start = perf_counter()
            predictions = model.predict(model_x_test) * prediction_scale(
                model_x_test
            )
            inference_total_time = perf_counter() - inference_start
            metrics = evaluate_predictions(config, x_test, y_test, predictions)
            results["gpr"].append({
                "training_size": n_train,
                "training_total_time": training_total_time,
                "label_generation_time": label_generation_time,
                "model_fit_time": model_fit_time,
                "inference_total_time": inference_total_time,
                "inference_seconds_per_query": inference_total_time / len(x_test),
                "kernel": str(model.model.kernel_),
                "metrics": metrics,
            })

    results["headline"] = headline(results)
    write_results(results, output)


def headline(results):
    """The three requested numbers per model: effective rank, MAE, query time."""
    oracle = results["oracle_interpolation"]
    rows = [{
        "model": "oracle interpolation floor",
        "effective_rank": None,
        "mae": oracle["metrics"]["global"]["mae"],
        "normalized_mae": oracle["metrics"]["global"]["normalized_mae"],
        "seconds_per_query": oracle["inference_seconds_per_query"],
    }]
    for entry in results["tt"]:
        rows.append({
            "model": f"TT budget {entry['budget']}",
            "function_evaluations": entry["function_evaluations"],
            "effective_rank": entry["effective_rank"],
            "mae": entry["metrics"]["global"]["mae"],
            "normalized_mae": entry["metrics"]["global"]["normalized_mae"],
            "seconds_per_query": entry["inference_seconds_per_query"],
        })
    for entry in results["gpr"]:
        rows.append({
            "model": f"GPR n={entry['training_size']}",
            "effective_rank": None,
            "mae": entry["metrics"]["global"]["mae"],
            "normalized_mae": entry["metrics"]["global"]["normalized_mae"],
            "seconds_per_query": entry["inference_seconds_per_query"],
        })
    return rows


def write_results(results, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["headline"], indent=2))
    print(f"\nfull results written to {output}")


if __name__ == "__main__":
    main()
