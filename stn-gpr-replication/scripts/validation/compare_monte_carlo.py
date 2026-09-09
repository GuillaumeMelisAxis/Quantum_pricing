"""Geometric basket: QTT surrogate versus Monte Carlo, accuracy and speed.

The geometric basket has a closed form, so both methods can be scored against
exact truth rather than against each other. Monte Carlo simulates the five
correlated assets to maturity in a single step, which is what a general engine
pays for a European basket; the geometric case also admits a one-dimensional
reduction, but pricing a basket that way is a special-case optimisation.

Two comparisons matter and they answer different questions:

* per query, at the accuracy each method happens to deliver;
* at matched accuracy, by asking how many paths Monte Carlo would need to reach
  the surrogate's error, since Monte Carlo error falls only as 1/sqrt(N).

The surrogate also carries a one-off build cost, so the report includes the
number of queries after which it becomes cheaper than Monte Carlo.

Runtime: about 2 minutes at the default settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import VolatilityExtendedConfig
from stngpr.coordinates import TransformedPricer, oracle_multilinear_predict
from stngpr.pricers import EuropeanGeometricBasketVolPricer, basket_put_monte_carlo
from stngpr.tt_surrogate import TTPriceSurrogate
from stngpr.vol_extension import (
    CUBIC_AXIS_NAMES,
    BasketScaledModelPricer,
    build_vol_extended_grid,
    resolve_cubic_columns,
    sample_market_points,
)


def score(y_true, y_pred):
    errors = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "p95_absolute_error": float(np.quantile(np.abs(errors), 0.95)),
        "max_absolute_error": float(np.max(np.abs(errors))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument("--layout", default="blocked", choices=("interleaved", "blocked"))
    parser.add_argument("--grid-mode", default="moneyness_adaptive")
    parser.add_argument("--truncation", type=float, default=1e-6)
    parser.add_argument(
        "--cubic-axes", nargs="+", default=["all"], choices=CUBIC_AXIS_NAMES,
        help="Axes interpolated cubically rather than linearly",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/monte_carlo_comparison.json")
    )
    args = parser.parse_args()

    started = perf_counter()
    config = VolatilityExtendedConfig(layout=args.layout)
    grid, transform, description = build_vol_extended_grid(
        config, args.grid_mode, basket_kind="geometric"
    )
    market = EuropeanGeometricBasketVolPricer(config)
    scaled = BasketScaledModelPricer(
        TransformedPricer(market, transform), config.spot_columns
    )

    rng = np.random.default_rng(config.seed)
    x_test = sample_market_points(config, args.test_size, rng)
    model_x_test = transform.to_model(x_test)
    basket_scale = scaled.basket(model_x_test)

    start = perf_counter()
    y_exact = market(x_test)
    closed_form_time = perf_counter() - start

    # ---------------------------------------------------------------- surrogate
    surrogate = TTPriceSurrogate(grid, scaled, seed=config.seed)
    start = perf_counter()
    diag = surrogate.fit(args.budget, anova_samples=2_000, log=False)
    build_time = perf_counter() - start
    if args.truncation:
        import teneva

        surrogate.cores = teneva.truncate(surrogate.cores, args.truncation)

    cubic_columns = resolve_cubic_columns(config, args.cubic_axes)
    start = perf_counter()
    tt_predictions = (
        surrogate.predict_factorized(model_x_test, cubic_columns=cubic_columns)
        * basket_scale
    )
    tt_time = perf_counter() - start
    tt = score(y_exact, tt_predictions)
    tt_seconds_per_query = tt_time / len(x_test)
    print(f"surrogate: build {build_time:.1f}s, MAE {tt['mae']:.4f}, "
          f"{tt_seconds_per_query:.2e} s/query")

    # the same train read with plain multilinear interpolation, for contrast
    start = perf_counter()
    linear_predictions = surrogate.predict_factorized(model_x_test) * basket_scale
    linear_time = perf_counter() - start
    linear = score(y_exact, linear_predictions)
    linear["seconds_per_query"] = linear_time / len(x_test)

    start = perf_counter()
    floor = oracle_multilinear_predict(
        grid, scaled, model_x_test, batch_size=8
    ) * basket_scale
    floor_time = perf_counter() - start

    # -------------------------------------------------------------- monte carlo
    monte_carlo = []
    for n_paths in args.paths:
        start = perf_counter()
        prices, standard_errors = basket_put_monte_carlo(
            x_test[:, list(config.spot_columns)],
            x_test[:, config.strike_column],
            x_test[:, config.rate_column],
            x_test[:, config.maturity_column],
            x_test[:, list(config.volatility_columns)],
            config.correlation,
            n_paths,
            basket_kind="geometric",
            dividends=config.dividends,
            seed=config.seed + n_paths,
        )
        elapsed = perf_counter() - start
        entry = score(y_exact, prices)
        entry["paths"] = n_paths
        entry["seconds_per_query"] = elapsed / len(x_test)
        entry["mean_standard_error"] = float(np.mean(standard_errors))
        # Monte Carlo error falls as 1/sqrt(N): paths needed to match the surrogate
        ratio = (entry["rmse"] / tt["rmse"]) ** 2 if tt["rmse"] > 0 else np.inf
        entry["paths_to_match_surrogate_rmse"] = float(n_paths * ratio)
        entry["seconds_per_query_at_matched_accuracy"] = (
            entry["seconds_per_query"] * ratio
        )
        # one-off build cost pays back after this many queries
        gap = entry["seconds_per_query"] - tt_seconds_per_query
        entry["queries_to_amortize_surrogate_build"] = (
            build_time / gap if gap > 0 else None
        )
        monte_carlo.append(entry)
        print(f"monte carlo {n_paths:>7} paths: MAE {entry['mae']:.4f}, "
              f"{entry['seconds_per_query']:.2e} s/query")

    results = {
        "experiment": "geometric basket: QTT surrogate versus Monte Carlo",
        "settings": {
            "test_size": args.test_size,
            "budget": args.budget,
            "layout": args.layout,
            "grid_mode": args.grid_mode,
            "truncation": args.truncation,
            "cubic_axes": args.cubic_axes,
            "cubic_columns": list(cubic_columns),
            "paths": args.paths,
            "coordinate_grid": description,
        },
        "reference": {
            "closed_form_seconds_per_query": closed_form_time / len(x_test),
            "mean_absolute_price": float(np.mean(np.abs(y_exact))),
        },
        "surrogate": {
            **tt,
            "function_evaluations": diag.function_evaluations,
            "effective_rank": diag.effective_rank,
            "build_time": build_time,
            "seconds_per_query": tt_seconds_per_query,
        },
        "surrogate_multilinear": linear,
        "interpolation_floor": {
            **score(y_exact, floor),
            "seconds_per_query": floor_time / len(x_test),
        },
        "monte_carlo": monte_carlo,
        "total_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    render(results)
    print(f"\nfull results written to {args.output}")


def render(results):
    surrogate = results["surrogate"]
    print(f"\n{'=' * 88}")
    print(f"Geometric basket, {results['settings']['test_size']} test points, "
          f"mean price {results['reference']['mean_absolute_price']:.3f}")
    print(f"{'method':<26}{'MAE':>9}{'RMSE':>9}{'bias':>9}"
          f"{'s/query':>11}{'speedup':>10}")
    slowest = max(m["seconds_per_query"] for m in results["monte_carlo"])
    print(f"{'QTT surrogate':<26}{surrogate['mae']:>9.4f}{surrogate['rmse']:>9.4f}"
          f"{surrogate['bias']:>9.4f}{surrogate['seconds_per_query']:>11.2e}"
          f"{slowest / surrogate['seconds_per_query']:>9.0f}x")
    linear = results["surrogate_multilinear"]
    print(f"{'  same TT, multilinear':<26}{linear['mae']:>9.4f}{linear['rmse']:>9.4f}"
          f"{linear['bias']:>9.4f}{linear['seconds_per_query']:>11.2e}"
          f"{slowest / linear['seconds_per_query']:>9.0f}x")
    floor = results["interpolation_floor"]
    print(f"{'  multilinear grid floor':<26}{floor['mae']:>9.4f}{floor['rmse']:>9.4f}"
          f"{floor['bias']:>9.4f}{'':>11}{'':>10}")
    for entry in results["monte_carlo"]:
        label = f"Monte Carlo {entry['paths']:,} paths"
        print(f"{label:<26}{entry['mae']:>9.4f}{entry['rmse']:>9.4f}"
              f"{entry['bias']:>9.4f}{entry['seconds_per_query']:>11.2e}"
              f"{slowest / entry['seconds_per_query']:>9.1f}x")

    print(f"\n{'matched-accuracy comparison':<40}")
    for entry in results["monte_carlo"]:
        print(f"  from {entry['paths']:,} paths, reaching the surrogate's RMSE "
              f"({surrogate['rmse']:.4f}) needs "
              f"{entry['paths_to_match_surrogate_rmse']:.3e} paths")
        print(f"    -> {entry['seconds_per_query_at_matched_accuracy']:.3e} s/query, "
              f"{entry['seconds_per_query_at_matched_accuracy'] / surrogate['seconds_per_query']:.3e}x "
              f"the surrogate")
        if entry["queries_to_amortize_surrogate_build"]:
            print(f"    -> build cost ({surrogate['build_time']:.1f}s) amortizes after "
                  f"{entry['queries_to_amortize_surrogate_build']:,.0f} queries "
                  f"at {entry['paths']:,} paths")


if __name__ == "__main__":
    main()
