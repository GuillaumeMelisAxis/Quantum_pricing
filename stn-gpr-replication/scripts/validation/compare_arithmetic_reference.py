"""Arithmetic basket: Levy and the QTT surrogate against a 10^6-path reference.

The arithmetic basket has no closed form, so every earlier arithmetic number in
this repository was measured against the Levy lognormal moment match - the
approximation the surrogate is *fitted to*, not the truth. This script supplies
the missing reference by pricing the same test set with a high-path Monte Carlo
run and scoring all three against it.

That makes the error decomposition explicit. For each test point,

    surrogate - reference  =  (surrogate - Levy)  +  (Levy - reference)
    ^ total                   ^ reconstruction      ^ model

The reconstruction term is what more TT-cross budget or a finer grid can fix.
The model term is a property of the label generator and no amount of surrogate
budget touches it. Reporting them apart says which one is worth spending on.

The reference carries its own Monte Carlo noise, reported as the mean standard
error, and that is the resolution floor of every number here.

Runtime: about 6 minutes at the default settings, dominated by the 10^6-path
reference (roughly 0.14 s per test point).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import teneva

from stngpr.config import VolatilityExtendedConfig
from stngpr.coordinates import TransformedPricer
from stngpr.pricers import EuropeanArithmeticBasketVolPricer, basket_put_monte_carlo
from stngpr.tt_surrogate import TTPriceSurrogate
from stngpr.vol_extension import (
    CUBIC_AXIS_NAMES,
    BasketScaledModelPricer,
    build_vol_extended_grid,
    resolve_cubic_columns,
    sample_market_points,
)

TOTAL_VOLATILITY_BUCKETS = (
    ("low <0.10", -np.inf, 0.10),
    ("0.10-0.20", 0.10, 0.20),
    ("0.20-0.35", 0.20, 0.35),
    ("high >0.35", 0.35, np.inf),
)


def score(reference, prediction):
    errors = np.asarray(prediction, dtype=float) - np.asarray(reference, dtype=float)
    absolute = np.abs(errors)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "max_absolute_error": float(np.max(absolute)),
    }


def total_volatility(config, x):
    """sigma_B * sqrt(T), the parameter the Levy moment match degrades in."""
    sigma = x[:, list(config.volatility_columns)]
    weights = np.full(config.n_assets, 1.0 / config.n_assets)
    scaled = sigma * weights
    sigma_b = np.sqrt(np.sum((scaled @ config.correlation) * scaled, axis=1))
    return sigma_b * np.sqrt(x[:, config.maturity_column])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-paths", type=int, default=1_000_000)
    parser.add_argument("--paths", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument(
        "--layout", default="interleaved", choices=("interleaved", "blocked")
    )
    parser.add_argument("--grid-mode", default="moneyness_adaptive")
    parser.add_argument("--truncation", type=float, default=1e-6)
    parser.add_argument(
        "--cubic-axes", nargs="+", default=["all"], choices=CUBIC_AXIS_NAMES
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/arithmetic_reference_comparison.json"),
    )
    args = parser.parse_args()

    started = perf_counter()
    config = VolatilityExtendedConfig(layout=args.layout)
    grid, transform, description = build_vol_extended_grid(
        config, args.grid_mode, basket_kind="arithmetic"
    )
    market = EuropeanArithmeticBasketVolPricer(config)
    scaled = BasketScaledModelPricer(
        TransformedPricer(market, transform),
        config.spot_columns,
        basket_kind="arithmetic",
    )

    rng = np.random.default_rng(config.seed)
    x_test = sample_market_points(config, args.test_size, rng)
    model_x_test = transform.to_model(x_test)
    basket_scale = scaled.basket(model_x_test)

    spots = x_test[:, list(config.spot_columns)]
    sigma = x_test[:, list(config.volatility_columns)]
    strikes = x_test[:, config.strike_column]
    rates = x_test[:, config.rate_column]
    maturities = x_test[:, config.maturity_column]

    # ------------------------------------------------------------------- Levy
    start = perf_counter()
    levy = market(x_test)
    levy_time = perf_counter() - start

    # -------------------------------------------------------------- surrogate
    surrogate = TTPriceSurrogate(grid, scaled, seed=config.seed)
    start = perf_counter()
    diag = surrogate.fit(args.budget, anova_samples=2_000, log=False)
    build_time = perf_counter() - start
    if args.truncation:
        surrogate.cores = teneva.truncate(surrogate.cores, args.truncation)
    cubic_columns = resolve_cubic_columns(config, args.cubic_axes)

    start = perf_counter()
    predictions = (
        surrogate.predict_factorized(model_x_test, cubic_columns=cubic_columns)
        * basket_scale
    )
    surrogate_time = perf_counter() - start
    print(f"surrogate: build {build_time:.1f}s, rank {diag.effective_rank:.2f}, "
          f"{surrogate_time / len(x_test):.2e} s/query")

    # -------------------------------------------------------------- reference
    def run_monte_carlo(n_paths):
        start = perf_counter()
        prices, standard_errors = basket_put_monte_carlo(
            spots, strikes, rates, maturities, sigma, config.correlation,
            n_paths, basket_kind="arithmetic", dividends=config.dividends,
            seed=config.seed + n_paths,
        )
        return prices, standard_errors, perf_counter() - start

    print(f"pricing the {args.reference_paths:,}-path reference "
          f"({args.test_size} points, this is the slow part)...")
    reference, reference_se, reference_time = run_monte_carlo(args.reference_paths)
    print(f"reference done in {reference_time:.0f}s, "
          f"mean standard error {np.mean(reference_se):.4f}")

    # ----------------------------------------------------------- decomposition
    total = predictions - reference
    reconstruction = predictions - levy
    model_error = levy - reference

    results = {
        "experiment": "arithmetic basket: Levy and QTT surrogate vs a Monte Carlo reference",
        "settings": {
            "reference_paths": args.reference_paths,
            "comparison_paths": args.paths,
            "test_size": args.test_size,
            "budget": args.budget,
            "layout": args.layout,
            "grid_mode": args.grid_mode,
            "truncation": args.truncation,
            "cubic_axes": args.cubic_axes,
            "coordinate_grid": description,
        },
        "reference": {
            "paths": args.reference_paths,
            "mean_standard_error": float(np.mean(reference_se)),
            "max_standard_error": float(np.max(reference_se)),
            "seconds_per_query": reference_time / len(x_test),
            "mean_absolute_price": float(np.mean(np.abs(reference))),
        },
        "against_reference": {
            "levy": {**score(reference, levy),
                     "seconds_per_query": levy_time / len(x_test)},
            "surrogate": {
                **score(reference, predictions),
                "seconds_per_query": surrogate_time / len(x_test),
                "build_time": build_time,
                "effective_rank": diag.effective_rank,
                "function_evaluations": diag.function_evaluations,
            },
        },
        "error_decomposition": {
            "note": "total = reconstruction + model, exactly, point by point",
            "total_mae": float(np.mean(np.abs(total))),
            "reconstruction_mae": float(np.mean(np.abs(reconstruction))),
            "model_mae": float(np.mean(np.abs(model_error))),
            "total_bias": float(np.mean(total)),
            "reconstruction_bias": float(np.mean(reconstruction)),
            "model_bias": float(np.mean(model_error)),
            "identity_max_residual": float(
                np.max(np.abs(total - (reconstruction + model_error)))
            ),
        },
        "monte_carlo": [],
        "by_total_volatility": {},
    }

    for n_paths in args.paths:
        prices, standard_errors, elapsed = run_monte_carlo(n_paths)
        results["monte_carlo"].append({
            "paths": n_paths,
            **score(reference, prices),
            "mean_standard_error": float(np.mean(standard_errors)),
            "seconds_per_query": elapsed / len(x_test),
        })
        print(f"monte carlo {n_paths:>9,} paths: "
              f"MAE {results['monte_carlo'][-1]['mae']:.4f}")

    vol = total_volatility(config, x_test)
    for name, lower, upper in TOTAL_VOLATILITY_BUCKETS:
        mask = (vol >= lower) & (vol < upper)
        if not np.any(mask):
            continue
        results["by_total_volatility"][name] = {
            "count": int(mask.sum()),
            "mean_price": float(np.mean(reference[mask])),
            "levy_mae": float(np.mean(np.abs(levy[mask] - reference[mask]))),
            "surrogate_mae": float(np.mean(np.abs(predictions[mask] - reference[mask]))),
            "reconstruction_mae": float(np.mean(np.abs(reconstruction[mask]))),
            "reference_standard_error": float(np.mean(reference_se[mask])),
        }

    results["total_seconds"] = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    render(results)
    print(f"\nfull results written to {args.output}")


def render(results):
    reference = results["reference"]
    print(f"\n{'=' * 92}")
    print(f"ARITHMETIC basket | reference = {reference['paths']:,}-path Monte Carlo "
          f"| mean price {reference['mean_absolute_price']:.3f}")
    print(f"reference noise: mean standard error {reference['mean_standard_error']:.4f} "
          f"(max {reference['max_standard_error']:.4f}) - the resolution floor below")
    print(f"\n{'method':<28}{'MAE':>9}{'RMSE':>9}{'bias':>10}{'p95':>9}"
          f"{'max':>9}{'s/query':>11}")
    for label, key in (("Levy approximation", "levy"), ("QTT surrogate", "surrogate")):
        e = results["against_reference"][key]
        print(f"{label:<28}{e['mae']:>9.4f}{e['rmse']:>9.4f}{e['bias']:>+10.4f}"
              f"{e['p95_absolute_error']:>9.4f}{e['max_absolute_error']:>9.4f}"
              f"{e['seconds_per_query']:>11.2e}")
    for entry in results["monte_carlo"]:
        label = f"Monte Carlo {entry['paths']:,} paths"
        print(f"{label:<28}{entry['mae']:>9.4f}{entry['rmse']:>9.4f}"
              f"{entry['bias']:>+10.4f}{entry['p95_absolute_error']:>9.4f}"
              f"{entry['max_absolute_error']:>9.4f}{entry['seconds_per_query']:>11.2e}")

    d = results["error_decomposition"]
    print(f"\nsurrogate error decomposition (identity residual "
          f"{d['identity_max_residual']:.2e}):")
    print(f"  total          |surrogate - reference|  MAE {d['total_mae']:.4f}  "
          f"bias {d['total_bias']:+.4f}")
    print(f"  reconstruction |surrogate - Levy|       MAE {d['reconstruction_mae']:.4f}  "
          f"bias {d['reconstruction_bias']:+.4f}   <- more budget fixes this")
    print(f"  model          |Levy - reference|       MAE {d['model_mae']:.4f}  "
          f"bias {d['model_bias']:+.4f}   <- budget cannot fix this")

    print(f"\n{'bucket (sigma_B sqrt(T))':<26}{'count':>7}{'price':>9}{'Levy':>9}"
          f"{'surrogate':>11}{'recon':>9}{'MC se':>8}")
    for name, entry in results["by_total_volatility"].items():
        print(f"{name:<26}{entry['count']:>7}{entry['mean_price']:>9.3f}"
              f"{entry['levy_mae']:>9.4f}{entry['surrogate_mae']:>11.4f}"
              f"{entry['reconstruction_mae']:>9.4f}"
              f"{entry['reference_standard_error']:>8.4f}")


if __name__ == "__main__":
    main()
