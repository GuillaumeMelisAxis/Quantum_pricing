"""Grid rank and storage efficiency: arithmetic versus geometric basket.

Measures how well the QTT surrogate compresses the volatility-extended price
surface for each basket type under both mode orderings.

Why the basket type matters. Under the ``P / B(s)`` target the geometric price
is exactly spot-free, so its spot modes collapse to rank one and a blocked
ordering leaves them there. The arithmetic price keeps a genuine dependence on
basket composition, and its variance carries ``S_i sigma_i`` cross terms
through ``exp(sigma_i sigma_j C_ij T)``. That is the coupling an interleaved
ordering is meant to exploit, so this is the case that can reverse the ranking.

The arithmetic labels come from the Levy lognormal moment match, which is what
makes a hundred thousand pricer calls affordable. It carries the correct
coupling structure but is itself an approximation (about 1.6% from Monte Carlo
over this domain, and worse above total volatility 0.35), so the reported error
is surrogate-versus-Levy, not surrogate-versus-truth.

Runtime: about 6 minutes at the default settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import teneva

from stngpr.config import VolatilityExtendedConfig
from stngpr.coordinates import TransformedPricer, oracle_multilinear_predict
from stngpr.pricers import (
    EuropeanArithmeticBasketVolPricer,
    EuropeanGeometricBasketVolPricer,
)
from stngpr.tt_surrogate import TTPriceSurrogate
from stngpr.vol_extension import (
    CUBIC_AXIS_NAMES,
    BasketScaledModelPricer,
    build_vol_extended_grid,
    resolve_cubic_columns,
    sample_market_points,
)

PRICERS = {
    "geometric": EuropeanGeometricBasketVolPricer,
    "arithmetic": EuropeanArithmeticBasketVolPricer,
}


def build(basket, layout, grid_mode):
    config = VolatilityExtendedConfig(layout=layout)
    grid, transform, description = build_vol_extended_grid(
        config, grid_mode, basket_kind=basket
    )
    market = PRICERS[basket](config)
    model = TransformedPricer(market, transform)
    scaled = BasketScaledModelPricer(model, config.spot_columns, basket_kind=basket)
    return config, grid, transform, market, scaled, description


def spot_block_rank(cores, grid, config):
    """Largest bond rank on any QTT core belonging to a spot mode."""
    ranks, start = [], 0
    for column, bits in enumerate(grid.bits):
        block = cores[start : start + bits]
        if column in config.spot_columns:
            ranks.append(max(core.shape[2] for core in block))
        start += bits
    return int(max(ranks))


def parameter_count(cores):
    return int(sum(np.prod(core.shape) for core in cores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baskets", nargs="+", default=["arithmetic", "geometric"], choices=PRICERS
    )
    parser.add_argument(
        "--layouts", nargs="+", default=["interleaved", "blocked"],
        choices=("interleaved", "blocked"),
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[50_000, 100_000])
    parser.add_argument(
        "--truncations", type=float, nargs="+",
        default=[1e-8, 1e-6, 1e-5, 1e-4, 1e-3],
    )
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument(
        "--cubic-axes", nargs="+", default=["all"], choices=CUBIC_AXIS_NAMES,
        help="Axes interpolated cubically rather than linearly",
    )
    parser.add_argument("--grid-mode", default="moneyness_adaptive")
    parser.add_argument("--anova-samples", type=int, default=2_000)
    parser.add_argument(
        "--output", type=Path, default=Path("results/basket_layout_comparison.json")
    )
    args = parser.parse_args()

    started = perf_counter()
    results = {
        "experiment": "basket type x mode ordering, volatility-extended surface",
        "settings": {
            "baskets": args.baskets,
            "layouts": args.layouts,
            "budgets": args.budgets,
            "truncations": args.truncations,
            "test_size": args.test_size,
            "grid_mode": args.grid_mode,
            "cubic_axes": args.cubic_axes,
            "arithmetic_labels": "Levy lognormal moment match (approximate)",
        },
        "oracle": {},
        "runs": [],
    }

    for basket in args.baskets:
        # the interpolation floor depends on the basket, not on the mode ordering
        config, grid, transform, market, scaled, description = build(
            basket, args.layouts[0], args.grid_mode
        )
        rng = np.random.default_rng(config.seed)
        x_test = sample_market_points(config, args.test_size, rng)
        model_x_test = transform.to_model(x_test)
        y_test = market(x_test)
        basket_scale = scaled.basket(model_x_test)

        start = perf_counter()
        floor = oracle_multilinear_predict(
            grid, scaled, model_x_test, batch_size=8
        ) * basket_scale
        floor_time = perf_counter() - start
        results["oracle"][basket] = {
            "mae": float(np.mean(np.abs(floor - y_test))),
            "mean_absolute_price": float(np.mean(np.abs(y_test))),
            "seconds_per_query": floor_time / len(x_test),
            "coordinate_grid": description,
        }
        print(
            f"[{basket}] interpolation floor MAE "
            f"{results['oracle'][basket]['mae']:.4f} "
            f"(mean price {results['oracle'][basket]['mean_absolute_price']:.3f})"
        )

        for layout in args.layouts:
            config, grid, transform, market, scaled, _ = build(
                basket, layout, args.grid_mode
            )
            rng = np.random.default_rng(config.seed)
            x_test = sample_market_points(config, args.test_size, rng)
            model_x_test = transform.to_model(x_test)
            y_test = market(x_test)
            basket_scale = scaled.basket(model_x_test)

            for budget in args.budgets:
                surrogate = TTPriceSurrogate(grid, scaled, seed=config.seed)
                start = perf_counter()
                diag = surrogate.fit(
                    budget, anova_samples=args.anova_samples, log=False
                )
                fit_time = perf_counter() - start
                fitted = surrogate.cores

                sweep = []
                for tolerance in args.truncations:
                    surrogate.cores = teneva.truncate(fitted, tolerance)
                    start = perf_counter()
                    predictions = (
                        surrogate.predict_factorized(
                            model_x_test,
                            cubic_columns=resolve_cubic_columns(
                                config, args.cubic_axes
                            ),
                        )
                        * basket_scale
                    )
                    query_time = perf_counter() - start
                    sweep.append({
                        "truncation": tolerance,
                        "effective_rank": float(teneva.erank(surrogate.cores)),
                        "parameters": parameter_count(surrogate.cores),
                        "kib": parameter_count(surrogate.cores) * 8 / 1024,
                        "spot_block_rank": spot_block_rank(
                            surrogate.cores, grid, config
                        ),
                        "mae": float(np.mean(np.abs(predictions - y_test))),
                        "seconds_per_query": query_time / len(x_test),
                    })
                results["runs"].append({
                    "basket": basket,
                    "layout": layout,
                    "budget": budget,
                    "function_evaluations": diag.function_evaluations,
                    "fit_time": fit_time,
                    "fitted_effective_rank": diag.effective_rank,
                    "truncation_sweep": sweep,
                })
                print(
                    f"[{basket}/{layout}] {diag.function_evaluations} evals, "
                    f"fit {fit_time:.1f}s, rank {diag.effective_rank:.2f}"
                )

    results["total_seconds"] = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    render(results)
    print(f"\nfull results written to {args.output}")


def render(results):
    for basket, floor in results["oracle"].items():
        print(f"\n{'=' * 92}")
        print(
            f"{basket.upper()} basket | interpolation floor MAE {floor['mae']:.4f} "
            f"| mean price {floor['mean_absolute_price']:.3f}"
        )
        print(
            f"{'layout':<13}{'evals':>8}{'trunc':>9}{'rank':>8}"
            f"{'params':>9}{'KiB':>8}{'spot r':>8}{'MAE':>9}{'s/query':>11}"
        )
        for run in results["runs"]:
            if run["basket"] != basket:
                continue
            for row in run["truncation_sweep"]:
                print(
                    f"{run['layout']:<13}{run['function_evaluations']:>8}"
                    f"{row['truncation']:>9.0e}{row['effective_rank']:>8.2f}"
                    f"{row['parameters']:>9d}{row['kib']:>8.1f}"
                    f"{row['spot_block_rank']:>8d}{row['mae']:>9.4f}"
                    f"{row['seconds_per_query']:>11.2e}"
                )


if __name__ == "__main__":
    main()
