from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.baselines import ExactLaplacianGPR
from stngpr.config import PaperConfig
from stngpr.coordinates import GRID_MODES, TransformedPricer, build_coordinate_grid
from stngpr.pricers import AmericanArithmeticBasketLSMC
from stngpr.tt_surrogate import TTPriceSurrogate


PROFILES = {
    "smoke": {
        "paths": 500,
        "steps": 8,
        "budgets": [1_000, 2_000],
        "anova": 300,
        "gpr": [100, 250],
        "test": 100,
    },
    "intermediate": {
        "paths": 2_000,
        "steps": 15,
        "budgets": [2_000, 5_000, 10_000],
        "anova": 1_000,
        "gpr": [250, 500, 1_000],
        "test": 500,
    },
    "paper": {
        "paths": 10_000,
        "steps": 30,
        "budgets": [1_000, 2_000, 5_000, 10_000, 20_000, 50_000],
        "anova": 2_000,
        "gpr": [100, 250, 500, 1_000, 2_000, 5_000, 10_000],
        "test": 1_000,
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


def uniform_points(config, n, rng):
    return np.column_stack([rng.uniform(a, b, n) for a, b in config.bounds])


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


def evaluate_predictions(config, x_test, y_test, predictions):
    n = config.n_assets
    arithmetic_spot = np.mean(x_test[:, :n], axis=1)
    strike_over_basket = x_test[:, n] / arithmetic_spot
    maturity = x_test[:, n + 2]
    return {
        "global": error_metrics(y_test, predictions),
        "by_moneyness_k_over_arithmetic_spot": bucketed_metrics(
            strike_over_basket, MONEYNESS_BUCKETS, y_test, predictions
        ),
        "by_maturity_years": bucketed_metrics(
            maturity, MATURITY_BUCKETS, y_test, predictions
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--grid-mode", choices=GRID_MODES, default="paper")
    parser.add_argument(
        "--output", type=Path, default=None
    )
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    output = args.output or Path(f"results/american_{args.grid_mode}.json")

    config = PaperConfig()
    rng = np.random.default_rng(config.seed + 1)
    grid, transform, grid_description = build_coordinate_grid(
        config, args.grid_mode, basket_kind="arithmetic"
    )
    market_pricer = AmericanArithmeticBasketLSMC(
        config,
        n_paths=profile["paths"],
        n_steps=profile["steps"],
    )
    model_pricer = TransformedPricer(market_pricer, transform)

    x_test = uniform_points(config, profile["test"], rng)
    model_x_test = transform.to_model(x_test)
    start = perf_counter()
    y_test = market_pricer(x_test)
    test_label_generation_time = perf_counter() - start

    results = {
        "profile": args.profile,
        "lsmc": {
            "paths": profile["paths"],
            "steps": profile["steps"],
            "crn": True,
        },
        "assumptions": {
            "grid": list(config.physical_shape),
            "qtt_cores": len(config.qtt_shape),
            "test_size": int(profile["test"]),
            "coordinate_grid": grid_description,
        },
        "test_label_generation_time": test_label_generation_time,
        "tt": [],
        "gpr": [],
    }

    for budget in profile["budgets"]:
        model = TTPriceSurrogate(grid, model_pricer, seed=config.seed)
        total_start = perf_counter()
        diag = model.fit(
            budget,
            anova_samples=profile["anova"],
            log=True,
        )
        training_total_time = perf_counter() - total_start
        initialization_time = max(training_total_time - diag.wall_time, 0.0)

        inference_start = perf_counter()
        predictions = model.predict(model_x_test)
        inference_total_time = perf_counter() - inference_start
        metrics = evaluate_predictions(
            config, x_test, y_test, predictions
        )
        results["tt"].append({
            "budget": budget,
            "function_evaluations": diag.function_evaluations,
            "training_total_time": training_total_time,
            "initialization_time_including_anova_labels": initialization_time,
            "cross_time_including_adaptive_labels": diag.wall_time,
            "inference_total_time": inference_total_time,
            "inference_seconds_per_query": inference_total_time / len(x_test),
            "sweeps": diag.sweeps,
            "stop": diag.stop,
            "effective_rank": diag.effective_rank,
            "metrics": metrics,
        })

    for n_train in profile["gpr"]:
        x_train = uniform_points(config, n_train, rng)
        model_x_train = transform.to_model(x_train)
        total_start = perf_counter()
        label_start = perf_counter()
        y_train = market_pricer(x_train)
        label_generation_time = perf_counter() - label_start
        fit_start = perf_counter()
        model = ExactLaplacianGPR(
            optimize=True, seed=config.seed
        ).fit(model_x_train, y_train)
        model_fit_time = perf_counter() - fit_start
        training_total_time = perf_counter() - total_start

        inference_start = perf_counter()
        predictions = model.predict(model_x_test)
        inference_total_time = perf_counter() - inference_start
        metrics = evaluate_predictions(
            config, x_test, y_test, predictions
        )
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

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
