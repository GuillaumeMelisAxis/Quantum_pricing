from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import PaperConfig
from stngpr.coordinates import GRID_MODES, TransformedPricer, build_coordinate_grid
from stngpr.pricers import AmericanArithmeticBasketLSMC
from stngpr.tt_surrogate import TTPriceSurrogate
from stngpr.validation import (
    metrics_by_labels,
    multi_seed_american_prices,
    reference_summary,
    stratified_american_points,
    uncertainty_diagnostics,
)


PROFILES = {
    "smoke": {
        "paths": 500, "steps": 8, "budget": 2_000, "anova": 300,
        "n_per_moneyness": 2,
        "reference_paths": 5_000, "reference_steps": 30,
        "reference_seeds": [901, 902, 903],
    },
    "intermediate": {
        "paths": 2_000, "steps": 15, "budget": 10_000, "anova": 1_000,
        "n_per_moneyness": 8,
        "reference_paths": 25_000, "reference_steps": 60,
        "reference_seeds": [901, 902, 903, 904, 905],
    },
    "paper": {
        "paths": 10_000, "steps": 30, "budget": 20_000, "anova": 2_000,
        "n_per_moneyness": 20,
        "reference_paths": 100_000, "reference_steps": 90,
        "reference_seeds": list(range(901, 911)),
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Separate TT error from common-random-number LSMC label noise."
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--grid-mode", choices=GRID_MODES, default="moneyness_adaptive")
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/validation/american_independent.json"),
    )
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    config = PaperConfig()
    rng = np.random.default_rng(config.seed + 30)
    parameters, labels = stratified_american_points(
        config, profile["n_per_moneyness"], rng
    )

    grid, transform, grid_description = build_coordinate_grid(
        config, args.grid_mode, basket_kind="arithmetic"
    )
    training_pricer = AmericanArithmeticBasketLSMC(
        config, n_paths=profile["paths"], n_steps=profile["steps"], seed=config.seed
    )
    model = TTPriceSurrogate(
        grid, TransformedPricer(training_pricer, transform), seed=config.seed
    )
    total_start = perf_counter()
    fit_start = perf_counter()
    diagnostics = model.fit(
        profile["budget"], anova_samples=profile["anova"], log=True
    )
    fit_time = perf_counter() - fit_start

    prediction_start = perf_counter()
    predictions = model.predict(transform.to_model(parameters))
    prediction_time = perf_counter() - prediction_start
    same_seed_start = perf_counter()
    same_seed_labels = training_pricer(parameters)
    same_seed_time = perf_counter() - same_seed_start
    reference_start = perf_counter()
    reference_matrix = multi_seed_american_prices(
        config, parameters,
        profile["reference_paths"], profile["reference_steps"],
        profile["reference_seeds"],
    )
    reference_time = perf_counter() - reference_start
    reference_mean, reference_std, reference_se = reference_summary(reference_matrix)

    result = {
        "profile": args.profile,
        "grid": grid_description,
        "test_design": {
            "points": len(parameters),
            "n_per_moneyness": profile["n_per_moneyness"],
            "labels": labels,
        },
        "training_lsmc": {
            "paths": profile["paths"], "steps": profile["steps"],
            "seed": config.seed,
        },
        "tt": {
            "budget": profile["budget"], "anova": profile["anova"],
            "function_evaluations": diagnostics.function_evaluations,
            "sweeps": diagnostics.sweeps, "stop": diagnostics.stop,
            "effective_rank": diagnostics.effective_rank,
            "fit_time_including_labels": fit_time,
            "prediction_time": prediction_time,
        },
        "independent_reference": {
            "paths": profile["reference_paths"],
            "steps": profile["reference_steps"],
            "seeds": profile["reference_seeds"],
            "wall_time": reference_time,
            "mean_pointwise_std": float(np.mean(reference_std)),
            "mean_pointwise_standard_error": float(np.mean(reference_se)),
        },
        "surrogate_vs_same_seed_lsmc": metrics_by_labels(
            same_seed_labels, predictions, labels
        ),
        "same_seed_lsmc_vs_independent": metrics_by_labels(
            reference_mean, same_seed_labels, labels
        ),
        "surrogate_vs_independent": metrics_by_labels(
            reference_mean, predictions, labels
        ),
        "uncertainty": uncertainty_diagnostics(
            predictions, reference_mean, reference_se
        ),
        "timing": {
            "same_seed_test_labels": same_seed_time,
            "total": perf_counter() - total_start,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
