from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import PaperConfig
from stngpr.validation import (
    error_metrics,
    multi_seed_american_prices,
    reference_summary,
    stratified_american_points,
)


PROFILES = {
    "smoke": {
        "n_per_moneyness": 1,
        "path_counts": [500, 2_000],
        "step_counts": [8, 15],
        "seeds": [101, 102, 103],
        "reference_paths": 5_000,
        "reference_steps": 30,
        "reference_seeds": [901, 902, 903],
    },
    "intermediate": {
        "n_per_moneyness": 4,
        "path_counts": [2_000, 5_000, 10_000],
        "step_counts": [15, 30],
        "seeds": [101, 102, 103, 104, 105],
        "reference_paths": 25_000,
        "reference_steps": 60,
        "reference_seeds": [901, 902, 903, 904, 905],
    },
    "paper": {
        "n_per_moneyness": 10,
        "path_counts": [2_000, 5_000, 10_000, 25_000, 50_000],
        "step_counts": [15, 30, 60],
        "seeds": list(range(101, 111)),
        "reference_paths": 100_000,
        "reference_steps": 90,
        "reference_seeds": list(range(901, 911)),
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/validation/lsmc_convergence.json"),
    )
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    config = PaperConfig()
    rng = np.random.default_rng(config.seed + 20)
    parameters, labels = stratified_american_points(
        config, profile["n_per_moneyness"], rng
    )

    total_start = perf_counter()
    ref_start = perf_counter()
    reference_matrix = multi_seed_american_prices(
        config, parameters,
        profile["reference_paths"], profile["reference_steps"],
        profile["reference_seeds"],
    )
    reference_time = perf_counter() - ref_start
    reference_mean, reference_std, reference_se = reference_summary(reference_matrix)
    configurations = []
    for n_paths in profile["path_counts"]:
        for n_steps in profile["step_counts"]:
            start = perf_counter()
            matrix = multi_seed_american_prices(
                config, parameters, n_paths, n_steps, profile["seeds"]
            )
            elapsed = perf_counter() - start
            mean, std, se = reference_summary(matrix)
            configurations.append({
                "paths": n_paths,
                "steps": n_steps,
                "seeds": profile["seeds"],
                "wall_time": elapsed,
                "error_against_reference_mean": error_metrics(reference_mean, mean),
                "mean_std_across_seeds": float(np.mean(std)),
                "mean_standard_error": float(np.mean(se)),
                "fraction_reference_inside_ci95": float(np.mean(
                    np.abs(mean - reference_mean) <= 1.96 * se
                )),
            })
    result = {
        "profile": args.profile,
        "test_design": {
            "points": len(parameters),
            "n_per_moneyness": profile["n_per_moneyness"],
            "labels": labels,
        },
        "reference": {
            "paths": profile["reference_paths"],
            "steps": profile["reference_steps"],
            "seeds": profile["reference_seeds"],
            "wall_time": reference_time,
            "mean_pointwise_std": float(np.mean(reference_std)),
            "mean_pointwise_standard_error": float(np.mean(reference_se)),
        },
        "configurations": configurations,
        "wall_time": perf_counter() - total_start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
