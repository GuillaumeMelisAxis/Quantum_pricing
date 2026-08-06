from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.baselines import ExactLaplacianGPR
from stngpr.config import PaperConfig
from stngpr.grids import QTTGrid
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.tt_surrogate import TTPriceSurrogate


PROFILES = {
    "smoke": {"budgets": [2_000, 5_000], "anova": 500, "gpr": [200, 500], "test": 200},
    "intermediate": {"budgets": [5_000, 10_000, 20_000, 50_000],"test": 2_000},
    "paper": {
        "budgets": [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 500_000],
        "anova": 2_000,
        "gpr": [100, 250, 500, 1_000, 2_000, 5_000, 10_000],
        "test": 1_000,
    },
}


def uniform_points(config, n, rng):
    return np.column_stack([rng.uniform(a, b, n) for a, b in config.bounds])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--output", type=Path, default=Path("results/european.json"))
    args = parser.parse_args()
    profile = PROFILES[args.profile]

    config = PaperConfig()
    rng = np.random.default_rng(config.seed)
    grid = QTTGrid(config.bounds, config.physical_shape)
    pricer = EuropeanGeometricBasketPricer(config)
    x_test = uniform_points(config, profile["test"], rng)
    y_test = pricer(x_test)
    results = {"profile": args.profile, "assumptions": {
        "volatilities": config.volatilities.tolist(),
        "correlation": config.correlation.tolist(),
        "grid": list(config.physical_shape),
        "qtt_cores": len(config.qtt_shape),
    }, "tt": [], "gpr": []}

    for budget in profile["budgets"]:
        model = TTPriceSurrogate(grid, pricer, seed=config.seed)
        diag = model.fit(budget, anova_samples=profile["anova"], log=True)
        pred = model.predict(x_test)
        results["tt"].append({
            "budget": budget, "mae": float(np.mean(np.abs(pred - y_test))), **diag.__dict__
        })

    for n_train in profile["gpr"]:
        x_train = uniform_points(config, n_train, rng)
        y_train = pricer(x_train)
        start = perf_counter()
        model = ExactLaplacianGPR(optimize=True, seed=config.seed).fit(x_train, y_train)
        elapsed = perf_counter() - start
        pred = model.predict(x_test)
        results["gpr"].append({
            "training_size": n_train,
            "wall_time": elapsed,
            "mae": float(np.mean(np.abs(pred - y_test))),
            "kernel": str(model.model.kernel_),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

