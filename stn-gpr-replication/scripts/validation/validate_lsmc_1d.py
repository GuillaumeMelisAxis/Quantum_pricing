from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from stngpr.validation import (
    american_put_binomial,
    american_put_lsmc_1d,
    black_scholes_put,
    summarize_replications,
)


PROFILES = {
    "smoke": {
        "spot_ratios": [0.8, 1.0, 1.2],
        "maturities": [0.25, 1.0],
        "volatilities": [0.20],
        "paths": 5_000,
        "steps": 30,
        "binomial_steps": 1_000,
        "seeds": list(range(3)),
    },
    "intermediate": {
        "spot_ratios": [0.7, 0.85, 1.0, 1.15, 1.3],
        "maturities": [0.1, 0.5, 1.0, 2.0],
        "volatilities": [0.15, 0.25, 0.40],
        "paths": 25_000,
        "steps": 60,
        "binomial_steps": 3_000,
        "seeds": list(range(10)),
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/validation/lsmc_1d.json"),
    )
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    strike, rate = 100.0, 0.03
    rows = []
    start_all = perf_counter()
    for ratio in profile["spot_ratios"]:
        for maturity in profile["maturities"]:
            for volatility in profile["volatilities"]:
                spot = strike * ratio
                binomial = american_put_binomial(
                    spot, strike, rate, volatility, maturity,
                    steps=profile["binomial_steps"],
                )
                values = [
                    american_put_lsmc_1d(
                        spot, strike, rate, volatility, maturity,
                        n_paths=profile["paths"], n_steps=profile["steps"],
                        seed=seed,
                    )
                    for seed in profile["seeds"]
                ]
                european = black_scholes_put(
                    spot, strike, rate, volatility, maturity
                )
                summary = summarize_replications(values, reference=binomial)
                rows.append({
                    "spot": spot,
                    "strike": strike,
                    "spot_over_strike": ratio,
                    "rate": rate,
                    "volatility": volatility,
                    "maturity": maturity,
                    "intrinsic": max(strike - spot, 0.0),
                    "european_put": float(european),
                    "american_binomial": binomial,
                    "lsmc": summary,
                    "sanity": {
                        "mean_above_intrinsic": bool(
                            summary["mean"] >= max(strike - spot, 0.0)
                        ),
                        "mean_above_european": bool(summary["mean"] >= european),
                    },
                })
    result = {
        "profile": args.profile,
        "settings": {
            "paths": profile["paths"],
            "lsmc_steps": profile["steps"],
            "binomial_steps": profile["binomial_steps"],
            "seeds": profile["seeds"],
        },
        "wall_time": perf_counter() - start_all,
        "scenarios": rows,
        "aggregate": {
            "max_absolute_bias": max(row["lsmc"]["absolute_bias"] for row in rows),
            "mean_absolute_bias": sum(row["lsmc"]["absolute_bias"] for row in rows) / len(rows),
            "fraction_binomial_in_ci95": sum(
                row["lsmc"]["reference_in_ci95"] for row in rows
            ) / len(rows),
            "all_above_intrinsic": all(
                row["sanity"]["mean_above_intrinsic"] for row in rows
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
