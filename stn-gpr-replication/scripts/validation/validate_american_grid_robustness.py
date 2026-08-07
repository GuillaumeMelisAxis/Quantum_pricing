from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import PaperConfig
from stngpr.coordinates import TransformedPricer, build_coordinate_grid
from stngpr.pricers import AmericanArithmeticBasketLSMC
from stngpr.tt_surrogate import TTPriceSurrogate
from stngpr.validation import (
    metrics_by_labels,
    multi_seed_american_prices,
    reference_summary,
    scalar_summary,
    stratified_american_points,
    uncertainty_diagnostics,
)


MODES = (
    "moneyness_adaptive_uniform_maturity",
    "moneyness_adaptive",
)

PROFILES = {
    "smoke": {
        "paths": 500,
        "steps": 8,
        "anova": 300,
        "n_per_moneyness": 2,
        "budgets": [2_000],
        "tt_seeds": [20260327, 20260328],
        "reference_paths": 5_000,
        "reference_steps": 30,
        "reference_seeds": [901, 902, 903],
    },
    "intermediate": {
        "paths": 2_000,
        "steps": 15,
        "anova": 1_000,
        "n_per_moneyness": 40,
        "budgets": [7_500, 9_000, 12_000],
        "tt_seeds": [20260327, 20260328, 20260329, 20260330, 20260331],
        "reference_paths": 25_000,
        "reference_steps": 60,
        "reference_seeds": [901, 902, 903, 904, 905],
    },
    "paper": {
        "paths": 10_000,
        "steps": 30,
        "anova": 2_000,
        "n_per_moneyness": 100,
        "budgets": [9_000, 12_000, 15_000],
        "tt_seeds": list(range(20260327, 20260337)),
        "reference_paths": 100_000,
        "reference_steps": 90,
        "reference_seeds": list(range(901, 911)),
    },
}


def atomic_json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def experiment_signature(args, profile, budgets, tt_seeds):
    return {
        "profile": args.profile,
        "modes": list(args.modes),
        "budgets": list(budgets),
        "tt_seeds": list(tt_seeds),
        "n_per_moneyness": profile["n_per_moneyness"],
        "training_paths": profile["paths"],
        "training_steps": profile["steps"],
        "anova": profile["anova"],
        "reference_paths": profile["reference_paths"],
        "reference_steps": profile["reference_steps"],
        "reference_seeds": profile["reference_seeds"],
    }


def aggregate_runs(runs, modes, budgets, tt_seeds):
    summaries = []
    for mode in modes:
        for budget in budgets:
            group = [
                run for run in runs
                if run["mode"] == mode and run["budget"] == budget
            ]
            group.sort(key=lambda run: run["tt_seed"])
            summaries.append({
                "mode": mode,
                "budget": budget,
                "completed_seeds": [run["tt_seed"] for run in group],
                "mae": scalar_summary([
                    run["surrogate_vs_independent"]["global"]["mae"]
                    for run in group
                ]),
                "rmse": scalar_summary([
                    run["surrogate_vs_independent"]["global"]["rmse"]
                    for run in group
                ]),
                "normalized_mae": scalar_summary([
                    run["surrogate_vs_independent"]["global"]["normalized_mae"]
                    for run in group
                ]),
                "bias": scalar_summary([
                    run["surrogate_vs_independent"]["global"]["mean_error_bias"]
                    for run in group
                ]),
                "effective_rank": scalar_summary([
                    run["effective_rank"] for run in group
                ]),
                "fit_time": scalar_summary([
                    run["fit_time_including_labels"] for run in group
                ]),
                "atm_mae": scalar_summary([
                    run["surrogate_vs_independent"]["by_moneyness"]["atm"]["mae"]
                    for run in group
                ]),
            })

    paired = []
    baseline_mode, adaptive_mode = MODES
    if baseline_mode in modes and adaptive_mode in modes:
        for budget in budgets:
            baseline = {
                run["tt_seed"]: run for run in runs
                if run["mode"] == baseline_mode and run["budget"] == budget
            }
            adaptive = {
                run["tt_seed"]: run for run in runs
                if run["mode"] == adaptive_mode and run["budget"] == budget
            }
            common = sorted(set(baseline) & set(adaptive) & set(tt_seeds))
            improvements = []
            adaptive_maes = []
            for seed in common:
                base_mae = baseline[seed]["surrogate_vs_independent"]["global"]["mae"]
                adaptive_mae = adaptive[seed]["surrogate_vs_independent"]["global"]["mae"]
                adaptive_maes.append(adaptive_mae)
                improvements.append(100.0 * (base_mae - adaptive_mae) / base_mae)
            adaptive_median = float(np.median(adaptive_maes)) if adaptive_maes else None
            adaptive_max_to_median = (
                float(np.max(adaptive_maes) / adaptive_median)
                if adaptive_median is not None and adaptive_median > 0.0 else None
            )
            paired.append({
                "budget": budget,
                "paired_seeds": common,
                "adaptive_win_rate": (
                    float(np.mean(np.asarray(improvements) > 0.0))
                    if improvements else None
                ),
                "adaptive_mae_improvement_percent": scalar_summary(improvements),
                "adaptive_mae_max_to_median": adaptive_max_to_median,
            })
    decision_rows = [row for row in paired if row["budget"] >= 9_000]
    gate_complete = bool(decision_rows) and all(
        len(row["paired_seeds"]) == len(tt_seeds) for row in decision_rows
    )
    gate_passed = (
        all(
            row["adaptive_win_rate"] >= 0.80
            and row["adaptive_mae_improvement_percent"]["mean"] > 0.0
            and row["adaptive_mae_max_to_median"] <= 2.0
            for row in decision_rows
        )
        if gate_complete else None
    )
    return {
        "by_mode_and_budget": summaries,
        "paired_grid_comparison": paired,
        "robustness_gate": {
            "budgets_at_or_above_9000": [row["budget"] for row in decision_rows],
            "required_paired_win_rate": 0.80,
            "required_mean_improvement_percent": "> 0",
            "maximum_allowed_adaptive_mae_max_to_median": 2.0,
            "complete": gate_complete,
            "passed": gate_passed,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Publication-grade seed robustness test for the two moneyness grids."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--budgets", nargs="+", type=int, default=None)
    parser.add_argument("--tt-seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/validation/american_grid_robustness.json"),
    )
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument(
        "--restart", action="store_true",
        help="Ignore compatible checkpoints and recompute every fit.",
    )
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    budgets = args.budgets or profile["budgets"]
    tt_seeds = args.tt_seeds or profile["tt_seeds"]
    cache_path = args.cache or args.output.with_suffix(".cache.npz")
    signature = experiment_signature(args, profile, budgets, tt_seeds)
    signature_json = json.dumps(signature, sort_keys=True)

    config = PaperConfig()
    rng = np.random.default_rng(config.seed + 50)
    parameters, labels = stratified_american_points(
        config, profile["n_per_moneyness"], rng
    )
    training_pricer = AmericanArithmeticBasketLSMC(
        config, n_paths=profile["paths"], n_steps=profile["steps"], seed=config.seed
    )

    if cache_path.exists() and not args.restart:
        cache = np.load(cache_path)
        cached_signature = str(cache["experiment_signature"].item())
        if cached_signature != signature_json:
            raise RuntimeError("label cache is incompatible; use --restart")
        cached_parameters = cache["parameters"]
        if not np.allclose(cached_parameters, parameters, rtol=0.0, atol=0.0):
            raise RuntimeError("cache test design does not match; use --restart")
        same_seed_labels = cache["same_seed_labels"]
        reference_matrix = cache["reference_matrix"]
        same_seed_time = float(cache["same_seed_time"])
        reference_time = float(cache["reference_time"])
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        start = perf_counter()
        same_seed_labels = training_pricer(parameters)
        same_seed_time = perf_counter() - start
        start = perf_counter()
        reference_matrix = multi_seed_american_prices(
            config, parameters,
            profile["reference_paths"], profile["reference_steps"],
            profile["reference_seeds"],
        )
        reference_time = perf_counter() - start
        np.savez_compressed(
            cache_path,
            parameters=parameters,
            same_seed_labels=same_seed_labels,
            reference_matrix=reference_matrix,
            same_seed_time=np.asarray(same_seed_time),
            reference_time=np.asarray(reference_time),
            experiment_signature=np.asarray(signature_json),
        )

    reference_mean, reference_std, reference_se = reference_summary(reference_matrix)
    runs = []
    if args.output.exists() and not args.restart:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("experiment_signature") != signature:
            raise RuntimeError("output checkpoint is incompatible; use --restart")
        runs = existing.get("runs", [])
    completed = {
        (run["mode"], run["budget"], run["tt_seed"]) for run in runs
    }
    grid_descriptions = {}
    expected_runs = len(args.modes) * len(budgets) * len(tt_seeds)
    session_start = perf_counter()

    def checkpoint(status):
        result = {
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_signature": signature,
            "test_design": {
                "points": len(parameters),
                "n_per_moneyness": profile["n_per_moneyness"],
                "labels": labels,
            },
            "training_lsmc": {
                "paths": profile["paths"],
                "steps": profile["steps"],
                "seed": config.seed,
            },
            "independent_reference": {
                "paths": profile["reference_paths"],
                "steps": profile["reference_steps"],
                "seeds": profile["reference_seeds"],
                "wall_time": reference_time,
                "mean_pointwise_std": float(np.mean(reference_std)),
                "mean_pointwise_standard_error": float(np.mean(reference_se)),
            },
            "same_seed_lsmc_vs_independent": metrics_by_labels(
                reference_mean, same_seed_labels, labels
            ),
            "grid_descriptions": grid_descriptions,
            "completed_runs": len(runs),
            "expected_runs": expected_runs,
            "runs": runs,
            "summary": aggregate_runs(runs, args.modes, budgets, tt_seeds),
            "timing": {
                "same_seed_test_labels": same_seed_time,
                "independent_reference": reference_time,
                "current_session": perf_counter() - session_start,
            },
        }
        atomic_json_write(args.output, result)
        return result

    checkpoint("in_progress")
    for mode in args.modes:
        grid, transform, description = build_coordinate_grid(
            config, mode, basket_kind="arithmetic"
        )
        grid_descriptions[mode] = description
        model_pricer = TransformedPricer(training_pricer, transform)
        model_test = transform.to_model(parameters)
        for budget in budgets:
            for tt_seed in tt_seeds:
                key = (mode, budget, tt_seed)
                if key in completed:
                    continue
                print(f"running mode={mode} budget={budget} tt_seed={tt_seed}")
                model = TTPriceSurrogate(grid, model_pricer, seed=tt_seed)
                start = perf_counter()
                diagnostics = model.fit(
                    budget, anova_samples=profile["anova"], log=True
                )
                fit_time = perf_counter() - start
                start = perf_counter()
                predictions = model.predict(model_test)
                prediction_time = perf_counter() - start
                runs.append({
                    "mode": mode,
                    "budget": budget,
                    "tt_seed": tt_seed,
                    "function_evaluations": diagnostics.function_evaluations,
                    "sweeps": diagnostics.sweeps,
                    "stop": diagnostics.stop,
                    "effective_rank": diagnostics.effective_rank,
                    "fit_time_including_labels": fit_time,
                    "prediction_time": prediction_time,
                    "surrogate_vs_same_seed_lsmc": metrics_by_labels(
                        same_seed_labels, predictions, labels
                    ),
                    "surrogate_vs_independent": metrics_by_labels(
                        reference_mean, predictions, labels
                    ),
                    "uncertainty": uncertainty_diagnostics(
                        predictions, reference_mean, reference_se
                    ),
                })
                completed.add(key)
                result = checkpoint(
                    "complete" if len(runs) == expected_runs else "in_progress"
                )
                global_mae = runs[-1]["surrogate_vs_independent"]["global"]["mae"]
                print(
                    f"completed {len(runs)}/{expected_runs} "
                    f"mae={global_mae:.6g} checkpoint={args.output}"
                )

    final = checkpoint("complete")
    print(json.dumps({
        "status": final["status"],
        "output": str(args.output),
        "completed_runs": final["completed_runs"],
        "summary": final["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
