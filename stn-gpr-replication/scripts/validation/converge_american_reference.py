from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from stngpr.config import PaperConfig
from stngpr.convergence import (
    component_metrics,
    components_to_json,
    deterministic_moneyness_panel,
    finite_difference_component_arrays,
    gamma_matrices,
    hessian_shape_diagnostics,
    hessian_uncertainty_summary,
    mean_and_standard_error,
    uncertainty_summary,
)
from stngpr.pricers import AmericanArithmeticBasketLSMC


PROFILES = {
    "smoke": {
        "paths": [500, 2_000],
        "steps": [8, 20],
        "relative_bumps": [0.005, 0.01],
        "seeds": [9101, 9102, 9103],
        "policy_modes": ["refit", "frozen", "frozen_oos"],
        "moneyness": [0.0],
        "maturity_days": [30.0],
        "risk_columns": [0, 1],
        "anchor_bump": 0.005,
    },
    "intermediate": {
        "paths": [2_000, 10_000],
        "steps": [20, 40],
        "relative_bumps": [0.005, 0.01],
        "seeds": [9101, 9102, 9103, 9104, 9105],
        "policy_modes": ["refit", "frozen", "frozen_oos"],
        "moneyness": [0.0],
        "maturity_days": [30.0, 90.0],
        "risk_columns": [0, 1],
        "anchor_bump": 0.005,
    },
    "paper": {
        "paths": [10_000, 50_000],
        "steps": [40, 60],
        "relative_bumps": [0.002, 0.005, 0.01],
        "seeds": [9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108],
        "policy_modes": ["refit", "frozen", "frozen_oos"],
        "moneyness": [-0.05, 0.0, 0.05],
        "maturity_days": [30.0, 90.0],
        "risk_columns": [0, 1],
        "anchor_bump": 0.005,
    },
}


def _bump_key(value):
    return f"{float(value):.10g}"


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Converge American arithmetic-basket LSMC Greeks in path count, "
            "exercise dates, bump size and exercise-policy treatment."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--paths", nargs="+", type=int, default=None)
    parser.add_argument("--steps", nargs="+", type=int, default=None)
    parser.add_argument("--relative-bumps", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--policy-modes",
        nargs="+",
        choices=("refit", "frozen", "frozen_oos"),
        default=None,
    )
    parser.add_argument("--moneyness", nargs="+", type=float, default=None)
    parser.add_argument("--maturity-days", nargs="+", type=float, default=None)
    parser.add_argument("--risk-columns", nargs="+", type=int, default=None)
    parser.add_argument("--anchor-bump", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/american_reference_convergence.json"),
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    paths = sorted(set(args.paths or profile["paths"]))
    steps = sorted(set(args.steps or profile["steps"]))
    bumps = sorted(set(args.relative_bumps or profile["relative_bumps"]))
    seeds = args.seeds or profile["seeds"]
    policy_modes = args.policy_modes or profile["policy_modes"]
    moneyness = args.moneyness or profile["moneyness"]
    maturity_days = args.maturity_days or profile["maturity_days"]
    risk_columns = tuple(args.risk_columns or profile["risk_columns"])
    anchor_bump = (
        float(profile["anchor_bump"])
        if args.anchor_bump is None
        else float(args.anchor_bump)
    )
    config = PaperConfig()
    if not paths or min(paths) <= 2:
        parser.error("path counts must exceed two")
    if not steps or min(steps) <= 1:
        parser.error("exercise-step counts must exceed one")
    if not bumps or any(not 0.0 < value < 1.0 for value in bumps):
        parser.error("relative bumps must lie in (0, 1)")
    if anchor_bump not in bumps:
        parser.error("anchor bump must be present in --relative-bumps")
    if len(seeds) < 2:
        parser.error("at least two seeds are required to estimate uncertainty")
    if (
        not risk_columns
        or len(set(risk_columns)) != len(risk_columns)
        or any(column < 0 or column >= config.n_assets for column in risk_columns)
    ):
        parser.error("risk columns must be unique valid spot columns")

    points, m_values, t_values = deterministic_moneyness_panel(
        config,
        moneyness,
        maturity_days,
    )
    pairs = [
        [left, right]
        for position, left in enumerate(risk_columns)
        for right in risk_columns[position + 1 :]
    ]
    results = {
        "schema_version": 2,
        "experiment": "american_arithmetic_reference_convergence",
        "product": "american_arithmetic_basket_put",
        "profile": args.profile,
        "design": {
            "market_parameters": points.tolist(),
            "log_moneyness": m_values.tolist(),
            "maturity_days": t_values.tolist(),
            "risk_columns": list(risk_columns),
            "cross_gamma_pairs": pairs,
            "stencil_price_count_per_point": (
                1
                + 2 * len(risk_columns)
                + 4 * len(risk_columns) * (len(risk_columns) - 1) // 2
            ),
        },
        "settings": {
            "path_counts": paths,
            "exercise_steps": steps,
            "relative_bumps": bumps,
            "seeds": list(seeds),
            "policy_modes": list(policy_modes),
            "anchor": {
                "path_count": max(paths),
                "exercise_steps": max(steps),
                "relative_bump": anchor_bump,
            },
        },
        "interpretation": {
            "refit": "continuation regressions are re-estimated after every spot bump",
            "frozen": "the base-scenario policy is reused on the same common paths",
            "frozen_oos": (
                "the base policy is trained on one sample and evaluated on an "
                "independent sample using seed + 104729"
            ),
            "scope": (
                "frozen-policy modes are variance-diagnostic ablations; agreement "
                "with refit must be checked before treating them as estimators"
            ),
        },
        "runs": {},
        "comparisons": {},
    }
    means = {}

    for policy_mode in policy_modes:
        results["runs"][policy_mode] = {}
        means[policy_mode] = {}
        for n_paths in paths:
            path_key = str(n_paths)
            results["runs"][policy_mode][path_key] = {}
            means[policy_mode][path_key] = {}
            for n_steps in steps:
                step_key = str(n_steps)
                results["runs"][policy_mode][path_key][step_key] = {}
                means[policy_mode][path_key][step_key] = {}
                for bump in bumps:
                    replications = []
                    start = perf_counter()
                    for seed in seeds:
                        pricer = AmericanArithmeticBasketLSMC(
                            config,
                            n_paths=n_paths,
                            n_steps=n_steps,
                            seed=seed,
                            policy_mode=policy_mode,
                        )
                        replications.append(
                            finite_difference_component_arrays(
                                pricer,
                                points,
                                risk_columns,
                                bump,
                            )
                        )
                    mean, standard_error = mean_and_standard_error(replications)
                    bump_key = _bump_key(bump)
                    means[policy_mode][path_key][step_key][bump_key] = mean
                    results["runs"][policy_mode][path_key][step_key][bump_key] = {
                        "path_count": int(n_paths),
                        "exercise_steps": int(n_steps),
                        "relative_bump": float(bump),
                        "seed_count": len(seeds),
                        "wall_time_seconds": perf_counter() - start,
                        "uncertainty": uncertainty_summary(mean, standard_error),
                        "hessian_uncertainty": hessian_uncertainty_summary(
                            mean,
                            standard_error,
                        ),
                        "hessian_shape": hessian_shape_diagnostics(
                            gamma_matrices(mean)
                        ),
                        "components": components_to_json(mean),
                        "component_standard_errors": components_to_json(
                            standard_error
                        ),
                    }
                    _write(args.output, results)
                    print(
                        f"policy={policy_mode} paths={n_paths} steps={n_steps} "
                        f"bump={bump:g} complete"
                    )

    anchor_path = str(max(paths))
    anchor_steps = str(max(steps))
    anchor_bump_key = _bump_key(anchor_bump)
    for policy_mode in policy_modes:
        anchor = means[policy_mode][anchor_path][anchor_steps][anchor_bump_key]
        for n_paths in paths:
            path_key = str(n_paths)
            for n_steps in steps:
                step_key = str(n_steps)
                for bump in bumps:
                    bump_key = _bump_key(bump)
                    run = results["runs"][policy_mode][path_key][step_key][bump_key]
                    run["error_against_mode_anchor"] = component_metrics(
                        anchor,
                        means[policy_mode][path_key][step_key][bump_key],
                    )
                    previous = [value for value in paths if value < n_paths]
                    if previous:
                        previous_key = str(max(previous))
                        run["error_against_previous_path_count"] = component_metrics(
                            means[policy_mode][previous_key][step_key][bump_key],
                            means[policy_mode][path_key][step_key][bump_key],
                        )

    if "refit" in policy_modes:
        refit_anchor = means["refit"][anchor_path][anchor_steps][anchor_bump_key]
        for policy_mode in policy_modes:
            if policy_mode == "refit":
                continue
            results["comparisons"][f"{policy_mode}_against_refit_at_anchor"] = (
                component_metrics(
                    refit_anchor,
                    means[policy_mode][anchor_path][anchor_steps][anchor_bump_key],
                )
            )

    results["acceptance_targets"] = {
        "price_normalized_standard_error": 0.005,
        "delta_normalized_standard_error": 0.02,
        "hessian_normalized_standard_error_frobenius": 0.15,
        "note": (
            "A frozen policy is acceptable only if it reduces dispersion without "
            "creating a material bias relative to the converged refit estimator."
        ),
    }
    _write(args.output, results)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
