from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

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
from stngpr.pricers import EuropeanArithmeticBasketQMC


PROFILES = {
    "smoke": {
        "paths": [512, 2_048],
        "relative_bumps": [0.005, 0.01],
        "seeds": [8101, 8102, 8103],
        "beta_modes": ["estimated", "unit"],
        "moneyness": [0.0],
        "maturity_days": [7.0, 30.0],
        "anchor_bump": 0.005,
    },
    "intermediate": {
        "paths": [4_096, 16_384, 65_536],
        "relative_bumps": [0.002, 0.005, 0.01],
        "seeds": [8101, 8102, 8103, 8104, 8105],
        "beta_modes": ["estimated", "unit"],
        "moneyness": [0.0],
        "maturity_days": [7.0, 30.0],
        "anchor_bump": 0.005,
    },
    "paper": {
        "paths": [16_384, 65_536, 131_072],
        "relative_bumps": [0.001, 0.002, 0.005, 0.01],
        "seeds": [8101, 8102, 8103, 8104, 8105, 8106, 8107, 8108],
        "beta_modes": ["estimated", "unit"],
        "moneyness": [-0.05, 0.0, 0.05],
        "maturity_days": [7.0, 30.0, 90.0],
        "anchor_bump": 0.005,
    },
}


def _power_of_two(value):
    value = int(value)
    return value > 1 and value & (value - 1) == 0


def _bump_key(value):
    return f"{float(value):.10g}"


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Converge European arithmetic-basket QMC Greeks jointly in path "
            "count, finite-difference bump and control-variate coefficient."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--paths", nargs="+", type=int, default=None)
    parser.add_argument("--relative-bumps", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--beta-modes",
        nargs="+",
        choices=("estimated", "unit"),
        default=None,
    )
    parser.add_argument("--moneyness", nargs="+", type=float, default=None)
    parser.add_argument("--maturity-days", nargs="+", type=float, default=None)
    parser.add_argument("--anchor-bump", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/european_arithmetic_reference_convergence.json"),
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    paths = sorted(set(args.paths or profile["paths"]))
    bumps = sorted(set(args.relative_bumps or profile["relative_bumps"]))
    seeds = args.seeds or profile["seeds"]
    beta_modes = args.beta_modes or profile["beta_modes"]
    moneyness = args.moneyness or profile["moneyness"]
    maturity_days = args.maturity_days or profile["maturity_days"]
    anchor_bump = (
        float(profile["anchor_bump"])
        if args.anchor_bump is None
        else float(args.anchor_bump)
    )
    if not all(_power_of_two(value) for value in paths):
        parser.error("every path count must be a power of two greater than one")
    if not bumps or any(not 0.0 < value < 1.0 for value in bumps):
        parser.error("relative bumps must lie in (0, 1)")
    if anchor_bump not in bumps:
        parser.error("anchor bump must be present in --relative-bumps")
    if len(seeds) < 2:
        parser.error("at least two scrambles are required to estimate uncertainty")
    if args.batch_size <= 0:
        parser.error("batch size must be positive")

    config = PaperConfig()
    points, m_values, t_values = deterministic_moneyness_panel(
        config,
        moneyness,
        maturity_days,
    )
    risk_columns = tuple(range(config.n_assets))
    results = {
        "schema_version": 2,
        "experiment": "european_arithmetic_reference_convergence",
        "product": "european_arithmetic_basket_put",
        "profile": args.profile,
        "design": {
            "market_parameters": points.tolist(),
            "log_moneyness": m_values.tolist(),
            "maturity_days": t_values.tolist(),
            "risk_columns": list(risk_columns),
            "cross_gamma_pairs": [
                [left, right]
                for left in risk_columns
                for right in risk_columns
                if left < right
            ],
        },
        "settings": {
            "path_counts": paths,
            "relative_bumps": bumps,
            "seeds": list(seeds),
            "control_variate_beta_modes": list(beta_modes),
            "anchor": {
                "path_count": max(paths),
                "relative_bump": anchor_bump,
            },
        },
        "interpretation": {
            "estimated": "sample-optimal beta is recomputed at every market scenario",
            "unit": "beta=1 is fixed across the complete finite-difference stencil",
            "hessian_uncertainty": (
                "Frobenius norm of componentwise standard errors; covariance "
                "between Hessian entries is not assumed"
            ),
        },
        "runs": {},
        "comparisons": {},
    }
    means = {}

    for beta_mode in beta_modes:
        results["runs"][beta_mode] = {}
        means[beta_mode] = {}
        for n_paths in paths:
            path_key = str(int(n_paths))
            results["runs"][beta_mode][path_key] = {}
            means[beta_mode][path_key] = {}
            for bump in bumps:
                replications = []
                start = perf_counter()
                for seed in seeds:
                    pricer = EuropeanArithmeticBasketQMC(
                        config,
                        n_paths=n_paths,
                        seed=seed,
                        control_variate=True,
                        control_variate_beta=beta_mode,
                        batch_size=args.batch_size,
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
                key = _bump_key(bump)
                means[beta_mode][path_key][key] = mean
                results["runs"][beta_mode][path_key][key] = {
                    "path_count": int(n_paths),
                    "relative_bump": float(bump),
                    "scramble_count": len(seeds),
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
                    f"beta={beta_mode} paths={n_paths} bump={bump:g} complete"
                )

    anchor_path = str(max(paths))
    anchor_key = _bump_key(anchor_bump)
    for beta_mode in beta_modes:
        anchor = means[beta_mode][anchor_path][anchor_key]
        for n_paths in paths:
            path_key = str(n_paths)
            for bump in bumps:
                key = _bump_key(bump)
                run = results["runs"][beta_mode][path_key][key]
                run["error_against_mode_anchor"] = component_metrics(
                    anchor,
                    means[beta_mode][path_key][key],
                )
                previous = [value for value in paths if value < n_paths]
                if previous:
                    previous_key = str(max(previous))
                    run["error_against_previous_path_count"] = component_metrics(
                        means[beta_mode][previous_key][key],
                        means[beta_mode][path_key][key],
                    )

    if "estimated" in beta_modes and "unit" in beta_modes:
        estimated = means["estimated"][anchor_path][anchor_key]
        unit = means["unit"][anchor_path][anchor_key]
        results["comparisons"]["unit_against_estimated_at_anchor"] = (
            component_metrics(estimated, unit)
        )

    results["acceptance_targets"] = {
        "price_normalized_standard_error": 0.005,
        "delta_normalized_standard_error": 0.02,
        "hessian_normalized_standard_error_frobenius": 0.15,
        "note": (
            "Targets are diagnostics, not automatic proof of convergence; "
            "bump stability and successive-path differences must also flatten."
        ),
    }
    _write(args.output, results)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
