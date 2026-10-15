from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from audit_price_greek_interpolant_consistency import (
    LOCAL_THRESHOLDS,
    _evaluate_mode,
    _record,
)
from validate_dense_greek_profiles import (
    SPOT_PANELS,
    analytical_components,
    dense_market_panel,
)
from validate_full_spot_hessian import _component_labels, _diagnostics
from validate_tt_greek_convergence import (
    ACCEPTANCE_THRESHOLDS,
    _power_of_two,
    _scalar_summary,
)

from stngpr.config import PaperConfig
from stngpr.convergence import (
    components_to_json,
    finite_difference_component_arrays,
)
from stngpr.coordinates import TransformedPricer
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk_grids import build_greek_coordinate_grid
from stngpr.tt_surrogate import TTPriceSurrogate

PROFILES = {
    "smoke": {
        "budget": 5_000,
        "anova_samples": 300,
        "tt_seeds": [20260401],
        "truncations": [0.0, 1e-10, 1e-8],
        "panels": ["oos_mid"],
        "maturity_days": [30],
        "moneyness": [-0.05, 0.0, 0.05],
    },
    "intermediate": {
        "budget": 100_000,
        "anova_samples": 1_000,
        "tt_seeds": [20260401, 20260402],
        "truncations": [0.0, 1e-12, 1e-10, 1e-9, 1e-8],
        "panels": ["oos_mid"],
        "maturity_days": [7, 30],
        "moneyness": [-0.1, -0.05, 0.0, 0.05, 0.1],
    },
    "paper": {
        "budget": 200_000,
        "anova_samples": 2_000,
        "tt_seeds": [20260401, 20260402, 20260403, 20260404, 20260405],
        "truncations": [
            0.0,
            1e-14,
            1e-12,
            1e-11,
            1e-10,
            1e-9,
            1e-8,
        ],
        "panels": ["oos_mid"],
        "maturity_days": [7, 30, 90],
        "moneyness": [-0.1, -0.05, 0.0, 0.05, 0.1],
    },
}


def _truncation_label(value):
    return "untruncated" if float(value) == 0.0 else f"{float(value):.0e}"


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _record_summary(record):
    global_error = record["global_error_against_analytical"]
    hessian = record["full_hessian_error_against_analytical"]
    local = record["local_error_audit"]
    return {
        "price_normalized_mae": global_error["price"]["normalized_mae"],
        "delta_normalized_mae": global_error["delta"]["normalized_mae"],
        "gamma_diagonal_normalized_mae": global_error["gamma_diagonal"][
            "normalized_mae"
        ],
        "cross_gamma_normalized_mae": global_error["cross_gamma"]["normalized_mae"],
        "hessian_relative_frobenius_error": hessian[
            "aggregate_relative_frobenius_error"
        ],
        "negative_part_relative_frobenius_norm": hessian[
            "negative_part_relative_frobenius_norm"
        ],
        **local,
        "wall_time_seconds": record["wall_time_seconds"],
        "effective_rank": record["core_diagnostics"]["effective_rank"],
        "parameter_count": record["core_diagnostics"]["parameter_count"],
        "parameter_retention_fraction": record["parameter_retention_fraction"],
    }


def _summarize_screening(payload, labels, seeds):
    output = {}
    for label in labels:
        records = [
            payload["screening_runs"].get(str(seed), {}).get("variants", {}).get(label)
            for seed in seeds
        ]
        records = [record for record in records if record is not None]
        summaries = [_record_summary(record) for record in records]
        complete = len(records) == len(seeds)
        robust = complete and all(
            record["greek_aware_local_acceptance"]["passed"] for record in records
        )
        output[label] = {
            "run_count": len(records),
            "expected_run_count": len(seeds),
            "complete": complete,
            "robustly_accepted": robust,
            "metrics": (
                {
                    name: _scalar_summary([summary[name] for summary in summaries])
                    for name in summaries[0]
                }
                if summaries
                else {}
            ),
        }
    feasible = [
        (label, record)
        for label, record in output.items()
        if record["robustly_accepted"]
    ]
    selected = min(
        feasible,
        key=lambda item: (
            item[1]["metrics"]["parameter_retention_fraction"]["mean"],
            item[1]["metrics"]["hessian_relative_frobenius_error"]["maximum"],
        ),
        default=None,
    )
    return {
        "selection_rule": (
            "most compressed tolerance passing every global and local "
            "Greek-aware criterion for every requested seed"
        ),
        "candidates": output,
        "selected_truncation": None if selected is None else selected[0],
    }


def _certification_summary(payload, labels, seeds):
    output = {}
    for label in labels:
        records = payload["certification"].get(label, {}).get("seeds", {})
        complete = all(str(seed) in records for seed in seeds)
        robust = complete and all(
            records[str(seed)]["greek_aware_local_acceptance"]["passed"]
            for seed in seeds
        )
        output[label] = {
            "run_count": len(records),
            "expected_run_count": len(seeds),
            "complete": complete,
            "robustly_accepted": robust,
        }
    feasible = [label for label in labels if output[label]["robustly_accepted"]]
    selected = feasible[0] if feasible else None
    return {
        "candidate_order": labels,
        "candidates": output,
        "selected_truncation": selected,
    }


def _cache_manifest(cache_dir, signature, overwrite):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "manifest.json"
    if path.exists() and not overwrite:
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != signature:
            raise RuntimeError(
                "the raw-core cache is incompatible; use another cache "
                "directory or pass --overwrite"
            )
    else:
        path.write_text(json.dumps(signature, indent=2), encoding="utf-8")


def _load_or_fit(
    grid,
    model_oracle,
    seed,
    budget,
    anova_samples,
    max_sweeps,
    rank_increment,
    cross_log,
    cache_dir,
    force_refit,
):
    path = cache_dir / f"seed_{seed}__budget_{budget}.npz"
    model = TTPriceSurrogate(grid, model_oracle, seed=seed)
    if path.exists() and not force_refit:
        diagnostics = model.load_untruncated_cores(path)
        return model, {
            "source": "raw_core_cache",
            "core_diagnostics": asdict(diagnostics),
        }
    start = perf_counter()
    fit = model.fit(
        budget,
        anova_samples=anova_samples,
        max_sweeps=max_sweeps,
        rank_increment=rank_increment,
        truncation=None,
        log=cross_log,
    )
    model.save_untruncated_cores(path)
    return model, {
        "source": "new_fit",
        "fit": {**asdict(fit), "total_time_seconds": perf_counter() - start},
        "core_diagnostics": asdict(model.apply_truncation(None)),
    }


def _evaluate_tt(
    model,
    truncation,
    mode,
    transform,
    points,
    risk_columns,
    relative_bump,
    analytical,
    component_labels,
    m_values,
    slices,
    batch_size,
):
    raw_diagnostics = model.apply_truncation(None)
    core_diagnostics = model.apply_truncation(truncation)

    def interpolator(model_points, cubic_columns=(), fitted=model):
        return fitted.predict_hybrid_cubic(
            model_points,
            cubic_columns=cubic_columns,
            batch_size=batch_size,
        )

    start = perf_counter()
    estimate = _evaluate_mode(
        mode,
        interpolator,
        transform,
        points,
        risk_columns,
        relative_bump,
    )
    record = _record(
        analytical,
        estimate,
        component_labels,
        m_values,
        slices,
        perf_counter() - start,
        mode,
    )
    record.update(
        {
            "truncation": None if float(truncation) == 0.0 else float(truncation),
            "core_diagnostics": asdict(core_diagnostics),
            "rank_retention_fraction": (
                core_diagnostics.effective_rank / raw_diagnostics.effective_rank
            ),
            "parameter_retention_fraction": (
                core_diagnostics.parameter_count / raw_diagnostics.parameter_count
            ),
        }
    )
    return record


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select a TT truncation with global and local Greek criteria, "
            "then certify it using one unified scalar cubic interpolant."
        )
    )
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--tt-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--truncations", nargs="+", type=float, default=None)
    parser.add_argument("--anova-samples", type=int, default=None)
    parser.add_argument("--relative-bump", type=float, default=0.002)
    parser.add_argument("--spot-nodes", type=int, default=64)
    parser.add_argument("--coordinate-nodes", type=int, default=512)
    parser.add_argument("--maturity-nodes", type=int, default=64)
    parser.add_argument("--screen-batch-size", type=int, default=16)
    parser.add_argument("--unified-batch-size", type=int, default=1)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--rank-increment", type=int, default=2)
    parser.add_argument("--cross-log", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--core-cache-dir",
        type=Path,
        default=Path("results/greeks_v97_raw_cores"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/greeks_v97_greek_aware_truncation.json"),
    )
    args = parser.parse_args()

    design = PROFILES[args.profile]
    budget = int(args.budget or design["budget"])
    anova_samples = int(args.anova_samples or design["anova_samples"])
    seeds = list(dict.fromkeys(args.tt_seeds or design["tt_seeds"]))
    truncations = sorted(set(args.truncations or design["truncations"]))
    if budget <= 0 or anova_samples <= 0:
        parser.error("budget and anova-samples must be positive")
    if not seeds:
        parser.error("at least one TT seed is required")
    if any(value < 0.0 for value in truncations) or 0.0 not in truncations:
        parser.error("truncations must be non-negative and include zero")
    if not 0.0 < args.relative_bump < 1.0:
        parser.error("relative-bump must lie in (0,1)")
    for name, value in (
        ("spot-nodes", args.spot_nodes),
        ("coordinate-nodes", args.coordinate_nodes),
        ("maturity-nodes", args.maturity_nodes),
    ):
        if not _power_of_two(value):
            parser.error(f"{name} must be a power of two greater than one")

    base = PaperConfig()
    shape = list(base.physical_shape)
    shape[: base.n_assets] = [int(args.spot_nodes)] * base.n_assets
    shape[base.n_assets] = int(args.coordinate_nodes)
    shape[-1] = int(args.maturity_nodes)
    config = replace(base, physical_shape=tuple(shape))
    risk_columns = tuple(range(config.n_assets))
    component_labels = _component_labels(risk_columns)
    points, m_values, t_values, panel_labels, slices, rate = dense_market_panel(
        config,
        design["panels"],
        design["maturity_days"],
        np.asarray(design["moneyness"], dtype=float),
        args.relative_bump,
    )
    market_pricer = EuropeanGeometricBasketPricer(config)
    analytical = analytical_components(
        config,
        market_pricer,
        points,
        risk_columns,
    )
    exact_fd = finite_difference_component_arrays(
        market_pricer,
        points,
        risk_columns,
        args.relative_bump,
    )
    grid, transform, coordinate_description = build_greek_coordinate_grid(
        config,
        "bounded_standardized_risk",
    )
    model_oracle = TransformedPricer(market_pricer, transform)
    labels = [_truncation_label(value) for value in truncations]
    label_to_value = dict(zip(labels, truncations, strict=True))
    signature = {
        "profile": args.profile,
        "physical_shape": list(config.physical_shape),
        "budget": budget,
        "anova_samples": anova_samples,
        "relative_bump": float(args.relative_bump),
        "moneyness": list(design["moneyness"]),
        "maturity_days": list(design["maturity_days"]),
        "panels": list(design["panels"]),
        "max_sweeps": int(args.max_sweeps),
        "rank_increment": int(args.rank_increment),
    }
    _cache_manifest(args.core_cache_dir, signature, args.overwrite)

    if args.output.exists() and not args.overwrite:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != "greeks-greek-aware-truncation-v9.7"
            or payload.get("experiment_signature") != signature
        ):
            parser.error(
                "the output checkpoint is incompatible; choose another path "
                "or pass --overwrite"
            )
        print(f"Resuming checkpoint {args.output}")
    else:
        payload = {
            "schema_version": "greeks-greek-aware-truncation-v9.7",
            "profile": args.profile,
            "hypothesis": (
                "the largest price-norm truncation passing local Greek criteria "
                "can be certified across seeds using one scalar cubic surface"
            ),
            "experiment_signature": signature,
            "settings": {
                "physical_shape": list(config.physical_shape),
                "qtt_core_count": int(sum(config.qtt_bits)),
                "logical_tensor_entries": int(np.prod(config.physical_shape)),
                "coordinate_mode": "bounded_standardized_risk",
                "screening_interpolation": "component_hybrid",
                "certification_interpolation": "unified_risk_cubic",
                "budget": budget,
                "tt_seeds": seeds,
                "truncations": truncations,
                "anova_samples": anova_samples,
                "relative_bump": float(args.relative_bump),
                "rate": rate,
                "monte_carlo": False,
            },
            "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
            "local_acceptance_thresholds": LOCAL_THRESHOLDS,
            "test_design": {
                "market_parameters": points.tolist(),
                "log_moneyness": m_values.tolist(),
                "maturity_days": t_values.tolist(),
                "spot_panel_labels": panel_labels.tolist(),
                "curve_slices": slices,
                "spot_panels": {name: SPOT_PANELS[name] for name in design["panels"]},
                "component_labels": component_labels,
            },
            "coordinate_description": coordinate_description,
            "analytical_reference": {"components": components_to_json(analytical)},
            "exact_price_finite_difference_control": {
                "components": components_to_json(exact_fd),
                **_diagnostics(
                    analytical,
                    exact_fd,
                    component_labels,
                    m_values,
                    slices,
                ),
            },
            "screening_runs": {},
            "screening_selection": {},
            "certification": {},
            "final_selection": {},
        }
        _write(args.output, payload)

    payload["settings"].update(
        {
            "tt_seeds": seeds,
            "truncations": truncations,
        }
    )
    models = {}
    for seed in seeds:
        model, fit_record = _load_or_fit(
            grid,
            model_oracle,
            seed,
            budget,
            anova_samples,
            args.max_sweeps,
            args.rank_increment,
            args.cross_log,
            args.core_cache_dir,
            args.overwrite,
        )
        models[seed] = model
        run = payload["screening_runs"].setdefault(
            str(seed),
            {"tt_seed": int(seed), "fit": fit_record, "variants": {}},
        )
        for truncation, label in zip(truncations, labels, strict=True):
            if label in run["variants"]:
                continue
            print(f"screening seed={seed} truncation={label}")
            run["variants"][label] = _evaluate_tt(
                model,
                truncation,
                "component_hybrid",
                transform,
                points,
                risk_columns,
                args.relative_bump,
                analytical,
                component_labels,
                m_values,
                slices,
                args.screen_batch_size,
            )
            _write(args.output, payload)

    payload["screening_selection"] = _summarize_screening(
        payload,
        labels,
        seeds,
    )
    robust_screen = [
        label
        for label, record in payload["screening_selection"]["candidates"].items()
        if record["robustly_accepted"]
    ]
    robust_screen.sort(
        key=lambda label: payload["screening_selection"]["candidates"][label][
            "metrics"
        ]["parameter_retention_fraction"]["mean"]
    )
    certified_labels = []
    selected = None
    for label in robust_screen:
        certified_labels.append(label)
        truncation = label_to_value[label]
        certification = payload["certification"].setdefault(
            label,
            {"truncation": truncation, "seeds": {}},
        )
        for seed in seeds:
            if str(seed) in certification["seeds"]:
                continue
            print(f"certifying seed={seed} truncation={label}")
            certification["seeds"][str(seed)] = _evaluate_tt(
                models[seed],
                truncation,
                "unified_risk_cubic",
                transform,
                points,
                risk_columns,
                args.relative_bump,
                analytical,
                component_labels,
                m_values,
                slices,
                args.unified_batch_size,
            )
            _write(args.output, payload)
        summary = _certification_summary(payload, certified_labels, seeds)
        if summary["candidates"][label]["robustly_accepted"]:
            selected = label
            break

    if not robust_screen:
        label = "untruncated"
        certification = payload["certification"].setdefault(
            label,
            {
                "truncation": 0.0,
                "seeds": {},
                "diagnostic_only": True,
            },
        )
        anchor = seeds[0]
        if str(anchor) not in certification["seeds"]:
            print(f"diagnostic certification seed={anchor} truncation={label}")
            certification["seeds"][str(anchor)] = _evaluate_tt(
                models[anchor],
                0.0,
                "unified_risk_cubic",
                transform,
                points,
                risk_columns,
                args.relative_bump,
                analytical,
                component_labels,
                m_values,
                slices,
                args.unified_batch_size,
            )

    boundary = None
    if selected is not None:
        selected_value = label_to_value[selected]
        rejected = [
            label
            for label in labels
            if label_to_value[label] > selected_value
            and (
                not payload["screening_selection"]["candidates"][label][
                    "robustly_accepted"
                ]
                or (
                    label in certified_labels
                    and not _certification_summary(
                        payload,
                        certified_labels,
                        seeds,
                    )["candidates"][label]["robustly_accepted"]
                )
            )
        ]
        if rejected:
            boundary = min(rejected, key=lambda label: label_to_value[label])
            certification = payload["certification"].setdefault(
                boundary,
                {
                    "truncation": label_to_value[boundary],
                    "seeds": {},
                    "boundary_rejection_control": True,
                },
            )
            anchor = seeds[0]
            if str(anchor) not in certification["seeds"]:
                print(f"boundary control seed={anchor} truncation={boundary}")
                certification["seeds"][str(anchor)] = _evaluate_tt(
                    models[anchor],
                    label_to_value[boundary],
                    "unified_risk_cubic",
                    transform,
                    points,
                    risk_columns,
                    args.relative_bump,
                    analytical,
                    component_labels,
                    m_values,
                    slices,
                    args.unified_batch_size,
                )

    payload["final_selection"] = {
        "selection_rule": (
            "most compressed screening candidate that also passes the unified "
            "scalar-surface certification for every requested seed"
        ),
        "certified_candidate_order": certified_labels,
        "selected_truncation": selected,
        "boundary_rejection_control": boundary,
        "screening_is_final": False,
    }
    payload["certification_summary"] = _certification_summary(
        payload,
        certified_labels,
        seeds,
    )
    _write(args.output, payload)
    print(f"Selected truncation: {selected}")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
