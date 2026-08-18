from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import PaperConfig
from stngpr.coordinates import (
    GRID_MODES,
    MarketCoordinatePricer,
    TransformedPricer,
    build_coordinate_grid,
    oracle_multilinear_predict,
)
from stngpr.greeks import (
    finite_difference_greeks,
    geometric_basket_put_spot_greeks,
)
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.tt_surrogate import TTPriceSurrogate


PROFILES = {
    "smoke": {"n_points": 12, "budget": 5_000, "anova_samples": 500},
    "intermediate": {
        "n_points": 60,
        "budget": 20_000,
        "anova_samples": 2_000,
    },
    "paper": {
        "n_points": 200,
        "budget": 50_000,
        "anova_samples": 2_000,
    },
}

DEFAULT_BUMPS = (1e-1, 5e-2, 3e-2, 2e-2, 1e-2, 5e-3)


def validation_points(config, n_points: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Build an interior design spanning moneyness and maturity regimes."""
    m_levels = np.array([-0.35, -0.15, -0.05, 0.0, 0.05, 0.15, 0.35])
    maturity_levels = np.array([7.0 / 365.0, 30.0 / 365.0, 0.25, 0.5, 1.0, 2.5])
    index = np.arange(n_points)
    log_moneyness = m_levels[index % m_levels.size]
    maturities = maturity_levels[(5 * index) % maturity_levels.size]

    spots = rng.uniform(45.0, 115.0, size=(n_points, config.n_assets))
    basket = np.exp(np.mean(np.log(spots), axis=1))
    strikes = basket * np.exp(log_moneyness)
    rates = rng.uniform(0.01, 0.06, size=n_points)
    points = np.column_stack((spots, strikes, rates, maturities))

    for column, (lower, upper) in enumerate(config.bounds):
        if np.any(points[:, column] <= lower) or np.any(points[:, column] >= upper):
            raise RuntimeError("validation design must remain inside grid bounds")
    return points, log_moneyness


def _cell_index(axis: np.ndarray, value: float) -> int:
    index = int(np.searchsorted(axis, value, side="right") - 1)
    return int(np.clip(index, 0, axis.size - 2))


def grid_resolution_diagnostics(
    grid,
    transform,
    points,
    relative_bumps,
    n_assets,
):
    model_points = transform.to_model(points)
    relative_widths = []
    for j in range(n_assets):
        axis = grid.axes[j]
        values = model_points[:, j]
        upper = np.searchsorted(axis, values, side="right")
        upper = np.clip(upper, 1, axis.size - 1)
        lower = upper - 1
        relative_widths.extend(
            ((axis[upper] - axis[lower]) / values).tolist()
        )

    crossing = {}
    for relative_bump in relative_bumps:
        spot_crossings = 0
        moneyness_crossings = 0
        either_crossings = 0
        total = 0
        for point in points:
            base = transform.to_model(point)[0]
            for j in range(n_assets):
                up, down = point.copy(), point.copy()
                up[j] *= 1.0 + relative_bump
                down[j] *= 1.0 - relative_bump
                model_up = transform.to_model(up)[0]
                model_down = transform.to_model(down)[0]

                spot_base = _cell_index(grid.axes[j], base[j])
                spot_cross = (
                    _cell_index(grid.axes[j], model_up[j]) != spot_base
                    or _cell_index(grid.axes[j], model_down[j]) != spot_base
                )
                if transform.use_moneyness:
                    m_column = n_assets
                    m_base = _cell_index(
                        grid.axes[m_column],
                        base[m_column],
                    )
                    m_cross = (
                        _cell_index(
                            grid.axes[m_column],
                            model_up[m_column],
                        )
                        != m_base
                        or _cell_index(
                            grid.axes[m_column],
                            model_down[m_column],
                        )
                        != m_base
                    )
                else:
                    m_cross = False

                spot_crossings += int(spot_cross)
                moneyness_crossings += int(m_cross)
                either_crossings += int(spot_cross or m_cross)
                total += 1

        crossing[f"{relative_bump:.10g}"] = {
            "relative_bump": float(relative_bump),
            "stencil_count": total,
            "spot_axis_crossing_fraction": spot_crossings / total,
            "moneyness_axis_crossing_fraction": moneyness_crossings / total,
            "either_axis_crossing_fraction": either_crossings / total,
        }

    widths = np.asarray(relative_widths)
    return {
        "spot_cell_relative_width": {
            "mean": float(np.mean(widths)),
            "median": float(np.median(widths)),
            "minimum": float(np.min(widths)),
            "maximum": float(np.max(widths)),
        },
        "central_stencil_cell_crossing": crossing,
    }


def error_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict:
    reference = np.asarray(reference, dtype=float).reshape(-1)
    estimate = np.asarray(estimate, dtype=float).reshape(-1)
    error = estimate - reference
    absolute = np.abs(error)
    reference_scale = float(np.mean(np.abs(reference)))
    mae = float(np.mean(absolute))
    return {
        "count": int(reference.size),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_absolute_error": float(np.max(absolute)),
        "mean_error_bias": float(np.mean(error)),
        "mean_absolute_reference": reference_scale,
        "normalized_mae": (
            mae / reference_scale if reference_scale > 1e-14 else None
        ),
    }


def analytical_references(config, market_pricer, points: np.ndarray) -> dict:
    n_assets = config.n_assets
    delta, gamma = geometric_basket_put_spot_greeks(
        points[:, :n_assets],
        points[:, n_assets],
        points[:, n_assets + 1],
        points[:, n_assets + 2],
        config.volatilities,
        config.correlation,
        config.dividends,
    )
    diagonal = np.column_stack(
        [gamma[:, i, i] for i in range(n_assets)]
    )
    pairs = [
        (i, j)
        for i in range(n_assets)
        for j in range(i + 1, n_assets)
    ]
    cross = np.column_stack([gamma[:, i, j] for i, j in pairs])
    return {
        "price": np.asarray(market_pricer(points), dtype=float),
        "delta": delta,
        "gamma_diagonal": diagonal,
        "cross_gamma": cross,
        "cross_gamma_pairs": pairs,
    }


def finite_difference_arrays(pricer, points, n_assets, relative_bump):
    records = [
        finite_difference_greeks(
            pricer,
            point,
            spot_columns=range(n_assets),
            relative_bump=relative_bump,
        )
        for point in points
    ]
    pairs = [
        (i, j)
        for i in range(n_assets)
        for j in range(i + 1, n_assets)
    ]
    return {
        "price": np.array([record["price"] for record in records]),
        "delta": np.array([
            [record["delta"][i] for i in range(n_assets)]
            for record in records
        ]),
        "gamma_diagonal": np.array([
            [record["gamma"][(i, i)] for i in range(n_assets)]
            for record in records
        ]),
        "cross_gamma": np.array([
            [record["gamma"][pair] for pair in pairs]
            for record in records
        ]),
    }


def component_metrics(reference: dict, estimate: dict, mask=None) -> dict:
    if mask is None:
        mask = np.ones(reference["price"].shape[0], dtype=bool)
    return {
        name: error_metrics(reference[name][mask], estimate[name][mask])
        for name in ("price", "delta", "gamma_diagonal", "cross_gamma")
    }


def regional_metrics(reference, estimate, log_moneyness, maturities):
    regions = {
        "by_moneyness": {
            "put_otm": log_moneyness < -0.1,
            "atm": np.abs(log_moneyness) <= 0.1,
            "put_itm": log_moneyness > 0.1,
        },
        "by_maturity": {
            "short": maturities <= 0.25,
            "medium": (maturities > 0.25) & (maturities <= 1.0),
            "long": maturities > 1.0,
        },
    }
    output = {}
    for family, masks in regions.items():
        output[family] = {
            name: component_metrics(reference, estimate, mask)
            for name, mask in masks.items()
            if np.any(mask)
        }
    return output


def convergence_score(metrics: dict) -> float:
    values = []
    for component in ("delta", "gamma_diagonal", "cross_gamma"):
        value = metrics[component]["normalized_mae"]
        if value is not None:
            values.append(value)
    return float(sum(values))


def evaluate_bumps(
    stage_name,
    pricer,
    points,
    log_moneyness,
    references,
    relative_bumps,
    n_assets,
):
    output = {}
    maturities = points[:, n_assets + 2]
    for relative_bump in relative_bumps:
        start = perf_counter()
        estimate = finite_difference_arrays(
            pricer,
            points,
            n_assets,
            relative_bump,
        )
        elapsed = perf_counter() - start
        global_metrics = component_metrics(references, estimate)
        key = f"{relative_bump:.10g}"
        output[key] = {
            "relative_bump": float(relative_bump),
            "wall_time_seconds": elapsed,
            "global": global_metrics,
            "regions": regional_metrics(
                references,
                estimate,
                log_moneyness,
                maturities,
            ),
            "selection_score": convergence_score(global_metrics),
        }
        print(
            f"{stage_name:>5s}  h/S={relative_bump:7.4f}  "
            f"Delta MAE={global_metrics['delta']['mae']:.3e}  "
            f"Gamma MAE={global_metrics['gamma_diagonal']['mae']:.3e}  "
            f"Cross MAE={global_metrics['cross_gamma']['mae']:.3e}"
        )
    best_key = min(output, key=lambda key: output[key]["selection_score"])
    return {
        "best_relative_bump_by_normalized_mae_sum": output[best_key][
            "relative_bump"
        ],
        "results": output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--stage",
        choices=("exact", "grid", "tt", "all"),
        default="exact",
    )
    parser.add_argument(
        "--grid-mode",
        choices=GRID_MODES,
        default="moneyness_adaptive",
    )
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--anova-samples", type=int, default=None)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument(
        "--relative-bumps",
        nargs="+",
        type=float,
        default=DEFAULT_BUMPS,
    )
    parser.add_argument("--cross-log", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if any(bump <= 0.0 or bump >= 1.0 for bump in args.relative_bumps):
        parser.error("relative bumps must lie in (0, 1)")

    profile = PROFILES[args.profile]
    n_points = args.n_points or profile["n_points"]
    budget = args.budget or profile["budget"]
    anova_samples = args.anova_samples or profile["anova_samples"]
    output = args.output or Path(
        f"results/european_greeks_{args.stage}_{args.grid_mode}_{args.profile}.json"
    )

    config = PaperConfig()
    rng = np.random.default_rng(config.seed + 101)
    points, log_moneyness = validation_points(config, n_points, rng)
    grid, transform, grid_description = build_coordinate_grid(
        config,
        args.grid_mode,
        basket_kind="geometric",
    )
    market_pricer = EuropeanGeometricBasketPricer(config)
    model_oracle = TransformedPricer(market_pricer, transform)
    references = analytical_references(config, market_pricer, points)

    results = {
        "profile": args.profile,
        "stage": args.stage,
        "seed": config.seed + 101,
        "n_points": n_points,
        "relative_bumps": [float(value) for value in args.relative_bumps],
        "test_design": {
            "market_parameters": points.tolist(),
            "log_moneyness": log_moneyness.tolist(),
        },
        "assumptions": {
            "n_assets": config.n_assets,
            "volatilities": config.volatilities.tolist(),
            "correlation": config.correlation.tolist(),
            "dividends": config.dividends.tolist(),
            "coordinate_grid": grid_description,
        },
        "reference": "analytical geometric-basket put spot Greeks",
        "grid_resolution": grid_resolution_diagnostics(
            grid,
            transform,
            points,
            args.relative_bumps,
            config.n_assets,
        ),
    }

    if args.stage in ("exact", "all"):
        exact_market_adapter = MarketCoordinatePricer(model_oracle, transform)
        results["exact"] = evaluate_bumps(
            "exact",
            exact_market_adapter,
            points,
            log_moneyness,
            references,
            args.relative_bumps,
            config.n_assets,
        )

    if args.stage in ("grid", "all"):
        def grid_model_pricer(model_parameters):
            return oracle_multilinear_predict(
                grid,
                model_oracle,
                model_parameters,
            )

        grid_market_adapter = MarketCoordinatePricer(
            grid_model_pricer,
            transform,
        )
        results["grid"] = evaluate_bumps(
            "grid",
            grid_market_adapter,
            points,
            log_moneyness,
            references,
            args.relative_bumps,
            config.n_assets,
        )

    if args.stage in ("tt", "all"):
        model = TTPriceSurrogate(grid, model_oracle, seed=config.seed)
        fit_start = perf_counter()
        diagnostics = model.fit(
            budget,
            anova_samples=anova_samples,
            log=args.cross_log,
        )
        fit_total_time = perf_counter() - fit_start
        market_surrogate = MarketCoordinatePricer(model.predict, transform)
        tt_results = evaluate_bumps(
            "tt",
            market_surrogate,
            points,
            log_moneyness,
            references,
            args.relative_bumps,
            config.n_assets,
        )
        tt_results["fit"] = {
            "budget": budget,
            "anova_samples": anova_samples,
            "total_time_seconds": fit_total_time,
            "cross_time_seconds": diagnostics.wall_time,
            "function_evaluations": diagnostics.function_evaluations,
            "sweeps": diagnostics.sweeps,
            "stop": diagnostics.stop,
            "effective_rank": diagnostics.effective_rank,
        }
        results["tt"] = tt_results

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
