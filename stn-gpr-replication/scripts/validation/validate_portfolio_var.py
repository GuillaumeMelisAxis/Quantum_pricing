"""Monte-Carlo VaR of a basket-option book under full grid revaluation.

The closed-form pricer supplies the trusted loss distribution. The same
scenarios are then repriced with the grid interpolation oracle and with the TT
surrogate, which separates the irreducible off-grid interpolation error from the
TT-cross reconstruction error in a risk metric rather than in a price metric.
"""

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
from stngpr.portfolio import Portfolio
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk import (
    MonteCarloVaREngine,
    compare_loss_distributions,
    domain_coverage,
)
from stngpr.scenarios import GBMScenarioGenerator
from stngpr.tt_surrogate import TTPriceSurrogate


PROFILES = {
    "smoke": {"scenarios": 5_000, "budget": 5_000, "anova": 500},
    "intermediate": {"scenarios": 50_000, "budget": 20_000, "anova": 2_000},
    "paper": {"scenarios": 250_000, "budget": 100_000, "anova": 2_000},
}

BASE_SPOTS = (95.0, 100.0, 105.0, 90.0, 110.0)
LEVELS = (0.95, 0.975, 0.99)


def build_portfolio(base_basket: float) -> Portfolio:
    """Heterogeneous book: long puts financed by short shorter-dated puts.

    The one-week position is deliberate. It survives a one-day horizon and is
    dropped by a ten-day one, so the horizon sensitivity of the excluded set is
    visible in the JSON rather than assumed away.
    """
    moneyness = (0.85, 0.95, 1.00, 1.05, 1.15, 0.90, 1.00, 1.10, 1.00)
    maturities = (0.25, 0.50, 1.00, 2.00, 3.00, 0.75, 1.50, 0.50, 1.0 / 52.0)
    quantities = (100.0, -150.0, 200.0, -80.0, 60.0, 120.0, -100.0, 90.0, 140.0)
    labels = tuple(
        f"put_k{ratio:.2f}_t{maturity:.2f}"
        for ratio, maturity in zip(moneyness, maturities)
    )
    return Portfolio.from_arrays(
        strikes=[base_basket * ratio for ratio in moneyness],
        maturities=maturities,
        quantities=quantities,
        labels=labels,
    )


def break_even_revaluations(exact_result, surrogate_result, build_time: float):
    """Revaluations at which the surrogate amortizes its construction cost."""
    exact_unit = exact_result.revaluation_time / max(exact_result.pricer_evaluations, 1)
    surrogate_unit = surrogate_result.revaluation_time / max(
        surrogate_result.pricer_evaluations, 1
    )
    saving = exact_unit - surrogate_unit
    return {
        "surrogate_build_time": float(build_time),
        "exact_seconds_per_revaluation": float(exact_unit),
        "surrogate_seconds_per_revaluation": float(surrogate_unit),
        "speedup": float(exact_unit / surrogate_unit) if surrogate_unit > 0.0 else None,
        "break_even_revaluations": (
            float(build_time / saving) if saving > 0.0 else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--grid-mode", choices=GRID_MODES, default="moneyness_adaptive")
    parser.add_argument("--horizon-days", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=0.03)
    parser.add_argument(
        "--drift",
        type=float,
        default=None,
        help="Annualized expected log-return; default is the risk-neutral drift",
    )
    parser.add_argument(
        "--skip-tt",
        action="store_true",
        help="Only compare the grid interpolation oracle against the exact pricer",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    output = args.output or Path(f"results/portfolio_var_{args.grid_mode}.json")
    horizon = args.horizon_days / 252.0

    config = PaperConfig()
    grid, transform, grid_description = build_coordinate_grid(
        config, args.grid_mode, basket_kind="geometric"
    )
    market_pricer = EuropeanGeometricBasketPricer(config)
    model_pricer = TransformedPricer(market_pricer, transform)

    spots = np.asarray(BASE_SPOTS, dtype=float)
    base_basket = float(np.exp(np.mean(np.log(spots))))
    book = build_portfolio(base_basket)
    # Positions expiring inside the horizon need a payoff treatment that full
    # revaluation of a live option cannot provide, so they are excluded here and
    # reported. The VaR below is the VaR of the remaining book, not of the book.
    portfolio, expiring = book.split_at_horizon(horizon)
    labels = [trade.label for trade in portfolio.trades]

    generator = GBMScenarioGenerator.from_config(
        config,
        drift=None if args.drift is None else np.full(config.n_assets, args.drift),
    )
    engine = MonteCarloVaREngine(
        portfolio, generator, horizon=horizon, levels=LEVELS
    )

    scenario_start = perf_counter()
    scenario_spots = engine.simulate_spots(
        spots, profile["scenarios"], args.rate, seed=config.seed
    )
    scenario_time = perf_counter() - scenario_start

    scenario_parameters = engine.scenario_parameters(scenario_spots, args.rate)
    model_points = transform.to_model(
        scenario_parameters.reshape(-1, scenario_parameters.shape[-1])
    )

    exact = engine.run(
        market_pricer, spots, args.rate, scenario_spots=scenario_spots
    )

    def oracle_pricer(model_parameters):
        return oracle_multilinear_predict(grid, model_pricer, model_parameters)

    oracle = engine.run(
        MarketCoordinatePricer(oracle_pricer, transform),
        spots,
        args.rate,
        scenario_spots=scenario_spots,
    )

    results = {
        "profile": args.profile,
        "setup": {
            "base_spots": spots.tolist(),
            "base_geometric_basket": base_basket,
            "rate": args.rate,
            "horizon_days": args.horizon_days,
            "horizon_years": horizon,
            "scenarios": int(profile["scenarios"]),
            "levels": list(LEVELS),
            "portfolio": portfolio.describe(),
            "excluded_expiring_trades": [trade.describe() for trade in expiring],
            "excluded_trade_count": len(expiring),
            "coordinate_grid": grid_description,
            "scenario_generation_time": scenario_time,
            "scenario_domain_coverage": domain_coverage(model_points, grid.bounds),
        },
        "exact": exact.to_dict(labels=labels),
        "oracle_interpolation": {
            "description": "Exact grid corners, off-grid multilinear interpolation",
            "risk": oracle.to_dict(labels=labels),
            "accuracy": compare_loss_distributions(
                exact.losses, oracle.losses, levels=LEVELS
            ),
        },
    }

    if not args.skip_tt:
        surrogate = TTPriceSurrogate(grid, model_pricer, seed=config.seed)
        diagnostics = surrogate.fit(
            profile["budget"], anova_samples=profile["anova"], log=True
        )
        tt = engine.run(
            MarketCoordinatePricer(surrogate.predict, transform),
            spots,
            args.rate,
            scenario_spots=scenario_spots,
        )
        results["tt"] = {
            "budget": int(profile["budget"]),
            "function_evaluations": diagnostics.function_evaluations,
            "build_time": diagnostics.wall_time,
            "sweeps": diagnostics.sweeps,
            "stop": diagnostics.stop,
            "effective_rank": diagnostics.effective_rank,
            "risk": tt.to_dict(labels=labels),
            "accuracy_versus_exact": compare_loss_distributions(
                exact.losses, tt.losses, levels=LEVELS
            ),
            "accuracy_versus_oracle": compare_loss_distributions(
                oracle.losses, tt.losses, levels=LEVELS
            ),
            "cost": break_even_revaluations(exact, tt, diagnostics.wall_time),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
