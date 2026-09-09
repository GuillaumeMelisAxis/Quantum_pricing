from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.stats import binom, spearmanr, wasserstein_distance

from .portfolio import Portfolio
from .scenarios import GBMScenarioGenerator


DEFAULT_LEVELS = (0.95, 0.975, 0.99)


def _level_key(level: float) -> str:
    return f"{float(level):g}"


def var_es(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """Empirical VaR and ES using the loss-positive convention."""
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    var = float(np.quantile(losses, alpha, method="higher"))
    tail = losses[losses >= var]
    return var, float(np.mean(tail))


def var_es_uncertainty(
    losses: np.ndarray,
    alpha: float = 0.99,
    confidence: float = 0.95,
) -> dict:
    """Monte-Carlo uncertainty of one empirical VaR/ES pair.

    The VaR interval is the exact order-statistic interval: the number of
    scenarios below the true quantile is Binomial(n, alpha), so the binomial
    quantiles index the confidence bounds directly. The ES uncertainty is the
    usual tail-mean standard error, which is asymptotic and understates the
    error when the tail sample is small.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    n = losses.size
    var, expected_shortfall = var_es(losses, alpha)
    ordered = np.sort(losses)
    tail_probability = 0.5 * (1.0 - confidence)
    lower = int(np.clip(binom.ppf(tail_probability, n, alpha), 1, n)) - 1
    upper = int(np.clip(binom.ppf(1.0 - tail_probability, n, alpha), 1, n)) - 1

    tail = losses[losses >= var]
    if tail.size > 1:
        es_standard_error = float(np.std(tail, ddof=1) / np.sqrt(tail.size))
    else:
        es_standard_error = float("nan")
    half_width = 1.96 * es_standard_error
    return {
        "level": float(alpha),
        "confidence": float(confidence),
        "scenarios": int(n),
        "tail_scenarios": int(tail.size),
        "var": var,
        "var_ci": [float(ordered[lower]), float(ordered[upper])],
        "expected_shortfall": expected_shortfall,
        "expected_shortfall_standard_error": es_standard_error,
        "expected_shortfall_ci": [
            expected_shortfall - half_width,
            expected_shortfall + half_width,
        ],
    }


def expected_shortfall_contributions(
    losses: np.ndarray,
    trade_losses: np.ndarray,
    alpha: float = 0.99,
) -> np.ndarray:
    """Per-trade Euler decomposition of the portfolio expected shortfall.

    ES is positively homogeneous in the position vector, so the conditional mean
    of each position loss over the scenarios that drive the portfolio tail sums
    exactly to the portfolio ES.
    """
    losses = np.asarray(losses, dtype=float)
    trade_losses = np.asarray(trade_losses, dtype=float)
    if trade_losses.ndim != 2 or trade_losses.shape[0] != losses.size:
        raise ValueError("trade_losses must be (n_scenarios, n_trades)")
    var = float(np.quantile(losses, alpha, method="higher"))
    tail = losses >= var
    return np.mean(trade_losses[tail], axis=0)


@dataclass(frozen=True)
class ScenarioValuation:
    """Base and scenario values of every trade, before risk aggregation."""

    base_trade_values: np.ndarray
    scenario_trade_values: np.ndarray
    quantities: np.ndarray
    revaluation_time: float = 0.0
    pricer_evaluations: int = 0

    @property
    def base_value(self) -> float:
        return float(self.base_trade_values @ self.quantities)

    @property
    def scenario_values(self) -> np.ndarray:
        return self.scenario_trade_values @ self.quantities

    @property
    def trade_pnl(self) -> np.ndarray:
        """Position-weighted P&L of every trade, shape (n_scenarios, n_trades)."""
        return (
            self.scenario_trade_values - self.base_trade_values[None, :]
        ) * self.quantities[None, :]

    @property
    def pnl(self) -> np.ndarray:
        return self.scenario_values - self.base_value

    @property
    def losses(self) -> np.ndarray:
        return -self.pnl


@dataclass(frozen=True)
class VaRResult:
    """Monte-Carlo VaR and ES of one portfolio under one pricer."""

    base_value: float
    losses: np.ndarray
    trade_losses: np.ndarray
    horizon: float
    levels: tuple[float, ...] = DEFAULT_LEVELS
    revaluation_time: float = 0.0
    pricer_evaluations: int = 0
    scenario_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def n_scenarios(self) -> int:
        return int(np.asarray(self.losses).size)

    @property
    def pnl(self) -> np.ndarray:
        return -np.asarray(self.losses, dtype=float)

    def var(self, level: float) -> float:
        return var_es(self.losses, level)[0]

    def expected_shortfall(self, level: float) -> float:
        return var_es(self.losses, level)[1]

    def contributions(self, level: float) -> np.ndarray:
        return expected_shortfall_contributions(
            self.losses, self.trade_losses, level
        )

    def to_dict(self, labels=None, include_losses: bool = False) -> dict:
        n_trades = np.asarray(self.trade_losses).shape[1]
        if labels is None:
            labels = [f"trade_{j}" for j in range(n_trades)]
        measures = {}
        for level in self.levels:
            summary = var_es_uncertainty(self.losses, level)
            summary["expected_shortfall_contributions"] = {
                str(label): float(value)
                for label, value in zip(labels, self.contributions(level))
            }
            measures[_level_key(level)] = summary
        pnl = self.pnl
        output = {
            "base_value": self.base_value,
            "horizon_years": float(self.horizon),
            "scenarios": self.n_scenarios,
            "pricer_evaluations": int(self.pricer_evaluations),
            "scenario_time": float(self.scenario_time),
            "revaluation_time": float(self.revaluation_time),
            "revaluation_seconds_per_pricer_evaluation": (
                float(self.revaluation_time) / self.pricer_evaluations
                if self.pricer_evaluations
                else None
            ),
            "pnl": {
                "mean": float(np.mean(pnl)),
                "std": float(np.std(pnl, ddof=1)) if self.n_scenarios > 1 else 0.0,
                "min": float(np.min(pnl)),
                "max": float(np.max(pnl)),
            },
            "measures": measures,
            "metadata": dict(self.metadata),
        }
        if include_losses:
            output["losses"] = np.asarray(self.losses).tolist()
        return output


class MonteCarloVaREngine:
    """Full-revaluation Monte-Carlo VaR for a book of basket options.

    Spots are diffused to the risk horizon under a correlated GBM and every
    trade is repriced from scratch in every scenario. The pricer is any callable
    mapping market parameter rows (S_1, ..., S_d, K, r, T) to prices, so the
    closed-form pricer, the grid interpolation oracle and a TT surrogate wrapped
    in MarketCoordinatePricer are interchangeable without touching the engine.

    Scenario generation is deliberately separate from revaluation: calling
    simulate_spots once and passing the result to several run calls turns the
    comparison of a trusted pricer against a surrogate into a paired experiment
    on identical scenarios.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        generator: GBMScenarioGenerator,
        horizon: float,
        levels=DEFAULT_LEVELS,
        age_maturities: bool = True,
        minimum_residual_maturity: float = 1.0 / 365.0,
        batch_size: int = 200_000,
    ):
        if not isinstance(portfolio, Portfolio):
            raise ValueError("portfolio must be a Portfolio")
        if not isinstance(generator, GBMScenarioGenerator):
            raise ValueError("generator must be a GBMScenarioGenerator")
        horizon = float(horizon)
        if horizon <= 0.0:
            raise ValueError("horizon must be strictly positive")
        levels = tuple(float(level) for level in levels)
        if not levels or any(not 0.0 < level < 1.0 for level in levels):
            raise ValueError("every confidence level must lie in (0, 1)")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least one")

        self.portfolio = portfolio
        self.generator = generator
        self.horizon = horizon
        self.levels = levels
        self.age_maturities = bool(age_maturities)
        self.minimum_residual_maturity = float(minimum_residual_maturity)
        self.batch_size = int(batch_size)

    @property
    def maturity_shift(self) -> float:
        return self.horizon if self.age_maturities else 0.0

    def simulate_spots(
        self,
        spots: np.ndarray,
        n_scenarios: int,
        rate: float,
        seed=None,
        rng=None,
        antithetic: bool = True,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed)
        return self.generator.simulate(
            spots,
            self.horizon,
            n_scenarios,
            rate=rate,
            rng=rng,
            antithetic=antithetic,
        )

    def scenario_parameters(
        self,
        scenario_spots: np.ndarray,
        rates,
    ) -> np.ndarray:
        """Aged valuation tensor of the scenarios, for domain diagnostics."""
        return self.portfolio.market_parameters(
            scenario_spots,
            rates,
            maturity_shift=self.maturity_shift,
            minimum_residual_maturity=self.minimum_residual_maturity,
        )

    def _price(self, pricer, parameters: np.ndarray) -> tuple[np.ndarray, float]:
        flat = parameters.reshape(-1, parameters.shape[-1])
        values = np.empty(flat.shape[0], dtype=float)
        elapsed = 0.0
        for start in range(0, flat.shape[0], self.batch_size):
            batch = flat[start : start + self.batch_size]
            begin = perf_counter()
            priced = np.asarray(pricer(batch), dtype=float).reshape(-1)
            elapsed += perf_counter() - begin
            if priced.size != batch.shape[0]:
                raise ValueError("pricer must return one value per parameter row")
            values[start : start + batch.shape[0]] = priced
        if not np.all(np.isfinite(values)):
            raise ValueError("pricer returned non-finite values")
        return values.reshape(parameters.shape[:2]), elapsed

    def revalue(
        self,
        pricer,
        spots: np.ndarray,
        rate: float,
        scenario_spots: np.ndarray,
        scenario_rates=None,
    ) -> ScenarioValuation:
        """Reprice the whole book at the base state and in every scenario."""
        spots = np.asarray(spots, dtype=float).reshape(1, -1)
        scenario_spots = np.atleast_2d(np.asarray(scenario_spots, dtype=float))
        if scenario_spots.shape[1] != spots.shape[1]:
            raise ValueError("scenario spots must have one column per asset")
        if scenario_rates is None:
            scenario_rates = rate

        base_parameters = self.portfolio.market_parameters(
            spots,
            rate,
            maturity_shift=0.0,
            minimum_residual_maturity=self.minimum_residual_maturity,
        )
        scenario_parameters = self.scenario_parameters(
            scenario_spots, scenario_rates
        )
        base_values, base_time = self._price(pricer, base_parameters)
        scenario_values, scenario_time = self._price(pricer, scenario_parameters)
        return ScenarioValuation(
            base_trade_values=base_values[0],
            scenario_trade_values=scenario_values,
            quantities=self.portfolio.quantities,
            revaluation_time=base_time + scenario_time,
            pricer_evaluations=int(
                base_parameters.shape[0] * base_parameters.shape[1]
                + scenario_parameters.shape[0] * scenario_parameters.shape[1]
            ),
        )

    def run(
        self,
        pricer,
        spots: np.ndarray,
        rate: float,
        n_scenarios: int = 10_000,
        seed=None,
        scenario_spots: np.ndarray | None = None,
        scenario_rates=None,
        antithetic: bool = True,
    ) -> VaRResult:
        """Simulate (unless scenarios are supplied), revalue and aggregate."""
        scenario_time = 0.0
        if scenario_spots is None:
            start = perf_counter()
            scenario_spots = self.simulate_spots(
                spots, n_scenarios, rate, seed=seed, antithetic=antithetic
            )
            scenario_time = perf_counter() - start

        valuation = self.revalue(
            pricer, spots, rate, scenario_spots, scenario_rates=scenario_rates
        )
        return VaRResult(
            base_value=valuation.base_value,
            losses=valuation.losses,
            trade_losses=-valuation.trade_pnl,
            horizon=self.horizon,
            levels=self.levels,
            revaluation_time=valuation.revaluation_time,
            pricer_evaluations=valuation.pricer_evaluations,
            scenario_time=scenario_time,
            metadata={
                "trades": self.portfolio.describe(),
                "scenario_model": self.generator.describe(rate, self.horizon),
                "base_rate": float(rate),
                "maturities_aged_by_horizon": self.age_maturities,
                "antithetic": bool(antithetic),
            },
        )


def compare_loss_distributions(
    reference_losses: np.ndarray,
    approximate_losses: np.ndarray,
    levels=DEFAULT_LEVELS,
    tail_fraction: float = 0.01,
) -> dict:
    """Accuracy of a surrogate loss distribution against a trusted one.

    Both loss vectors must come from the same scenarios, in the same order: the
    rank correlation and the tail-set overlap are only meaningful for paired
    revaluations.
    """
    reference = np.asarray(reference_losses, dtype=float).reshape(-1)
    approximate = np.asarray(approximate_losses, dtype=float).reshape(-1)
    if reference.size != approximate.size or reference.size == 0:
        raise ValueError("both loss vectors must be non-empty and of equal size")
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must lie in (0, 1)")

    error = approximate - reference
    n = reference.size
    tail_size = max(int(np.ceil(tail_fraction * n)), 1)
    worst_reference = set(np.argsort(reference)[-tail_size:].tolist())
    worst_approximate = set(np.argsort(approximate)[-tail_size:].tolist())

    measures = {}
    for level in levels:
        reference_var, reference_es = var_es(reference, level)
        surrogate_var, surrogate_es = var_es(approximate, level)
        measures[_level_key(level)] = {
            "reference_var": reference_var,
            "surrogate_var": surrogate_var,
            "var_absolute_error": abs(surrogate_var - reference_var),
            "var_relative_error": (
                abs(surrogate_var - reference_var) / abs(reference_var)
                if abs(reference_var) > 1e-14
                else None
            ),
            "reference_expected_shortfall": reference_es,
            "surrogate_expected_shortfall": surrogate_es,
            "expected_shortfall_absolute_error": abs(surrogate_es - reference_es),
            "expected_shortfall_relative_error": (
                abs(surrogate_es - reference_es) / abs(reference_es)
                if abs(reference_es) > 1e-14
                else None
            ),
        }

    return {
        "scenarios": int(n),
        "measures": measures,
        "loss_mae": float(np.mean(np.abs(error))),
        "loss_rmse": float(np.sqrt(np.mean(error**2))),
        "loss_max_absolute_error": float(np.max(np.abs(error))),
        "loss_mean_bias": float(np.mean(error)),
        "wasserstein_distance": float(wasserstein_distance(reference, approximate)),
        "spearman_rank_correlation": float(
            spearmanr(reference, approximate).statistic
        ),
        "tail_fraction": float(tail_fraction),
        "tail_size": int(tail_size),
        "worst_scenario_overlap": len(worst_reference & worst_approximate) / tail_size,
    }


def domain_coverage(parameters: np.ndarray, bounds) -> dict:
    """Fraction of valuation points falling outside the grid box.

    Scenario spots are unbounded under GBM while the grid is not, so every
    interpolant silently clamps extreme states. Tail risk is exactly where that
    happens, which makes this diagnostic a precondition for trusting a surrogate
    VaR number.
    """
    bounds = tuple(bounds)
    points = np.asarray(parameters, dtype=float).reshape(-1, len(bounds))
    lower = np.asarray([b[0] for b in bounds], dtype=float)
    upper = np.asarray([b[1] for b in bounds], dtype=float)
    outside = (points < lower[None, :]) | (points > upper[None, :])
    return {
        "points": int(points.shape[0]),
        "fraction_outside_any_axis": float(np.mean(np.any(outside, axis=1))),
        "fraction_outside_by_axis": np.mean(outside, axis=0).tolist(),
        "observed_bounds": [
            [float(np.min(points[:, j])), float(np.max(points[:, j]))]
            for j in range(points.shape[1])
        ],
    }
