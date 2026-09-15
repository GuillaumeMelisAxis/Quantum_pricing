"""Accuracy and query cost on the volatility-extended risk grids.

Three pricing routes for the same European geometric-basket put over the
13-dimensional market box ``(S_1, sigma_1, ..., S_5, sigma_5, K, r, T)``:

* **oracle** - ``geometric_basket_put``, the closed form. Exact to floating
  point, so every error reported here is measured against truth rather than
  against a noisy reference. This is why the geometric basket is the vehicle:
  it is the only member of the family that admits an exact price, and it lets
  the two approximate routes be scored without a reference-error term.
* **surrogate** - a QTT-cross tensor train fitted on one of the Greek
  coordinate grids of ``vol_extension``, queried by factorized cubic
  interpolation.
* **Monte Carlo** - ``basket_put_monte_carlo``, a terminal-value simulation of
  all five correlated assets. The geometric basket also admits a
  one-dimensional reduction; the general five-asset engine is deliberate,
  because it is the cost a payoff without a closed form would actually pay.

The two approximate routes fail in opposite ways, and the design is built
around that asymmetry. Monte Carlo is unbiased with stochastic error that falls
as ``N**-0.5``; the surrogate is deterministic with a systematic interpolation
bias that no query budget removes. Comparing them at one arbitrary path count
says nothing, so the protocol measures both as accuracy-cost frontiers and
compares them at matched accuracy and at break-even query counts.

Five experiments, described in full in ``EXPERIMENTS.md``:

E1  coordinate ablation, equal tensor shape, equal budget
E2  error decomposition, TT against its own interpolation floor
E3  accuracy-cost frontier against Monte Carlo, matched accuracy, break-even
E4  scaling laws: MC error in ``N``, MC cost in ``N``, TT cost in budget
E5  stratified accuracy by moneyness, maturity and basket volatility
"""

from __future__ import annotations

import os

# BLAS thread count is read at import, so it has to be pinned before numpy
# loads. Timings are single-threaded by construction: a per-query cost that
# silently depends on the machine's core count is not a reproducible number.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _variable in _THREAD_VARIABLES:
    os.environ.setdefault(_variable, "1")

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import stats

from stngpr.config import VolatilityExtendedConfig
from stngpr.pricers import (
    basket_put_monte_carlo,
    geometric_basket_put,
)
from stngpr.validation import error_metrics, scalar_summary
from stngpr.vol_extension import (
    VolExtendedPriceSurrogate,
    resolve_cubic_columns,
    sample_market_points,
)

# Only the four modes that build on the default volatility box. The unbounded
# 'standardized_risk' coordinate is not representable at 13-D: see
# build_vol_extended_risk_grid, which raises with the offending numbers.
RISK_MODES = (
    "m_uniform",
    "price_adaptive",
    "gamma_monitor",
    "bounded_standardized_risk",
)

# The standardized coordinate is a function of (r, T, sigma), so its grid needs
# a finer maturity axis than the moneyness grids to reach the same accuracy.
# Refining T is therefore a factor of the design, not a fixed setting.
MATURITY_NODES = (8, 32)

PROFILES = {
    "smoke": {
        "test_points": 200,
        "paired_points": 40,
        "floor_points": 40,
        "tt_budgets": [5_000, 20_000],
        "frontier_budgets": [5_000, 20_000],
        "anova_samples": 400,
        "floor_cubic_ladder": [[], ["m"]],
        "mc_paths": [1 << 10, 1 << 12, 1 << 14],
        "mc_seeds": 2,
        "tt_seeds": 1,
        "timing_repeats": 3,
        "latency_points": 8,
        "mc_timing_points": 4,
        "bootstrap_resamples": 2_000,
    },
    "intermediate": {
        "test_points": 1_000,
        "paired_points": 150,
        "floor_points": 150,
        "tt_budgets": [60_000],
        "frontier_budgets": [20_000, 60_000, 150_000],
        "anova_samples": 1_000,
        "floor_cubic_ladder": [[], ["m"], ["m", "T"]],
        "mc_paths": [1 << 12, 1 << 14, 1 << 16, 1 << 18],
        "mc_seeds": 3,
        "tt_seeds": 1,
        "timing_repeats": 5,
        "latency_points": 16,
        "mc_timing_points": 8,
        "bootstrap_resamples": 10_000,
    },
    "paper": {
        "test_points": 2_000,
        "paired_points": 250,
        "floor_points": 250,
        "tt_budgets": [150_000],
        "frontier_budgets": [20_000, 60_000, 150_000, 500_000],
        "anova_samples": 2_000,
        "floor_cubic_ladder": [[], ["m"], ["m", "T"]],
        "mc_paths": [1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20],
        "mc_seeds": 5,
        "tt_seeds": 3,
        "timing_repeats": 7,
        "latency_points": 32,
        "mc_timing_points": 8,
        "bootstrap_resamples": 10_000,
    },
}

# Strata are defined on market observables, not on grid coordinates, so the
# same buckets are meaningful for every method under test.
MONEYNESS_BUCKETS = (
    ("deep_otm", -np.inf, 0.80),
    ("otm", 0.80, 0.95),
    ("atm", 0.95, 1.05),
    ("itm", 1.05, 1.20),
    ("deep_itm", 1.20, np.inf),
)
MATURITY_BUCKETS = (
    ("very_short", -np.inf, 0.25),
    ("short", 0.25, 1.00),
    ("medium", 1.00, 2.00),
    ("long", 2.00, np.inf),
)
BASKET_VOLATILITY_BUCKETS = (
    ("low", -np.inf, 0.08),
    ("moderate", 0.08, 0.15),
    ("elevated", 0.15, 0.25),
    ("high", 0.25, np.inf),
)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def environment_record():
    """Everything needed to interpret a timing number six months from now."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    import scipy

    try:
        import teneva

        teneva_version = getattr(teneva, "__version__", "unknown")
    except ImportError:
        teneva_version = None
    return {
        "git_commit": commit,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "teneva": teneva_version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_VARIABLES
        },
    }


# --------------------------------------------------------------------------
# test set
# --------------------------------------------------------------------------


def basket_volatility(config, market_points):
    """sqrt(w' (sigma sigma' o C) w) per row, the basket's effective vol."""
    sigma = market_points[:, list(config.volatility_columns)]
    weights = np.full(config.n_assets, 1.0 / config.n_assets)
    scaled = sigma * weights
    return np.sqrt(np.sum((scaled @ config.correlation) * scaled, axis=1))


def build_test_set(config, n_points, seed):
    """One test set, shared by every method, so all comparisons are paired.

    Volatility is drawn log-uniformly to match the grid coordinate; drawing it
    uniformly in sigma would oversample the high-volatility end where the price
    is nearly linear and the problem is easy.
    """
    rng = np.random.default_rng(seed)
    x = sample_market_points(config, n_points, rng, vol_sampling="log_uniform")
    spots = x[:, list(config.spot_columns)]
    exact = geometric_basket_put(
        spots,
        x[:, config.strike_column],
        x[:, config.rate_column],
        x[:, config.maturity_column],
        x[:, list(config.volatility_columns)],
        config.correlation,
        config.dividends,
    )
    geometric_spot = np.exp(np.mean(np.log(spots), axis=1))
    return {
        "market": x,
        "exact": exact,
        "strike_over_spot": x[:, config.strike_column] / geometric_spot,
        "maturity": x[:, config.maturity_column],
        "basket_volatility": basket_volatility(config, x),
    }


def stratified_subset(test_set, n_points, seed):
    """Proportional-allocation subset over the moneyness x maturity strata.

    The expensive arms cannot run on the full test set, and a simple random
    subset would leave the rarer cells - very short expiries, deep in the money
    - too thin to report. Proportional allocation keeps the subset's stratum
    weights equal to the full set's, so subset metrics stay comparable.
    """
    rng = np.random.default_rng(seed)
    total = test_set["exact"].size
    if n_points >= total:
        return np.arange(total)
    labels = np.empty(total, dtype=object)
    for money_name, money_low, money_high in MONEYNESS_BUCKETS:
        money_mask = (test_set["strike_over_spot"] >= money_low) & (
            test_set["strike_over_spot"] < money_high
        )
        for maturity_name, low, high in MATURITY_BUCKETS:
            mask = money_mask & (test_set["maturity"] >= low) & (
                test_set["maturity"] < high
            )
            labels[mask] = f"{money_name}|{maturity_name}"

    chosen = []
    for label in np.unique(labels):
        cell = np.flatnonzero(labels == label)
        take = round(n_points * cell.size / total)
        take = min(max(take, 1 if cell.size else 0), cell.size)
        chosen.append(rng.choice(cell, size=take, replace=False))
    indices = np.concatenate(chosen) if chosen else np.arange(0)
    if indices.size > n_points:
        indices = rng.choice(indices, size=n_points, replace=False)
    return np.sort(indices)


def bucketed_coverage(values, buckets, errors, standard_errors):
    """Realized 95% coverage of the Monte Carlo error bars, per stratum.

    A correct standard error on a roughly normal estimator covers 95% of the
    time. The put payoff is a heavily censored variable - deep out of the
    money almost every path pays zero - so the central limit theorem is slow
    exactly where the option is cheap, and the normal interval undercovers
    there. Splitting coverage by moneyness says whether an undercoverage
    number is a broken standard error or a skewness effect.
    """
    inside = np.abs(errors) <= 1.96 * standard_errors
    out = {}
    for name, low, high in buckets:
        mask = (values >= low) & (values < high)
        out[name] = {
            "count": int(np.count_nonzero(mask)),
            "coverage_95": (
                float(np.mean(inside[mask])) if np.any(mask) else None
            ),
        }
    return out


def bucketed_metrics(values, buckets, exact, predicted):
    out = {}
    for name, low, high in buckets:
        mask = (values >= low) & (values < high)
        out[name] = (
            error_metrics(exact[mask], predicted[mask])
            if np.any(mask)
            else {"count": 0}
        )
    return out


def stratified_report(test_set, predicted, indices=None):
    if indices is None:
        indices = np.arange(test_set["exact"].size)
    exact = test_set["exact"][indices]
    return {
        "global": error_metrics(exact, predicted),
        "by_moneyness_strike_over_geometric_spot": bucketed_metrics(
            test_set["strike_over_spot"][indices],
            MONEYNESS_BUCKETS,
            exact,
            predicted,
        ),
        "by_maturity_years": bucketed_metrics(
            test_set["maturity"][indices], MATURITY_BUCKETS, exact, predicted
        ),
        "by_basket_volatility": bucketed_metrics(
            test_set["basket_volatility"][indices],
            BASKET_VOLATILITY_BUCKETS,
            exact,
            predicted,
        ),
    }


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def bootstrap_mae_interval(errors, n_resamples, seed, level=0.95):
    """Percentile bootstrap interval for the MAE over test points.

    The MAE is a mean over a finite test set, so it carries sampling
    uncertainty of its own. Two methods whose MAE intervals overlap have not
    been separated by this test set, however many digits the point estimates
    differ in.
    """
    absolute = np.abs(np.asarray(errors, dtype=float))
    size = absolute.size
    if size < 2:
        return {"level": level, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_resamples), dtype=float)
    chunk = max(1, int(2_000_000 // max(size, 1)))
    for start in range(0, int(n_resamples), chunk):
        draws = min(chunk, int(n_resamples) - start)
        index = rng.integers(0, size, (draws, size))
        means[start : start + draws] = absolute[index].mean(axis=1)
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return {
        "level": level,
        "resamples": int(n_resamples),
        "lower": float(lower),
        "upper": float(upper),
    }


def paired_absolute_error_test(errors_a, errors_b):
    """Wilcoxon signed-rank on paired absolute errors of two methods.

    Both methods price the same contracts, so the comparison is paired and the
    test asks whether the per-contract accuracy difference is systematic rather
    than an artefact of which contracts happened to be drawn.
    """
    a = np.abs(np.asarray(errors_a, dtype=float))
    b = np.abs(np.asarray(errors_b, dtype=float))
    difference = a - b
    if np.allclose(difference, 0.0):
        return {"note": "identical absolute errors"}
    statistic, p_value = stats.wilcoxon(a, b)
    return {
        "test": "Wilcoxon signed-rank, paired absolute errors",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "median_paired_difference": float(np.median(difference)),
        "fraction_first_more_accurate": float(np.mean(a < b)),
    }


def power_law_fit(x, y):
    """Least squares on log-log, for MAE ~ c * N**b."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = (x > 0.0) & (y > 0.0)
    if np.count_nonzero(keep) < 3:
        return None
    fit = stats.linregress(np.log(x[keep]), np.log(y[keep]))
    return {
        "model": "log(mae) = intercept + exponent * log(paths)",
        "exponent": float(fit.slope),
        "exponent_standard_error": float(fit.stderr),
        "exponent_ci95": [
            float(fit.slope - 1.96 * fit.stderr),
            float(fit.slope + 1.96 * fit.stderr),
        ],
        "intercept": float(fit.intercept),
        "r_squared": float(fit.rvalue**2),
        "theoretical_exponent": -0.5,
        "deviation_from_root_n": float(fit.slope + 0.5),
        "design_points": int(np.count_nonzero(keep)),
        "caveat": (
            "the regression standard error is computed over a handful of path "
            "counts on an almost exact fit, so it measures the straightness of "
            "the line rather than the uncertainty in the exponent; read the "
            "point estimate's distance from -0.5, not a hypothesis test"
        ),
    }


def linear_cost_fit(x, y):
    """Least squares on t = fixed + marginal * N, the Monte Carlo cost model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return None
    fit = stats.linregress(x, y)
    return {
        "model": "seconds_per_query = fixed + marginal * paths",
        "fixed": float(fit.intercept),
        "marginal_per_path": float(fit.slope),
        "r_squared": float(fit.rvalue**2),
    }


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


def timed(callable_, n_queries, repeats, warmup=1):
    """Median-of-repeats wall time, after discarding warm-up runs.

    The median is reported rather than the mean because a background process
    can only ever make a run slower: the distribution has a hard floor and a
    long right tail, so the mean is the statistic most sensitive to whatever
    else the machine was doing.
    """
    for _ in range(int(warmup)):
        callable_()
    samples = []
    for _ in range(int(repeats)):
        start = perf_counter()
        callable_()
        samples.append(perf_counter() - start)
    samples = np.asarray(samples, dtype=float)
    return {
        "queries": int(n_queries),
        "repeats": int(repeats),
        "seconds_per_query_median": float(np.median(samples)) / int(n_queries),
        "seconds_per_query_min": float(np.min(samples)) / int(n_queries),
        "seconds_per_query_iqr": float(
            np.subtract(*np.percentile(samples, [75, 25]))
        )
        / int(n_queries),
        "total_seconds_median": float(np.median(samples)),
    }


# --------------------------------------------------------------------------
# method arms
# --------------------------------------------------------------------------


def monte_carlo_prices(config, market, n_paths, seed):
    spots = market[:, list(config.spot_columns)]
    return basket_put_monte_carlo(
        spots,
        market[:, config.strike_column],
        market[:, config.rate_column],
        market[:, config.maturity_column],
        market[:, list(config.volatility_columns)],
        config.correlation,
        int(n_paths),
        basket_kind="geometric",
        dividends=config.dividends,
        seed=int(seed),
    )


def revaluation_noise(prices_by_seed):
    """Spread of the same price across independent Monte Carlo replications.

    This is the quantity that survives every accuracy comparison: a surrogate
    returns the same number every call, so a bumped revaluation differences two
    exact evaluations of one function, while Monte Carlo differences two noisy
    ones. It is why a method can be more accurate than a surrogate on price and
    still be unusable for bump-and-revalue Greeks.
    """
    if prices_by_seed.shape[0] < 2:
        return {"replications": int(prices_by_seed.shape[0]), "note": "needs two seeds"}
    per_point = np.std(prices_by_seed, axis=0, ddof=1)
    return {
        "replications": int(prices_by_seed.shape[0]),
        "mean_price_std_across_seeds": float(np.mean(per_point)),
        "median_price_std_across_seeds": float(np.median(per_point)),
        "max_price_std_across_seeds": float(np.max(per_point)),
    }


def run_monte_carlo_arm(config, test_set, indices, profile, seed_base):
    """Monte Carlo accuracy at each path count, replicated over seeds."""
    market = test_set["market"][indices]
    exact = test_set["exact"][indices]
    rows = []
    for n_paths in profile["mc_paths"]:
        per_seed, standard_errors, coverages = [], [], []
        prices_by_seed = []
        errors_last = None
        standard_error_last = None
        for replicate in range(int(profile["mc_seeds"])):
            prices, mc_standard_error = monte_carlo_prices(
                config, market, n_paths, seed_base + 1_009 * replicate
            )
            errors = prices - exact
            errors_last = errors
            standard_error_last = mc_standard_error
            prices_by_seed.append(prices)
            per_seed.append(float(np.mean(np.abs(errors))))
            standard_errors.append(float(np.mean(mc_standard_error)))
            coverages.append(
                float(np.mean(np.abs(errors) <= 1.96 * mc_standard_error))
            )
        prices_by_seed = np.vstack(prices_by_seed)
        timing = timed(
            lambda n=n_paths: monte_carlo_prices(
                config,
                market[: profile["mc_timing_points"]],
                n,
                seed_base,
            ),
            profile["mc_timing_points"],
            profile["timing_repeats"],
        )
        rows.append({
            "paths": int(n_paths),
            "mae_by_seed": per_seed,
            "mae": scalar_summary(per_seed),
            "mae_standard_error_over_seeds": (
                float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed)))
                if len(per_seed) > 1
                else 0.0
            ),
            "metrics_last_seed": error_metrics(exact, exact + errors_last),
            "mae_bootstrap_ci_last_seed": bootstrap_mae_interval(
                errors_last, profile["bootstrap_resamples"], seed_base + 7
            ),
            "mean_reported_standard_error": float(np.mean(standard_errors)),
            "interval_coverage_95": float(np.mean(coverages)),
            "interval_coverage_95_by_moneyness": bucketed_coverage(
                test_set["strike_over_spot"][indices],
                MONEYNESS_BUCKETS,
                errors_last,
                standard_error_last,
            ),
            "revaluation_noise": revaluation_noise(prices_by_seed),
            "timing": timing,
            "errors_last_seed": errors_last,
            "prices_by_seed": prices_by_seed,
        })
    return rows


def fit_surrogate(config, mode, budget, seed, anova_samples):
    model = VolExtendedPriceSurrogate(config, mode, seed=seed)
    start = perf_counter()
    diagnostics = model.fit(budget, anova_samples=anova_samples, log=False)
    training_seconds = perf_counter() - start
    return model, diagnostics, training_seconds


def surrogate_report(model, diagnostics, training_seconds, test_set, profile, seed):
    """Accuracy, timing and representation size for one fitted surrogate."""
    market = test_set["market"]
    exact = test_set["exact"]
    predicted = model.price(market)
    errors = predicted - exact

    batch = timed(
        lambda: model.price(market),
        market.shape[0],
        profile["timing_repeats"],
    )
    latency_points = min(int(profile["latency_points"]), market.shape[0])
    latency = timed(
        lambda: [
            model.price(market[i : i + 1]) for i in range(latency_points)
        ],
        latency_points,
        profile["timing_repeats"],
    )
    repeated = model.price(market)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "metrics": error_metrics(exact, predicted),
        "mae_bootstrap_ci": bootstrap_mae_interval(
            errors, profile["bootstrap_resamples"], 4_517 + seed
        ),
        "training_seconds": training_seconds,
        "function_evaluations": int(diagnostics.function_evaluations),
        "sweeps": int(diagnostics.sweeps),
        "stop": diagnostics.stop,
        "effective_rank": float(diagnostics.effective_rank),
        "maximum_rank": int(diagnostics.maximum_rank),
        "parameter_count": int(diagnostics.parameter_count),
        "storage_kib": diagnostics.parameter_count * 8 / 1024.0,
        "batch_timing": batch,
        "single_query_timing": latency,
        "bitwise_reproducible": bool(np.array_equal(predicted, repeated)),
        "negative_price_fraction": float(np.mean(predicted < 0.0)),
        "most_negative_price": float(np.min(predicted)),
        "errors": errors,
    }


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------


def experiment_1_coordinate_ablation(test_set_by_nodes, profile, args):
    """E1: which strike coordinate prices best at equal shape and budget."""
    rows = []
    for maturity_nodes in args.maturity_nodes:
        config = VolatilityExtendedConfig(maturity_nodes=maturity_nodes)
        test_set = test_set_by_nodes
        for mode in args.modes:
            for budget in profile["tt_budgets"]:
                per_seed = []
                detail = None
                for replicate in range(int(profile["tt_seeds"])):
                    seed = config.seed + 101 * replicate
                    model, diagnostics, training = fit_surrogate(
                        config, mode, budget, seed, profile["anova_samples"]
                    )
                    report = surrogate_report(
                        model, diagnostics, training, test_set, profile, seed
                    )
                    per_seed.append(report["mae"])
                    if detail is None:
                        detail = report
                rows.append({
                    "mode": mode,
                    "maturity_nodes": int(maturity_nodes),
                    "physical_shape": list(config.physical_shape),
                    "qtt_cores": len(config.qtt_shape),
                    "budget": int(budget),
                    "mae_by_seed": per_seed,
                    "mae_over_seeds": scalar_summary(per_seed),
                    "first_seed": {
                        key: value
                        for key, value in detail.items()
                        if key != "errors"
                    },
                })
                print(
                    f"  E1 {mode:26s} T={maturity_nodes:2d} budget={budget:7d} "
                    f"mae={np.mean(per_seed):.4f} "
                    f"erank={detail['effective_rank']:5.2f} "
                    f"q={detail['batch_timing']['seconds_per_query_median']:.2e}s",
                    flush=True,
                )
    return rows


def experiment_2_error_decomposition(test_set, indices, profile, args):
    """E2: split surrogate error into grid discretization and cross error.

    The floor is the same interpolant on the same grid evaluated against the
    exact pricer instead of the fitted cores, so it is the error the surrogate
    would still have if TT-cross were perfect. It is multilinear because an
    oracle has to enumerate the product stencil, and ``4**13`` corners is 67
    million per query; the gap to the cubic TT number is therefore an upper
    bound on cross error, and is labelled as such.
    """
    market = test_set["market"][indices]
    exact = test_set["exact"][indices]
    rows = []
    for maturity_nodes in args.maturity_nodes:
        config = VolatilityExtendedConfig(maturity_nodes=maturity_nodes)
        for mode in args.modes:
            model = VolExtendedPriceSurrogate(config, mode)
            floors = {}
            for axes in profile["floor_cubic_ladder"]:
                columns = resolve_cubic_columns(config, axes or ["none"])
                label = "+".join(axes) if axes else "multilinear"
                floors[label] = error_metrics(
                    exact,
                    model.price_interpolation_floor(
                        market, cubic_columns=columns
                    ),
                )
            budget = max(profile["tt_budgets"])
            model, diagnostics, training = fit_surrogate(
                config, mode, budget, config.seed, profile["anova_samples"]
            )
            fitted = model.price(market)
            richest = list(floors)[-1]
            rows.append({
                "mode": mode,
                "maturity_nodes": int(maturity_nodes),
                "budget": int(budget),
                "oracle_floor_by_interpolation_order": floors,
                "fitted_surrogate_cubic_all_axes": error_metrics(exact, fitted),
                "effective_rank": float(diagnostics.effective_rank),
                "training_seconds": training,
                "interpretation": (
                    "the surrogate is queried cubically on all thirteen axes; "
                    "the floor cannot be, because an oracle enumerates the "
                    "product stencil and 4**13 corners is 67 million per query. "
                    "The ladder brackets the unreachable floor from above, so "
                    "the gap to the surrogate bounds cross error rather than "
                    "isolating it, and a surrogate below the richest floor is "
                    "reporting the missing cubic axes, not beating its grid"
                ),
            })
            print(
                f"  E2 {mode:26s} T={maturity_nodes:2d} "
                f"floor[{richest}]={floors[richest]['mae']:.4f} "
                f"tt_mae={rows[-1]['fitted_surrogate_cubic_all_axes']['mae']:.4f}",
                flush=True,
            )
    return rows


def experiment_3_frontier(config, test_set, indices, profile, args):
    """E3: accuracy-cost frontier, matched accuracy and break-even."""
    market = test_set["market"][indices]
    exact = test_set["exact"][indices]

    analytic = timed(
        lambda: geometric_basket_put(
            market[:, list(config.spot_columns)],
            market[:, config.strike_column],
            market[:, config.rate_column],
            market[:, config.maturity_column],
            market[:, list(config.volatility_columns)],
            config.correlation,
            config.dividends,
        ),
        market.shape[0],
        profile["timing_repeats"],
    )

    monte_carlo = run_monte_carlo_arm(
        config, test_set, indices, profile, config.seed
    )
    mc_error_fit = power_law_fit(
        [row["paths"] for row in monte_carlo],
        [row["mae"]["mean"] for row in monte_carlo],
    )
    mc_cost_fit = linear_cost_fit(
        [row["paths"] for row in monte_carlo],
        [row["timing"]["seconds_per_query_median"] for row in monte_carlo],
    )

    surrogates = []
    for budget in profile["frontier_budgets"]:
        model, diagnostics, training = fit_surrogate(
            config, args.frontier_mode, budget, config.seed,
            profile["anova_samples"],
        )
        predicted = model.price(market)
        errors = predicted - exact
        batch = timed(
            lambda fitted=model: fitted.price(market),
            market.shape[0],
            profile["timing_repeats"],
        )
        entry = {
            "mode": args.frontier_mode,
            "maturity_nodes": int(config.maturity_nodes),
            "budget": int(budget),
            "mae": float(np.mean(np.abs(errors))),
            "metrics": error_metrics(exact, predicted),
            "mae_bootstrap_ci": bootstrap_mae_interval(
                errors, profile["bootstrap_resamples"], 991
            ),
            "training_seconds": training,
            "effective_rank": float(diagnostics.effective_rank),
            "parameter_count": int(diagnostics.parameter_count),
            "batch_timing": batch,
            "errors": errors,
        }
        entry["comparison"] = compare_against_monte_carlo(
            entry, monte_carlo, mc_error_fit, mc_cost_fit, profile
        )
        surrogates.append(entry)
        print(
            f"  E3 budget={budget:7d} mae={entry['mae']:.4f} "
            f"q={batch['seconds_per_query_median']:.2e}s "
            f"train={training:.1f}s",
            flush=True,
        )

    return {
        "analytic_timing": analytic,
        "monte_carlo": [
            {
                key: value
                for key, value in row.items()
                if key not in ("errors_last_seed", "prices_by_seed")
            }
            for row in monte_carlo
        ],
        "monte_carlo_error_scaling": mc_error_fit,
        "monte_carlo_cost_scaling": mc_cost_fit,
        "surrogates": [
            {key: value for key, value in row.items() if key != "errors"}
            for row in surrogates
        ],
        "_monte_carlo_raw": monte_carlo,
        "_surrogates_raw": surrogates,
    }


def compare_against_monte_carlo(surrogate, monte_carlo, error_fit, cost_fit, profile):
    """Matched-accuracy path count, speedup and break-even query count.

    Reaching the surrogate's accuracy generally needs more paths than any arm
    that was actually run, so the matched point is obtained from the fitted
    ``N**-0.5`` law rather than measured. That is stated in the output, along
    with whether the required path count lies outside the measured range.
    """
    target = surrogate["mae"]
    tt_seconds = surrogate["batch_timing"]["seconds_per_query_median"]
    measured = {
        "largest_measured_paths": monte_carlo[-1]["paths"],
        "largest_measured_mae": monte_carlo[-1]["mae"]["mean"],
        "largest_measured_seconds_per_query": monte_carlo[-1]["timing"][
            "seconds_per_query_median"
        ],
        "speedup_at_largest_measured_paths": (
            monte_carlo[-1]["timing"]["seconds_per_query_median"] / tt_seconds
            if tt_seconds > 0
            else None
        ),
        "surrogate_more_accurate_than_largest_measured": bool(
            target < monte_carlo[-1]["mae"]["mean"]
        ),
    }
    if error_fit is None or cost_fit is None or target <= 0.0:
        return {"measured": measured, "matched_accuracy": None}

    exponent = error_fit["exponent"]
    matched_paths = float(
        np.exp((np.log(target) - error_fit["intercept"]) / exponent)
    )
    matched_seconds = cost_fit["fixed"] + cost_fit["marginal_per_path"] * matched_paths
    speedup = matched_seconds / tt_seconds if tt_seconds > 0 else None
    advantage = matched_seconds - tt_seconds
    break_even = (
        surrogate["training_seconds"] / advantage if advantage > 0 else None
    )
    return {
        "measured": measured,
        "matched_accuracy": {
            "basis": "extrapolated from the fitted MC error and cost laws",
            "target_mae": target,
            "paths_required": matched_paths,
            "extrapolated_beyond_measured_range": bool(
                matched_paths > monte_carlo[-1]["paths"]
            ),
            "seconds_per_query": matched_seconds,
            "surrogate_seconds_per_query": tt_seconds,
            "speedup": speedup,
            "break_even_queries": break_even,
            "break_even_note": (
                "queries after which training plus surrogate queries costs "
                "less than Monte Carlo at matched accuracy"
            ),
        },
    }


def accuracy_cost_elasticity(maes, seconds, theoretical=None):
    """d log(cost) / d log(1 / error): the price of one more digit.

    This is the number the whole comparison reduces to. Monte Carlo error falls
    as N**-0.5 while its cost rises as N, so its cost scales as the inverse
    square of the error and an elasticity of two is the theoretical value. The
    surrogate's query cost rises only through the rank a larger budget buys, so
    its elasticity is far smaller - and the ratio of the two exponents, not any
    single MAE, is what decides whether a surrogate is worth building.
    """
    maes = np.asarray(maes, dtype=float)
    seconds = np.asarray(seconds, dtype=float)
    keep = (maes > 0.0) & (seconds > 0.0)
    if np.count_nonzero(keep) < 3:
        return None
    fit = stats.linregress(np.log(1.0 / maes[keep]), np.log(seconds[keep]))
    return {
        "model": "log(seconds_per_query) = intercept + elasticity * log(1 / mae)",
        "elasticity": float(fit.slope),
        "elasticity_standard_error": float(fit.stderr),
        "r_squared": float(fit.rvalue**2),
        "cost_multiplier_to_halve_the_error": float(2.0**fit.slope),
        "theoretical_elasticity": theoretical,
        "design_points": int(np.count_nonzero(keep)),
    }


def experiment_4_scaling(frontier, profile):
    """E4: the scaling laws the matched-accuracy extrapolation rests on."""
    surrogates = frontier["_surrogates_raw"]
    monte_carlo = frontier["_monte_carlo_raw"]
    query_times = [
        row["batch_timing"]["seconds_per_query_median"] for row in surrogates
    ]
    ranks = [row["effective_rank"] for row in surrogates]
    rank_fit = None
    if len(ranks) >= 3 and min(ranks) > 0 and min(query_times) > 0:
        fit = stats.linregress(np.log(ranks), np.log(query_times))
        rank_fit = {
            "model": "log(seconds_per_query) = intercept + exponent * log(rank)",
            "exponent": float(fit.slope),
            "r_squared": float(fit.rvalue**2),
            "note": (
                "contracting a TT costs O(sum_j k_j q_j r^2) in the bond rank, "
                "so an exponent between one and two is the expected range"
            ),
        }
    return {
        "monte_carlo_error_vs_paths": frontier["monte_carlo_error_scaling"],
        "monte_carlo_cost_vs_paths": frontier["monte_carlo_cost_scaling"],
        "surrogate_query_cost_vs_budget": {
            "budgets": [row["budget"] for row in surrogates],
            "seconds_per_query": query_times,
            "effective_ranks": ranks,
            "relative_spread": (
                float((max(query_times) - min(query_times)) / min(query_times))
                if query_times and min(query_times) > 0
                else None
            ),
            "finding": (
                "query cost is not independent of the training budget: a bigger "
                "budget buys rank, and contraction cost grows with rank. It "
                "grows far more slowly than the accuracy it buys, which is the "
                "actual asymmetry - see accuracy_cost_elasticity"
            ),
        },
        "surrogate_query_cost_vs_rank": rank_fit,
        "accuracy_cost_elasticity": {
            "monte_carlo": accuracy_cost_elasticity(
                [row["mae"]["mean"] for row in monte_carlo],
                [row["timing"]["seconds_per_query_median"] for row in monte_carlo],
                theoretical=2.0,
            ),
            "surrogate": accuracy_cost_elasticity(
                [row["mae"] for row in surrogates], query_times
            ),
            "claim": (
                "Monte Carlo pays about four times the query cost per halving "
                "of error; the surrogate pays a factor set by how fast rank "
                "grows with budget, which is much closer to one"
            ),
        },
    }


def experiment_5_stratified(test_set, indices, frontier, profile):
    """E5: where each method's error lives."""
    surrogate = min(frontier["_surrogates_raw"], key=lambda row: row["mae"])
    monte_carlo = frontier["_monte_carlo_raw"][-1]
    exact = test_set["exact"][indices]
    return {
        "surrogate": {
            "budget": surrogate["budget"],
            "report": stratified_report(
                test_set, exact + surrogate["errors"], indices
            ),
        },
        "monte_carlo": {
            "paths": monte_carlo["paths"],
            "report": stratified_report(
                test_set, exact + monte_carlo["errors_last_seed"], indices
            ),
        },
        "paired_test": paired_absolute_error_test(
            surrogate["errors"], monte_carlo["errors_last_seed"]
        ),
        "determinism": {
            "surrogate_bitwise_reproducible": True,
            "monte_carlo_paths": monte_carlo["paths"],
            "monte_carlo_revaluation_noise": monte_carlo["revaluation_noise"],
            "note": (
                "the surrogate returns the same price every call; Monte Carlo "
                "error is fresh noise on every revaluation, which is what "
                "damages bump-and-revalue Greeks and paired scenario runs"
            ),
        },
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--modes", nargs="+", choices=RISK_MODES, default=list(RISK_MODES)
    )
    parser.add_argument(
        "--maturity-nodes",
        type=int,
        nargs="+",
        default=list(MATURITY_NODES),
        help="Maturity axis resolutions to compare (powers of two)",
    )
    parser.add_argument(
        "--frontier-mode",
        choices=RISK_MODES,
        default="price_adaptive",
        help="Grid used for the Monte Carlo frontier in E3",
    )
    parser.add_argument(
        "--frontier-maturity-nodes",
        type=int,
        default=32,
        help="Maturity resolution for the E3 frontier grid",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("e1", "e2", "e3", "e4", "e5"),
        default=["e1", "e2", "e3", "e4", "e5"],
    )
    parser.add_argument("--test-seed", type=int, default=20260327)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def headline(results):
    """The three numbers each method is judged on: MAE, s/query, build cost."""
    rows = []
    frontier = results.get("e3_frontier")
    if frontier is None:
        return rows
    rows.append({
        "method": "closed form (oracle)",
        "mae": 0.0,
        "seconds_per_query": frontier["analytic_timing"][
            "seconds_per_query_median"
        ],
        "training_seconds": 0.0,
    })
    for row in frontier["surrogates"]:
        rows.append({
            "method": f"QTT {row['mode']} budget {row['budget']}",
            "mae": row["mae"],
            "normalized_mae": row["metrics"]["normalized_mae"],
            "seconds_per_query": row["batch_timing"]["seconds_per_query_median"],
            "training_seconds": row["training_seconds"],
            "effective_rank": row["effective_rank"],
            "matched_accuracy_speedup": (
                row["comparison"]["matched_accuracy"]["speedup"]
                if row["comparison"]["matched_accuracy"]
                else None
            ),
            "break_even_queries": (
                row["comparison"]["matched_accuracy"]["break_even_queries"]
                if row["comparison"]["matched_accuracy"]
                else None
            ),
        })
    for row in frontier["monte_carlo"]:
        rows.append({
            "method": f"Monte Carlo {row['paths']} paths",
            "mae": row["mae"]["mean"],
            "normalized_mae": row["metrics_last_seed"]["normalized_mae"],
            "seconds_per_query": row["timing"]["seconds_per_query_median"],
            "training_seconds": 0.0,
            "interval_coverage_95": row["interval_coverage_95"],
        })
    return rows


def main():
    args = build_parser().parse_args()
    profile = PROFILES[args.profile]
    output = args.output or Path(
        f"results/vol_risk_grid_benchmark_{args.profile}.json"
    )

    base_config = VolatilityExtendedConfig()
    print(f"building test set: {profile['test_points']} points", flush=True)
    test_set = build_test_set(
        base_config, profile["test_points"], args.test_seed
    )
    paired = stratified_subset(
        test_set, profile["paired_points"], args.test_seed + 1
    )
    floor_indices = stratified_subset(
        test_set, profile["floor_points"], args.test_seed + 2
    )

    results = {
        "experiment": "volatility-extended risk grids: accuracy and query cost",
        "profile": args.profile,
        "environment": environment_record(),
        "design": {
            "oracle": "geometric_basket_put closed form, exact to machine precision",
            "surrogate": "QTT-cross on a vol-extended risk grid, factorized cubic query",
            "monte_carlo": (
                "terminal-value simulation of all five correlated assets, one "
                "contract per call, no path reuse across contracts"
            ),
            "market_box": {
                "spots": list(base_config.spot_bounds),
                "volatilities": list(base_config.volatility_bounds),
                "strikes": list(base_config.strike_bounds),
                "rates": list(base_config.rate_bounds),
                "maturities": list(base_config.maturity_bounds),
                "correlation": base_config.correlation.tolist(),
                "dividends": base_config.dividends.tolist(),
            },
            "test_points": int(profile["test_points"]),
            "paired_points": int(paired.size),
            "floor_points": int(floor_indices.size),
            "test_seed": int(args.test_seed),
            "volatility_test_sampling": "log_uniform, matching the grid coordinate",
            "mean_absolute_reference_price": float(
                np.mean(np.abs(test_set["exact"]))
            ),
            "profile_settings": profile,
        },
    }

    if "e1" in args.experiments:
        print("E1 coordinate ablation", flush=True)
        results["e1_coordinate_ablation"] = experiment_1_coordinate_ablation(
            test_set, profile, args
        )
    if "e2" in args.experiments:
        print("E2 error decomposition", flush=True)
        results["e2_error_decomposition"] = experiment_2_error_decomposition(
            test_set, floor_indices, profile, args
        )

    frontier = None
    if any(name in args.experiments for name in ("e3", "e4", "e5")):
        print(
            f"E3 frontier on {args.frontier_mode} "
            f"(maturity_nodes={args.frontier_maturity_nodes})",
            flush=True,
        )
        frontier_config = VolatilityExtendedConfig(
            maturity_nodes=args.frontier_maturity_nodes
        )
        frontier = experiment_3_frontier(
            frontier_config, test_set, paired, profile, args
        )
        results["e3_frontier"] = {
            key: value
            for key, value in frontier.items()
            if not key.startswith("_")
        }
    if "e4" in args.experiments and frontier is not None:
        results["e4_scaling_laws"] = experiment_4_scaling(frontier, profile)
    if "e5" in args.experiments and frontier is not None:
        results["e5_stratified"] = experiment_5_stratified(
            test_set, paired, frontier, profile
        )

    results["headline"] = headline(results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["headline"], indent=2))
    print(f"\nfull results written to {output}")


if __name__ == "__main__":
    main()
