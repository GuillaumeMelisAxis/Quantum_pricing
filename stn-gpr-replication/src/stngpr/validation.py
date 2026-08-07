from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .pricers import AmericanArithmeticBasketLSMC


MONEYNESS_RANGES = {
    "deep_otm": (0.50, 0.80),
    "otm": (0.80, 0.95),
    "atm": (0.95, 1.05),
    "itm": (1.05, 1.20),
    "deep_itm": (1.20, 2.00),
}

MATURITY_RANGES = {
    "very_short": (1.0 / 365.0, 0.25),
    "short": (0.25, 1.00),
    "medium": (1.00, 2.00),
    "long": (2.00, 3.00),
}


def error_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {"count": 0}
    error = y_pred - y_true
    absolute = np.abs(error)
    mae = float(np.mean(absolute))
    scale = float(np.mean(np.abs(y_true)))
    return {
        "count": int(y_true.size),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(absolute)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "max_absolute_error": float(np.max(absolute)),
        "mean_absolute_reference": scale,
        "normalized_mae": mae / scale if scale > 1e-14 else None,
        "mean_error_bias": float(np.mean(error)),
    }


def black_scholes_put(spot, strike, rate, volatility, maturity):
    std = volatility * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / std
    d2 = d1 - std
    return strike * np.exp(-rate * maturity) * ndtr(-d2) - spot * ndtr(-d1)


def american_put_binomial(spot, strike, rate, volatility, maturity, steps=2_000):
    """Cox-Ross-Rubinstein American put benchmark."""
    dt = maturity / int(steps)
    up = np.exp(volatility * np.sqrt(dt))
    down = 1.0 / up
    growth = np.exp(rate * dt)
    probability = (growth - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("invalid CRR probability; increase the number of steps")
    j = np.arange(steps + 1)
    terminal_spot = spot * up**j * down ** (steps - j)
    value = np.maximum(strike - terminal_spot, 0.0)
    discount = np.exp(-rate * dt)
    for step in range(steps - 1, -1, -1):
        value = discount * (
            probability * value[1 : step + 2]
            + (1.0 - probability) * value[: step + 1]
        )
        node = np.arange(step + 1)
        node_spot = spot * up**node * down ** (step - node)
        value = np.maximum(value, strike - node_spot)
    return float(value[0])


def american_put_lsmc_1d(
    spot,
    strike,
    rate,
    volatility,
    maturity,
    n_paths=10_000,
    n_steps=30,
    seed=0,
    antithetic=True,
):
    """One-factor LSMC using a quadratic continuation basis."""
    n_paths = int(n_paths)
    n_steps = int(n_steps)
    dt = maturity / n_steps
    rng = np.random.default_rng(seed)
    if antithetic:
        half = (n_paths + 1) // 2
        z_half = rng.standard_normal((n_steps, half))
        z = np.concatenate((z_half, -z_half), axis=1)[:, :n_paths]
    else:
        z = rng.standard_normal((n_steps, n_paths))
    increments = (
        (rate - 0.5 * volatility**2) * dt
        + volatility * np.sqrt(dt) * z
    )
    paths = np.empty((n_steps + 1, n_paths), dtype=float)
    paths[0] = spot
    paths[1:] = spot * np.exp(np.cumsum(increments, axis=0))
    payoff = np.maximum(strike - paths, 0.0)
    cashflow = payoff[-1].copy()
    exercise_time = np.full(n_paths, n_steps, dtype=np.int64)
    for step in range(n_steps - 1, 0, -1):
        itm = payoff[step] > 0.0
        if np.count_nonzero(itm) < 3:
            continue
        state = paths[step, itm] / strike
        discounted = cashflow[itm] * np.exp(
            -rate * dt * (exercise_time[itm] - step)
        )
        design = np.column_stack((np.ones_like(state), state, state**2))
        beta, *_ = np.linalg.lstsq(design, discounted, rcond=None)
        continuation = design @ beta
        local_exercise = payoff[step, itm] > continuation
        exercise_idx = np.flatnonzero(itm)[local_exercise]
        cashflow[exercise_idx] = payoff[step, exercise_idx]
        exercise_time[exercise_idx] = step
    estimate = np.mean(cashflow * np.exp(-rate * dt * exercise_time))
    return max(float(estimate), max(strike - spot, 0.0))


def summarize_replications(values, reference=None):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    standard_error = std / np.sqrt(values.size) if values.size > 1 else 0.0
    result = {
        "replications": int(values.size),
        "mean": mean,
        "std_across_seeds": std,
        "standard_error": standard_error,
        "ci95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
    }
    if reference is not None:
        result["reference"] = float(reference)
        result["bias"] = mean - float(reference)
        result["absolute_bias"] = abs(mean - float(reference))
        result["reference_in_ci95"] = bool(
            result["ci95"][0] <= reference <= result["ci95"][1]
        )
    return result


def stratified_american_points(config, n_per_moneyness, rng):
    """Balanced arithmetic-basket test design inside the paper market bounds."""
    rows, labels = [], []
    maturity_names = tuple(MATURITY_RANGES)
    for money_name, (ratio_low, ratio_high) in MONEYNESS_RANGES.items():
        for index in range(int(n_per_moneyness)):
            for _ in range(10_000):
                basket_level = rng.uniform(20.0, 110.0)
                relative = np.exp(rng.normal(0.0, 0.15, config.n_assets))
                spots = basket_level * relative / np.mean(relative)
                ratio = rng.uniform(ratio_low, ratio_high)
                strike = basket_level * ratio
                if (
                    np.all(spots >= config.spot_bounds[0])
                    and np.all(spots <= config.spot_bounds[1])
                    and config.strike_bounds[0] <= strike <= config.strike_bounds[1]
                ):
                    break
            else:
                raise RuntimeError("failed to sample a valid stratified point")
            maturity_name = maturity_names[index % len(maturity_names)]
            maturity = rng.uniform(*MATURITY_RANGES[maturity_name])
            rate = rng.uniform(*config.rate_bounds)
            rows.append(np.r_[spots, strike, rate, maturity])
            labels.append((money_name, maturity_name))
    return np.asarray(rows, dtype=float), labels


def multi_seed_american_prices(config, parameters, n_paths, n_steps, seeds):
    prices = []
    for seed in seeds:
        pricer = AmericanArithmeticBasketLSMC(
            config, n_paths=n_paths, n_steps=n_steps, seed=int(seed)
        )
        prices.append(pricer(parameters))
    return np.asarray(prices, dtype=float)


def reference_summary(price_matrix):
    """Pointwise mean and Monte-Carlo uncertainty across independent seeds."""
    values = np.asarray(price_matrix, dtype=float)
    mean = np.mean(values, axis=0)
    if values.shape[0] > 1:
        std = np.std(values, axis=0, ddof=1)
        se = std / np.sqrt(values.shape[0])
    else:
        std = np.zeros(values.shape[1])
        se = np.zeros(values.shape[1])
    return mean, std, se


def metrics_by_labels(y_true, y_pred, labels):
    result = {"global": error_metrics(y_true, y_pred)}
    money, maturity = zip(*labels)
    result["by_moneyness"] = {}
    for name in MONEYNESS_RANGES:
        mask = np.asarray(money) == name
        result["by_moneyness"][name] = error_metrics(y_true[mask], y_pred[mask])
    result["by_maturity"] = {}
    for name in MATURITY_RANGES:
        mask = np.asarray(maturity) == name
        result["by_maturity"][name] = error_metrics(y_true[mask], y_pred[mask])
    return result


def uncertainty_diagnostics(predictions, reference_mean, reference_se):
    predictions = np.asarray(predictions)
    reference_mean = np.asarray(reference_mean)
    reference_se = np.asarray(reference_se)
    absolute = np.abs(predictions - reference_mean)
    band = 1.96 * reference_se
    return {
        "mean_reference_standard_error": float(np.mean(reference_se)),
        "median_reference_standard_error": float(np.median(reference_se)),
        "fraction_predictions_inside_reference_ci95": float(np.mean(absolute <= band)),
        "fraction_errors_below_one_reference_se": float(
            np.mean(absolute <= reference_se)
        ),
    }


def scalar_summary(values):
    """JSON-friendly descriptive statistics for repeated validation runs."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
