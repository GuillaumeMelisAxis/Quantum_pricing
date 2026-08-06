from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def geometric_basket_put(
    spots: np.ndarray,
    strikes: np.ndarray,
    rates: np.ndarray,
    maturities: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    dividends: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form put on an equally weighted geometric basket under GBM."""
    spots = np.atleast_2d(np.asarray(spots, dtype=float))
    m, n_assets = spots.shape
    strikes = np.broadcast_to(np.asarray(strikes, dtype=float), (m,))
    rates = np.broadcast_to(np.asarray(rates, dtype=float), (m,))
    maturities = np.broadcast_to(np.asarray(maturities, dtype=float), (m,))
    sigma = np.asarray(volatilities, dtype=float)
    q = np.zeros(n_assets) if dividends is None else np.asarray(dividends, dtype=float)
    weights = np.full(n_assets, 1.0 / n_assets)

    covariance = np.outer(sigma, sigma) * np.asarray(correlation, dtype=float)
    basket_variance = float(weights @ covariance @ weights)
    basket_sigma = np.sqrt(basket_variance)
    g0 = np.exp(np.log(spots) @ weights)
    carry = rates - weights @ q - 0.5 * weights @ sigma**2 + 0.5 * basket_variance

    sqrt_t = np.sqrt(np.maximum(maturities, 1e-16))
    std = basket_sigma * sqrt_t
    d1 = (np.log(g0 / strikes) + (carry + 0.5 * basket_variance) * maturities) / std
    d2 = d1 - std
    discounted_k = strikes * np.exp(-rates * maturities)
    discounted_forward_component = g0 * np.exp((carry - rates) * maturities)
    return discounted_k * ndtr(-d2) - discounted_forward_component * ndtr(-d1)


class EuropeanGeometricBasketPricer:
    def __init__(self, config):
        self.config = config

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(parameters, dtype=float))
        n = self.config.n_assets
        return geometric_basket_put(
            x[:, :n], x[:, n], x[:, n + 1], x[:, n + 2],
            self.config.volatilities, self.config.correlation, self.config.dividends,
        )


class AmericanArithmeticBasketLSMC:
    """Reference NumPy LSMC pricer with deterministic common random numbers."""

    def __init__(self, config, n_paths=10_000, n_steps=30, seed=None):
        self.config = config
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.seed = config.seed if seed is None else int(seed)
        self._chol = np.linalg.cholesky(config.correlation)

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(parameters, dtype=float))
        return np.asarray([self._price_one(row) for row in x])

    def _price_one(self, x: np.ndarray) -> float:
        n = self.config.n_assets
        s0, strike, rate, maturity = x[:n], x[n], x[n + 1], x[n + 2]
        dt = maturity / self.n_steps
        rng = np.random.default_rng(self.seed)  # CRN makes the learned surface smoother.
        z = rng.standard_normal((self.n_steps, self.n_paths, n)) @ self._chol.T
        drift = (rate - self.config.dividends - 0.5 * self.config.volatilities**2) * dt
        diffusion = self.config.volatilities * np.sqrt(dt)
        log_returns = drift[None, None, :] + diffusion[None, None, :] * z
        paths = s0[None, None, :] * np.exp(np.cumsum(log_returns, axis=0))
        baskets = paths.mean(axis=2)
        payoff = np.maximum(strike - baskets, 0.0)

        cashflow = payoff[-1].copy()
        exercise_time = np.full(self.n_paths, self.n_steps, dtype=np.int64)
        for step in range(self.n_steps - 1, 0, -1):
            itm = payoff[step] > 0.0
            if np.count_nonzero(itm) < 3:
                continue
            state = baskets[step, itm]
            discounted = cashflow[itm] * np.exp(-rate * dt * (exercise_time[itm] - step))
            scale = max(strike, 1e-12)
            a = state / scale
            design = np.column_stack((np.ones_like(a), a, a * a))
            beta, *_ = np.linalg.lstsq(design, discounted, rcond=None)
            continuation = design @ beta
            exercise_local = payoff[step, itm] > continuation
            exercise_idx = np.flatnonzero(itm)[exercise_local]
            cashflow[exercise_idx] = payoff[step, exercise_idx]
            exercise_time[exercise_idx] = step

        discounted = cashflow * np.exp(-rate * dt * exercise_time)
        immediate = max(strike - float(np.mean(s0)), 0.0)
        return max(immediate, float(np.mean(discounted)))

