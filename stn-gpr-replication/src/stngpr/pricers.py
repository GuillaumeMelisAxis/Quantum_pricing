from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm, qmc


def geometric_basket_put(
    spots: np.ndarray,
    strikes: np.ndarray,
    rates: np.ndarray,
    maturities: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    dividends: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form put on an equally weighted geometric basket under GBM.

    ``volatilities`` is either one vector shared by every row, or a
    ``(m, n_assets)`` array giving each row its own volatility vector. The
    latter is what the volatility-extended experiment needs, since sigma is
    then a sampled coordinate rather than a fixed model assumption.
    """
    spots = np.atleast_2d(np.asarray(spots, dtype=float))
    m, n_assets = spots.shape
    strikes = np.broadcast_to(np.asarray(strikes, dtype=float), (m,))
    rates = np.broadcast_to(np.asarray(rates, dtype=float), (m,))
    maturities = np.broadcast_to(np.asarray(maturities, dtype=float), (m,))
    sigma = np.atleast_2d(np.asarray(volatilities, dtype=float))
    if sigma.shape[1] != n_assets or sigma.shape[0] not in (1, m):
        raise ValueError("volatilities must be one vector or one vector per row")
    q = np.zeros(n_assets) if dividends is None else np.asarray(dividends, dtype=float)
    weights = np.full(n_assets, 1.0 / n_assets)

    # w_i sigma_i C_ij w_j sigma_j, kept factored so no (m, n, n) array is built.
    scaled = sigma * weights
    basket_variance = np.sum(
        (scaled @ np.asarray(correlation, dtype=float)) * scaled, axis=1
    )
    basket_sigma = np.sqrt(basket_variance)
    g0 = np.exp(np.log(spots) @ weights)
    carry = rates - weights @ q - 0.5 * (sigma**2 @ weights) + 0.5 * basket_variance

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


class EuropeanArithmeticBasketQMC:
    """Deterministic Sobol-QMC pricer for an arithmetic-basket put.

    The same scrambled Sobol points are reused for every call.  This common-
    random-number contract is essential when the pricer is differentiated by
    bump-and-revalue or sampled by TT-cross.  A geometric-basket control
    variate is enabled by default and uses its exact closed-form expectation.
    """

    def __init__(
        self,
        config,
        n_paths: int = 2_048,
        seed: int | None = None,
        control_variate: bool = True,
        control_variate_beta: str | float = "estimated",
        batch_size: int = 64,
    ):
        self.config = config
        self.n_paths = int(n_paths)
        self.seed = config.seed if seed is None else int(seed)
        self.control_variate = bool(control_variate)
        if isinstance(control_variate_beta, str):
            if control_variate_beta not in {"estimated", "unit"}:
                raise ValueError(
                    "control_variate_beta must be 'estimated', 'unit' or a finite scalar"
                )
            self.control_variate_beta = control_variate_beta
        else:
            beta = float(control_variate_beta)
            if not np.isfinite(beta):
                raise ValueError("control_variate_beta must be finite")
            self.control_variate_beta = beta
        self.batch_size = int(batch_size)
        if self.n_paths <= 1 or self.n_paths & (self.n_paths - 1):
            raise ValueError("n_paths must be a power of two greater than one")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        exponent = int(np.log2(self.n_paths))
        engine = qmc.Sobol(
            d=config.n_assets,
            scramble=True,
            seed=self.seed,
        )
        uniforms = engine.random_base2(exponent)
        eps = np.finfo(float).eps
        normals = norm.ppf(np.clip(uniforms, eps, 1.0 - eps))
        self._correlated_normals = normals @ np.linalg.cholesky(
            config.correlation
        ).T

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(parameters, dtype=float))
        n = self.config.n_assets
        if x.shape[1] != n + 3:
            raise ValueError("parameters must contain spots, strike, rate and maturity")
        if np.any(x[:, : n + 1] <= 0.0) or np.any(x[:, n + 2] <= 0.0):
            raise ValueError("spots, strikes and maturities must be positive")

        values = np.empty(x.shape[0], dtype=float)
        sigma = np.asarray(self.config.volatilities, dtype=float)
        dividends = np.asarray(self.config.dividends, dtype=float)
        z = self._correlated_normals
        for start in range(0, len(x), self.batch_size):
            batch = x[start : start + self.batch_size]
            spots = batch[:, :n]
            strikes = batch[:, n]
            rates = batch[:, n + 1]
            maturities = batch[:, n + 2]
            drift = (
                rates[:, None]
                - dividends[None, :]
                - 0.5 * sigma[None, :] ** 2
            ) * maturities[:, None]
            diffusion = (
                np.sqrt(maturities)[:, None, None]
                * sigma[None, None, :]
                * z[None, :, :]
            )
            log_terminal = (
                np.log(spots)[:, None, :]
                + drift[:, None, :]
                + diffusion
            )
            terminal = np.exp(log_terminal)
            arithmetic = np.mean(terminal, axis=2)
            geometric = np.exp(np.mean(log_terminal, axis=2))
            discount = np.exp(-rates * maturities)
            arithmetic_payoff = discount[:, None] * np.maximum(
                strikes[:, None] - arithmetic,
                0.0,
            )
            estimate = np.mean(arithmetic_payoff, axis=1)

            if self.control_variate:
                geometric_payoff = discount[:, None] * np.maximum(
                    strikes[:, None] - geometric,
                    0.0,
                )
                if self.control_variate_beta == "estimated":
                    centered_a = arithmetic_payoff - np.mean(
                        arithmetic_payoff,
                        axis=1,
                        keepdims=True,
                    )
                    centered_g = geometric_payoff - np.mean(
                        geometric_payoff,
                        axis=1,
                        keepdims=True,
                    )
                    variance_g = np.mean(centered_g * centered_g, axis=1)
                    covariance = np.mean(centered_a * centered_g, axis=1)
                    beta = np.divide(
                        covariance,
                        variance_g,
                        out=np.zeros_like(covariance),
                        where=variance_g > 1e-20,
                    )
                elif self.control_variate_beta == "unit":
                    beta = np.ones(len(batch), dtype=float)
                else:
                    beta = np.full(
                        len(batch),
                        float(self.control_variate_beta),
                        dtype=float,
                    )
                exact_geometric = geometric_basket_put(
                    spots,
                    strikes,
                    rates,
                    maturities,
                    sigma,
                    self.config.correlation,
                    dividends,
                )
                estimate -= beta * (
                    np.mean(geometric_payoff, axis=1) - exact_geometric
                )

            values[start : start + len(batch)] = np.maximum(estimate, 0.0)
        return values


class AmericanArithmeticBasketLSMC:
    """Reference LSMC pricer with deterministic common random numbers.

    ``policy_mode='refit'`` reproduces the standard bump-and-revalue contract:
    every scenario estimates its own continuation regressions. ``'frozen'``
    estimates the policy on the first row of a vectorized call and reuses it for
    all remaining fixed-contract spot scenarios. ``'frozen_oos'`` additionally
    evaluates that policy on an independent path sample. The latter two modes
    are designed for Greek-noise ablations, not as automatic replacements for
    the standard LSMC estimator.
    """

    def __init__(
        self,
        config,
        n_paths=10_000,
        n_steps=30,
        seed=None,
        policy_mode="refit",
        evaluation_seed=None,
        evaluation_paths=None,
    ):
        self.config = config
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.seed = config.seed if seed is None else int(seed)
        self.policy_mode = str(policy_mode)
        if self.policy_mode not in {"refit", "frozen", "frozen_oos"}:
            raise ValueError(
                "policy_mode must be 'refit', 'frozen' or 'frozen_oos'"
            )
        if self.n_paths <= 2 or self.n_steps <= 1:
            raise ValueError("n_paths must exceed two and n_steps must exceed one")
        self.evaluation_paths = (
            self.n_paths if evaluation_paths is None else int(evaluation_paths)
        )
        if self.evaluation_paths <= 2:
            raise ValueError("evaluation_paths must exceed two")
        if evaluation_seed is None:
            evaluation_seed = (
                self.seed + 104_729
                if self.policy_mode == "frozen_oos"
                else self.seed
            )
        self.evaluation_seed = int(evaluation_seed)
        self._chol = np.linalg.cholesky(config.correlation)
        self._training_normals = self._draw_normals(self.n_paths, self.seed)
        self._evaluation_normals = (
            self._draw_normals(self.evaluation_paths, self.evaluation_seed)
            if self.policy_mode == "frozen_oos"
            else self._training_normals
        )

    def _draw_normals(self, n_paths, seed):
        rng = np.random.default_rng(seed)
        return (
            rng.standard_normal((self.n_steps, n_paths, self.config.n_assets))
            @ self._chol.T
        )

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(parameters, dtype=float))
        n = self.config.n_assets
        if x.shape[1] != n + 3:
            raise ValueError("parameters must contain spots, strike, rate and maturity")
        if np.any(x[:, : n + 1] <= 0.0) or np.any(x[:, n + 2] <= 0.0):
            raise ValueError("spots, strikes and maturities must be positive")
        if self.policy_mode == "refit":
            return np.asarray(
                [self._price_one(row, self._training_normals)[0] for row in x]
            )

        if not np.allclose(x[:, n:], x[0, n:], rtol=0.0, atol=1e-13):
            raise ValueError(
                "a frozen-policy batch must keep strike, rate and maturity fixed"
            )
        _, policy = self._price_one(
            x[0],
            self._training_normals,
            fit_policy=True,
        )
        return np.asarray([
            self._price_one(
                row,
                self._evaluation_normals,
                policy=policy,
            )[0]
            for row in x
        ])

    def _price_one(
        self,
        x: np.ndarray,
        normals: np.ndarray,
        policy=None,
        fit_policy=False,
    ):
        n = self.config.n_assets
        s0, strike, rate, maturity = x[:n], x[n], x[n + 1], x[n + 2]
        dt = maturity / self.n_steps
        drift = (rate - self.config.dividends - 0.5 * self.config.volatilities**2) * dt
        diffusion = self.config.volatilities * np.sqrt(dt)
        log_returns = drift[None, None, :] + diffusion[None, None, :] * normals
        paths = s0[None, None, :] * np.exp(np.cumsum(log_returns, axis=0))
        baskets = paths.mean(axis=2)
        payoff = np.maximum(strike - baskets, 0.0)

        cashflow = payoff[-1].copy()
        exercise_time = np.full(normals.shape[1], self.n_steps, dtype=np.int64)
        fitted_policy = [None] * (self.n_steps + 1) if fit_policy else None
        for step in range(self.n_steps - 1, 0, -1):
            itm = payoff[step] > 0.0
            if np.count_nonzero(itm) < 3:
                continue
            state = baskets[step, itm]
            scale = max(strike, 1e-12)
            a = state / scale
            design = np.column_stack((np.ones_like(a), a, a * a))
            if policy is None:
                discounted = cashflow[itm] * np.exp(
                    -rate * dt * (exercise_time[itm] - step)
                )
                beta, *_ = np.linalg.lstsq(design, discounted, rcond=None)
                if fitted_policy is not None:
                    fitted_policy[step] = beta.copy()
            else:
                beta = policy[step]
                if beta is None:
                    continue
            continuation = design @ beta
            exercise_local = payoff[step, itm] > continuation
            exercise_idx = np.flatnonzero(itm)[exercise_local]
            cashflow[exercise_idx] = payoff[step, exercise_idx]
            exercise_time[exercise_idx] = step

        discounted = cashflow * np.exp(-rate * dt * exercise_time)
        immediate = max(strike - float(np.mean(s0)), 0.0)
        value = max(immediate, float(np.mean(discounted)))
        return value, fitted_policy


class EuropeanGeometricBasketVolPricer:
    """Closed-form pricer over interleaved (S_1, sigma_1, ..., K, r, T) inputs.

    Each underlying carries its own volatility coordinate, so the price is no
    longer conditioned on a fixed volatility vector taken from the config.
    """

    def __init__(self, config):
        self.config = config
        self.spot_columns = list(config.spot_columns)
        self.volatility_columns = list(config.volatility_columns)

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(parameters, dtype=float))
        if x.shape[1] != self.config.n_dimensions:
            raise ValueError("wrong parameter dimension for the extended pricer")
        return geometric_basket_put(
            x[:, self.spot_columns],
            x[:, self.config.strike_column],
            x[:, self.config.rate_column],
            x[:, self.config.maturity_column],
            x[:, self.volatility_columns],
            self.config.correlation,
            self.config.dividends,
        )


def arithmetic_basket_put_levy(
    spots: np.ndarray,
    strikes: np.ndarray,
    rates: np.ndarray,
    maturities: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    dividends: np.ndarray | None = None,
) -> np.ndarray:
    """Levy (1992) lognormal moment-matched put on an arithmetic basket.

    The arithmetic basket has no closed form. Matching the first two moments of
    the terminal basket to a lognormal gives a Black-76 formula that is exact in
    its homogeneity and carries the true ``S_i sigma_i`` cross terms through
    ``exp(sigma_i sigma_j C_ij T)``. That coupling structure, not the last digit
    of the price, is what the tensor-rank experiments measure.
    """
    spots = np.atleast_2d(np.asarray(spots, dtype=float))
    m, n_assets = spots.shape
    strikes = np.broadcast_to(np.asarray(strikes, dtype=float), (m,))
    rates = np.broadcast_to(np.asarray(rates, dtype=float), (m,))
    maturities = np.broadcast_to(np.asarray(maturities, dtype=float), (m,))
    sigma = np.atleast_2d(np.asarray(volatilities, dtype=float))
    if sigma.shape[1] != n_assets or sigma.shape[0] not in (1, m):
        raise ValueError("volatilities must be one vector or one vector per row")
    q = np.zeros(n_assets) if dividends is None else np.asarray(dividends, dtype=float)
    weights = np.full(n_assets, 1.0 / n_assets)
    corr = np.asarray(correlation, dtype=float)

    t = maturities[:, None]
    forwards = weights * spots * np.exp((rates[:, None] - q) * t)   # (m, n)
    forward = forwards.sum(axis=1)
    # second moment of the terminal basket, sum_ij F_i F_j exp(sigma_i sigma_j C_ij T)
    covariance = sigma[:, :, None] * sigma[:, None, :] * corr
    second_moment = np.einsum(
        "mi,mj,mij->m",
        forwards,
        forwards,
        np.exp(covariance * maturities[:, None, None]),
    )
    variance = np.log(np.maximum(second_moment / forward**2, 1.0 + 1e-14))
    std = np.sqrt(variance)
    d1 = (np.log(forward / strikes) + 0.5 * variance) / std
    d2 = d1 - std
    discount = np.exp(-rates * maturities)
    return discount * (strikes * ndtr(-d2) - forward * ndtr(-d1))


class EuropeanArithmeticBasketVolPricer:
    """Levy arithmetic-basket pricer over interleaved or blocked coordinates."""

    def __init__(self, config):
        self.config = config
        self.spot_columns = list(config.spot_columns)
        self.volatility_columns = list(config.volatility_columns)

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(parameters, dtype=float))
        if x.shape[1] != self.config.n_dimensions:
            raise ValueError("wrong parameter dimension for the extended pricer")
        return arithmetic_basket_put_levy(
            x[:, self.spot_columns],
            x[:, self.config.strike_column],
            x[:, self.config.rate_column],
            x[:, self.config.maturity_column],
            x[:, self.volatility_columns],
            self.config.correlation,
            self.config.dividends,
        )


def basket_put_monte_carlo(
    spots: np.ndarray,
    strikes: np.ndarray,
    rates: np.ndarray,
    maturities: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    n_paths: int,
    basket_kind: str = "geometric",
    dividends: np.ndarray | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
):
    """Terminal-value Monte Carlo for a European basket put, one row at a time.

    A European basket needs only the terminal joint distribution, so this is a
    single-step simulation of the five correlated assets. The geometric case
    also admits a one-dimensional reduction, but a general engine pays for all
    the assets, which is the cost this benchmark is meant to measure.

    Returns the price estimate and its Monte Carlo standard error.
    """
    spots = np.atleast_2d(np.asarray(spots, dtype=float))
    m, n_assets = spots.shape
    strikes = np.broadcast_to(np.asarray(strikes, dtype=float), (m,))
    rates = np.broadcast_to(np.asarray(rates, dtype=float), (m,))
    maturities = np.broadcast_to(np.asarray(maturities, dtype=float), (m,))
    sigma = np.atleast_2d(np.asarray(volatilities, dtype=float))
    if sigma.shape[0] == 1:
        sigma = np.broadcast_to(sigma, (m, n_assets))
    q = np.zeros(n_assets) if dividends is None else np.asarray(dividends, dtype=float)
    weights = np.full(n_assets, 1.0 / n_assets)
    chol = np.linalg.cholesky(np.asarray(correlation, dtype=float))
    generator = np.random.default_rng(seed) if rng is None else rng

    prices = np.empty(m, dtype=float)
    errors = np.empty(m, dtype=float)
    for i in range(m):
        t = maturities[i]
        z = generator.standard_normal((int(n_paths), n_assets)) @ chol.T
        drift = (rates[i] - q - 0.5 * sigma[i] ** 2) * t
        terminal = spots[i] * np.exp(drift + sigma[i] * np.sqrt(t) * z)
        if basket_kind == "geometric":
            basket = np.exp(np.log(terminal) @ weights)
        elif basket_kind == "arithmetic":
            basket = terminal @ weights
        else:
            raise ValueError("basket_kind must be 'geometric' or 'arithmetic'")
        payoff = np.maximum(strikes[i] - basket, 0.0) * np.exp(-rates[i] * t)
        prices[i] = payoff.mean()
        errors[i] = payoff.std(ddof=1) / np.sqrt(int(n_paths))
    return prices, errors
