"""
monte_carlo_pricer.py
======================

Monte Carlo benchmark for discretely monitored Down-and-Out Call options
under geometric Brownian motion (GBM):

    dS_t = r S_t dt + sigma S_t dW_t

Used in Part 3 (Experiments 2 and 4) to validate the discrete
Hamiltonian pricing scheme of `hamiltonian_pricer.py`.

Author: Guillaume Melis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from hamiltonian_pricer import MonitoringFrequency, build_monitoring_schedule


@dataclass
class MonteCarloResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int


@dataclass
class MonteCarloPricer:
    """Monte Carlo pricer for discretely monitored Down-and-Out Calls
    under GBM, with exact (lognormal) path simulation at the monitoring
    dates.

    Parameters
    ----------
    S0, K, B, sigma, r, T : float
        Standard contract / market parameters.
    seed : int
        Random seed for reproducibility.
    """

    S0: float
    K: float
    B: float
    sigma: float
    r: float
    T: float
    seed: int = 42

    def simulate_paths(
        self,
        n_paths: int,
        frequency: MonitoringFrequency = "daily",
        n_monitoring: int | None = None,
    ) -> np.ndarray:
        """Simulate GBM paths sampled exactly at the monitoring dates.

        Returns an array of shape (n_paths, M+1) including S0 at t=0.
        """
        schedule = build_monitoring_schedule(self.T, frequency, n_monitoring)
        dts = np.diff(schedule)
        M = len(dts)

        rng = np.random.default_rng(self.seed)
        Z = rng.standard_normal(size=(n_paths, M))

        log_increments = (self.r - 0.5 * self.sigma ** 2) * dts + self.sigma * np.sqrt(dts) * Z
        log_S = np.log(self.S0) + np.cumsum(log_increments, axis=1)
        log_S = np.concatenate([np.full((n_paths, 1), np.log(self.S0)), log_S], axis=1)
        return np.exp(log_S)

    def price(
        self,
        n_paths: int = 100_000,
        frequency: MonitoringFrequency = "daily",
        n_monitoring: int | None = None,
        confidence: float = 0.95,
        chunk_size: int = 100_000,
    ) -> MonteCarloResult:
        """Price a Down-and-Out Call via Monte Carlo and return price,
        standard error and confidence interval.

        Simulation is processed in chunks of `chunk_size` paths to keep
        memory usage bounded for large `n_paths` (e.g. 500_000 -
        1_000_000), while remaining exactly reproducible (a single RNG,
        seeded once, is advanced across chunks).
        """
        schedule = build_monitoring_schedule(self.T, frequency, n_monitoring)
        dts = np.diff(schedule)
        rng = np.random.default_rng(self.seed)

        total = 0.0
        total_sq = 0.0
        remaining = n_paths
        while remaining > 0:
            batch = min(chunk_size, remaining)
            Z = rng.standard_normal(size=(batch, len(dts)))
            log_increments = (self.r - 0.5 * self.sigma ** 2) * dts + self.sigma * np.sqrt(dts) * Z
            log_S = np.log(self.S0) + np.cumsum(log_increments, axis=1)

            knocked_out = np.any(log_S <= np.log(self.B), axis=1)
            terminal = np.exp(log_S[:, -1])
            payoff = np.where(knocked_out, 0.0, np.maximum(terminal - self.K, 0.0))
            discounted = np.exp(-self.r * self.T) * payoff

            total += discounted.sum()
            total_sq += (discounted ** 2).sum()
            remaining -= batch

        price = total / n_paths
        var = total_sq / n_paths - price ** 2
        var = max(var, 0.0) * n_paths / (n_paths - 1)
        se = np.sqrt(var / n_paths)

        from scipy.stats import norm

        z = norm.ppf(0.5 + confidence / 2)
        ci_low, ci_high = price - z * se, price + z * se

        return MonteCarloResult(
            price=price, std_error=se, ci_low=ci_low, ci_high=ci_high, n_paths=n_paths
        )