"""
analytic_pricing.py
=====================

Closed-form Black-Scholes benchmarks used in Experiment 1 (Part 3):

* `bs_call_price`: standard European call (used as a sanity check that
  the Hamiltonian scheme collapses to vanilla Black-Scholes when the
  barrier is pushed far away / never breached).
* `down_and_out_call_price`: classical continuously-monitored
  Down-and-Out Call closed form (Merton / Reiner-Rubinstein), used as
  the continuous-time reference against which the discretely monitored
  Hamiltonian scheme is expected to converge as the monitoring
  frequency increases.

Author: Guillaume Melis
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Standard Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def down_and_out_call_price(
    S: float, K: float, B: float, r: float, sigma: float, T: float
) -> float:
    """Closed-form price of a continuously-monitored Down-and-Out Call.

    Standard Reiner-Rubinstein / Merton formula. Valid for B <= K and
    B < S (the case relevant to the numerical experiments below); if
    the spot has already breached the barrier the option is worthless.
    """
    if S <= B:
        return 0.0

    lam = (r + 0.5 * sigma ** 2) / sigma ** 2
    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
    y1 = np.log(B ** 2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)

    if B <= K:
        x2 = np.log(S / B) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        y2 = np.log(B / S) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)

        c = S * norm.cdf(x1) - K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
        c_di = S * (B / S) ** (2 * lam) * norm.cdf(y1) - K * np.exp(-r * T) * (
            B / S
        ) ** (2 * lam - 2) * norm.cdf(y1 - sigma * np.sqrt(T))
        # Down-and-in call (B <= K case)
        c_di = S * (B / S) ** (2 * lam) * norm.cdf(y1) - K * np.exp(-r * T) * (
            B / S
        ) ** (2 * lam - 2) * norm.cdf(y1 - sigma * np.sqrt(T))
        return max(c - c_di, 0.0)
    else:
        # B > K case decomposes differently (down-and-out via vanilla call
        # minus a down-and-in built from a digital + asset-or-nothing
        # combination). For the numerical experiments here we restrict
        # to B <= K which covers all tested barrier ratios.
        raise NotImplementedError("Closed form only implemented for B <= K in this study")
