from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def geometric_basket_effective_parameters(
    rate: float,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    dividends: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return effective volatility and carry of a geometric basket under GBM."""
    sigma = np.asarray(volatilities, dtype=float)
    n_assets = sigma.size
    corr = np.asarray(correlation, dtype=float)
    if corr.shape != (n_assets, n_assets):
        raise ValueError("correlation has an incompatible shape")
    q = (
        np.zeros(n_assets, dtype=float)
        if dividends is None
        else np.asarray(dividends, dtype=float)
    )
    if q.shape != (n_assets,):
        raise ValueError("dividends has an incompatible shape")
    w = (
        np.full(n_assets, 1.0 / n_assets, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if w.shape != (n_assets,) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must have one entry per asset and sum to one")

    covariance = np.outer(sigma, sigma) * corr
    variance = float(w @ covariance @ w)
    if variance <= 0.0:
        raise ValueError("the effective basket variance must be positive")
    carry = float(rate - w @ q - 0.5 * w @ sigma**2 + 0.5 * variance)
    return float(np.sqrt(variance)), carry


def geometric_basket_normalized_put(
    log_moneyness: np.ndarray,
    maturities: np.ndarray,
    rate: float,
    basket_volatility: float,
    basket_carry: float,
) -> np.ndarray:
    """Normalized geometric-basket put price u=P/G0 on broadcast inputs."""
    m, t = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(maturities, dtype=float),
    )
    t = np.maximum(t, np.finfo(float).tiny)
    sigma = float(basket_volatility)
    std = sigma * np.sqrt(t)
    d1 = (-m + (float(basket_carry) + 0.5 * sigma**2) * t) / std
    d2 = d1 - std
    return (
        np.exp(m - float(rate) * t) * ndtr(-d2)
        - np.exp((float(basket_carry) - float(rate)) * t) * ndtr(-d1)
    )


def geometric_basket_log_moneyness_convexity(
    log_moneyness: np.ndarray,
    maturities: np.ndarray,
    rate: float,
    basket_volatility: float,
    basket_carry: float,
) -> np.ndarray:
    r"""Return chi=(K^2/G0) d^2P/dK^2 = u_mm-u_m for a geometric put."""
    m, t = np.broadcast_arrays(
        np.asarray(log_moneyness, dtype=float),
        np.asarray(maturities, dtype=float),
    )
    t = np.maximum(t, np.finfo(float).tiny)
    sigma = float(basket_volatility)
    std = sigma * np.sqrt(t)
    d1 = (-m + (float(basket_carry) + 0.5 * sigma**2) * t) / std
    d2 = d1 - std
    density = np.exp(-0.5 * d2**2) / np.sqrt(2.0 * np.pi)
    return np.exp(m - float(rate) * t) * density / std


def geometric_basket_convexity_ridge(
    maturities: np.ndarray,
    basket_volatility: float,
    basket_carry: float,
) -> np.ndarray:
    """Log-moneyness location maximizing the dimensionless strike convexity."""
    t = np.asarray(maturities, dtype=float)
    sigma = float(basket_volatility)
    return (float(basket_carry) + 0.5 * sigma**2) * t
