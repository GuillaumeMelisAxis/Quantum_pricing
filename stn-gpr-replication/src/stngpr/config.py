from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def equicorrelation(n_assets: int, rho: float) -> np.ndarray:
    corr = np.full((n_assets, n_assets), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    return corr


@dataclass(frozen=True)
class PaperConfig:
    """Reconstruction assumptions for parameters omitted by the paper."""

    n_assets: int = 5
    volatilities: np.ndarray = field(
        default_factory=lambda: np.full(5, 0.20, dtype=float)
    )
    dividends: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=float)
    )
    correlation: np.ndarray = field(
        default_factory=lambda: equicorrelation(5, 0.30)
    )
    spot_bounds: tuple[float, float] = (5.0, 150.0)
    strike_bounds: tuple[float, float] = (1.0, 200.0)
    rate_bounds: tuple[float, float] = (0.005, 0.08)
    maturity_bounds: tuple[float, float] = (1.0 / 365.0, 3.0)
    physical_shape: tuple[int, ...] = (32, 32, 32, 32, 32, 64, 8, 8)
    seed: int = 20260327

    def __post_init__(self) -> None:
        if len(self.physical_shape) != self.n_assets + 3:
            raise ValueError("physical_shape must contain spots, strike, rate and T")
        if any(n <= 1 or n & (n - 1) for n in self.physical_shape):
            raise ValueError("QTT requires every physical mode size to be a power of two")
        if self.volatilities.shape != (self.n_assets,):
            raise ValueError("invalid volatility vector")
        if self.dividends.shape != (self.n_assets,):
            raise ValueError("invalid dividend vector")
        if self.correlation.shape != (self.n_assets, self.n_assets):
            raise ValueError("invalid correlation matrix")
        np.linalg.cholesky(self.correlation)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return (
            *((self.spot_bounds,) * self.n_assets),
            self.strike_bounds,
            self.rate_bounds,
            self.maturity_bounds,
        )

    @property
    def qtt_bits(self) -> tuple[int, ...]:
        return tuple(int(np.log2(n)) for n in self.physical_shape)

    @property
    def qtt_shape(self) -> tuple[int, ...]:
        return (2,) * sum(self.qtt_bits)

