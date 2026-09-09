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



@dataclass(frozen=True)
class VolatilityExtendedConfig:
    """Paper setup extended with one volatility coordinate per underlying.

    Market coordinates interleave spots and volatilities,
    ``(S_1, sigma_1, ..., S_n, sigma_n, K, r, T)``, so each asset's volatility
    sits next to its own spot in the QTT core ordering: the pair (S_i, sigma_i)
    is where the strongest local coupling lives, and adjacent modes are the ones
    a tensor train can correlate at low rank.

    Volatilities are gridded on ``u_i = log(sigma_i)`` because the range spans
    more than one decade; a uniform sigma grid would waste nodes at high
    volatility, where the price is nearly linear in sigma, and starve the low
    volatility end, where it is not.

    ``layout`` selects the mode ordering. "interleaved" pairs each spot with its
    own volatility; "blocked" puts all spots first, then all volatilities. Which
    one compresses better depends on the payoff: a coupling costs rank on every
    bond it has to cross, so contiguity of the dominant coupling is what matters.
    """

    n_assets: int = 5
    layout: str = "interleaved"
    dividends: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=float)
    )
    correlation: np.ndarray = field(
        default_factory=lambda: equicorrelation(5, 0.30)
    )
    spot_bounds: tuple[float, float] = (5.0, 150.0)
    volatility_bounds: tuple[float, float] = (0.05, 0.80)
    strike_bounds: tuple[float, float] = (1.0, 200.0)
    rate_bounds: tuple[float, float] = (0.005, 0.08)
    maturity_bounds: tuple[float, float] = (1.0 / 365.0, 3.0)
    spot_nodes: int = 32
    volatility_nodes: int = 16
    strike_nodes: int = 64
    rate_nodes: int = 8
    maturity_nodes: int = 8
    seed: int = 20260327

    def __post_init__(self) -> None:
        if self.n_assets < 1:
            raise ValueError("n_assets must be positive")
        if self.layout not in ("interleaved", "blocked"):
            raise ValueError("layout must be 'interleaved' or 'blocked'")
        if any(n <= 1 or n & (n - 1) for n in self.physical_shape):
            raise ValueError("QTT requires every physical mode size to be a power of two")
        if not 0.0 < self.volatility_bounds[0] < self.volatility_bounds[1]:
            raise ValueError("volatility bounds must be positive and increasing")
        if self.dividends.shape != (self.n_assets,):
            raise ValueError("invalid dividend vector")
        if self.correlation.shape != (self.n_assets, self.n_assets):
            raise ValueError("invalid correlation matrix")
        np.linalg.cholesky(self.correlation)

    @property
    def n_dimensions(self) -> int:
        return 2 * self.n_assets + 3

    @property
    def spot_columns(self) -> tuple[int, ...]:
        if self.layout == "blocked":
            return tuple(range(self.n_assets))
        return tuple(2 * i for i in range(self.n_assets))

    @property
    def volatility_columns(self) -> tuple[int, ...]:
        if self.layout == "blocked":
            return tuple(range(self.n_assets, 2 * self.n_assets))
        return tuple(2 * i + 1 for i in range(self.n_assets))

    @property
    def strike_column(self) -> int:
        return 2 * self.n_assets

    @property
    def rate_column(self) -> int:
        return 2 * self.n_assets + 1

    @property
    def maturity_column(self) -> int:
        return 2 * self.n_assets + 2

    @property
    def log_volatility_bounds(self) -> tuple[float, float]:
        low, high = self.volatility_bounds
        return (float(np.log(low)), float(np.log(high)))

    def _by_column(self, spot_value, volatility_value):
        """Place per-asset values at the columns the layout assigns them."""
        out = [None] * self.n_dimensions
        for column in self.spot_columns:
            out[column] = spot_value
        for column in self.volatility_columns:
            out[column] = volatility_value
        return out

    @property
    def physical_shape(self) -> tuple[int, ...]:
        out = self._by_column(self.spot_nodes, self.volatility_nodes)
        out[self.strike_column] = self.strike_nodes
        out[self.rate_column] = self.rate_nodes
        out[self.maturity_column] = self.maturity_nodes
        return tuple(out)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Market-coordinate bounds, with sigma (not log sigma) for volatility."""
        out = self._by_column(self.spot_bounds, self.volatility_bounds)
        out[self.strike_column] = self.strike_bounds
        out[self.rate_column] = self.rate_bounds
        out[self.maturity_column] = self.maturity_bounds
        return tuple(out)

    @property
    def qtt_bits(self) -> tuple[int, ...]:
        return tuple(int(np.log2(n)) for n in self.physical_shape)

    @property
    def qtt_shape(self) -> tuple[int, ...]:
        return (2,) * sum(self.qtt_bits)
