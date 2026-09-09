from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GBMScenarioGenerator:
    """Correlated geometric Brownian motion for risk-factor scenarios.

    The horizon distribution of a GBM is known in closed form, so scenarios are
    drawn in a single exact step rather than by discretizing a path. This is
    exact for any payoff that only depends on the state at the risk horizon,
    which covers full revaluation of European and American basket options.
    """

    volatilities: np.ndarray
    correlation: np.ndarray
    dividends: np.ndarray | None = None
    drift: np.ndarray | None = None
    _cholesky: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sigma = np.asarray(self.volatilities, dtype=float)
        if sigma.ndim != 1 or sigma.size < 1:
            raise ValueError("volatilities must be a non-empty vector")
        if np.any(sigma < 0.0):
            raise ValueError("volatilities must be nonnegative")
        n_assets = sigma.size

        corr = np.asarray(self.correlation, dtype=float)
        if corr.shape != (n_assets, n_assets):
            raise ValueError("correlation has an incompatible shape")
        chol = np.linalg.cholesky(corr)

        q = (
            np.zeros(n_assets, dtype=float)
            if self.dividends is None
            else np.asarray(self.dividends, dtype=float)
        )
        if q.shape != (n_assets,):
            raise ValueError("dividends has an incompatible shape")

        drift = self.drift
        if drift is not None:
            drift = np.broadcast_to(
                np.asarray(drift, dtype=float), (n_assets,)
            ).copy()

        object.__setattr__(self, "volatilities", sigma)
        object.__setattr__(self, "correlation", corr)
        object.__setattr__(self, "dividends", q)
        object.__setattr__(self, "drift", drift)
        object.__setattr__(self, "_cholesky", chol)

    @classmethod
    def from_config(cls, config, drift=None) -> "GBMScenarioGenerator":
        """Build a generator from the pricing assumptions of a PaperConfig."""
        return cls(
            volatilities=config.volatilities,
            correlation=config.correlation,
            dividends=config.dividends,
            drift=drift,
        )

    @property
    def n_assets(self) -> int:
        return int(np.asarray(self.volatilities).size)

    def log_drift(self, rate: float) -> np.ndarray:
        """Annualized drift of the log-spot.

        With `drift=None` the risk-neutral drift `r - q - sigma^2 / 2` is used.
        Supplying an explicit real-world expected log-return is the correct
        choice for long horizons; over a ten-day horizon the two differ by an
        amount that is small relative to the diffusion term.
        """
        sigma = np.asarray(self.volatilities, dtype=float)
        if self.drift is not None:
            return np.asarray(self.drift, dtype=float)
        q = np.asarray(self.dividends, dtype=float)
        return float(rate) - q - 0.5 * sigma**2

    def simulate(
        self,
        spots: np.ndarray,
        horizon: float,
        n_scenarios: int,
        rate: float = 0.0,
        rng=None,
        antithetic: bool = True,
    ) -> np.ndarray:
        """Return (n_scenarios, n_assets) spots at the risk horizon."""
        spots = np.asarray(spots, dtype=float).reshape(-1)
        if spots.size != self.n_assets:
            raise ValueError("spots must contain one level per asset")
        if np.any(spots <= 0.0):
            raise ValueError("spots must be strictly positive")
        horizon = float(horizon)
        if horizon <= 0.0:
            raise ValueError("horizon must be strictly positive")
        n_scenarios = int(n_scenarios)
        if n_scenarios < 1:
            raise ValueError("n_scenarios must be at least one")
        rng = np.random.default_rng() if rng is None else rng

        if antithetic:
            half = (n_scenarios + 1) // 2
            base = rng.standard_normal((half, self.n_assets))
            normals = np.concatenate((base, -base), axis=0)[:n_scenarios]
        else:
            normals = rng.standard_normal((n_scenarios, self.n_assets))

        shocks = normals @ np.asarray(self._cholesky).T
        log_returns = (
            self.log_drift(rate) * horizon
            + np.asarray(self.volatilities, dtype=float)
            * np.sqrt(horizon)
            * shocks
        )
        return spots[None, :] * np.exp(log_returns)

    def describe(self, rate: float, horizon: float) -> dict:
        return {
            "model": "one-step exact correlated GBM",
            "horizon_years": float(horizon),
            "volatilities": np.asarray(self.volatilities).tolist(),
            "dividends": np.asarray(self.dividends).tolist(),
            "log_drift": np.asarray(self.log_drift(rate)).tolist(),
            "measure": "risk-neutral" if self.drift is None else "user-supplied drift",
        }
