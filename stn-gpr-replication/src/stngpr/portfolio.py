from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True) #@dataclass est un décorateur qui permet de simplifier la création de classe qund on représente une clase avec principalement des données  
class BasketOptionTrade:
    """One basket-option position described in market coordinates.

    The contractual strike and maturity are fixed attributes of the trade. The
    spots, the rate and the residual maturity are supplied by the risk engine at
    valuation time, so the same trade can be repriced in any market state.
    """

    strike: float
    maturity: float
    quantity: float = 1.0
    label: str | None = None

    def __post_init__(self) -> None:
        for name in ("strike", "maturity", "quantity"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.strike <= 0.0:
            raise ValueError("strike must be strictly positive")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be strictly positive")

    def describe(self) -> dict:
        return {
            "label": self.label,
            "strike": self.strike,
            "maturity": self.maturity,
            "quantity": self.quantity,
        }


@dataclass(frozen=True)
class Portfolio:
    """Book of basket options sharing one set of underlyings."""

    trades: tuple[BasketOptionTrade, ...]

    def __post_init__(self) -> None:
        trades = tuple(self.trades)
        if not trades:
            raise ValueError("a portfolio must contain at least one trade")
        if any(not isinstance(trade, BasketOptionTrade) for trade in trades):
            raise ValueError("every position must be a BasketOptionTrade")
        object.__setattr__(self, "trades", trades)

    @classmethod
    def from_arrays(
        cls, strikes, maturities, quantities=1.0, labels=None
    ) -> "Portfolio":
        strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
        maturities = np.broadcast_to(
            np.asarray(maturities, dtype=float), strikes.shape
        )
        quantities = np.broadcast_to(
            np.asarray(quantities, dtype=float), strikes.shape
        )
        if labels is None:
            labels = [None] * strikes.size
        elif len(labels) != strikes.size:
            raise ValueError("labels must contain one entry per trade")
        return cls(tuple(
            BasketOptionTrade(k, t, q, label)
            for k, t, q, label in zip(strikes, maturities, quantities, labels)
        ))

    def __len__(self) -> int:
        return len(self.trades)

    @property
    def strikes(self) -> np.ndarray:
        return np.asarray([trade.strike for trade in self.trades], dtype=float)

    @property
    def maturities(self) -> np.ndarray:
        return np.asarray([trade.maturity for trade in self.trades], dtype=float)

    @property
    def quantities(self) -> np.ndarray:
        return np.asarray([trade.quantity for trade in self.trades], dtype=float)

    def split_at_horizon(
        self,
        horizon: float,
        minimum_residual_maturity: float = 1.0 / 365.0,
    ) -> tuple["Portfolio", tuple[BasketOptionTrade, ...]]:
        """Separate positions still alive after the horizon from expiring ones.

        Returns the sub-portfolio that full revaluation can price and the trades
        it cannot. The split is returned rather than applied silently: dropping
        a position changes the book, so the caller must record what was removed
        before reading any risk number from the remainder.
        """
        horizon = float(horizon)
        if horizon < 0.0:
            raise ValueError("horizon must be nonnegative")
        residual = self.maturities - horizon
        alive = residual >= float(minimum_residual_maturity)
        if not np.any(alive):
            raise ValueError("every trade expires within the risk horizon")
        live = Portfolio(tuple(
            trade for trade, keep in zip(self.trades, alive) if keep
        ))
        expiring = tuple(
            trade for trade, keep in zip(self.trades, alive) if not keep
        )
        return live, expiring

    def market_parameters(
        self,
        spots: np.ndarray,
        rates,
        maturity_shift: float = 0.0,
        minimum_residual_maturity: float = 1.0 / 365.0,
    ) -> np.ndarray:
        """Return the (n_states, n_trades, n_assets + 3) valuation tensor.

        Column layout is the package-wide market convention
        (S_1, ..., S_d, K, r, T). `maturity_shift` ages every trade by the risk
        horizon; trades whose residual maturity would fall below
        `minimum_residual_maturity` are rejected rather than clipped, because a
        position expiring inside the horizon requires a payoff treatment that
        full revaluation of a live option cannot provide.
        """
        spots = np.atleast_2d(np.asarray(spots, dtype=float))
        if spots.ndim != 2 or spots.shape[1] < 1:
            raise ValueError("spots must be a (n_states, n_assets) array")
        if np.any(spots <= 0.0):
            raise ValueError("spots must be strictly positive")
        n_states, n_assets = spots.shape
        rates = np.broadcast_to(
            np.asarray(rates, dtype=float), (n_states,)
        )
        shift = float(maturity_shift)
        if shift < 0.0:
            raise ValueError("maturity_shift must be nonnegative")
        residual = self.maturities - shift
        if np.any(residual < float(minimum_residual_maturity)):
            raise ValueError(
                "at least one trade expires within the risk horizon; shorten "
                "the horizon or value expiring positions separately"
            )

        parameters = np.empty((n_states, len(self), n_assets + 3), dtype=float)
        parameters[:, :, :n_assets] = spots[:, None, :]
        parameters[:, :, n_assets] = self.strikes[None, :]
        parameters[:, :, n_assets + 1] = rates[:, None]
        parameters[:, :, n_assets + 2] = residual[None, :]
        return parameters

    def describe(self) -> list[dict]:
        return [trade.describe() for trade in self.trades]
