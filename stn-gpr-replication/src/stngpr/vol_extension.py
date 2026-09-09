"""Volatility-extended European experiment.

The paper prices a geometric basket on ``(S_1, ..., S_n, K, r, T)`` with a
volatility vector frozen in the config. This module adds one volatility
coordinate per underlying, so the surrogate learns a ``2n + 3`` dimensional
price surface instead of an ``n + 3`` dimensional slice of it.

Two conventions matter, and both are deliberate:

* **interleaving** - market coordinates are ordered
  ``(S_1, sigma_1, S_2, sigma_2, ..., S_n, sigma_n, K, r, T)`` rather than
  blocked as ``(S_1..S_n, sigma_1..sigma_n, K, r, T)``. A tensor train only
  compresses cheaply along the mode ordering it is given, and the dominant
  coupling is between an asset and its own volatility.
* **log volatility** - the grid is uniform in ``u_i = log(sigma_i)`` over
  ``[log 0.05, log 0.80]``. The price is far more sensitive to sigma at the low
  end of that range, and a log axis spends nodes accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coordinates import GRID_MODES
from .grids import QTTGrid, short_maturity_axis, sinh_centered_axis


@dataclass(frozen=True)
class InterleavedVolTransform:
    """Market <-> model transform for interleaved (spot, volatility) inputs.

    Market: ``(S_1, sigma_1, ..., S_n, sigma_n, K, r, T)``.
    Model:  ``(S_1, log sigma_1, ..., S_n, log sigma_n, K or log(K/G), r, T)``.
    """

    n_assets: int
    spot_columns: tuple[int, ...]
    volatility_columns: tuple[int, ...]
    strike_column: int
    basket_kind: str
    use_moneyness: bool
    log_volatility: bool = True

    def _basket(self, spots: np.ndarray) -> np.ndarray:
        if self.basket_kind == "geometric":
            return np.exp(np.mean(np.log(spots), axis=1))
        if self.basket_kind == "arithmetic":
            return np.mean(spots, axis=1)
        raise ValueError("basket_kind must be 'geometric' or 'arithmetic'")

    def to_model(self, market_parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(market_parameters, dtype=float)).copy()
        spots = x[:, list(self.spot_columns)]
        if self.log_volatility:
            x[:, list(self.volatility_columns)] = np.log(
                x[:, list(self.volatility_columns)]
            )
        if self.use_moneyness:
            x[:, self.strike_column] = np.log(
                x[:, self.strike_column] / self._basket(spots)
            )
        return x

    def to_market(self, model_parameters: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(model_parameters, dtype=float)).copy()
        spots = z[:, list(self.spot_columns)]
        if self.log_volatility:
            z[:, list(self.volatility_columns)] = np.exp(
                z[:, list(self.volatility_columns)]
            )
        if self.use_moneyness:
            z[:, self.strike_column] = self._basket(spots) * np.exp(
                z[:, self.strike_column]
            )
        return z


def build_vol_extended_grid(
    config,
    mode: str,
    basket_kind: str = "geometric",
    volatility_scale: str = "log",
):
    """QTT grid, transform and description for the volatility-extended domain."""
    if mode not in GRID_MODES:
        raise ValueError(f"unknown grid mode: {mode}")
    if volatility_scale not in ("log", "linear"):
        raise ValueError("volatility_scale must be 'log' or 'linear'")
    log_volatility = volatility_scale == "log"
    transform = InterleavedVolTransform(
        n_assets=config.n_assets,
        spot_columns=config.spot_columns,
        volatility_columns=config.volatility_columns,
        strike_column=config.strike_column,
        basket_kind=basket_kind,
        use_moneyness=mode.startswith("moneyness"),
        log_volatility=log_volatility,
    )

    u_min, u_max = (
        config.log_volatility_bounds if log_volatility else config.volatility_bounds
    )
    volatility_axis = np.linspace(u_min, u_max, config.volatility_nodes)
    spot_axis = np.linspace(*config.spot_bounds, config.spot_nodes)
    # positional fill, so the axes follow whichever layout the config selects
    axes: list = [None] * config.n_dimensions
    for column in config.spot_columns:
        axes[column] = spot_axis
    for column in config.volatility_columns:
        axes[column] = volatility_axis

    if mode in ("paper", "paper_adaptive_maturity"):
        strike_axis = np.linspace(*config.strike_bounds, config.strike_nodes)
        strike_coordinate = "strike"
        strike_nodes_description = "uniform"
        moneyness_bounds = None
    else:
        spot_min, spot_max = config.spot_bounds
        strike_min, strike_max = config.strike_bounds
        m_min = float(np.log(strike_min / spot_max))
        m_max = float(np.log(strike_max / spot_min))
        moneyness_bounds = [m_min, m_max]
        strike_coordinate = "log(strike / basket_spot)"
        if mode == "moneyness_uniform":
            strike_axis = np.linspace(m_min, m_max, config.strike_nodes)
            strike_nodes_description = "uniform"
        else:
            strike_axis = sinh_centered_axis(
                m_min, m_max, config.strike_nodes, concentration=3.0
            )
            strike_nodes_description = "asymmetric sinh concentration=3 around zero"

    if mode in (
        "paper",
        "moneyness_uniform",
        "moneyness_adaptive_uniform_maturity",
    ):
        maturity_axis = np.linspace(*config.maturity_bounds, config.maturity_nodes)
        maturity_nodes_description = "uniform"
    else:
        maturity_axis = short_maturity_axis(
            *config.maturity_bounds, config.maturity_nodes, power=2.0
        )
        maturity_nodes_description = "quadratic near T_min"

    axes[config.strike_column] = strike_axis
    axes[config.rate_column] = np.linspace(*config.rate_bounds, config.rate_nodes)
    axes[config.maturity_column] = maturity_axis
    grid = QTTGrid(axes=axes)
    description = {
        "mode": mode,
        "layout": config.layout,
        "spot_columns": list(config.spot_columns),
        "volatility_columns": list(config.volatility_columns),
        "strike_coordinate": strike_coordinate,
        "basket_kind": basket_kind,
        "strike_nodes": strike_nodes_description,
        "moneyness_bounds": moneyness_bounds,
        "volatility_coordinate": "log(sigma)" if log_volatility else "sigma",
        "volatility_bounds": list(config.volatility_bounds),
        "volatility_axis_bounds": [u_min, u_max],
        "volatility_axis": volatility_axis.tolist(),
        "maturity_nodes": maturity_nodes_description,
        "maturity_axis": maturity_axis.tolist(),
    }
    return grid, transform, description


def sample_market_points(config, n_samples, rng, vol_sampling="log_uniform"):
    """Uniform test points, with volatility drawn uniformly in sigma or log sigma."""
    if vol_sampling not in ("log_uniform", "uniform"):
        raise ValueError("vol_sampling must be 'log_uniform' or 'uniform'")
    volatility_columns = set(config.volatility_columns)
    u_min, u_max = config.log_volatility_bounds
    columns = []
    for j, (low, high) in enumerate(config.bounds):
        if j in volatility_columns and vol_sampling == "log_uniform":
            columns.append(np.exp(rng.uniform(u_min, u_max, n_samples)))
        else:
            columns.append(rng.uniform(low, high, n_samples))
    return np.column_stack(columns)


def basket_spot_from_model(model_points, spot_columns, basket_kind="geometric"):
    """Basket spot G(S); spots are untouched by the model transform."""
    z = np.atleast_2d(np.asarray(model_points, dtype=float))
    spots = z[:, list(spot_columns)]
    if basket_kind == "geometric":
        return np.exp(np.mean(np.log(spots), axis=1))
    if basket_kind == "arithmetic":
        return np.mean(spots, axis=1)
    raise ValueError("basket_kind must be 'geometric' or 'arithmetic'")


class BasketScaledModelPricer:
    """Model-coordinate pricer returning P / G(S) instead of the raw price P.

    On a moneyness grid the corners reach strikes G(S) exp(m_max), far outside
    the market strike box, and the raw price there scales with G. That region
    is unused at query time, yet the cross must spend rank on it because the
    volatility modes couple into it through exp(T sigma_B^2 / 2). Dividing by
    the basket spot removes the G factor: for the geometric closed form,
    P / G in moneyness coordinates does not depend on the spots at all, so the
    scaled target collapses the five spot modes to rank one and the cross
    budget goes to the volatility structure instead. Predictions are
    re-multiplied by the exact G of the query point.
    """

    def __init__(self, model_pricer, spot_columns, basket_kind="geometric"):
        self.model_pricer = model_pricer
        self.spot_columns = tuple(int(j) for j in spot_columns)
        self.basket_kind = basket_kind

    def basket(self, model_points):
        return basket_spot_from_model(
            model_points, self.spot_columns, self.basket_kind
        )

    def __call__(self, model_points):
        z = np.atleast_2d(np.asarray(model_points, dtype=float))
        values = np.asarray(self.model_pricer(z), dtype=float).reshape(-1)
        return values / self.basket(z)


CUBIC_AXIS_NAMES = ("none", "all", "spots", "vols", "m", "r", "T")


def resolve_cubic_columns(config, names):
    """Map axis names to physical column indices for cubic interpolation.

    Accepts any combination of ``spots``, ``vols``, ``m``, ``r``, ``T``, plus
    the shorthands ``all`` and ``none``. Returns a sorted tuple of columns.
    """
    if isinstance(names, str):
        names = [names]
    names = [str(name) for name in names]
    unknown = set(names) - set(CUBIC_AXIS_NAMES)
    if unknown:
        raise ValueError(f"unknown cubic axis names: {sorted(unknown)}")
    if "none" in names:
        if len(names) > 1:
            raise ValueError("'none' cannot be combined with other axis names")
        return ()
    if "all" in names:
        return tuple(range(config.n_dimensions))
    columns: set[int] = set()
    for name in names:
        if name == "spots":
            columns.update(config.spot_columns)
        elif name == "vols":
            columns.update(config.volatility_columns)
        elif name == "m":
            columns.add(config.strike_column)
        elif name == "r":
            columns.add(config.rate_column)
        elif name == "T":
            columns.add(config.maturity_column)
    return tuple(sorted(columns))
