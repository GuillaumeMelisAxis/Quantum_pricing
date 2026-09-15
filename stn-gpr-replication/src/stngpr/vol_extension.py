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

Both the pricing coordinate families of ``coordinates`` and the Greek
coordinate families of ``risk_grids`` are available over this domain, through
``build_vol_grid`` and priced by :class:`VolExtendedPriceSurrogate`. Lifting the
Greek families here is not a rename: they were written against a frozen
volatility vector, and the standardized coordinate in particular divides by
``sigma_B sqrt(T)``, which is now a per-row quantity rather than a constant.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .coordinates import (
    GRID_MODES,
    TransformedPricer,
    oracle_hybrid_cubic_predict,
)
from .diagnostics import geometric_basket_effective_parameters
from .grids import QTTGrid, short_maturity_axis, sinh_centered_axis
from .risk_grids import (
    TENSOR_COMPATIBLE_GREEK_GRID_MODES,
    _basket_log_moneyness_bounds,
    gamma_monitor_axis,
)
from .tt_surrogate import TTPriceSurrogate

# The Greek-coordinate families from risk_grids, lifted onto the
# volatility-extended domain, plus the pricing families from coordinates.
# Every mode here accepts the same 2n+3 market vector.
VOL_RISK_GRID_MODES = TENSOR_COMPATIBLE_GREEK_GRID_MODES
VOL_GRID_MODES = (*GRID_MODES, *VOL_RISK_GRID_MODES)
PAPER_STRIKE_MODES = ("paper", "paper_adaptive_maturity")


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


def market_axes(config, log_volatility: bool):
    """Spot, volatility and rate axes placed at the columns the layout assigns.

    Returned as a list whose strike and maturity slots are still ``None``:
    those two are what the grid families disagree about, everything else is
    shared between them.
    """
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
    axes[config.rate_column] = np.linspace(*config.rate_bounds, config.rate_nodes)
    return axes, volatility_axis, (float(u_min), float(u_max))


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

    axes, volatility_axis, (u_min, u_max) = market_axes(config, log_volatility)

    if mode in PAPER_STRIKE_MODES:
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



@dataclass(frozen=True)
class VolStandardizedRiskTransform:
    r"""Standardized convexity coordinate with volatility as a live input.

    ``risk_grids.StandardizedRiskTransform`` reads sigma_B off a frozen config
    vector, which is exactly what the volatility extension removes. Here the
    basket volatility is recomputed per row from that row's own volatility
    coordinates,

    ``sigma_B(sigma) = sqrt(w' (sigma sigma' o C) w)``,

    so the coordinate

    ``z = asinh((m - m_ridge(r, T, sigma)) / (sigma_B(sigma) sqrt(T)))``

    still measures log-moneyness in standard deviations of the terminal basket.
    That is the whole point of it: without the sigma dependence, a node placed
    two standard deviations out at sigma = 0.05 sits a quarter of a standard
    deviation out at sigma = 0.80, and one strike axis cannot serve both ends.

    The inverse is applied after every market-space bump, so spot Greeks stay
    fixed-strike derivatives, exactly as in the fixed-volatility case.

    The ridge uses the geometric-basket carry even when ``basket_kind`` is
    arithmetic. The coordinate only has to be invertible and centred near the
    convexity peak, and the geometric proxy is accurate enough for that; the
    pricer is unaffected either way.
    """

    n_assets: int
    spot_columns: tuple[int, ...]
    volatility_columns: tuple[int, ...]
    strike_column: int
    rate_column: int
    maturity_column: int
    correlation: np.ndarray
    dividends: np.ndarray
    basket_kind: str = "geometric"
    log_volatility: bool = True

    def __post_init__(self) -> None:
        if self.basket_kind not in ("geometric", "arithmetic"):
            raise ValueError("basket_kind must be 'geometric' or 'arithmetic'")
        if len(self.spot_columns) != self.n_assets:
            raise ValueError("one spot column per asset is required")
        if len(self.volatility_columns) != self.n_assets:
            raise ValueError("one volatility column per asset is required")
        if np.asarray(self.correlation, dtype=float).shape != (
            self.n_assets,
            self.n_assets,
        ):
            raise ValueError("invalid correlation matrix")
        if np.asarray(self.dividends, dtype=float).shape != (self.n_assets,):
            raise ValueError("invalid dividend vector")

    @property
    def weights(self) -> np.ndarray:
        return np.full(self.n_assets, 1.0 / self.n_assets)

    def basket_volatility(self, volatilities: np.ndarray) -> np.ndarray:
        """Row-wise sqrt(w' (sigma sigma' o C) w), kept factored over rows."""
        sigma = np.atleast_2d(np.asarray(volatilities, dtype=float))
        scaled = sigma * self.weights
        variance = np.sum(
            (scaled @ np.asarray(self.correlation, dtype=float)) * scaled,
            axis=1,
        )
        return np.sqrt(variance)

    def _basket(self, spots: np.ndarray) -> np.ndarray:
        if self.basket_kind == "geometric":
            return np.exp(np.mean(np.log(spots), axis=1))
        return np.mean(spots, axis=1)

    def _ridge_and_scale(self, volatilities, rates, maturities):
        """Convexity ridge in log-moneyness, and the terminal basket std dev."""
        sigma = np.atleast_2d(np.asarray(volatilities, dtype=float))
        weights = self.weights
        variance = self.basket_volatility(sigma) ** 2
        carry = (
            rates
            - float(weights @ np.asarray(self.dividends, dtype=float))
            - 0.5 * (sigma**2 @ weights)
            + 0.5 * variance
        )
        ridge = (carry + 0.5 * variance) * maturities
        scale = np.sqrt(variance) * np.sqrt(maturities)
        return ridge, scale

    def _coordinate(self, log_moneyness, ridge, scale):
        return np.arcsinh((log_moneyness - ridge) / scale)

    def _log_moneyness(self, coordinate, ridge, scale):
        return ridge + scale * np.sinh(coordinate)

    def to_model(self, market_parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(market_parameters, dtype=float)).copy()
        volatility_columns = list(self.volatility_columns)
        sigma = x[:, volatility_columns]
        basket = self._basket(x[:, list(self.spot_columns)])
        log_moneyness = np.log(x[:, self.strike_column] / basket)
        ridge, scale = self._ridge_and_scale(
            sigma, x[:, self.rate_column], x[:, self.maturity_column]
        )
        x[:, self.strike_column] = self._coordinate(log_moneyness, ridge, scale)
        if self.log_volatility:
            x[:, volatility_columns] = np.log(sigma)
        return x

    def to_market(self, model_parameters: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(model_parameters, dtype=float)).copy()
        volatility_columns = list(self.volatility_columns)
        sigma = (
            np.exp(z[:, volatility_columns])
            if self.log_volatility
            else z[:, volatility_columns]
        )
        basket = self._basket(z[:, list(self.spot_columns)])
        ridge, scale = self._ridge_and_scale(
            sigma, z[:, self.rate_column], z[:, self.maturity_column]
        )
        log_moneyness = self._log_moneyness(
            z[:, self.strike_column], ridge, scale
        )
        z[:, self.strike_column] = basket * np.exp(log_moneyness)
        z[:, volatility_columns] = sigma
        return z


@dataclass(frozen=True)
class BoundedVolStandardizedRiskTransform(VolStandardizedRiskTransform):
    r"""Tensor-compatible volatility-aware standardized coordinate on ``[-1, 1]``.

    A tensor grid takes the Cartesian product of its axes, so a strike axis in
    the raw standardized coordinate is used at every ``(r, T, sigma)`` the
    other axes offer. The standardized interval the market box actually
    occupies moves with all three, and with sigma spanning a factor of sixteen
    it moves much further than in the fixed-volatility setting: one global
    interval either clips the short-dated, low-volatility corner or spends most
    of its nodes on strikes outside the market box.

    Normalizing that interval to ``[-1, 1]`` fixes the endpoints. Coordinate
    ``-1`` and ``+1`` map to ``moneyness_bounds`` for every rate, maturity and
    volatility vector, while the interior keeps standardized-risk node density.

    The price is what pays for that. A fixed coordinate now names a different
    strike at every maturity, so the surface moves along the maturity axis far
    faster than it does on a moneyness grid, and the maturity axis is the one
    the config gives eight nodes. Measured at a 150k budget on the default
    domain, this grid stalls at MAE 0.35 with eight maturity nodes - its own
    interpolation floor, not a cross failure, since the rank keeps climbing -
    and reaches 0.0086 with thirty-two, while ``price_adaptive`` sits at 0.0054
    either way. Raise ``maturity_nodes`` before reading anything into a
    standardized-coordinate error.
    """

    moneyness_bounds: tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        lower, upper = self.moneyness_bounds
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("invalid log-moneyness bounds")

    def standardized_bounds(self, ridge, scale):
        """Standardized images of the two global log-moneyness bounds."""
        lower, upper = self.moneyness_bounds
        return (
            np.arcsinh((float(lower) - ridge) / scale),
            np.arcsinh((float(upper) - ridge) / scale),
        )

    def _coordinate(self, log_moneyness, ridge, scale):
        standardized = np.arcsinh((log_moneyness - ridge) / scale)
        lower, upper = self.standardized_bounds(ridge, scale)
        return 2.0 * (standardized - lower) / (upper - lower) - 1.0

    def _log_moneyness(self, coordinate, ridge, scale):
        lower, upper = self.standardized_bounds(ridge, scale)
        standardized = lower + 0.5 * (coordinate + 1.0) * (upper - lower)
        return ridge + scale * np.sinh(standardized)


def _volatility_box_vertices(config, max_assets: int = 12) -> np.ndarray:
    """Volatility vectors at the corners of the volatility box.

    ``sigma_B**2`` is a convex quadratic form in sigma, so its maximum over the
    box is attained at a vertex and this enumeration is exact at the upper end.
    It is exact at the lower end too whenever sigma_B is componentwise
    increasing, which holds for any non-negative correlation matrix. Past
    ``max_assets`` the ``2**n`` enumeration is dropped for the two uniform
    corners, which remain exact under that same monotonicity.
    """
    low, high = config.volatility_bounds
    if config.n_assets > max_assets:
        return np.array(
            [[low] * config.n_assets, [high] * config.n_assets], dtype=float
        )
    return np.array(
        list(itertools.product((low, high), repeat=config.n_assets)),
        dtype=float,
    )


def _vol_standardized_bounds(config, transform, moneyness_bounds):
    """Envelope of the unbounded standardized coordinate over the market box.

    The coordinate is spot-free once expressed through ``m``, so the reference
    spots below only serve to build valid market vectors.
    """
    reference_spot = 100.0
    rows = []
    for sigma in _volatility_box_vertices(config):
        for m in moneyness_bounds:
            for rate in config.rate_bounds:
                for maturity in config.maturity_bounds:
                    row = np.empty(config.n_dimensions, dtype=float)
                    row[list(config.spot_columns)] = reference_spot
                    row[list(config.volatility_columns)] = sigma
                    row[config.strike_column] = reference_spot * np.exp(m)
                    row[config.rate_column] = rate
                    row[config.maturity_column] = maturity
                    rows.append(row)
    coordinate = transform.to_model(np.asarray(rows))[:, config.strike_column]
    return float(np.min(coordinate)), float(np.max(coordinate))


def _standardized_extreme_log_moneyness(config, transform, coordinate_bounds):
    """Largest ``|m|`` the unbounded standardized axis reaches over the box.

    Evaluated through the ridge and scale directly rather than through
    :meth:`to_market`, so the strike never has to be exponentiated to find out
    whether exponentiating it would overflow.
    """
    extreme = 0.0
    for sigma in _volatility_box_vertices(config):
        volatilities = np.atleast_2d(sigma)
        for rate in config.rate_bounds:
            for maturity in config.maturity_bounds:
                ridge, scale = transform._ridge_and_scale(
                    volatilities,
                    np.asarray([float(rate)]),
                    np.asarray([float(maturity)]),
                )
                for z in coordinate_bounds:
                    log_moneyness = transform._log_moneyness(
                        np.asarray([float(z)]), ridge, scale
                    )
                    extreme = max(extreme, float(np.max(np.abs(log_moneyness))))
    return extreme


def _monitor_regimes(config, monitor_rate, monitor_maturities, monitor_volatilities):
    """Flattened ``(T, sigma_B, carry)`` regimes for the gamma monitor.

    The fourth derivative that sets cubic interpolation error moves with both
    maturity and basket volatility, and here sigma is a coordinate rather than a
    constant. Every maturity is therefore paired with every reference
    volatility, and the monitor takes the pointwise maximum over the whole set.
    """
    if monitor_maturities is None:
        monitor_maturities = np.asarray([3.0, 7.0, 14.0, 30.0, 90.0, 365.0]) / 365.0
    if monitor_volatilities is None:
        low, high = config.log_volatility_bounds
        monitor_volatilities = np.exp(np.linspace(low, high, 3))
    monitor_maturities = np.asarray(monitor_maturities, dtype=float).reshape(-1)
    monitor_volatilities = np.asarray(monitor_volatilities, dtype=float).reshape(-1)

    basket_volatilities, carries = [], []
    for sigma in monitor_volatilities:
        volatility, carry = geometric_basket_effective_parameters(
            float(monitor_rate),
            np.full(config.n_assets, float(sigma)),
            config.correlation,
            config.dividends,
        )
        basket_volatilities.append(volatility)
        carries.append(carry)

    n_regimes = monitor_volatilities.size
    return (
        np.repeat(monitor_maturities, n_regimes),
        np.tile(np.asarray(basket_volatilities), monitor_maturities.size),
        np.tile(np.asarray(carries), monitor_maturities.size),
        monitor_volatilities,
    )


def build_vol_extended_risk_grid(
    config,
    mode: str,
    basket_kind: str = "geometric",
    volatility_scale: str = "log",
    monitor_maturities=None,
    monitor_rate: float = 0.03,
    monitor_volatilities=None,
):
    """Greek coordinate grid from ``risk_grids``, on the extended domain.

    Builds the same five strike-coordinate families that
    ``risk_grids.build_greek_coordinate_grid`` builds for the paper's frozen
    volatility vector, but over ``(S_1, sigma_1, ..., S_n, sigma_n, K, r, T)``:
    the spot, volatility and rate axes come from the layout, the maturity axis
    is quadratic near ``T_min`` as on every Greek grid, and only the strike
    coordinate distinguishes the modes.

    The two standardized modes need a finer maturity axis than the config
    default to price competitively - see
    :class:`BoundedVolStandardizedRiskTransform`.
    """
    if mode not in VOL_RISK_GRID_MODES:
        raise ValueError(f"unknown risk grid mode: {mode}")
    if volatility_scale not in ("log", "linear"):
        raise ValueError("volatility_scale must be 'log' or 'linear'")
    log_volatility = volatility_scale == "log"

    axes, volatility_axis, (u_min, u_max) = market_axes(config, log_volatility)
    maturity_axis = short_maturity_axis(
        *config.maturity_bounds, config.maturity_nodes, power=2.0
    )
    m_bounds = _basket_log_moneyness_bounds(config)
    coordinate_nodes = config.strike_nodes

    if mode in ("standardized_risk", "bounded_standardized_risk"):
        common = {
            "n_assets": config.n_assets,
            "spot_columns": config.spot_columns,
            "volatility_columns": config.volatility_columns,
            "strike_column": config.strike_column,
            "rate_column": config.rate_column,
            "maturity_column": config.maturity_column,
            "correlation": config.correlation,
            "dividends": config.dividends,
            "basket_kind": basket_kind,
            "log_volatility": log_volatility,
        }
        if mode == "standardized_risk":
            transform = VolStandardizedRiskTransform(**common)
            coordinate_bounds = _vol_standardized_bounds(config, transform, m_bounds)
            extreme = _standardized_extreme_log_moneyness(
                config, transform, coordinate_bounds
            )
            # One global z interval has to cover every (r, T, sigma) the other
            # axes offer. At fixed volatility that only wastes nodes on strikes
            # outside the market box; once sigma spans a factor of sixteen the
            # interval widens until the opposite corner of the box maps to a
            # strike that is not a floating point number at all, and the grid
            # cannot be built rather than merely being inefficient.
            if extreme > np.log(np.finfo(float).max):
                raise ValueError(
                    "the unbounded standardized risk coordinate is not "
                    f"representable on this domain: the envelope "
                    f"[{coordinate_bounds[0]:.2f}, {coordinate_bounds[1]:.2f}] "
                    f"reaches |log(K / basket_spot)| = {extreme:.3g} at the "
                    f"corners of the (r, T, sigma) box, against a market box of "
                    f"[{m_bounds[0]:.2f}, {m_bounds[1]:.2f}]. Use "
                    "'bounded_standardized_risk', which keeps the same node "
                    "density and pins the endpoints to the market box, or "
                    "narrow volatility_bounds."
                )
            coordinate_definition = (
                "asinh((m - convexity_ridge(r,T,sigma)) / (sigma_B(sigma) sqrt(T)))"
            )
            construction = {
                "type": "uniform in the volatility-aware standardized coordinate",
                "mapped_full_moneyness_bounds": list(m_bounds),
                "envelope": "over the corners of the (m, r, T, sigma) box",
                "extreme_physical_log_moneyness": extreme,
            }
        else:
            transform = BoundedVolStandardizedRiskTransform(
                **common, moneyness_bounds=m_bounds
            )
            coordinate_bounds = (-1.0, 1.0)
            coordinate_definition = (
                "normalized volatility-aware standardized risk coordinate: the "
                "(r,T,sigma)-dependent z interval mapped to [-1,1]"
            )
            construction = {
                "type": "bounded tensor-compatible standardized risk coordinate",
                "mapped_full_moneyness_bounds": list(m_bounds),
                "endpoint_invariant": (
                    "coordinate -1 and +1 map to the fixed global log-moneyness "
                    "bounds for every (r, T, sigma)"
                ),
            }
        coordinate_axis = np.linspace(*coordinate_bounds, coordinate_nodes)
    else:
        transform = InterleavedVolTransform(
            n_assets=config.n_assets,
            spot_columns=config.spot_columns,
            volatility_columns=config.volatility_columns,
            strike_column=config.strike_column,
            basket_kind=basket_kind,
            use_moneyness=True,
            log_volatility=log_volatility,
        )
        coordinate_definition = "m = log(K / basket_spot)"
        if mode == "m_uniform":
            coordinate_axis = np.linspace(*m_bounds, coordinate_nodes)
            construction = {"type": "uniform log-moneyness"}
        elif mode == "price_adaptive":
            coordinate_axis = sinh_centered_axis(
                *m_bounds, coordinate_nodes, concentration=3.0
            )
            construction = {
                "type": "price-adaptive sinh concentration around m=0",
                "concentration": 3.0,
            }
        else:
            maturities, volatilities, carries, reference_sigma = _monitor_regimes(
                config, monitor_rate, monitor_maturities, monitor_volatilities
            )
            coordinate_axis, construction = gamma_monitor_axis(
                *m_bounds,
                coordinate_nodes,
                maturities,
                float(monitor_rate),
                volatilities,
                carries,
            )
            construction["type"] = (
                "Greek-adaptive fourth-derivative monitor over (T, sigma) regimes"
            )
            construction["reference_volatilities"] = reference_sigma.tolist()

    axes[config.strike_column] = coordinate_axis
    axes[config.maturity_column] = maturity_axis
    grid = QTTGrid(axes=axes)
    description = {
        "mode": mode,
        "family": "risk",
        "layout": config.layout,
        "spot_columns": list(config.spot_columns),
        "volatility_columns": list(config.volatility_columns),
        "basket_kind": basket_kind,
        "physical_shape": list(grid.shape),
        "qtt_core_count": int(sum(grid.bits)),
        "strike_coordinate": coordinate_definition,
        "coordinate_bounds": [float(coordinate_axis[0]), float(coordinate_axis[-1])],
        "coordinate_axis": coordinate_axis.tolist(),
        "moneyness_bounds": list(m_bounds),
        "volatility_coordinate": "log(sigma)" if log_volatility else "sigma",
        "volatility_bounds": list(config.volatility_bounds),
        "volatility_axis_bounds": [u_min, u_max],
        "volatility_axis": volatility_axis.tolist(),
        "maturity_nodes": "quadratic near T_min",
        "maturity_axis": maturity_axis.tolist(),
        "construction": construction,
    }
    return grid, transform, description


def build_vol_grid(config, mode: str, **kwargs):
    """Dispatch to the pricing or the risk grid family by mode name.

    ``GRID_MODES`` are the pricing coordinate families from ``coordinates``;
    ``VOL_RISK_GRID_MODES`` are the Greek families from ``risk_grids``. Both
    return ``(grid, transform, description)`` over the same market vector, so a
    caller that only wants a price never has to know which family it asked for.
    """
    if mode in VOL_RISK_GRID_MODES:
        return build_vol_extended_risk_grid(config, mode, **kwargs)
    if mode in GRID_MODES:
        unsupported = set(kwargs) - {"basket_kind", "volatility_scale"}
        if unsupported:
            raise TypeError(f"pricing grid modes do not accept: {sorted(unsupported)}")
        return build_vol_extended_grid(config, mode, **kwargs)
    raise ValueError(f"unknown grid mode: {mode}")


def strike_axis_log_moneyness_nodes(
    config,
    grid,
    transform,
    rate: float,
    maturity: float,
    volatility: float | None = None,
    reference_spot: float = 100.0,
):
    """Physical log-moneyness of the strike axis nodes at one market regime.

    Every mode places its nodes in a different coordinate, so the only way to
    compare them is to push each axis back to ``m = log(K / G(S))`` at a fixed
    ``(r, T, sigma)``. The spots cancel in that ratio, so ``reference_spot``
    does not affect the result.
    """
    coordinate = np.asarray(grid.axes[config.strike_column], dtype=float)
    if volatility is None:
        volatility = float(np.exp(np.mean(config.log_volatility_bounds)))
    if not np.isfinite(rate) or maturity <= 0.0 or volatility <= 0.0:
        raise ValueError("rate, maturity and volatility are invalid")

    model = np.empty((coordinate.size, config.n_dimensions), dtype=float)
    model[:, list(config.spot_columns)] = float(reference_spot)
    model[:, list(config.volatility_columns)] = (
        np.log(volatility) if transform.log_volatility else volatility
    )
    model[:, config.strike_column] = coordinate
    model[:, config.rate_column] = float(rate)
    model[:, config.maturity_column] = float(maturity)

    market = transform.to_market(model)
    basket = np.exp(np.mean(np.log(market[:, list(config.spot_columns)]), axis=1))
    nodes = np.log(market[:, config.strike_column] / basket)
    if np.any(~np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("the physical log-moneyness nodes are not strictly increasing")
    return nodes


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


def default_market_pricer(config, basket_kind: str = "geometric"):
    """Reference pricer over the extended market vector for either basket."""
    from .pricers import (
        EuropeanArithmeticBasketVolPricer,
        EuropeanGeometricBasketVolPricer,
    )

    if basket_kind == "geometric":
        return EuropeanGeometricBasketVolPricer(config)
    if basket_kind == "arithmetic":
        return EuropeanArithmeticBasketVolPricer(config)
    raise ValueError("basket_kind must be 'geometric' or 'arithmetic'")


class VolExtendedPriceSurrogate:
    """Price options on any volatility-extended grid, pricing or risk.

    The experiment scripts assemble grid, coordinate transform, basket-spot
    scaling, TT-cross and cubic interpolation by hand every time. This wires
    them together once: market coordinates
    ``(S_1, sigma_1, ..., S_n, sigma_n, K, r, T)`` go in, prices come out, and
    the coordinate family is a constructor argument rather than a rewrite.

    ``target='auto'`` fits ``P / G(S)`` on every grid whose strike coordinate
    is a moneyness or standardized coordinate - which is all of them except the
    two ``paper`` modes - because those grids reach strikes far outside the
    market box at their corners, where the raw price still scales with the
    basket spot. Predictions are re-multiplied by the exact ``G(S)`` of the
    query point, so this changes what the cross has to represent, not what is
    returned.
    """

    def __init__(
        self,
        config,
        mode: str = "bounded_standardized_risk",
        basket_kind: str = "geometric",
        volatility_scale: str = "log",
        target: str = "auto",
        cubic_axes=("all",),
        market_pricer=None,
        seed: int | None = None,
        **grid_kwargs,
    ):
        if target not in ("auto", "price", "price_over_basket"):
            raise ValueError(
                "target must be 'auto', 'price' or 'price_over_basket'"
            )
        self.config = config
        self.mode = mode
        self.basket_kind = basket_kind
        self.grid, self.transform, self.grid_description = build_vol_grid(
            config,
            mode,
            basket_kind=basket_kind,
            volatility_scale=volatility_scale,
            **grid_kwargs,
        )
        self.market_pricer = (
            default_market_pricer(config, basket_kind)
            if market_pricer is None
            else market_pricer
        )
        self.model_pricer = TransformedPricer(self.market_pricer, self.transform)
        self.target = (
            ("price" if mode in PAPER_STRIKE_MODES else "price_over_basket")
            if target == "auto"
            else target
        )
        self.fit_pricer = (
            BasketScaledModelPricer(
                self.model_pricer, config.spot_columns, basket_kind
            )
            if self.target == "price_over_basket"
            else self.model_pricer
        )
        self.cubic_columns = resolve_cubic_columns(config, cubic_axes)
        self.seed = int(config.seed if seed is None else seed)
        self.surrogate = TTPriceSurrogate(
            self.grid, self.fit_pricer, seed=self.seed
        )

    def to_model(self, market_points: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(market_points, dtype=float))
        if x.shape[1] != self.config.n_dimensions:
            raise ValueError(
                "market points must contain spots, volatilities, strike, "
                "rate and maturity"
            )
        return self.transform.to_model(x)

    def _scale(self, model_points: np.ndarray):
        """Undo the fit-target scaling at the query point."""
        if self.target != "price_over_basket":
            return 1.0
        return basket_spot_from_model(
            model_points, self.config.spot_columns, self.basket_kind
        )

    def fit(self, max_evals: int, anova_samples: int = 2_000, **kwargs):
        """Run TT-cross on the fit target; returns the fit diagnostics."""
        return self.surrogate.fit(max_evals, anova_samples=anova_samples, **kwargs)

    def price(self, market_points: np.ndarray, batch_size: int = 2_048):
        """Surrogate price at market points, in market coordinates."""
        model_points = self.to_model(market_points)
        values = self.surrogate.predict_factorized(
            model_points,
            cubic_columns=self.cubic_columns,
            batch_size=batch_size,
        )
        return values * self._scale(model_points)

    def price_interpolation_floor(
        self,
        market_points,
        cubic_columns=(),
        batch_size: int = 8,
        max_stencil: int = 1 << 20,
    ):
        """Price from exact node values instead of TT ones.

        The same interpolant on the same grid, evaluated against the oracle
        rather than the fitted cores: the gap to :meth:`price` is TT error
        alone, and the gap to :meth:`reference_price` is the grid's own
        interpolation error. It calls the pricer at every stencil node, so it
        is a diagnostic, not a pricing path.

        ``cubic_columns`` defaults to none rather than to
        :attr:`cubic_columns`. An oracle has no cores to contract mode by mode,
        so it must enumerate the product stencil - ``2**d`` corners multilinear
        against ``4**d`` fully cubic, which at ``d = 13`` is eight thousand
        against sixty-seven million. The cubic floor is reachable a few axes at
        a time; the fully cubic one is not, and ``max_stencil`` refuses it
        rather than exhausting memory.
        """
        model_points = self.to_model(market_points)
        cubic = tuple(int(column) for column in cubic_columns)
        stencil = int(
            np.prod([4 if j in cubic else 2 for j in range(len(self.grid.shape))])
        )
        if stencil > int(max_stencil):
            raise ValueError(
                f"the requested oracle stencil is {stencil} corners per query, "
                f"above max_stencil={int(max_stencil)}; the oracle floor "
                "enumerates the product stencil, so restrict cubic_columns to "
                "a few axes (the TT path in price() has no such limit)"
            )
        values = oracle_hybrid_cubic_predict(
            self.grid,
            self.fit_pricer,
            model_points,
            cubic_columns=cubic,
            batch_size=batch_size,
        )
        return values * self._scale(model_points)

    def reference_price(self, market_points: np.ndarray):
        """Closed-form price at market points, bypassing the grid entirely."""
        x = np.atleast_2d(np.asarray(market_points, dtype=float))
        return np.asarray(self.market_pricer(x), dtype=float).reshape(-1)
