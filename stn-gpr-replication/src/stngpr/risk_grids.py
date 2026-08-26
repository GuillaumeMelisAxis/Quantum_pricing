from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

from .diagnostics import (
    geometric_basket_effective_parameters,
    geometric_basket_normalized_put,
)
from .grids import QTTGrid, short_maturity_axis, sinh_centered_axis


GREEK_GRID_MODES = (
    "m_uniform",
    "price_adaptive",
    "gamma_monitor",
    "standardized_risk",
)


def _basket_log_moneyness_bounds(config) -> tuple[float, float]:
    return (
        float(np.log(config.strike_bounds[0] / config.spot_bounds[1])),
        float(np.log(config.strike_bounds[1] / config.spot_bounds[0])),
    )


def gamma_monitor_axis(
    lower: float,
    upper: float,
    n_nodes: int,
    maturities: np.ndarray,
    rate: float,
    basket_volatility: float,
    basket_carry: float,
    dense_nodes: int = 8193,
    smoothing_window: int = 41,
    floor: float = 0.02,
) -> tuple[np.ndarray, dict]:
    r"""Equidistribute a monitor for cubic-interpolation Gamma error.

    Cubic interpolation has a second-derivative error proportional to
    ``h**2 * |u''''|``.  The monitor therefore uses ``sqrt(|u''''|)`` as a
    node-density proxy.  Each maturity is normalized before taking the
    pointwise maximum so that short expiries do not erase longer regimes.
    """
    if not lower < upper:
        raise ValueError("lower must be smaller than upper")
    if n_nodes < 4:
        raise ValueError("gamma-monitor interpolation requires at least four nodes")
    if dense_nodes < max(257, n_nodes * 4):
        raise ValueError("dense_nodes is too small for the requested axis")
    if smoothing_window < 7 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be an odd integer at least seven")
    if floor <= 0.0:
        raise ValueError("floor must be positive")

    maturities = np.asarray(maturities, dtype=float).reshape(-1)
    if maturities.size == 0 or np.any(maturities <= 0.0):
        raise ValueError("maturities must be non-empty and positive")

    dense = np.linspace(float(lower), float(upper), int(dense_nodes))
    step = float(dense[1] - dense[0])
    window = min(int(smoothing_window), dense.size - (1 - dense.size % 2))
    signals = []
    fourth_derivative_scales = []
    for maturity in maturities:
        values = geometric_basket_normalized_put(
            dense,
            float(maturity),
            float(rate),
            float(basket_volatility),
            float(basket_carry),
        )
        fourth = savgol_filter(
            values,
            window_length=window,
            polyorder=6,
            deriv=4,
            delta=step,
            mode="interp",
        )
        absolute = np.abs(fourth)
        scale = float(np.quantile(absolute, 0.995))
        fourth_derivative_scales.append(scale)
        if scale > np.finfo(float).tiny:
            signals.append(np.sqrt(np.minimum(absolute / scale, 1.0)))
        else:
            signals.append(np.zeros_like(absolute))

    aggregate = np.max(np.vstack(signals), axis=0)
    monitor = float(floor) + aggregate
    cumulative = cumulative_trapezoid(monitor, dense, initial=0.0)
    cumulative /= cumulative[-1]
    probabilities = np.linspace(0.0, 1.0, int(n_nodes))
    axis = np.interp(probabilities, cumulative, dense)
    axis[0], axis[-1] = float(lower), float(upper)
    if np.any(np.diff(axis) <= 0.0):
        raise RuntimeError("the gamma-monitor axis is not strictly increasing")

    return axis, {
        "definition": "equidistribution of floor + max_T normalized sqrt(abs(d4u/dm4))",
        "maturities": maturities.tolist(),
        "rate": float(rate),
        "dense_nodes": int(dense_nodes),
        "smoothing_window": int(window),
        "monitor_floor": float(floor),
        "fourth_derivative_quantile_scales": fourth_derivative_scales,
    }


@dataclass(frozen=True)
class StandardizedRiskTransform:
    r"""Transform ``m`` into a maturity-standardized convexity coordinate.

    The coordinate is

    ``z = asinh((m - m_ridge(r,T)) / (sigma_B sqrt(T)))``.

    The inverse is evaluated after every market-space spot bump, so all spot
    Greeks remain fixed-strike derivatives.
    """

    n_assets: int
    volatilities: np.ndarray
    correlation: np.ndarray
    dividends: np.ndarray
    basket_kind: str = "geometric"
    use_moneyness: bool = True

    def __post_init__(self) -> None:
        if self.basket_kind != "geometric":
            raise ValueError("standardized risk coordinates currently require a geometric basket")
        sigma = np.asarray(self.volatilities, dtype=float)
        corr = np.asarray(self.correlation, dtype=float)
        dividends = np.asarray(self.dividends, dtype=float)
        if sigma.shape != (self.n_assets,):
            raise ValueError("invalid volatility vector")
        if corr.shape != (self.n_assets, self.n_assets):
            raise ValueError("invalid correlation matrix")
        if dividends.shape != (self.n_assets,):
            raise ValueError("invalid dividend vector")

    @property
    def basket_volatility(self) -> float:
        sigma = np.asarray(self.volatilities, dtype=float)
        weights = np.full(self.n_assets, 1.0 / self.n_assets)
        covariance = np.outer(sigma, sigma) * np.asarray(
            self.correlation,
            dtype=float,
        )
        return float(np.sqrt(weights @ covariance @ weights))

    def _basket(self, spots: np.ndarray) -> np.ndarray:
        return np.exp(np.mean(np.log(spots), axis=1))

    def _ridge(self, rates: np.ndarray, maturities: np.ndarray) -> np.ndarray:
        weights = np.full(self.n_assets, 1.0 / self.n_assets)
        sigma = np.asarray(self.volatilities, dtype=float)
        variance = self.basket_volatility**2
        carry = (
            rates
            - float(weights @ np.asarray(self.dividends, dtype=float))
            - 0.5 * float(weights @ sigma**2)
            + 0.5 * variance
        )
        return (carry + 0.5 * variance) * maturities

    def to_model(self, market_parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(market_parameters, dtype=float)).copy()
        basket = self._basket(x[:, : self.n_assets])
        m = np.log(x[:, self.n_assets] / basket)
        rates = x[:, self.n_assets + 1]
        maturities = x[:, self.n_assets + 2]
        scale = self.basket_volatility * np.sqrt(maturities)
        x[:, self.n_assets] = np.arcsinh(
            (m - self._ridge(rates, maturities)) / scale
        )
        return x

    def to_market(self, model_parameters: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(model_parameters, dtype=float)).copy()
        basket = self._basket(z[:, : self.n_assets])
        rates = z[:, self.n_assets + 1]
        maturities = z[:, self.n_assets + 2]
        scale = self.basket_volatility * np.sqrt(maturities)
        m = self._ridge(rates, maturities) + scale * np.sinh(
            z[:, self.n_assets]
        )
        z[:, self.n_assets] = basket * np.exp(m)
        return z


def _standardized_bounds(config, transform, m_bounds) -> tuple[float, float]:
    values = []
    reference_spots = np.full(config.n_assets, 100.0)
    basket = float(np.exp(np.mean(np.log(reference_spots))))
    for m in m_bounds:
        for rate in config.rate_bounds:
            for maturity in config.maturity_bounds:
                market = np.concatenate(
                    (reference_spots, [basket * np.exp(m), rate, maturity])
                )
                values.append(transform.to_model(market)[0, config.n_assets])
    return float(np.min(values)), float(np.max(values))


def build_greek_coordinate_grid(
    config,
    mode: str,
    monitor_maturities: np.ndarray | None = None,
    monitor_rate: float = 0.03,
) -> tuple[QTTGrid, object, dict]:
    """Build one equal-size coordinate candidate for the v9 Greek ablation."""
    if mode not in GREEK_GRID_MODES:
        raise ValueError(f"unknown Greek grid mode: {mode}")

    from .coordinates import CoordinateTransform

    m_bounds = _basket_log_moneyness_bounds(config)
    coordinate_nodes = config.physical_shape[config.n_assets]
    maturity_axis = short_maturity_axis(
        *config.maturity_bounds,
        config.physical_shape[-1],
        power=2.0,
    )
    spot_axes = [
        np.linspace(*config.spot_bounds, config.physical_shape[j])
        for j in range(config.n_assets)
    ]
    rate_axis = np.linspace(*config.rate_bounds, config.physical_shape[-2])

    if mode == "standardized_risk":
        transform = StandardizedRiskTransform(
            n_assets=config.n_assets,
            volatilities=config.volatilities,
            correlation=config.correlation,
            dividends=config.dividends,
        )
        coordinate_bounds = _standardized_bounds(config, transform, m_bounds)
        coordinate_axis = np.linspace(*coordinate_bounds, coordinate_nodes)
        coordinate_definition = (
            "asinh((m - convexity_ridge(r,T)) / (sigma_B sqrt(T)))"
        )
        construction = {
            "type": "uniform in standardized risk coordinate",
            "basket_volatility": transform.basket_volatility,
            "mapped_full_moneyness_bounds": list(m_bounds),
        }
    else:
        transform = CoordinateTransform(
            n_assets=config.n_assets,
            basket_kind="geometric",
            use_moneyness=True,
        )
        coordinate_bounds = m_bounds
        coordinate_definition = "m = log(K / geometric_basket_spot)"
        if mode == "m_uniform":
            coordinate_axis = np.linspace(*m_bounds, coordinate_nodes)
            construction = {"type": "uniform log-moneyness"}
        elif mode == "price_adaptive":
            coordinate_axis = sinh_centered_axis(
                *m_bounds,
                coordinate_nodes,
                concentration=3.0,
            )
            construction = {
                "type": "price-adaptive sinh concentration around m=0",
                "concentration": 3.0,
            }
        else:
            if monitor_maturities is None:
                monitor_maturities = np.asarray(
                    [3.0, 7.0, 14.0, 30.0, 90.0, 365.0]
                ) / 365.0
            basket_sigma, basket_carry = geometric_basket_effective_parameters(
                float(monitor_rate),
                config.volatilities,
                config.correlation,
                config.dividends,
            )
            coordinate_axis, construction = gamma_monitor_axis(
                *m_bounds,
                coordinate_nodes,
                np.asarray(monitor_maturities, dtype=float),
                float(monitor_rate),
                basket_sigma,
                basket_carry,
            )
            construction["type"] = "Greek-adaptive fourth-derivative monitor"

    grid = QTTGrid(
        axes=tuple(
            [
                *spot_axes,
                coordinate_axis,
                rate_axis,
                maturity_axis,
            ]
        )
    )
    return grid, transform, {
        "mode": mode,
        "physical_shape": list(grid.shape),
        "qtt_core_count": int(sum(grid.bits)),
        "coordinate_definition": coordinate_definition,
        "coordinate_bounds": [float(coordinate_axis[0]), float(coordinate_axis[-1])],
        "coordinate_axis": coordinate_axis.tolist(),
        "maturity_axis": maturity_axis.tolist(),
        "construction": construction,
    }
