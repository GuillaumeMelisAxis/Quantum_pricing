from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grids import QTTGrid, short_maturity_axis, sinh_centered_axis


GRID_MODES = (
    "paper",
    "paper_adaptive_maturity",
    "moneyness_uniform",
    "moneyness_adaptive_uniform_maturity",
    "moneyness_adaptive",
)


@dataclass(frozen=True)
class CoordinateTransform:
    """Strike or log-moneyness coordinate transform for basket options."""

    n_assets: int
    basket_kind: str
    use_moneyness: bool

    def _basket(self, spots: np.ndarray) -> np.ndarray:
        if self.basket_kind == "geometric":
            return np.exp(np.mean(np.log(spots), axis=1))
        if self.basket_kind == "arithmetic":
            return np.mean(spots, axis=1)
        raise ValueError("basket_kind must be 'geometric' or 'arithmetic'")

    def to_model(self, market_parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(market_parameters, dtype=float)).copy()
        if self.use_moneyness:
            basket = self._basket(x[:, : self.n_assets])
            x[:, self.n_assets] = np.log(x[:, self.n_assets] / basket)
        return x

    def to_market(self, model_parameters: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(model_parameters, dtype=float)).copy()
        if self.use_moneyness:
            basket = self._basket(z[:, : self.n_assets])
            z[:, self.n_assets] = basket * np.exp(z[:, self.n_assets])
        return z


class TransformedPricer:
    def __init__(self, market_pricer, transform: CoordinateTransform):
        self.market_pricer = market_pricer
        self.transform = transform

    def __call__(self, model_parameters: np.ndarray) -> np.ndarray:
        return self.market_pricer(self.transform.to_market(model_parameters))


class MarketCoordinatePricer:
    """Expose a model-coordinate pricer through market coordinates.

    The wrapped pricer accepts (S_1, ..., S_d, m, r, T), while this adapter
    accepts (S_1, ..., S_d, K, r, T). Recomputing log-moneyness after every
    market-space bump is essential for spot Greeks defined at fixed strike.
    """

    def __init__(self, model_pricer, transform: CoordinateTransform):
        self.model_pricer = model_pricer
        self.transform = transform

    def __call__(self, market_parameters: np.ndarray) -> np.ndarray:
        market = np.atleast_2d(np.asarray(market_parameters, dtype=float))
        values = np.asarray(
            self.model_pricer(self.transform.to_model(market)),
            dtype=float,
        ).reshape(-1)
        if values.size != market.shape[0]:
            raise ValueError(
                "model_pricer must return one value per parameter vector"
            )
        return values


def build_coordinate_grid(config, mode: str, basket_kind: str):
    if mode not in GRID_MODES:
        raise ValueError(f"unknown grid mode: {mode}")
    transform = CoordinateTransform(
        n_assets=config.n_assets,
        basket_kind=basket_kind,
        use_moneyness=mode.startswith("moneyness"),
    )
    if mode in ("paper", "paper_adaptive_maturity"):
        if mode == "paper":
            maturity_axis = np.linspace(
                *config.maturity_bounds, config.physical_shape[-1]
            )
            maturity_nodes = "uniform"
        else:
            maturity_axis = short_maturity_axis(
                *config.maturity_bounds, config.physical_shape[-1], power=2.0
            )
            maturity_nodes = "quadratic near T_min"
        axes = [
            np.linspace(a, b, size)
            for (a, b), size in zip(config.bounds, config.physical_shape)
        ]
        axes[-1] = maturity_axis
        grid = QTTGrid(axes=axes)
        return grid, transform, {
            "mode": mode,
            "strike_coordinate": "strike",
            "maturity_nodes": maturity_nodes,
            "maturity_axis": maturity_axis.tolist(),
        }

    spot_min, spot_max = config.spot_bounds
    strike_min, strike_max = config.strike_bounds
    m_min = float(np.log(strike_min / spot_max))
    m_max = float(np.log(strike_max / spot_min))
    axes = [
        np.linspace(*config.spot_bounds, config.physical_shape[j])
        for j in range(config.n_assets)
    ]
    if mode == "moneyness_uniform":
        m_axis = np.linspace(m_min, m_max, config.physical_shape[config.n_assets])
        moneyness_nodes = "uniform"
    else:
        m_axis = sinh_centered_axis(
            m_min, m_max, config.physical_shape[config.n_assets], concentration=3.0
        )
        moneyness_nodes = "asymmetric sinh concentration=3 around zero"
    if mode in ("moneyness_uniform", "moneyness_adaptive_uniform_maturity"):
        maturity_axis = np.linspace(
            *config.maturity_bounds, config.physical_shape[-1]
        )
        maturity_nodes = "uniform"
    else:
        maturity_axis = short_maturity_axis(
            *config.maturity_bounds, config.physical_shape[-1], power=2.0
        )
        maturity_nodes = "quadratic near T_min"
    axes.extend((
        m_axis,
        np.linspace(*config.rate_bounds, config.physical_shape[-2]),
        maturity_axis,
    ))
    grid = QTTGrid(axes=axes)
    return grid, transform, {
        "mode": mode,
        "strike_coordinate": "log(strike / basket_spot)",
        "basket_kind": basket_kind,
        "moneyness_bounds": [m_min, m_max],
        "moneyness_nodes": moneyness_nodes,
        "moneyness_axis": m_axis.tolist(),
        "maturity_nodes": maturity_nodes,
        "maturity_axis": maturity_axis.tolist(),
    }


def oracle_multilinear_predict(grid, pricer, points, batch_size=128):
    """Exact grid-corner interpolation floor, without a TT approximation."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    predictions = np.empty(len(points), dtype=float)
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        corners, weights = grid.multilinear_stencil(batch)
        flat_indices = corners.reshape(-1, corners.shape[-1])
        corner_points = grid.indices_to_points(flat_indices)
        corner_values = pricer(corner_points).reshape(corners.shape[:2])
        predictions[start : start + len(batch)] = np.sum(
            weights * corner_values, axis=1
        )
    return predictions
