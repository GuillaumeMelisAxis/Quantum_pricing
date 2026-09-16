r"""Additive volatility axes on top of the frozen-volatility shared axes.

``vol_extension`` promotes *every* asset volatility to a coordinate at once, so
its dimension count is pinned at ``2 n + 3`` and its ``layout`` field can only
interleave the volatility block with the spots or bury it between the spots and
``(K, r, T)``. Neither shape answers the question this module exists for: how
does the tensor cost of the surrogate grow as volatility factors are added one
at a time?

Answering it needs two properties that ``vol_extension`` does not have.

* **A free count.** ``d_sigma`` volatility axes are promoted and the remaining
  ``n - d_sigma`` volatilities stay frozen at ``sigma_reference``. The promoted
  ones are assets ``0 .. d_sigma - 1``, a fixed order, never permuted.
* **Additivity.** The shared axes ``(S_1, ..., S_n, K, r, T)`` occupy columns
  ``0 .. n + 2`` and keep their node arrays *identical* for every
  ``d_sigma``; the sigma block is appended after them. The QTT chain at
  ``d_sigma = 0`` is therefore exactly the chain at ``d_sigma = k`` with the
  sigma cores removed, which is what makes a rank profile comparable core by
  core across configurations. ``sigma_block`` can also place the block at the
  head or interleave it with the spots, but only as a control: those two orders
  break additivity by construction and exist to show the conclusion does not
  depend on the tail order being chosen favourably.

Volatility axes carry raw ``sigma`` on uniform nodes, not ``log sigma``. The
fit target is always ``P / G(S)``: on a moneyness or standardized-risk
coordinate the grid corners reach strikes far outside the market box where the
raw price still scales with the basket spot, and at ``d_sigma > 0`` the
volatility modes couple into that unused region. Keeping one target across
every ``d_sigma`` is also what stops the comparison from mixing a dimension
effect with a target effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import equicorrelation
from .grids import QTTGrid, short_maturity_axis, sinh_centered_axis
from .pricers import geometric_basket_put

SIGMA_BLOCK_POSITIONS = ("tail", "head", "interleaved")
SIGMA_AUGMENTED_MODES = ("bounded_standardized_risk", "price_adaptive")


@dataclass(frozen=True)
class SigmaAugmentedConfig:
    """Shared axes plus ``n_sigma_axes`` appended volatility axes.

    Every bound and node count on the shared axes matches ``PaperConfig``, so a
    ``n_sigma_axes = 0`` instance describes the same physical grid as the
    frozen-volatility setup and can be compared to it node by node.
    """

    n_assets: int = 5
    n_sigma_axes: int = 0
    sigma_block: str = "tail"
    sigma_reference: float = 0.20
    volatility_bounds: tuple[float, float] = (0.05, 0.60)
    volatility_nodes: int = 16
    spot_bounds: tuple[float, float] = (5.0, 150.0)
    strike_bounds: tuple[float, float] = (1.0, 200.0)
    rate_bounds: tuple[float, float] = (0.005, 0.08)
    maturity_bounds: tuple[float, float] = (1.0 / 365.0, 3.0)
    spot_nodes: int = 64
    strike_nodes: int = 512
    rate_nodes: int = 8
    maturity_nodes: int = 64
    dividends: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    correlation: np.ndarray = field(default_factory=lambda: equicorrelation(5, 0.30))
    seed: int = 20260327

    def __post_init__(self) -> None:
        if self.n_assets < 1:
            raise ValueError("n_assets must be positive")
        if not 0 <= self.n_sigma_axes <= self.n_assets:
            raise ValueError("n_sigma_axes must lie between zero and n_assets")
        if self.sigma_block not in SIGMA_BLOCK_POSITIONS:
            raise ValueError(f"sigma_block must be one of {SIGMA_BLOCK_POSITIONS}")
        if self.sigma_block == "interleaved" and self.n_sigma_axes not in (
            0,
            self.n_assets,
        ):
            raise ValueError(
                "the interleaved control order is only defined when every asset "
                "carries a volatility axis"
            )
        if any(n <= 1 or n & (n - 1) for n in self.physical_shape):
            raise ValueError("QTT requires every physical mode size to be a power of two")
        low, high = self.volatility_bounds
        if not 0.0 < low < high:
            raise ValueError("volatility bounds must be positive and increasing")
        if self.n_sigma_axes and not low < self.sigma_reference < high:
            raise ValueError("sigma_reference must lie strictly inside the sigma box")
        if np.asarray(self.dividends, dtype=float).shape != (self.n_assets,):
            raise ValueError("invalid dividend vector")
        if np.asarray(self.correlation, dtype=float).shape != (
            self.n_assets,
            self.n_assets,
        ):
            raise ValueError("invalid correlation matrix")
        np.linalg.cholesky(np.asarray(self.correlation, dtype=float))

    # ---- column layout -------------------------------------------------

    @property
    def n_dimensions(self) -> int:
        return self.n_assets + 3 + self.n_sigma_axes

    @property
    def promoted_assets(self) -> tuple[int, ...]:
        """Assets whose volatility is a coordinate: the first ``d_sigma``."""
        return tuple(range(self.n_sigma_axes))

    def _layout(self) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        """Return ``(spot_columns, sigma_columns, first_shared_tail_column)``."""
        n, d = self.n_assets, self.n_sigma_axes
        if self.sigma_block == "tail":
            return tuple(range(n)), tuple(range(n + 3, n + 3 + d)), n
        if self.sigma_block == "head":
            return tuple(range(d, d + n)), tuple(range(d)), d + n
        return (
            tuple(2 * i for i in range(n)),
            tuple(2 * i + 1 for i in range(n)) if d else (),
            n + d,
        )

    @property
    def spot_columns(self) -> tuple[int, ...]:
        return self._layout()[0]

    @property
    def sigma_columns(self) -> tuple[int, ...]:
        return self._layout()[1]

    @property
    def strike_column(self) -> int:
        return self._layout()[2]

    @property
    def rate_column(self) -> int:
        return self._layout()[2] + 1

    @property
    def maturity_column(self) -> int:
        return self._layout()[2] + 2

    @property
    def shared_columns(self) -> tuple[int, ...]:
        """The eight shared axes, in their canonical order."""
        return (
            *self.spot_columns,
            self.strike_column,
            self.rate_column,
            self.maturity_column,
        )

    @property
    def physical_shape(self) -> tuple[int, ...]:
        out: list[int | None] = [None] * self.n_dimensions
        for column in self.spot_columns:
            out[column] = self.spot_nodes
        for column in self.sigma_columns:
            out[column] = self.volatility_nodes
        out[self.strike_column] = self.strike_nodes
        out[self.rate_column] = self.rate_nodes
        out[self.maturity_column] = self.maturity_nodes
        return tuple(int(value) for value in out)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        out: list[tuple[float, float] | None] = [None] * self.n_dimensions
        for column in self.spot_columns:
            out[column] = self.spot_bounds
        for column in self.sigma_columns:
            out[column] = self.volatility_bounds
        out[self.strike_column] = self.strike_bounds
        out[self.rate_column] = self.rate_bounds
        out[self.maturity_column] = self.maturity_bounds
        return tuple(out)

    @property
    def qtt_bits(self) -> tuple[int, ...]:
        return tuple(int(np.log2(n)) for n in self.physical_shape)

    @property
    def n_qtt_cores(self) -> int:
        return int(sum(self.qtt_bits))

    @property
    def sigma_core_indices(self) -> tuple[int, ...]:
        """Indices, in the QTT chain, of the cores belonging to sigma axes."""
        sigma = set(self.sigma_columns)
        out, core = [], 0
        for column, bits in enumerate(self.qtt_bits):
            if column in sigma:
                out.extend(range(core, core + bits))
            core += bits
        return tuple(out)

    def sigma_matrix(self, parameters: np.ndarray) -> np.ndarray:
        """Full ``(m, n_assets)`` volatility matrix for market or model rows.

        Non-promoted assets take ``sigma_reference``. At ``n_sigma_axes = 0``
        this is the frozen volatility vector of the paper setup, broadcast.
        """
        x = np.atleast_2d(np.asarray(parameters, dtype=float))
        sigma = np.full((x.shape[0], self.n_assets), float(self.sigma_reference))
        if self.n_sigma_axes:
            sigma[:, list(self.promoted_assets)] = x[:, list(self.sigma_columns)]
        return sigma


def _basket_log_moneyness_bounds(config) -> tuple[float, float]:
    """Same global log-moneyness box as ``risk_grids``, on this config."""
    return (
        float(np.log(config.strike_bounds[0] / config.spot_bounds[1])),
        float(np.log(config.strike_bounds[1] / config.spot_bounds[0])),
    )


@dataclass(frozen=True)
class _SigmaAugmentedTransform:
    """Shared machinery: a basket spot, and sigma held per row."""

    config: SigmaAugmentedConfig

    def _basket(self, spots: np.ndarray) -> np.ndarray:
        return np.exp(np.mean(np.log(spots), axis=1))

    def basket_spot(self, parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(parameters, dtype=float))
        return self._basket(x[:, list(self.config.spot_columns)])

    @property
    def weights(self) -> np.ndarray:
        return np.full(self.config.n_assets, 1.0 / self.config.n_assets)

    def basket_volatility(self, sigma: np.ndarray) -> np.ndarray:
        """Row-wise ``sqrt(w' (sigma sigma' o C) w)``, kept factored over rows.

        Deliberately the same factored expression as
        ``vol_extension.VolStandardizedRiskTransform.basket_volatility``, so the
        two agree to the last bit when every asset is promoted.
        """
        s = np.atleast_2d(np.asarray(sigma, dtype=float))
        scaled = s * self.weights
        variance = np.sum(
            (scaled @ np.asarray(self.config.correlation, dtype=float)) * scaled,
            axis=1,
        )
        return np.sqrt(variance)


@dataclass(frozen=True)
class SigmaAugmentedMoneynessTransform(_SigmaAugmentedTransform):
    """``m = log(K / G(S))`` on the strike column; sigma columns untouched."""

    def to_model(self, market_parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(market_parameters, dtype=float)).copy()
        column = self.config.strike_column
        x[:, column] = np.log(x[:, column] / self.basket_spot(x))
        return x

    def to_market(self, model_parameters: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(model_parameters, dtype=float)).copy()
        column = self.config.strike_column
        z[:, column] = self.basket_spot(z) * np.exp(z[:, column])
        return z


@dataclass(frozen=True)
class SigmaAugmentedBoundedRiskTransform(_SigmaAugmentedTransform):
    r"""Volatility-aware standardized risk coordinate normalized to ``[-1, 1]``.

    Identical arithmetic to
    ``vol_extension.BoundedVolStandardizedRiskTransform``, with the volatility
    vector assembled from the promoted columns and ``sigma_reference``
    everywhere else. At ``d_sigma = 0`` the assembled vector is constant, so the
    coordinate reduces to the frozen-volatility one of
    ``risk_grids.BoundedStandardizedRiskTransform``.
    """

    moneyness_bounds: tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self) -> None:
        lower, upper = self.moneyness_bounds
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("invalid log-moneyness bounds")

    def _ridge_and_scale(self, sigma, rates, maturities):
        s = np.atleast_2d(np.asarray(sigma, dtype=float))
        weights = self.weights
        variance = self.basket_volatility(s) ** 2
        carry = (
            rates
            - float(weights @ np.asarray(self.config.dividends, dtype=float))
            - 0.5 * (s**2 @ weights)
            + 0.5 * variance
        )
        ridge = (carry + 0.5 * variance) * maturities
        scale = np.sqrt(variance) * np.sqrt(maturities)
        return ridge, scale

    def standardized_bounds(self, ridge, scale):
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

    def to_model(self, market_parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(market_parameters, dtype=float)).copy()
        config = self.config
        ridge, scale = self._ridge_and_scale(
            config.sigma_matrix(x),
            x[:, config.rate_column],
            x[:, config.maturity_column],
        )
        log_moneyness = np.log(x[:, config.strike_column] / self.basket_spot(x))
        x[:, config.strike_column] = self._coordinate(log_moneyness, ridge, scale)
        return x

    def to_market(self, model_parameters: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(model_parameters, dtype=float)).copy()
        config = self.config
        ridge, scale = self._ridge_and_scale(
            config.sigma_matrix(z),
            z[:, config.rate_column],
            z[:, config.maturity_column],
        )
        log_moneyness = self._log_moneyness(
            z[:, config.strike_column], ridge, scale
        )
        z[:, config.strike_column] = self.basket_spot(z) * np.exp(log_moneyness)
        return z


class SigmaAugmentedGeometricPricer:
    """Closed-form geometric-basket put over the augmented market vector."""

    def __init__(self, config: SigmaAugmentedConfig):
        self.config = config

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(parameters, dtype=float))
        config = self.config
        if x.shape[1] != config.n_dimensions:
            raise ValueError("wrong parameter dimension for the augmented pricer")
        return geometric_basket_put(
            x[:, list(config.spot_columns)],
            x[:, config.strike_column],
            x[:, config.rate_column],
            x[:, config.maturity_column],
            config.sigma_matrix(x),
            config.correlation,
            config.dividends,
        )


def build_sigma_augmented_grid(config: SigmaAugmentedConfig, mode: str):
    """QTT grid, transform and description on the augmented domain.

    The shared axes are built by the same expressions as
    ``risk_grids.build_greek_coordinate_grid``: uniform spots, uniform rate, a
    maturity axis quadratic near ``T_min``, and a strike coordinate set by the
    mode. Nothing in them depends on ``n_sigma_axes``, which is the additivity
    the whole comparison rests on.
    """
    if mode not in SIGMA_AUGMENTED_MODES:
        raise ValueError(f"unknown mode: {mode}; expected {SIGMA_AUGMENTED_MODES}")

    m_bounds = _basket_log_moneyness_bounds(config)
    spot_axis = np.linspace(*config.spot_bounds, config.spot_nodes)
    rate_axis = np.linspace(*config.rate_bounds, config.rate_nodes)
    maturity_axis = short_maturity_axis(
        *config.maturity_bounds, config.maturity_nodes, power=2.0
    )
    sigma_axis = np.linspace(*config.volatility_bounds, config.volatility_nodes)

    if mode == "bounded_standardized_risk":
        transform = SigmaAugmentedBoundedRiskTransform(
            config=config, moneyness_bounds=m_bounds
        )
        coordinate_axis = np.linspace(-1.0, 1.0, config.strike_nodes)
        coordinate_definition = (
            "normalized volatility-aware standardized risk coordinate: the "
            "(r,T,sigma)-dependent z interval mapped to [-1,1]"
        )
        construction = {"type": "bounded tensor-compatible standardized risk"}
    else:
        transform = SigmaAugmentedMoneynessTransform(config=config)
        coordinate_axis = sinh_centered_axis(
            *m_bounds, config.strike_nodes, concentration=3.0
        )
        coordinate_definition = "m = log(K / basket_spot)"
        construction = {
            "type": "price-adaptive sinh concentration around m=0",
            "concentration": 3.0,
        }

    axes: list = [None] * config.n_dimensions
    for column in config.spot_columns:
        axes[column] = spot_axis
    for column in config.sigma_columns:
        axes[column] = sigma_axis
    axes[config.strike_column] = coordinate_axis
    axes[config.rate_column] = rate_axis
    axes[config.maturity_column] = maturity_axis
    grid = QTTGrid(axes=axes)

    description = {
        "mode": mode,
        "n_sigma_axes": int(config.n_sigma_axes),
        "sigma_block": config.sigma_block,
        "promoted_assets": list(config.promoted_assets),
        "sigma_reference": float(config.sigma_reference),
        "physical_shape": list(grid.shape),
        "qtt_core_count": int(sum(grid.bits)),
        "log2_logical_entries": int(sum(grid.bits)),
        "spot_columns": list(config.spot_columns),
        "sigma_columns": list(config.sigma_columns),
        "strike_column": int(config.strike_column),
        "rate_column": int(config.rate_column),
        "maturity_column": int(config.maturity_column),
        "sigma_core_indices": list(config.sigma_core_indices),
        "strike_coordinate": coordinate_definition,
        "coordinate_bounds": [float(coordinate_axis[0]), float(coordinate_axis[-1])],
        "moneyness_bounds": list(m_bounds),
        "volatility_coordinate": "sigma (uniform nodes, linear scale)",
        "volatility_bounds": list(config.volatility_bounds),
        "volatility_axis": sigma_axis.tolist(),
        "maturity_nodes": "quadratic near T_min",
        "construction": construction,
    }
    return grid, transform, description
