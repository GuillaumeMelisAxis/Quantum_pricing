r"""End-to-end surrogate and interpolation floor on sigma-augmented grids.

Two pieces live here.

:class:`SigmaAugmentedSurrogate` wires grid, coordinate transform, ``P / G(S)``
scaling, TT-cross and the production interpolant together for a
:class:`~stngpr.sigma_axes.SigmaAugmentedConfig`. It refuses post-fit SVD
truncation outright rather than defaulting it off, because a truncated rank is
not the quantity the dimensional-cost experiment measures.

:func:`spot_collapsed_interpolation_floor` computes the *production*
interpolation floor — the production interpolant, cubic on every axis, fed the
exact oracle at the stencil nodes — which the straightforward path cannot do
past eight or nine dimensions. ``oracle_hybrid_cubic_predict`` enumerates the
product stencil, so a thirteen-axis cubic floor is ``4**13`` = 67 million
corners per query and is refused outright.

The way out is an exact identity, not an approximation. On a moneyness or
standardized-risk coordinate the log-moneyness ``m`` recovered by the inverse
transform depends on ``(coordinate, r, T, sigma)`` and not on the spots; the
geometric-basket price is homogeneous of degree one in ``(G, K)``; so the fit
target ``P / G`` takes the *same value at every combination of spot stencil
nodes*. The Lagrange weights on each axis sum to one exactly, so the spot axes
contribute a factor of one to the weighted sum and drop out. The effective
stencil falls from ``4**13`` to ``4**8`` = 65 536 with the value unchanged to
the last bit, not merely to some tolerance.

That identity is a property of this payoff and these coordinates, so it is
checked at run time on a subsample rather than assumed: the floor is evaluated
with the spot axes pinned to two different stencil candidates and the two must
agree. ``tests/test_sigma_axes.py`` additionally pins the reduced path against
``oracle_hybrid_cubic_predict`` on a toy grid small enough to enumerate in full.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict
from time import perf_counter

import numpy as np

from .coordinates import TransformedPricer
from .sigma_axes import SigmaAugmentedGeometricPricer, build_sigma_augmented_grid
from .tt_surrogate import TTPriceSurrogate
from .vol_extension import BasketScaledModelPricer


def _selector_table(sizes: tuple[int, ...]) -> np.ndarray:
    """All product-stencil selectors as an ``(n_corners, n_axes)`` table."""
    return np.array(
        list(itertools.product(*(range(size) for size in sizes))),
        dtype=np.int64,
    ).reshape(-1, len(sizes))


def _floor_with_pinned_spots(
    grid,
    fit_pricer,
    points,
    spot,
    cubic_columns,
    spot_candidate: int,
    rows_per_call: int,
):
    """Weighted stencil sum with every spot axis pinned to one candidate.

    Pinning is what makes the cost tractable, and it changes nothing only
    because the target is spot-free; the caller is responsible for checking
    that, which is why ``spot_candidate`` is an argument rather than hard-wired.
    """
    n_axes = len(grid.shape)
    supports, weights = grid.axis_stencils(points, cubic_columns)

    local_supports, sizes = [], []
    for j in range(n_axes):
        if j in spot:
            index = min(int(spot_candidate), supports[j].shape[1] - 1)
            local_supports.append(supports[j][:, index : index + 1])
            sizes.append(1)
        else:
            local_supports.append(supports[j])
            sizes.append(supports[j].shape[1])

    table = _selector_table(tuple(sizes))
    n_corners = table.shape[0]
    batch = max(1, int(rows_per_call) // n_corners)
    chunk = n_corners if batch > 1 else max(1, int(rows_per_call))
    out = np.empty(len(points), dtype=float)

    for start in range(0, len(points), batch):
        stop = min(start + batch, len(points))
        rows = np.arange(stop - start)[:, None]
        total = np.zeros(stop - start, dtype=float)
        for first in range(0, n_corners, chunk):
            selector = table[first : first + chunk]
            corners = selector.shape[0]
            indices = np.empty((stop - start, corners, n_axes), dtype=np.int64)
            weight = np.ones((stop - start, corners), dtype=float)
            for j in range(n_axes):
                column = selector[None, :, j]
                block = local_supports[j][start:stop]
                indices[:, :, j] = block[rows, column]
                if sizes[j] > 1:
                    weight *= weights[j][start:stop][rows, column]
            flat = indices.reshape(-1, n_axes)
            values = np.asarray(
                fit_pricer(grid.indices_to_points(flat)), dtype=float
            ).reshape(stop - start, corners)
            total += np.sum(weight * values, axis=1)
        out[start:stop] = total
    return out


def spot_collapsed_interpolation_floor(
    grid,
    fit_pricer,
    model_points,
    spot_columns,
    cubic_columns,
    rows_per_call: int = 1 << 18,
    verify_points: int = 32,
    verify_tolerance: float = 1e-12,
    diagnostics: dict | None = None,
):
    """Production interpolation floor with the spot stencil collapsed exactly.

    Returns the interpolated value of ``fit_pricer`` (i.e. of ``P / G``); the
    caller re-multiplies by ``G`` at the query point, exactly as
    :meth:`SigmaAugmentedSurrogate.price` does for the TT path.

    The collapse residual is measured on a subsample and, if ``diagnostics`` is
    given, recorded there so it can be reported rather than merely asserted.

    The residual is normalized by ``mean |value|`` over the subsample, the same
    normalization the reported NMAE uses, and not pointwise. Pointwise relative
    error is meaningless on this test domain: uniform sampling of the strike box
    reaches deep out-of-the-money puts where ``P / G`` underflows to ``1e-279``,
    and a last-bit absolute wobble there is a relative error of ``1e-11`` that
    says nothing about whether the identity holds. A genuine spot dependence is
    not a subtle effect - ``G`` spans a factor of thirty across the spot box - so
    this norm still separates the two cases by ten orders of magnitude.
    """
    points = np.atleast_2d(np.asarray(model_points, dtype=float))
    spot = tuple(int(j) for j in spot_columns)
    _, weights = grid.axis_stencils(points, cubic_columns)

    # A Lagrange (or linear) stencil is a partition of unity on every axis, and
    # that is the half of the identity that does not depend on the payoff.
    for j in spot:
        deviation = float(np.max(np.abs(np.sum(weights[j], axis=1) - 1.0)))
        if deviation > 1e-12:
            raise RuntimeError(
                f"spot axis {j} stencil weights do not sum to one "
                f"(max deviation {deviation:.3e}); the collapse is not valid"
            )

    estimate = _floor_with_pinned_spots(
        grid, fit_pricer, points, spot, cubic_columns, 0, rows_per_call
    )

    # The other half does depend on the payoff and the coordinate, so measure it
    # on a subsample instead of assuming it.
    if verify_points and spot:
        n_check = min(int(verify_points), len(points))
        check = np.unique(np.linspace(0, len(points) - 1, n_check).astype(int))
        subset = points[check]
        highest = max(
            grid.axis_stencils(subset, cubic_columns)[0][j].shape[1] for j in spot
        )
        pinned_high = _floor_with_pinned_spots(
            grid, fit_pricer, subset, spot, cubic_columns, highest - 1, rows_per_call
        )
        reference = estimate[check]
        scale = float(np.mean(np.abs(reference)))
        absolute = float(np.max(np.abs(pinned_high - reference)))
        residual = absolute / scale if scale > 0.0 else absolute
        if diagnostics is not None:
            diagnostics.update(
                {
                    "collapse_check_points": int(reference.size),
                    "collapse_max_absolute_residual": absolute,
                    "collapse_normalized_residual": residual,
                    "collapse_tolerance": float(verify_tolerance),
                    "collapsed_axes": list(spot),
                    "effective_corners_per_query": int(
                        np.prod(
                            [
                                1 if j in spot else w.shape[1]
                                for j, w in enumerate(weights)
                            ]
                        )
                    ),
                    "product_stencil_corners_per_query": int(
                        np.prod([w.shape[1] for w in weights])
                    ),
                }
            )
        if residual > float(verify_tolerance):
            raise RuntimeError(
                "the spot-collapse identity does not hold on this grid: pinning "
                "the spot axes to a different stencil candidate moves the floor "
                f"by {residual:.3e} in the NMAE norm, above "
                f"{verify_tolerance:.1e}. The fit target is not spot-free here, "
                "so the reduced floor is invalid and the full product stencil "
                "would be required."
            )

    return estimate


class SigmaAugmentedSurrogate:
    """Grid, ``P/G`` target, TT-cross and production interpolant, wired up."""

    def __init__(
        self,
        config,
        mode: str = "bounded_standardized_risk",
        seed: int | None = None,
    ):
        self.config = config
        self.mode = mode
        self.grid, self.transform, self.grid_description = build_sigma_augmented_grid(
            config, mode
        )
        self.market_pricer = SigmaAugmentedGeometricPricer(config)
        self.model_pricer = TransformedPricer(self.market_pricer, self.transform)
        # B3: one fit target for every d_sigma, so the comparison cannot mix a
        # dimension effect with a target effect.
        self.fit_pricer = BasketScaledModelPricer(
            self.model_pricer, config.spot_columns, "geometric"
        )
        self.cubic_columns = tuple(range(config.n_dimensions))
        self.seed = int(config.seed if seed is None else seed)
        self.surrogate = TTPriceSurrogate(self.grid, self.fit_pricer, seed=self.seed)

    # ---- coordinates ---------------------------------------------------

    def to_model(self, market_points: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(market_points, dtype=float))
        if x.shape[1] != self.config.n_dimensions:
            raise ValueError(
                f"market points must have {self.config.n_dimensions} columns"
            )
        return self.transform.to_model(x)

    # ---- fitting -------------------------------------------------------

    def fit(self, max_evals: int, anova_samples: int = 2_000, **kwargs):
        """Run TT-cross. Post-fit SVD truncation is refused, not defaulted off."""
        if kwargs.pop("truncation", None) is not None:
            raise ValueError(
                "post-fit SVD truncation is not part of this protocol; the cores "
                "are reported as TT-cross produced them"
            )
        start = perf_counter()
        diagnostics = self.surrogate.fit(
            max_evals, anova_samples=anova_samples, truncation=None, **kwargs
        )
        self.fit_seconds = perf_counter() - start
        return diagnostics

    def core_diagnostics(self):
        """Untruncated bond-rank profile and parameter count."""
        return self.surrogate.apply_truncation(None)

    def bond_rank_profile(self) -> list[int]:
        """``r_0 ... r_d`` over the QTT chain, both boundary ranks included."""
        return [1, *self.core_diagnostics().bond_ranks, 1]

    def fit_record(self) -> dict:
        cores = self.core_diagnostics()
        record = {
            **asdict(self.surrogate.diagnostics),
            "bond_ranks_full": self.bond_rank_profile(),
            "sigma_core_indices": list(self.config.sigma_core_indices),
        }
        record["core_parameter_count"] = cores.parameter_count
        return record

    # ---- prediction ----------------------------------------------------

    def _basket(self, model_points):
        return self.transform.basket_spot(model_points)

    def price(self, market_points: np.ndarray, batch_size: int = 2_048):
        model_points = self.to_model(market_points)
        values = self.surrogate.predict_factorized(
            model_points, cubic_columns=self.cubic_columns, batch_size=batch_size
        )
        return values * self._basket(model_points)

    def reference_price(self, market_points: np.ndarray):
        x = np.atleast_2d(np.asarray(market_points, dtype=float))
        return np.asarray(self.market_pricer(x), dtype=float).reshape(-1)

    def interpolation_floor(
        self,
        market_points: np.ndarray,
        rows_per_call: int = 1 << 18,
        verify_points: int = 32,
    ):
        """Production interpolation floor: cubic on every axis, oracle at nodes.

        The collapse-residual diagnostics of the last call are kept on
        :attr:`floor_diagnostics` so a run can report them.
        """
        model_points = self.to_model(market_points)
        self.floor_diagnostics: dict = {}
        values = spot_collapsed_interpolation_floor(
            self.grid,
            self.fit_pricer,
            model_points,
            self.config.spot_columns,
            self.cubic_columns,
            rows_per_call=rows_per_call,
            verify_points=verify_points,
            diagnostics=self.floor_diagnostics,
        )
        return values * self._basket(model_points)
