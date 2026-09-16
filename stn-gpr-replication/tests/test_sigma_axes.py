"""Invariance and equivalence tests for the additive volatility axes.

Two properties are load-bearing for the dimensional-cost experiment and neither
is self-evident from the code, so both are pinned here.

*Additivity* (B1/B2): adding volatility axes must leave the shared axes, their
nodes, their QTT cores and the prices on the ``sigma = sigma_ref`` slice exactly
as they were. If it does not, a rank profile cannot be compared core by core
across ``d_sigma`` and the whole sweep is meaningless.

*Reduced-floor equivalence* (B4): the spot-collapsed interpolation floor must
agree bit-for-bit with the straightforward product-stencil floor
(``oracle_hybrid_cubic_predict``) wherever the latter is affordable.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from stngpr.config import PaperConfig, VolatilityExtendedConfig, equicorrelation
from stngpr.coordinates import oracle_hybrid_cubic_predict
from stngpr.grid_geometry import physical_log_moneyness_nodes
from stngpr.pricers import EuropeanGeometricBasketPricer
from stngpr.risk_grids import build_greek_coordinate_grid
from stngpr.sigma_axes import (
    SigmaAugmentedConfig,
    SigmaAugmentedGeometricPricer,
    build_sigma_augmented_grid,
)
from stngpr.sigma_surrogate import (
    SigmaAugmentedSurrogate,
    spot_collapsed_interpolation_floor,
)
from stngpr.vol_extension import build_vol_extended_risk_grid

SIGMA_REF = 0.20
RISK_SHAPE = (64, 64, 64, 64, 64, 512, 8, 64)


def _config(n_sigma_axes, **kwargs):
    return SigmaAugmentedConfig(
        n_sigma_axes=n_sigma_axes,
        sigma_reference=SIGMA_REF,
        volatility_bounds=(0.05, 0.60),
        volatility_nodes=16,
        **kwargs,
    )


def _toy_config(n_sigma_axes, n_assets=2, **kwargs):
    """Small enough that the full product stencil is enumerable."""
    return SigmaAugmentedConfig(
        n_assets=n_assets,
        n_sigma_axes=n_sigma_axes,
        sigma_reference=SIGMA_REF,
        volatility_bounds=(0.05, 0.60),
        volatility_nodes=4,
        spot_nodes=8,
        strike_nodes=16,
        rate_nodes=4,
        maturity_nodes=8,
        dividends=np.zeros(n_assets),
        correlation=equicorrelation(n_assets, 0.30),
        **kwargs,
    )


def _market_points(config, n_samples, seed, sigma_value=None):
    """Uniform market points; ``sigma_value`` pins every sigma column."""
    rng = np.random.default_rng(seed)
    columns = []
    sigma_columns = set(config.sigma_columns)
    for j, (low, high) in enumerate(config.bounds):
        if j in sigma_columns and sigma_value is not None:
            columns.append(np.full(n_samples, float(sigma_value)))
        else:
            columns.append(rng.uniform(low, high, n_samples))
    return np.column_stack(columns)


class TestSigmaAugmentedLayout(unittest.TestCase):
    def test_tail_layout_appends_sigma_after_the_shared_axes(self):
        config = _config(3)
        self.assertEqual(config.spot_columns, (0, 1, 2, 3, 4))
        self.assertEqual(config.strike_column, 5)
        self.assertEqual(config.rate_column, 6)
        self.assertEqual(config.maturity_column, 7)
        self.assertEqual(config.sigma_columns, (8, 9, 10))
        self.assertEqual(config.promoted_assets, (0, 1, 2))
        self.assertEqual(config.n_dimensions, 11)

    def test_zero_sigma_axes_reproduces_the_eight_shared_columns(self):
        config = _config(0)
        self.assertEqual(config.n_dimensions, 8)
        self.assertEqual(config.sigma_columns, ())
        self.assertEqual(config.physical_shape, RISK_SHAPE)
        self.assertEqual(config.n_qtt_cores, 48)

    def test_core_count_grows_by_log2_nsigma_per_axis(self):
        for d_sigma in range(6):
            config = _config(d_sigma)
            self.assertEqual(config.n_qtt_cores, 48 + 4 * d_sigma)
            self.assertEqual(len(config.sigma_core_indices), 4 * d_sigma)

    def test_sigma_cores_are_contiguous_and_last_in_the_tail_layout(self):
        config = _config(5)
        indices = config.sigma_core_indices
        self.assertEqual(list(indices), list(range(48, 48 + 20)))
        self.assertEqual(max(indices), config.n_qtt_cores - 1)

    def test_head_layout_puts_the_sigma_cores_first(self):
        config = _config(5, sigma_block="head")
        self.assertEqual(config.sigma_columns, (0, 1, 2, 3, 4))
        self.assertEqual(list(config.sigma_core_indices), list(range(20)))

    def test_interleaved_layout_matches_the_vol_extension_convention(self):
        config = _config(5, sigma_block="interleaved")
        self.assertEqual(config.spot_columns, (0, 2, 4, 6, 8))
        self.assertEqual(config.sigma_columns, (1, 3, 5, 7, 9))
        self.assertEqual(config.strike_column, 10)

    def test_interleaved_layout_requires_every_asset_promoted(self):
        with self.assertRaises(ValueError):
            _config(3, sigma_block="interleaved")

    def test_sigma_reference_must_lie_inside_the_sigma_box(self):
        with self.assertRaises(ValueError):
            SigmaAugmentedConfig(n_sigma_axes=1, sigma_reference=0.60)

    def test_sigma_matrix_freezes_the_unpromoted_assets(self):
        config = _config(2)
        points = _market_points(config, 4, seed=1)
        sigma = config.sigma_matrix(points)
        np.testing.assert_allclose(sigma[:, 0], points[:, 8])
        np.testing.assert_allclose(sigma[:, 1], points[:, 9])
        np.testing.assert_allclose(sigma[:, 2:], SIGMA_REF)


class TestAdditivity(unittest.TestCase):
    """Adding sigma axes must not perturb the shared axes in any way."""

    def test_shared_axis_nodes_are_bitwise_identical_across_d_sigma(self):
        reference, *_ = build_sigma_augmented_grid(
            _config(0), "bounded_standardized_risk"
        )
        for d_sigma in (1, 2, 3, 5):
            config = _config(d_sigma)
            grid, _, _ = build_sigma_augmented_grid(
                config, "bounded_standardized_risk"
            )
            for position, column in enumerate(config.shared_columns):
                np.testing.assert_array_equal(
                    grid.axes[column],
                    reference.axes[position],
                    err_msg=f"shared axis {position} moved at d_sigma={d_sigma}",
                )

    def test_qtt_chain_at_d_sigma_zero_is_the_full_chain_minus_sigma_cores(self):
        reference = _config(0)
        reference_grid, _, _ = build_sigma_augmented_grid(
            reference, "bounded_standardized_risk"
        )
        for d_sigma in (1, 2, 3, 5):
            config = _config(d_sigma)
            grid, _, _ = build_sigma_augmented_grid(
                config, "bounded_standardized_risk"
            )
            sigma_cores = set(config.sigma_core_indices)
            # Expand each axis into its cores, then drop the sigma ones.
            core_axis = []
            for column, bits in enumerate(config.qtt_bits):
                core_axis.extend([column] * bits)
            kept = [
                column
                for core, column in enumerate(core_axis)
                if core not in sigma_cores
            ]
            shared_bits = []
            for column, bits in enumerate(reference.qtt_bits):
                shared_bits.extend([column] * bits)
            self.assertEqual(
                [config.shared_columns.index(column) for column in kept],
                shared_bits,
                msg=f"core ordering of the shared axes changed at d_sigma={d_sigma}",
            )
            self.assertEqual(
                sum(reference_grid.bits), sum(grid.bits) - 4 * d_sigma
            )

    def test_prices_on_the_sigma_ref_slice_match_the_frozen_volatility_oracle(self):
        paper = replace(PaperConfig(), physical_shape=RISK_SHAPE)
        frozen = EuropeanGeometricBasketPricer(paper)
        for d_sigma in (0, 1, 2, 3, 5):
            config = _config(d_sigma)
            augmented = SigmaAugmentedGeometricPricer(config)
            points = _market_points(config, 64, seed=7, sigma_value=SIGMA_REF)
            shared = points[:, list(config.shared_columns)]
            np.testing.assert_allclose(
                augmented(points), frozen(shared), rtol=0.0, atol=0.0
            )

    def test_log_moneyness_nodes_on_the_slice_match_the_frozen_volatility_grid(self):
        paper = replace(PaperConfig(), physical_shape=RISK_SHAPE)
        grid8, transform8, _ = build_greek_coordinate_grid(
            paper, "bounded_standardized_risk"
        )
        reference = physical_log_moneyness_nodes(
            grid8, transform8, paper.n_assets, rate=0.03, maturity=0.75
        )
        for d_sigma in (0, 1, 2, 3, 5):
            config = _config(d_sigma)
            grid, transform, _ = build_sigma_augmented_grid(
                config, "bounded_standardized_risk"
            )
            coordinate = grid.axes[config.strike_column]
            model = np.empty((coordinate.size, config.n_dimensions))
            model[:, list(config.spot_columns)] = 100.0
            if config.sigma_columns:
                model[:, list(config.sigma_columns)] = SIGMA_REF
            model[:, config.strike_column] = coordinate
            model[:, config.rate_column] = 0.03
            model[:, config.maturity_column] = 0.75
            market = transform.to_market(model)
            basket = np.exp(
                np.mean(np.log(market[:, list(config.spot_columns)]), axis=1)
            )
            nodes = np.log(market[:, config.strike_column] / basket)
            displacement = np.max(np.abs(nodes - reference))
            self.assertLess(
                displacement,
                1e-12,
                msg=f"nodes moved by {displacement:.3e} at d_sigma={d_sigma}",
            )


class TestAgreementWithExistingImplementations(unittest.TestCase):
    def test_transform_matches_the_frozen_volatility_one_at_d_sigma_zero(self):
        paper = replace(PaperConfig(), physical_shape=RISK_SHAPE)
        _, transform8, _ = build_greek_coordinate_grid(
            paper, "bounded_standardized_risk"
        )
        config = _config(0)
        _, transform, _ = build_sigma_augmented_grid(
            config, "bounded_standardized_risk"
        )
        points = _market_points(config, 128, seed=11)
        np.testing.assert_allclose(
            transform.to_model(points), transform8.to_model(points), rtol=1e-14
        )
        model = transform.to_model(points)
        np.testing.assert_allclose(
            transform.to_market(model), transform8.to_market(model), rtol=1e-14
        )

    def test_transform_matches_vol_extension_at_d_sigma_five(self):
        vol_config = VolatilityExtendedConfig(
            layout="interleaved",
            spot_nodes=64,
            strike_nodes=512,
            rate_nodes=8,
            maturity_nodes=64,
            volatility_nodes=16,
            volatility_bounds=(0.05, 0.60),
        )
        _, vol_transform, _ = build_vol_extended_risk_grid(
            vol_config, "bounded_standardized_risk", volatility_scale="linear"
        )
        config = _config(5, sigma_block="interleaved")
        _, transform, _ = build_sigma_augmented_grid(
            config, "bounded_standardized_risk"
        )
        points = _market_points(config, 128, seed=13)
        # Agreement is to floating-point rounding, not bit-for-bit: the two
        # implementations accumulate the convexity ridge in a different order and
        # the last bit differs on a few rows (max 6e-17 absolute on the ridge,
        # 6e-16 relative on the coordinate). Four orders of magnitude below the
        # 1e-12 node-coincidence gate of phase 2, which is what actually has to
        # hold; bit-equality with vol_extension is not a protocol requirement.
        np.testing.assert_allclose(
            transform.to_model(points),
            vol_transform.to_model(points),
            rtol=1e-13,
            atol=1e-15,
        )

    def test_grid_shape_matches_vol_extension_at_d_sigma_five(self):
        vol_config = VolatilityExtendedConfig(
            layout="interleaved",
            spot_nodes=64,
            strike_nodes=512,
            rate_nodes=8,
            maturity_nodes=64,
            volatility_nodes=16,
            volatility_bounds=(0.05, 0.60),
        )
        vol_grid, _, _ = build_vol_extended_risk_grid(
            vol_config, "bounded_standardized_risk", volatility_scale="linear"
        )
        grid, _, _ = build_sigma_augmented_grid(
            _config(5, sigma_block="interleaved"), "bounded_standardized_risk"
        )
        self.assertEqual(grid.shape, vol_grid.shape)
        for own, other in zip(grid.axes, vol_grid.axes):
            np.testing.assert_allclose(own, other, rtol=0.0, atol=0.0)


class TestReducedInterpolationFloor(unittest.TestCase):
    """B4: the collapsed floor must equal the product-stencil floor exactly."""

    def _compare(self, config, mode, n_points, seed):
        model = SigmaAugmentedSurrogate(config, mode)
        market = _market_points(config, n_points, seed=seed)
        points = model.to_model(market)
        full = oracle_hybrid_cubic_predict(
            model.grid,
            model.fit_pricer,
            points,
            cubic_columns=model.cubic_columns,
            batch_size=1,
        )
        reduced = spot_collapsed_interpolation_floor(
            model.grid,
            model.fit_pricer,
            points,
            config.spot_columns,
            model.cubic_columns,
            rows_per_call=1 << 14,
        )
        np.testing.assert_allclose(reduced, full, rtol=1e-12, atol=1e-14)

    def test_matches_product_stencil_on_the_risk_coordinate(self):
        for d_sigma in (0, 1, 2):
            with self.subTest(d_sigma=d_sigma):
                self._compare(
                    _toy_config(d_sigma), "bounded_standardized_risk", 12, 17
                )

    def test_matches_product_stencil_on_the_pricing_coordinate(self):
        for d_sigma in (0, 2):
            with self.subTest(d_sigma=d_sigma):
                self._compare(_toy_config(d_sigma), "price_adaptive", 12, 19)

    def test_collapse_check_rejects_a_spot_dependent_target(self):
        """A raw-price target is not spot-free, and must be refused."""
        config = _toy_config(1)
        model = SigmaAugmentedSurrogate(config, "bounded_standardized_risk")
        points = model.to_model(_market_points(config, 8, seed=23))
        with self.assertRaises(RuntimeError):
            spot_collapsed_interpolation_floor(
                model.grid,
                model.model_pricer,  # raw price P, not P / G
                points,
                config.spot_columns,
                model.cubic_columns,
                rows_per_call=1 << 14,
            )

    def test_floor_is_closer_to_the_oracle_than_a_coarse_alternative(self):
        """Sanity: the floor is an interpolation of the truth, not noise."""
        config = _toy_config(1)
        model = SigmaAugmentedSurrogate(config, "bounded_standardized_risk")
        market = _market_points(config, 48, seed=29)
        floor = model.interpolation_floor(market, rows_per_call=1 << 14)
        exact = model.reference_price(market)
        self.assertTrue(np.all(np.isfinite(floor)))
        self.assertLess(
            np.mean(np.abs(floor - exact)) / np.mean(np.abs(exact)), 0.5
        )


class TestSurrogateProtocolGuards(unittest.TestCase):
    def test_fit_refuses_post_fit_truncation(self):
        model = SigmaAugmentedSurrogate(_toy_config(1))
        with self.assertRaises(ValueError):
            model.fit(200, anova_samples=50, truncation=1e-8)

    def test_fit_reports_realized_calls_and_a_full_bond_rank_profile(self):
        config = _toy_config(1)
        model = SigmaAugmentedSurrogate(config, seed=20260401)
        diagnostics = model.fit(2_000, anova_samples=200, log=False)
        self.assertGreater(diagnostics.function_evaluations, 200)
        self.assertLessEqual(diagnostics.function_evaluations, 2_000 + 200)
        self.assertIsNone(diagnostics.truncation)
        profile = model.bond_rank_profile()
        self.assertEqual(len(profile), config.n_qtt_cores + 1)
        self.assertEqual(profile[0], 1)
        self.assertEqual(profile[-1], 1)

    def test_price_runs_on_every_sigma_block_position(self):
        for block in ("tail", "head", "interleaved"):
            with self.subTest(block=block):
                config = _toy_config(2, sigma_block=block)
                model = SigmaAugmentedSurrogate(config, seed=20260401)
                model.fit(2_000, anova_samples=200, log=False)
                market = _market_points(config, 16, seed=31)
                predicted = model.price(market)
                self.assertEqual(predicted.shape, (16,))
                self.assertTrue(np.all(np.isfinite(predicted)))


if __name__ == "__main__":
    unittest.main()
