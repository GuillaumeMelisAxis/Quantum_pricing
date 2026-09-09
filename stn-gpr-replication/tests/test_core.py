import math
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from stngpr.baselines import ManhattanLaplacian
from stngpr.config import PaperConfig, VolatilityExtendedConfig
from stngpr.convergence import (
    columnwise_error_metrics,
    component_error_decomposition,
    component_residual_metrics,
    cross_gamma_sign_diagnostics,
    curve_error_diagnostics,
    finite_difference_component_arrays,
    finite_difference_fixed_hybrid_component_arrays,
    full_hessian_error_metrics,
    gamma_matrices,
    mean_and_standard_error,
    tt_component_error_decomposition,
)
from stngpr.coordinates import (
    CachedGridInterpolator,
    CoordinateTransform,
    MarketCoordinatePricer,
    TransformedPricer,
    build_coordinate_grid,
    oracle_hybrid_cubic_predict,
    oracle_multilinear_predict,
)
from stngpr.diagnostics import (
    geometric_basket_convexity_ridge,
    geometric_basket_effective_parameters,
    geometric_basket_log_moneyness_convexity,
)
from stngpr.greeks import (
    finite_difference_greeks,
    finite_difference_hybrid_greeks,
    geometric_basket_put_spot_greeks,
    project_symmetric_matrix_psd,
)
from stngpr.grids import QTTGrid, sinh_centered_axis
from stngpr.portfolio import Portfolio
from stngpr.pricers import (
    AmericanArithmeticBasketLSMC,
    EuropeanArithmeticBasketQMC,
    EuropeanArithmeticBasketVolPricer,
    EuropeanGeometricBasketPricer,
    EuropeanGeometricBasketVolPricer,
    arithmetic_basket_put_levy,
    basket_put_monte_carlo,
    geometric_basket_put,
)
from stngpr.risk import (
    MonteCarloVaREngine,
    compare_loss_distributions,
    domain_coverage,
    var_es,
    var_es_uncertainty,
)
from stngpr.risk_grids import (
    BoundedStandardizedRiskTransform,
    StandardizedRiskTransform,
    build_greek_coordinate_grid,
    gamma_monitor_axis,
)
from stngpr.scenarios import GBMScenarioGenerator
from stngpr.tt_surrogate import TTPriceSurrogate
from stngpr.validation import (
    american_put_binomial,
    black_scholes_put,
    error_metrics,
    scalar_summary,
    stratified_american_points,
)
from stngpr.vol_extension import (
    BasketScaledModelPricer,
    build_vol_extended_grid,
    resolve_cubic_columns,
    sample_market_points,
)


class GridTests(unittest.TestCase):
    def test_qtt_round_trip(self):
        config = PaperConfig()
        grid = QTTGrid(config.bounds, config.physical_shape)
        rng = np.random.default_rng(7)
        indices = grid.random_physical_indices(100, rng)
        recovered = grid.qtt_indices_to_physical(grid.physical_indices_to_qtt(indices))
        np.testing.assert_array_equal(indices, recovered)

    def test_multilinear_weights_sum_to_one(self):
        grid = QTTGrid(((0, 1), (0, 1)), (4, 8))
        corners, weights = grid.multilinear_stencil(np.array([[0.31, 0.77]]))
        self.assertEqual(corners.shape, (1, 4, 2))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_tt_truncation_can_be_reapplied_from_raw_cores(self):
        grid = QTTGrid(axes=(np.array([0.0, 1.0]), np.array([0.0, 1.0])))
        model = TTPriceSurrogate(grid, lambda points: np.sum(points, axis=1))
        model.untruncated_cores = [
            np.ones((1, 2, 1)),
            np.ones((1, 2, 1)),
        ]
        raw = model.apply_truncation(None)
        truncated = model.apply_truncation(1e-12)
        self.assertEqual(raw.parameter_count, 4)
        self.assertEqual(truncated.parameter_count, 4)
        self.assertIsNone(raw.truncation)
        self.assertEqual(truncated.truncation, 1e-12)
        self.assertEqual(raw.bond_ranks, [1])
        self.assertEqual(raw.core_shapes, [[1, 2, 1], [1, 2, 1]])
        with self.assertRaises(ValueError):
            model.apply_truncation(-1e-8)

    def test_nonuniform_grid_interpolates_linear_function_exactly(self):
        grid = QTTGrid(
            axes=(np.array([0.0, 0.1, 0.4, 1.0]), np.array([0.0, 0.2, 1.0, 3.0]))
        )
        points = np.array([[0.25, 0.7], [0.9, 2.4]])
        pricer = lambda x: 2.0 + 3.0 * x[:, 0] - 0.5 * x[:, 1]
        predicted = oracle_multilinear_predict(grid, pricer, points)
        np.testing.assert_allclose(predicted, pricer(points), atol=1e-14)

    def test_hybrid_cubic_interpolates_cubic_linear_function_exactly(self):
        grid = QTTGrid(
            axes=(
                np.array([-1.0, -0.7, -0.2, 0.1, 0.4, 0.8, 1.3, 2.0]),
                np.array([0.0, 0.3, 1.1, 3.0]),
            )
        )
        points = np.array(
            [
                [-0.45, 0.7],
                [0.25, 1.8],
                [1.05, 2.4],
            ]
        )

        def pricer(x):
            return 1.0 + x[:, 0] ** 3 + 0.5 * x[:, 0] ** 2 * x[:, 1] - 2.0 * x[:, 1]

        predicted = oracle_hybrid_cubic_predict(
            grid,
            pricer,
            points,
            cubic_columns=(0,),
        )
        np.testing.assert_allclose(predicted, pricer(points), atol=2e-14)

    def test_fixed_hybrid_greeks_derive_one_cubic_surface(self):
        grid = QTTGrid(
            axes=(
                np.linspace(0.5, 2.0, 8),
                np.linspace(-1.0, 1.0, 8),
            )
        )

        class IdentityTransform:
            n_assets = 1

            @staticmethod
            def to_model(points):
                return np.asarray(points, dtype=float)

        def oracle(points, cubic_columns=()):
            return oracle_hybrid_cubic_predict(
                grid,
                lambda values: values[:, 0] ** 3 + values[:, 1],
                points,
                cubic_columns=cubic_columns,
            )

        point = np.array([[1.1, 0.2]])
        result = finite_difference_fixed_hybrid_component_arrays(
            oracle,
            IdentityTransform(),
            point,
            risk_columns=(0,),
            relative_bump=1e-3,
            cubic_columns=(0,),
        )
        np.testing.assert_allclose(result["delta"], [[3.0 * 1.1**2]], atol=2e-6)
        np.testing.assert_allclose(result["gamma_diagonal"], [[6.0 * 1.1]], atol=1e-8)

    def test_raw_tt_core_archive_round_trip(self):
        grid = QTTGrid(axes=(np.linspace(0.0, 1.0, 4),))
        model = TTPriceSurrogate(grid, lambda points: points[:, 0])
        model.untruncated_cores = [
            np.arange(4.0).reshape(1, 2, 2),
            np.arange(4.0, 8.0).reshape(2, 2, 1),
        ]
        with TemporaryDirectory() as directory:
            path = model.save_untruncated_cores(Path(directory) / "raw_cores.npz")
            restored = TTPriceSurrogate(grid, lambda points: points[:, 0])
            diagnostics = restored.load_untruncated_cores(path)
        self.assertEqual(diagnostics.parameter_count, 8)
        for expected, actual in zip(
            model.untruncated_cores,
            restored.untruncated_cores,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_cached_grid_interpolator_reuses_exact_node_values(self):
        grid = QTTGrid(
            axes=(
                np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
                np.array([0.0, 0.25, 0.5, 1.0]),
            )
        )
        calls = {"count": 0}

        def oracle(x):
            calls["count"] += len(x)
            return 1.0 + x[:, 0] ** 3 - 0.5 * x[:, 1]

        cached = CachedGridInterpolator(grid, oracle, evaluation_batch_size=3)
        points = np.array([[0.2, 0.4], [0.7, 0.8]])
        first = cached(points, cubic_columns=(0,))
        first_count = calls["count"]
        second = cached(points, cubic_columns=(0,))
        np.testing.assert_allclose(first, oracle(points), atol=2e-14)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(calls["count"], first_count + len(points))
        self.assertEqual(cached.function_evaluations, first_count)
        self.assertGreater(cached.cache_hit_count, 0)

    def test_adaptive_grid_concentrates_moneyness_nodes_at_atm(self):
        config = PaperConfig()
        grid, _, description = build_coordinate_grid(
            config, "moneyness_adaptive", "geometric"
        )
        m_axis = grid.axes[config.n_assets]
        self.assertEqual(m_axis.size, 64)
        uniform_step = (m_axis[-1] - m_axis[0]) / (m_axis.size - 1)
        atm_step = np.diff(m_axis)[np.argmin(np.abs(m_axis[:-1]))]
        self.assertLess(atm_step, uniform_step / 2.0)
        self.assertEqual(description["mode"], "moneyness_adaptive")

    def test_sinh_grid_atm_cell_matches_analytical_ratio(self):
        lower, upper, n_nodes, concentration = -5.0, 3.7, 64, 3.0
        axis = sinh_centered_axis(lower, upper, n_nodes, concentration)
        uniform_width = (upper - lower) / (n_nodes - 1)
        atm_width = axis[n_nodes // 2] - axis[n_nodes // 2 - 1]
        expected_ratio = (
            (n_nodes - 1)
            * np.sinh(concentration / (n_nodes - 1))
            / np.sinh(concentration)
        )
        np.testing.assert_allclose(atm_width / uniform_width, expected_ratio)
        self.assertTrue(np.all(np.diff(axis[: n_nodes // 2]) > 0.0))
        self.assertTrue(np.all(np.diff(axis[n_nodes // 2 :]) > 0.0))

    def test_grid_ablation_changes_one_axis_at_a_time(self):
        config = PaperConfig()
        paper, _, _ = build_coordinate_grid(config, "paper", "arithmetic")
        paper_t, _, _ = build_coordinate_grid(
            config, "paper_adaptive_maturity", "arithmetic"
        )
        money_t, _, _ = build_coordinate_grid(
            config, "moneyness_adaptive", "arithmetic"
        )
        money_u, _, _ = build_coordinate_grid(
            config, "moneyness_adaptive_uniform_maturity", "arithmetic"
        )
        np.testing.assert_allclose(
            paper.axes[config.n_assets], paper_t.axes[config.n_assets]
        )
        np.testing.assert_allclose(paper_t.axes[-1], money_t.axes[-1])
        np.testing.assert_allclose(paper.axes[-1], money_u.axes[-1])
        self.assertFalse(np.allclose(paper.axes[-1], paper_t.axes[-1]))

    def test_moneyness_and_maturity_resolutions_can_vary_independently(self):
        base = PaperConfig()
        shape = list(base.physical_shape)
        shape[base.n_assets] = 128
        shape[-1] = 32
        config = replace(base, physical_shape=tuple(shape))
        grid, _, _ = build_coordinate_grid(
            config,
            "moneyness_adaptive",
            "geometric",
        )
        self.assertEqual(grid.shape[base.n_assets], 128)
        self.assertEqual(grid.shape[-1], 32)
        self.assertEqual(grid.shape[-2], base.physical_shape[-2])

    def test_standardized_risk_coordinate_round_trip(self):
        config = PaperConfig()
        transform = StandardizedRiskTransform(
            config.n_assets,
            config.volatilities,
            config.correlation,
            config.dividends,
        )
        market = np.array(
            [
                [80.0, 90.0, 100.0, 110.0, 120.0, 105.0, 0.03, 14.0 / 365.0],
                [50.0, 65.0, 80.0, 95.0, 110.0, 75.0, 0.06, 2.0],
            ]
        )
        recovered = transform.to_market(transform.to_model(market))
        np.testing.assert_allclose(recovered, market, rtol=2e-14, atol=2e-14)

    def test_bounded_standardized_risk_is_tensor_compatible(self):
        config = PaperConfig()
        m_bounds = (
            np.log(config.strike_bounds[0] / config.spot_bounds[1]),
            np.log(config.strike_bounds[1] / config.spot_bounds[0]),
        )
        transform = BoundedStandardizedRiskTransform(
            config.n_assets,
            config.volatilities,
            config.correlation,
            config.dividends,
            moneyness_bounds=m_bounds,
        )
        market = np.array(
            [
                [80.0, 90.0, 100.0, 110.0, 120.0, 105.0, 0.03, 14.0 / 365.0],
                [50.0, 65.0, 80.0, 95.0, 110.0, 75.0, 0.06, 2.0],
            ]
        )
        model = transform.to_model(market)
        self.assertTrue(np.all(np.abs(model[:, config.n_assets]) <= 1.0))
        np.testing.assert_allclose(
            transform.to_market(model),
            market,
            rtol=2e-14,
            atol=2e-14,
        )

        grid, grid_transform, _ = build_greek_coordinate_grid(
            config,
            "bounded_standardized_risk",
        )
        indices = grid.random_physical_indices(256, np.random.default_rng(9))
        tensor_market = grid_transform.to_market(grid.indices_to_points(indices))
        basket = np.exp(np.mean(np.log(tensor_market[:, : config.n_assets]), axis=1))
        log_moneyness = np.log(tensor_market[:, config.n_assets] / basket)
        self.assertGreaterEqual(np.min(log_moneyness), m_bounds[0] - 1e-13)
        self.assertLessEqual(np.max(log_moneyness), m_bounds[1] + 1e-13)

    def test_gamma_monitor_axis_is_strict_and_preserves_bounds(self):
        config = PaperConfig()
        basket_sigma, basket_carry = geometric_basket_effective_parameters(
            0.03,
            config.volatilities,
            config.correlation,
            config.dividends,
        )
        axis, description = gamma_monitor_axis(
            -1.0,
            0.8,
            64,
            np.array([7.0, 30.0, 365.0]) / 365.0,
            0.03,
            basket_sigma,
            basket_carry,
            dense_nodes=1025,
        )
        self.assertEqual(axis.size, 64)
        self.assertEqual(axis[0], -1.0)
        self.assertEqual(axis[-1], 0.8)
        self.assertTrue(np.all(np.diff(axis) > 0.0))
        self.assertIn("d4u/dm4", description["definition"])

    def test_v9_grid_candidates_have_identical_tensor_shapes(self):
        config = PaperConfig()
        shapes = []
        for mode in (
            "m_uniform",
            "price_adaptive",
            "gamma_monitor",
            "standardized_risk",
            "bounded_standardized_risk",
        ):
            grid, _, description = build_greek_coordinate_grid(config, mode)
            shapes.append(grid.shape)
            self.assertEqual(description["mode"], mode)
        self.assertTrue(all(shape == shapes[0] for shape in shapes))

    def test_standardized_coordinate_axis_is_invariant_to_spot_resolution(self):
        base = PaperConfig()
        axes = []
        for n_spot in (16, 32, 64):
            shape = list(base.physical_shape)
            shape[: base.n_assets] = [n_spot] * base.n_assets
            config = replace(base, physical_shape=tuple(shape))
            grid, _, _ = build_greek_coordinate_grid(
                config,
                "standardized_risk",
            )
            axes.append(grid.axes[base.n_assets])
        np.testing.assert_array_equal(axes[0], axes[1])
        np.testing.assert_array_equal(axes[1], axes[2])


class PricingTests(unittest.TestCase):
    def test_one_asset_reduces_to_black_scholes_put(self):
        price = geometric_basket_put(
            [[100.0]], [100.0], [0.05], [1.0], np.array([0.2]), np.eye(1)
        )[0]
        self.assertAlmostEqual(price, 5.573526, places=5)

    def test_put_nonnegative(self):
        config = PaperConfig()
        prices = geometric_basket_put(
            [[50] * 5, [100] * 5],
            [80, 80],
            [0.03, 0.03],
            [1, 1],
            config.volatilities,
            config.correlation,
        )
        self.assertTrue(np.all(prices >= 0.0))

    def test_arithmetic_qmc_is_reproducible(self):
        config = PaperConfig()
        parameters = np.array([[80.0, 90.0, 100.0, 110.0, 120.0, 100.0, 0.03, 0.5]])
        first = EuropeanArithmeticBasketQMC(config, n_paths=512, seed=17)(parameters)
        second = EuropeanArithmeticBasketQMC(config, n_paths=512, seed=17)(parameters)
        np.testing.assert_array_equal(first, second)

    def test_arithmetic_qmc_respects_positive_homogeneity(self):
        config = PaperConfig()
        parameters = np.array([[70.0, 85.0, 100.0, 115.0, 130.0, 105.0, 0.03, 0.8]])
        pricer = EuropeanArithmeticBasketQMC(config, n_paths=1_024, seed=19)
        scaled = parameters.copy()
        scaled[:, : config.n_assets + 1] *= 3.7
        np.testing.assert_allclose(
            pricer(scaled),
            3.7 * pricer(parameters),
            rtol=2e-13,
            atol=2e-13,
        )

    def test_arithmetic_qmc_reduces_to_geometric_in_one_dimension(self):
        config = PaperConfig(
            n_assets=1,
            volatilities=np.array([0.2]),
            dividends=np.array([0.0]),
            correlation=np.eye(1),
            physical_shape=(32, 64, 8, 8),
        )
        parameters = np.array([[100.0, 105.0, 0.03, 0.75]])
        qmc_price = EuropeanArithmeticBasketQMC(config, n_paths=512, seed=23)(
            parameters
        )[0]
        exact = geometric_basket_put(
            [[100.0]], [105.0], [0.03], [0.75], np.array([0.2]), np.eye(1)
        )[0]
        self.assertAlmostEqual(qmc_price, exact, places=11)

    def test_arithmetic_qmc_unit_control_variate_is_exact_in_one_dimension(self):
        config = PaperConfig(
            n_assets=1,
            volatilities=np.array([0.2]),
            dividends=np.array([0.0]),
            correlation=np.eye(1),
            physical_shape=(32, 64, 8, 8),
        )
        parameters = np.array([[100.0, 105.0, 0.03, 0.75]])
        qmc_price = EuropeanArithmeticBasketQMC(
            config,
            n_paths=512,
            seed=23,
            control_variate_beta="unit",
        )(parameters)[0]
        exact = geometric_basket_put(
            [[100.0]], [105.0], [0.03], [0.75], np.array([0.2]), np.eye(1)
        )[0]
        self.assertAlmostEqual(qmc_price, exact, places=11)

    def test_arithmetic_qmc_rejects_invalid_control_beta(self):
        with self.assertRaisesRegex(ValueError, "control_variate_beta"):
            EuropeanArithmeticBasketQMC(
                PaperConfig(),
                n_paths=512,
                control_variate_beta="local",
            )

    def test_american_frozen_policy_is_reproducible_and_homogeneous(self):
        config = PaperConfig()
        base = np.array([80.0, 90.0, 100.0, 110.0, 120.0, 100.0, 0.03, 30 / 365])
        bumped = base.copy()
        bumped[0] *= 1.01
        scenarios = np.vstack((base, bumped))
        pricer = AmericanArithmeticBasketLSMC(
            config,
            n_paths=128,
            n_steps=4,
            seed=31,
            policy_mode="frozen",
        )
        values = pricer(scenarios)
        np.testing.assert_array_equal(values, pricer(scenarios))
        scaled = scenarios.copy()
        scaled[:, : config.n_assets + 1] *= 2.5
        np.testing.assert_allclose(pricer(scaled), 2.5 * values, rtol=1e-12, atol=1e-12)

    def test_american_frozen_policy_requires_one_fixed_contract(self):
        config = PaperConfig()
        scenarios = np.array(
            [
                [80.0, 90.0, 100.0, 110.0, 120.0, 100.0, 0.03, 30 / 365],
                [80.0, 90.0, 100.0, 110.0, 120.0, 101.0, 0.03, 30 / 365],
            ]
        )
        pricer = AmericanArithmeticBasketLSMC(
            config,
            n_paths=64,
            n_steps=4,
            seed=37,
            policy_mode="frozen",
        )
        with self.assertRaisesRegex(ValueError, "strike, rate and maturity"):
            pricer(scenarios)

    def test_american_frozen_out_of_sample_greeks_are_finite(self):
        config = PaperConfig()
        point = np.array([80.0, 90.0, 100.0, 110.0, 120.0, 100.0, 0.03, 30 / 365])
        pricer = AmericanArithmeticBasketLSMC(
            config,
            n_paths=128,
            n_steps=4,
            seed=41,
            policy_mode="frozen_oos",
        )
        components = finite_difference_component_arrays(
            pricer,
            point[None, :],
            risk_columns=(0, 1),
            relative_bump=0.01,
        )
        for name in ("price", "delta", "gamma_diagonal", "cross_gamma"):
            self.assertTrue(np.all(np.isfinite(components[name])))

    def test_geometric_convexity_is_positive_and_peaks_on_ridge(self):
        config = PaperConfig()
        rate = 0.03
        basket_sigma, basket_carry = geometric_basket_effective_parameters(
            rate,
            config.volatilities,
            config.correlation,
            config.dividends,
        )
        maturity = 0.5
        expected_peak = float(
            geometric_basket_convexity_ridge(maturity, basket_sigma, basket_carry)
        )
        m = np.linspace(expected_peak - 0.5, expected_peak + 0.5, 20_001)
        chi = geometric_basket_log_moneyness_convexity(
            m, maturity, rate, basket_sigma, basket_carry
        )
        self.assertTrue(np.all(chi >= 0.0))
        self.assertAlmostEqual(float(m[np.argmax(chi)]), expected_peak, places=4)

    def test_moneyness_coordinate_round_trip(self):
        transform = CoordinateTransform(5, "geometric", True)
        market = np.array(
            [
                [80.0, 90.0, 100.0, 110.0, 120.0, 105.0, 0.03, 1.2],
                [20.0, 30.0, 40.0, 50.0, 60.0, 35.0, 0.01, 0.2],
            ]
        )
        np.testing.assert_allclose(
            transform.to_market(transform.to_model(market)), market, rtol=1e-14
        )

    def test_american_binomial_respects_basic_bounds(self):
        value = american_put_binomial(100.0, 100.0, 0.03, 0.2, 1.0, steps=500)
        european = black_scholes_put(100.0, 100.0, 0.03, 0.2, 1.0)
        self.assertGreaterEqual(value, european - 1e-10)
        self.assertGreaterEqual(value, 0.0)

    def test_stratified_american_design_is_balanced_and_in_bounds(self):
        config = PaperConfig()
        points, labels = stratified_american_points(
            config, 3, np.random.default_rng(11)
        )
        self.assertEqual(points.shape, (15, config.n_assets + 3))
        counts = {
            name: sum(label[0] == name for label in labels)
            for name in {label[0] for label in labels}
        }
        self.assertTrue(all(count == 3 for count in counts.values()))
        for column, (lower, upper) in enumerate(config.bounds):
            self.assertTrue(np.all(points[:, column] >= lower))
            self.assertTrue(np.all(points[:, column] <= upper))

    def test_error_metrics_zero_error(self):
        metrics = error_metrics([1.0, 2.0], [1.0, 2.0])
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)

    def test_scalar_summary(self):
        summary = scalar_summary([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 3.0)


class RiskTests(unittest.TestCase):
    def test_var_es(self):
        var, es = var_es(np.arange(100.0), 0.95)
        self.assertEqual(var, 95.0)
        self.assertGreaterEqual(es, var)

    def test_var_es_uncertainty_brackets_the_point_estimate(self):
        rng = np.random.default_rng(3)
        losses = rng.standard_normal(20_000)
        summary = var_es_uncertainty(losses, 0.99)
        self.assertLessEqual(summary["var_ci"][0], summary["var"])
        self.assertLessEqual(summary["var"], summary["var_ci"][1])
        self.assertAlmostEqual(summary["var"], 2.3263, delta=0.06)
        self.assertEqual(summary["tail_scenarios"], 200)

    def test_portfolio_parameters_age_maturities_and_keep_strikes_fixed(self):
        portfolio = Portfolio.from_arrays(
            strikes=[90.0, 110.0], maturities=[1.0, 2.0], quantities=[1.0, -2.0]
        )
        spots = np.array([[100.0, 105.0], [80.0, 95.0]])
        parameters = portfolio.market_parameters(spots, 0.02, maturity_shift=0.25)
        self.assertEqual(parameters.shape, (2, 2, 5))
        np.testing.assert_allclose(
            parameters[:, :, :2], np.broadcast_to(spots[:, None, :], (2, 2, 2))
        )
        np.testing.assert_allclose(parameters[:, :, 2], [[90.0, 110.0]] * 2)
        np.testing.assert_allclose(parameters[:, :, 3], 0.02)
        np.testing.assert_allclose(parameters[:, :, 4], [[0.75, 1.75]] * 2)

    def test_portfolio_rejects_trades_expiring_inside_the_horizon(self):
        portfolio = Portfolio.from_arrays(strikes=[100.0], maturities=[0.05])
        with self.assertRaises(ValueError):
            portfolio.market_parameters(np.array([[100.0]]), 0.01, maturity_shift=0.05)

    def test_split_at_horizon_separates_expiring_positions(self):
        portfolio = Portfolio.from_arrays(
            strikes=[90.0, 100.0, 110.0],
            maturities=[1.0 / 365.0, 1.0 / 52.0, 1.0],
            quantities=[10.0, -20.0, 30.0],
            labels=["one_day", "one_week", "one_year"],
        )
        live, expiring = portfolio.split_at_horizon(1.0 / 252.0)
        self.assertEqual([trade.label for trade in live.trades], ["one_week", "one_year"])
        self.assertEqual([trade.label for trade in expiring], ["one_day"])
        np.testing.assert_allclose(live.quantities, [-20.0, 30.0])
        # The survivors must now price without tripping the maturity guard.
        live.market_parameters(
            np.array([[100.0]]), 0.01, maturity_shift=1.0 / 252.0
        )

    def test_split_at_horizon_keeps_the_whole_book_when_nothing_expires(self):
        portfolio = Portfolio.from_arrays(strikes=[100.0, 110.0], maturities=[1.0, 2.0])
        live, expiring = portfolio.split_at_horizon(1.0 / 252.0)
        self.assertEqual(live.trades, portfolio.trades)
        self.assertEqual(expiring, ())

    def test_split_at_horizon_refuses_an_entirely_expired_book(self):
        portfolio = Portfolio.from_arrays(strikes=[100.0], maturities=[1.0 / 365.0])
        with self.assertRaisesRegex(ValueError, "expires within"):
            portfolio.split_at_horizon(10.0 / 252.0)

    def test_gbm_scenarios_match_analytical_moments_and_correlation(self):
        config = PaperConfig()
        generator = GBMScenarioGenerator.from_config(config)
        spots = np.full(config.n_assets, 100.0)
        rate, horizon = 0.03, 0.5
        scenarios = generator.simulate(
            spots, horizon, 200_000, rate=rate, rng=np.random.default_rng(11)
        )
        forward = 100.0 * math.exp(rate * horizon)
        np.testing.assert_allclose(scenarios.mean(axis=0), forward, rtol=2e-3)

        log_returns = np.log(scenarios / spots[None, :])
        np.testing.assert_allclose(
            log_returns.std(axis=0, ddof=1),
            config.volatilities * math.sqrt(horizon),
            rtol=2e-2,
        )
        np.testing.assert_allclose(
            np.corrcoef(log_returns, rowvar=False), config.correlation, atol=1e-2
        )

    def test_antithetic_scenarios_pair_log_returns_exactly(self):
        config = PaperConfig()
        generator = GBMScenarioGenerator.from_config(config, drift=np.zeros(5))
        spots = np.full(config.n_assets, 100.0)
        scenarios = generator.simulate(
            spots, 0.1, 8, rate=0.0, rng=np.random.default_rng(5), antithetic=True
        )
        log_returns = np.log(scenarios / spots[None, :])
        np.testing.assert_allclose(log_returns[:4], -log_returns[4:], atol=1e-13)

    def test_full_revaluation_losses_are_exact_for_a_linear_pricer(self):
        config = PaperConfig()
        portfolio = Portfolio.from_arrays(
            strikes=[100.0, 120.0], maturities=[1.0, 2.0], quantities=[2.0, -1.0]
        )
        engine = MonteCarloVaREngine(
            portfolio,
            GBMScenarioGenerator.from_config(config),
            horizon=10.0 / 252.0,
            levels=(0.95, 0.99),
        )
        spots = np.full(config.n_assets, 100.0)

        def linear_pricer(parameters):
            # Value is the basket level, so P&L must not depend on strike or aging.
            return parameters[:, : config.n_assets].mean(axis=1)

        result = engine.run(
            linear_pricer, spots, 0.03, n_scenarios=2_000, seed=config.seed
        )
        scenario_baskets = engine.simulate_spots(
            spots, 2_000, 0.03, seed=config.seed
        ).mean(axis=1)
        expected_pnl = (scenario_baskets - 100.0) * portfolio.quantities.sum()
        np.testing.assert_allclose(result.pnl, expected_pnl, atol=1e-10)
        self.assertAlmostEqual(result.base_value, 100.0 * portfolio.quantities.sum())
        self.assertEqual(result.pricer_evaluations, 2 * (1 + 2_000))

    def test_expected_shortfall_contributions_add_up_to_portfolio_es(self):
        config = PaperConfig()
        portfolio = Portfolio.from_arrays(
            strikes=[95.0, 105.0, 115.0],
            maturities=[0.5, 1.0, 2.0],
            quantities=[100.0, -50.0, 25.0],
        )
        engine = MonteCarloVaREngine(
            portfolio,
            GBMScenarioGenerator.from_config(config),
            horizon=10.0 / 252.0,
            levels=(0.99,),
        )
        spots = np.array([95.0, 100.0, 105.0, 90.0, 110.0])
        result = engine.run(
            EuropeanGeometricBasketPricer(config),
            spots,
            0.03,
            n_scenarios=4_000,
            seed=config.seed,
        )
        self.assertAlmostEqual(
            float(result.contributions(0.99).sum()),
            result.expected_shortfall(0.99),
            places=9,
        )
        self.assertGreater(result.var(0.99), 0.0)
        self.assertGreaterEqual(result.expected_shortfall(0.99), result.var(0.99))

    def test_supplied_scenarios_make_two_pricers_a_paired_experiment(self):
        config = PaperConfig()
        portfolio = Portfolio.from_arrays(strikes=[100.0], maturities=[1.0])
        engine = MonteCarloVaREngine(
            portfolio,
            GBMScenarioGenerator.from_config(config),
            horizon=10.0 / 252.0,
            levels=(0.99,),
        )
        spots = np.full(config.n_assets, 100.0)
        scenario_spots = engine.simulate_spots(spots, 1_000, 0.03, seed=config.seed)
        exact = EuropeanGeometricBasketPricer(config)
        biased = lambda parameters: exact(parameters) * 1.01

        reference = engine.run(
            exact, spots, 0.03, scenario_spots=scenario_spots
        )
        surrogate = engine.run(
            biased, spots, 0.03, scenario_spots=scenario_spots
        )
        comparison = compare_loss_distributions(
            reference.losses, surrogate.losses, levels=(0.99,)
        )
        self.assertAlmostEqual(comparison["spearman_rank_correlation"], 1.0, places=12)
        self.assertEqual(comparison["worst_scenario_overlap"], 1.0)
        np.testing.assert_allclose(
            surrogate.losses, 1.01 * reference.losses, atol=1e-10
        )

    def test_identical_loss_vectors_report_no_error(self):
        rng = np.random.default_rng(1)
        losses = rng.standard_normal(500)
        comparison = compare_loss_distributions(losses, losses, levels=(0.95, 0.99))
        self.assertEqual(comparison["loss_mae"], 0.0)
        self.assertEqual(comparison["wasserstein_distance"], 0.0)
        self.assertEqual(comparison["worst_scenario_overlap"], 1.0)
        for summary in comparison["measures"].values():
            self.assertEqual(summary["var_absolute_error"], 0.0)
            self.assertEqual(summary["expected_shortfall_absolute_error"], 0.0)

    def test_domain_coverage_counts_points_outside_the_grid_box(self):
        points = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, -0.2], [2.0, 3.0]])
        coverage = domain_coverage(points, ((0.0, 1.0), (0.0, 1.0)))
        self.assertEqual(coverage["points"], 4)
        self.assertAlmostEqual(coverage["fraction_outside_any_axis"], 0.75)
        np.testing.assert_allclose(coverage["fraction_outside_by_axis"], [0.5, 0.5])

    def test_var_engine_on_the_grid_oracle_tracks_the_exact_loss_distribution(self):
        config = PaperConfig()
        grid, transform, _ = build_coordinate_grid(
            config, "moneyness_adaptive", "geometric"
        )
        market_pricer = EuropeanGeometricBasketPricer(config)
        model_pricer = TransformedPricer(market_pricer, transform)
        portfolio = Portfolio.from_arrays(
            strikes=[90.0, 100.0, 110.0],
            maturities=[0.5, 1.0, 2.0],
            quantities=[100.0, -75.0, 50.0],
        )
        engine = MonteCarloVaREngine(
            portfolio,
            GBMScenarioGenerator.from_config(config),
            horizon=10.0 / 252.0,
            levels=(0.99,),
        )
        spots = np.array([95.0, 100.0, 105.0, 90.0, 110.0])
        scenario_spots = engine.simulate_spots(spots, 2_000, 0.03, seed=config.seed)

        oracle = MarketCoordinatePricer(
            lambda points: oracle_multilinear_predict(grid, model_pricer, points),
            transform,
        )
        exact_result = engine.run(
            market_pricer, spots, 0.03, scenario_spots=scenario_spots
        )
        oracle_result = engine.run(
            oracle, spots, 0.03, scenario_spots=scenario_spots
        )
        comparison = compare_loss_distributions(
            exact_result.losses, oracle_result.losses, levels=(0.99,)
        )
        # Scenario ranking survives interpolation; the VaR level itself carries
        # the grid interpolation floor, which validate_portfolio_var.py measures.
        self.assertGreater(comparison["spearman_rank_correlation"], 0.99)
        self.assertEqual(comparison["worst_scenario_overlap"], 1.0)
        self.assertLess(comparison["measures"]["0.99"]["var_relative_error"], 0.15)


class GreekTests(unittest.TestCase):
    def test_curve_error_diagnostics_detects_shape_oscillation(self):
        moneyness = np.linspace(-1.0, 1.0, 5)
        reference = moneyness**2
        estimate = reference + np.array([0.0, -0.2, 0.2, -0.2, 0.0])
        diagnostics = curve_error_diagnostics(
            moneyness,
            reference,
            estimate,
        )
        self.assertEqual(diagnostics["point_count"], 5)
        self.assertEqual(diagnostics["moneyness_at_maximum_error"], -0.5)
        self.assertGreater(
            diagnostics["shape_by_component_column"][0]["estimate_turning_point_count"],
            diagnostics["shape_by_component_column"][0][
                "reference_turning_point_count"
            ],
        )

    def test_full_hessian_diagnostics_separate_sign_and_psd(self):
        reference = {
            "gamma_diagonal": np.array([[2.0, 1.0], [1.5, 0.8]]),
            "cross_gamma": np.array([[-0.5], [0.3]]),
        }
        estimate = {
            "gamma_diagonal": np.array([[2.01, 0.99], [1.49, 0.81]]),
            "cross_gamma": np.array([[-0.49], [0.29]]),
        }
        fidelity = full_hessian_error_metrics(reference, estimate)
        signs = cross_gamma_sign_diagnostics(
            reference["cross_gamma"],
            estimate["cross_gamma"],
            labels=["Gamma_12"],
        )
        self.assertGreater(fidelity["aggregate_relative_frobenius_error"], 0.0)
        self.assertEqual(fidelity["material_psd_violation_count"], 0)
        self.assertEqual(signs["sign_agreement_fraction"], 1.0)
        self.assertEqual(signs["by_component"][0]["reference_negative_count"], 1)

    def test_columnwise_error_metrics_preserve_labels(self):
        reference = np.array([[1.0, 2.0], [2.0, 4.0]])
        estimate = reference + np.array([[0.1, -0.2], [0.1, -0.2]])
        metrics = columnwise_error_metrics(
            reference,
            estimate,
            labels=["S1", "S2"],
        )
        self.assertEqual([record["label"] for record in metrics], ["S1", "S2"])
        self.assertAlmostEqual(metrics[0]["mae"], 0.1)
        self.assertAlmostEqual(metrics[1]["mae"], 0.2)

    def test_component_error_decomposition_closes_exactly(self):
        analytical = {
            "price": np.array([10.0, 12.0]),
            "delta": np.array([[-0.4, -0.2], [-0.3, -0.1]]),
            "gamma_diagonal": np.array([[0.02, 0.03], [0.04, 0.05]]),
            "cross_gamma": np.array([[0.01], [0.02]]),
        }
        exact_fd = {
            name: np.asarray(values) + 0.1 for name, values in analytical.items()
        }
        grid_fd = {
            name: np.asarray(values) - 0.025 for name, values in exact_fd.items()
        }
        decomposition = component_error_decomposition(
            analytical,
            exact_fd,
            grid_fd,
        )
        for name in ("price", "delta", "gamma_diagonal", "cross_gamma"):
            np.testing.assert_allclose(
                decomposition["total"]["residuals"][name],
                decomposition["finite_difference"]["residuals"][name]
                + decomposition["grid_interpolation"]["residuals"][name],
                atol=1e-15,
            )
            np.testing.assert_allclose(
                decomposition["closure"]["residuals"][name],
                0.0,
                atol=1e-15,
            )

    def test_tt_component_error_decomposition_closes_exactly(self):
        analytical = {
            "price": np.array([10.0]),
            "delta": np.array([[-0.4, -0.2]]),
            "gamma_diagonal": np.array([[0.02, 0.03]]),
            "cross_gamma": np.array([[0.01]]),
        }
        exact_fd = {
            name: np.asarray(values) + 0.01 for name, values in analytical.items()
        }
        grid_fd = {name: np.asarray(values) - 0.02 for name, values in exact_fd.items()}
        tt_fd = {name: np.asarray(values) + 0.03 for name, values in grid_fd.items()}
        decomposition = tt_component_error_decomposition(
            analytical,
            exact_fd,
            grid_fd,
            tt_fd,
        )
        for name in ("price", "delta", "gamma_diagonal", "cross_gamma"):
            np.testing.assert_allclose(
                decomposition["closure"]["residuals"][name],
                0.0,
                atol=1e-15,
            )

    def test_residual_layers_share_the_analytical_scale(self):
        reference = {
            "price": np.array([2.0]),
            "delta": np.array([[4.0, -4.0]]),
            "gamma_diagonal": np.array([[0.5, 1.5]]),
            "cross_gamma": np.array([[0.25]]),
        }
        residual = {
            name: 0.1 * np.asarray(values) for name, values in reference.items()
        }
        metrics = component_residual_metrics(reference, residual)
        for name in ("price", "delta", "gamma_diagonal", "cross_gamma"):
            self.assertAlmostEqual(metrics[name]["normalized_mae"], 0.1)

    def test_convergence_helpers_preserve_hessian_layout(self):
        first = {
            "price": np.array([1.0]),
            "delta": np.array([[2.0, 3.0]]),
            "gamma_diagonal": np.array([[4.0, 5.0]]),
            "cross_gamma": np.array([[6.0]]),
        }
        second = {name: np.asarray(value) + 2.0 for name, value in first.items()}
        mean, standard_error = mean_and_standard_error([first, second])
        np.testing.assert_allclose(mean["gamma_diagonal"], [[5.0, 6.0]])
        np.testing.assert_allclose(standard_error["price"], [1.0])
        np.testing.assert_allclose(
            gamma_matrices(mean),
            [[[5.0, 7.0], [7.0, 6.0]]],
        )

    def test_psd_projection_enforces_convexity_and_reduces_frobenius_error(self):
        raw = np.array([[1.0, 2.0], [2.0, 1.0]])
        reference = np.eye(2)
        projected = project_symmetric_matrix_psd(raw)
        self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(projected))), -1e-14)
        self.assertLessEqual(
            np.linalg.norm(projected - reference),
            np.linalg.norm(raw - reference),
        )

    def test_psd_projection_validates_inputs(self):
        with self.assertRaises(ValueError):
            project_symmetric_matrix_psd(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            project_symmetric_matrix_psd(np.eye(2), eigenvalue_floor=-1.0)

    def test_market_coordinate_greeks_hold_strike_fixed(self):
        transform = CoordinateTransform(2, "arithmetic", True)

        def market_pricer(parameters):
            x = np.atleast_2d(np.asarray(parameters, dtype=float))
            s1, s2, strike = x[:, 0], x[:, 1], x[:, 2]
            return s1**2 + 3.0 * s1 * s2 + 0.5 * s2**2 + strike * s1

        model_pricer = TransformedPricer(market_pricer, transform)
        market_adapter = MarketCoordinatePricer(model_pricer, transform)
        x = np.array([80.0, 120.0, 105.0, 0.03, 1.0])
        result = finite_difference_greeks(
            market_adapter,
            x,
            spot_columns=(0, 1),
            relative_bump=1e-3,
        )

        expected_delta = {
            0: 2.0 * x[0] + 3.0 * x[1] + x[2],
            1: 3.0 * x[0] + x[1],
        }
        expected_gamma = {(0, 0): 2.0, (1, 1): 1.0, (0, 1): 3.0}
        for column, expected in expected_delta.items():
            self.assertAlmostEqual(result["delta"][column], expected, places=8)
        for columns, expected in expected_gamma.items():
            self.assertAlmostEqual(result["gamma"][columns], expected, places=8)
        self.assertAlmostEqual(result["gamma"][(1, 0)], 3.0, places=8)

    def test_hybrid_greeks_hold_strike_fixed(self):
        transform = CoordinateTransform(2, "arithmetic", True)

        def market_pricer(parameters):
            x = np.atleast_2d(np.asarray(parameters, dtype=float))
            s1, s2, strike = x[:, 0], x[:, 1], x[:, 2]
            return s1**2 + 3.0 * s1 * s2 + 0.5 * s2**2 + strike * s1

        model_pricer = TransformedPricer(market_pricer, transform)

        def exact_interpolator(model_parameters, cubic_columns):
            return model_pricer(model_parameters)

        x = np.array([80.0, 120.0, 105.0, 0.03, 1.0])
        result = finite_difference_hybrid_greeks(
            exact_interpolator,
            transform,
            x,
            spot_columns=(0, 1),
            relative_bump=1e-3,
        )
        self.assertAlmostEqual(
            result["delta"][0],
            2.0 * x[0] + 3.0 * x[1] + x[2],
            places=8,
        )
        self.assertAlmostEqual(
            result["delta"][1],
            3.0 * x[0] + x[1],
            places=8,
        )
        self.assertAlmostEqual(result["gamma"][(0, 0)], 2.0, places=8)
        self.assertAlmostEqual(result["gamma"][(1, 1)], 1.0, places=8)
        self.assertAlmostEqual(result["gamma"][(0, 1)], 3.0, places=8)

    def test_geometric_basket_greeks_match_closed_form_derivatives(self):
        spots = np.array([80.0, 95.0, 110.0])
        strike, rate, maturity = 100.0, 0.03, 0.8
        volatilities = np.array([0.18, 0.22, 0.25])
        correlation = np.full((3, 3), 0.25)
        np.fill_diagonal(correlation, 1.0)
        weights = np.full(3, 1.0 / 3.0)

        def pricer(parameters):
            x = np.atleast_2d(np.asarray(parameters, dtype=float))
            return geometric_basket_put(
                x[:, :3],
                x[:, 3],
                x[:, 4],
                x[:, 5],
                volatilities,
                correlation,
            )

        x = np.concatenate((spots, [strike, rate, maturity]))
        result = finite_difference_greeks(
            pricer,
            x,
            spot_columns=(0, 1, 2),
            relative_bump=1e-3,
        )
        analytical_delta, analytical_gamma = geometric_basket_put_spot_greeks(
            spots[None, :],
            [strike],
            [rate],
            [maturity],
            volatilities,
            correlation,
        )

        covariance = np.outer(volatilities, volatilities) * correlation
        basket_variance = float(weights @ covariance @ weights)
        basket_sigma = np.sqrt(basket_variance)
        basket = float(np.exp(np.log(spots) @ weights))
        carry = rate - 0.5 * float(weights @ volatilities**2) + 0.5 * basket_variance
        std = basket_sigma * np.sqrt(maturity)
        d1 = (
            np.log(basket / strike) + (carry + 0.5 * basket_variance) * maturity
        ) / std
        normal_cdf_minus_d1 = 0.5 * (1.0 + math.erf(-d1 / np.sqrt(2.0)))
        normal_density_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
        discount_carry = np.exp((carry - rate) * maturity)
        basket_delta = -discount_carry * normal_cdf_minus_d1
        basket_gamma = discount_carry * normal_density_d1 / (basket * std)

        basket_first = weights * basket / spots
        expected_delta = basket_delta * basket_first
        np.testing.assert_allclose(
            analytical_delta[0],
            expected_delta,
            rtol=1e-13,
            atol=1e-14,
        )
        for i in range(3):
            np.testing.assert_allclose(
                result["delta"][i],
                expected_delta[i],
                rtol=1e-6,
                atol=1e-9,
            )
            basket_second = weights[i] * (weights[i] - 1.0) * basket / spots[i] ** 2
            expected = (
                basket_gamma * basket_first[i] ** 2 + basket_delta * basket_second
            )
            np.testing.assert_allclose(
                analytical_gamma[0, i, i],
                expected,
                rtol=1e-13,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                result["gamma"][(i, i)],
                expected,
                rtol=2e-5,
                atol=1e-8,
            )

        for i in range(3):
            for j in range(i + 1, 3):
                basket_mixed = weights[i] * weights[j] * basket / (spots[i] * spots[j])
                expected = (
                    basket_gamma * basket_first[i] * basket_first[j]
                    + basket_delta * basket_mixed
                )
                np.testing.assert_allclose(
                    analytical_gamma[0, i, j],
                    expected,
                    rtol=1e-13,
                    atol=1e-14,
                )
                np.testing.assert_allclose(
                    result["gamma"][(i, j)],
                    expected,
                    rtol=2e-5,
                    atol=1e-8,
                )

    def test_greek_input_validation(self):
        pricer = lambda x: np.sum(x, axis=1)
        with self.assertRaisesRegex(ValueError, "positive"):
            finite_difference_greeks(
                pricer,
                np.array([100.0, 90.0]),
                spot_columns=(0,),
                relative_bump=0.0,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            finite_difference_greeks(
                pricer,
                np.array([100.0, 90.0]),
                spot_columns=(0, 0),
            )


class VolatilityExtensionTests(unittest.TestCase):
    """Per-underlying volatility dimensions, interleaved and on a log axis."""

    def test_layout_interleaves_each_spot_with_its_own_volatility(self):
        config = VolatilityExtendedConfig()
        self.assertEqual(config.n_dimensions, 13)
        self.assertEqual(config.spot_columns, (0, 2, 4, 6, 8))
        self.assertEqual(config.volatility_columns, (1, 3, 5, 7, 9))
        self.assertEqual(
            config.physical_shape,
            (32, 16, 32, 16, 32, 16, 32, 16, 32, 16, 64, 8, 8),
        )
        self.assertEqual(len(config.qtt_shape), 57)

    def test_volatility_grid_is_uniform_in_log_sigma_over_the_stated_range(self):
        config = VolatilityExtendedConfig()
        grid, _, description = build_vol_extended_grid(config, "paper")
        axis = grid.axes[config.volatility_columns[0]]
        self.assertAlmostEqual(float(np.exp(axis[0])), 0.05)
        self.assertAlmostEqual(float(np.exp(axis[-1])), 0.80)
        np.testing.assert_allclose(np.diff(axis), np.diff(axis)[0])
        self.assertEqual(description["volatility_coordinate"], "log(sigma)")
        for column in config.volatility_columns:
            np.testing.assert_allclose(grid.axes[column], axis)

    def test_linear_volatility_axis_stays_in_sigma(self):
        config = VolatilityExtendedConfig()
        grid, transform, description = build_vol_extended_grid(
            config, "paper", volatility_scale="linear"
        )
        axis = grid.axes[config.volatility_columns[0]]
        self.assertAlmostEqual(float(axis[0]), 0.05)
        self.assertAlmostEqual(float(axis[-1]), 0.80)
        self.assertEqual(description["volatility_coordinate"], "sigma")
        rng = np.random.default_rng(3)
        x = sample_market_points(config, 8, rng)
        np.testing.assert_allclose(transform.to_model(x), x)

    def test_model_coordinates_round_trip(self):
        config = VolatilityExtendedConfig()
        rng = np.random.default_rng(11)
        for mode in ("paper", "moneyness_adaptive"):
            _, transform, _ = build_vol_extended_grid(config, mode)
            x = sample_market_points(config, 32, rng)
            np.testing.assert_allclose(
                transform.to_market(transform.to_model(x)), x, atol=1e-10
            )

    def test_extended_pricer_matches_the_fixed_volatility_pricer(self):
        paper = PaperConfig()
        config = VolatilityExtendedConfig()
        rng = np.random.default_rng(5)
        x_paper = np.column_stack([rng.uniform(a, b, 25) for a, b in paper.bounds])
        x = np.zeros((25, config.n_dimensions))
        x[:, list(config.spot_columns)] = x_paper[:, : paper.n_assets]
        x[:, list(config.volatility_columns)] = paper.volatilities
        x[:, config.strike_column] = x_paper[:, paper.n_assets]
        x[:, config.rate_column] = x_paper[:, paper.n_assets + 1]
        x[:, config.maturity_column] = x_paper[:, paper.n_assets + 2]
        np.testing.assert_allclose(
            EuropeanGeometricBasketVolPricer(config)(x),
            EuropeanGeometricBasketPricer(paper)(x_paper),
            atol=1e-10,
        )

    def test_price_increases_with_every_single_volatility(self):
        config = VolatilityExtendedConfig()
        pricer = EuropeanGeometricBasketVolPricer(config)
        rng = np.random.default_rng(17)
        base = sample_market_points(config, 64, rng)
        base[:, list(config.volatility_columns)] = 0.20
        base[:, config.maturity_column] = np.clip(
            base[:, config.maturity_column], 0.25, None
        )
        reference = pricer(base)
        for column in config.volatility_columns:
            bumped = base.copy()
            bumped[:, column] = 0.30
            self.assertTrue(np.all(pricer(bumped) > reference))

    def test_per_row_volatilities_agree_with_row_by_row_pricing(self):
        config = VolatilityExtendedConfig()
        rng = np.random.default_rng(23)
        x = sample_market_points(config, 12, rng)
        spots = x[:, list(config.spot_columns)]
        sigma = x[:, list(config.volatility_columns)]
        batched = geometric_basket_put(
            spots,
            x[:, config.strike_column],
            x[:, config.rate_column],
            x[:, config.maturity_column],
            sigma,
            config.correlation,
            config.dividends,
        )
        one_at_a_time = [
            float(
                geometric_basket_put(
                    spots[i : i + 1],
                    x[i, config.strike_column],
                    x[i, config.rate_column],
                    x[i, config.maturity_column],
                    sigma[i],
                    config.correlation,
                    config.dividends,
                )[0]
            )
            for i in range(len(x))
        ]
        np.testing.assert_allclose(batched, one_at_a_time, atol=1e-12)

    def test_geometric_basket_put_rejects_mismatched_volatility_shape(self):
        config = VolatilityExtendedConfig()
        with self.assertRaisesRegex(ValueError, "one vector per row"):
            geometric_basket_put(
                np.full((4, 5), 100.0),
                100.0,
                0.02,
                1.0,
                np.full((3, 5), 0.2),
                config.correlation,
            )

    def test_log_uniform_sampling_covers_the_volatility_range(self):
        config = VolatilityExtendedConfig()
        rng = np.random.default_rng(29)
        x = sample_market_points(config, 4_000, rng, vol_sampling="log_uniform")
        sigma = x[:, list(config.volatility_columns)]
        self.assertGreaterEqual(float(sigma.min()), 0.05)
        self.assertLessEqual(float(sigma.max()), 0.80)
        # log-uniform means the median sits at the geometric, not arithmetic, mid
        self.assertAlmostEqual(
            float(np.median(sigma)), math.sqrt(0.05 * 0.80), delta=0.02
        )


    def test_basket_scaled_target_is_spot_free_and_recovers_the_price(self):
        config = VolatilityExtendedConfig()
        _, transform, _ = build_vol_extended_grid(config, "moneyness_adaptive")
        model_pricer = TransformedPricer(
            EuropeanGeometricBasketVolPricer(config), transform
        )
        scaled = BasketScaledModelPricer(model_pricer, config.spot_columns)

        rng = np.random.default_rng(41)
        z = transform.to_model(sample_market_points(config, 20, rng))
        # rescaling by the basket reproduces the raw model-coordinate price
        np.testing.assert_allclose(
            scaled(z) * scaled.basket(z), model_pricer(z), rtol=1e-12
        )
        # in moneyness coordinates P / G does not depend on the spots at all
        shuffled = z.copy()
        shuffled[:, list(config.spot_columns)] = z[
            ::-1, list(config.spot_columns)
        ]
        np.testing.assert_allclose(scaled(shuffled), scaled(z), rtol=1e-10)


    def test_blocked_layout_reorders_columns_and_axes_consistently(self):
        blocked = VolatilityExtendedConfig(layout="blocked")
        self.assertEqual(blocked.spot_columns, (0, 1, 2, 3, 4))
        self.assertEqual(blocked.volatility_columns, (5, 6, 7, 8, 9))
        self.assertEqual(
            blocked.physical_shape,
            (32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 64, 8, 8),
        )
        grid, _, description = build_vol_extended_grid(blocked, "moneyness_adaptive")
        self.assertEqual(grid.shape, blocked.physical_shape)
        self.assertEqual(description["layout"], "blocked")
        # the same market point prices identically under either mode ordering
        interleaved = VolatilityExtendedConfig(layout="interleaved")
        rng = np.random.default_rng(61)
        x = sample_market_points(interleaved, 16, rng)
        reordered = np.empty_like(x)
        for source, target in zip(
            interleaved.spot_columns, blocked.spot_columns, strict=True
        ):
            reordered[:, target] = x[:, source]
        for source, target in zip(
            interleaved.volatility_columns, blocked.volatility_columns, strict=True
        ):
            reordered[:, target] = x[:, source]
        for column in (
            interleaved.strike_column,
            interleaved.rate_column,
            interleaved.maturity_column,
        ):
            reordered[:, column] = x[:, column]
        np.testing.assert_allclose(
            EuropeanGeometricBasketVolPricer(blocked)(reordered),
            EuropeanGeometricBasketVolPricer(interleaved)(x),
            rtol=1e-12,
        )

    def test_levy_arithmetic_is_homogeneous_and_below_the_geometric_put(self):
        config = VolatilityExtendedConfig()
        rng = np.random.default_rng(67)
        spots = rng.uniform(30.0, 120.0, (40, 5))
        sigma = np.full((40, 5), 0.20)   # low total vol, where Levy is tight
        strikes = rng.uniform(40.0, 110.0, 40)
        rates = rng.uniform(0.01, 0.05, 40)
        maturities = rng.uniform(0.1, 1.0, 40)
        levy = arithmetic_basket_put_levy(
            spots, strikes, rates, maturities, sigma, config.correlation
        )
        self.assertTrue(np.all(levy >= -1e-12))
        # AM-GM: the arithmetic basket dominates, so its put is worth less
        geometric = geometric_basket_put(
            spots, strikes, rates, maturities, sigma, config.correlation
        )
        self.assertTrue(np.all(levy <= geometric + 1e-9))
        # positive homogeneity of degree one in (spots, strike)
        scaled = arithmetic_basket_put_levy(
            3.0 * spots, 3.0 * strikes, rates, maturities, sigma, config.correlation
        )
        np.testing.assert_allclose(scaled, 3.0 * levy, rtol=1e-10)

    def test_monte_carlo_brackets_the_closed_form_geometric_price(self):
        config = VolatilityExtendedConfig()
        rng = np.random.default_rng(71)
        spots = rng.uniform(40.0, 110.0, (8, 5))
        sigma = np.full((8, 5), 0.25)
        strikes = rng.uniform(50.0, 100.0, 8)
        rates = np.full(8, 0.03)
        maturities = np.full(8, 1.0)
        exact = geometric_basket_put(
            spots, strikes, rates, maturities, sigma, config.correlation
        )
        prices, standard_errors = basket_put_monte_carlo(
            spots, strikes, rates, maturities, sigma, config.correlation,
            60_000, basket_kind="geometric", seed=5,
        )
        # every estimate within four standard errors of the exact price
        self.assertTrue(np.all(np.abs(prices - exact) < 4.0 * standard_errors + 1e-9))

    def test_monte_carlo_respects_am_gm_between_basket_kinds(self):
        config = VolatilityExtendedConfig()
        rng = np.random.default_rng(73)
        spots = rng.uniform(40.0, 110.0, (6, 5))
        sigma = np.full((6, 5), 0.30)
        strikes = rng.uniform(50.0, 100.0, 6)
        rates = np.full(6, 0.02)
        maturities = np.full(6, 1.5)
        arithmetic, _ = basket_put_monte_carlo(
            spots, strikes, rates, maturities, sigma, config.correlation,
            40_000, basket_kind="arithmetic", seed=9,
        )
        geometric, _ = basket_put_monte_carlo(
            spots, strikes, rates, maturities, sigma, config.correlation,
            40_000, basket_kind="geometric", seed=9,
        )
        self.assertTrue(np.all(arithmetic <= geometric + 1e-9))

    def test_arithmetic_vol_pricer_reads_the_configured_columns(self):
        config = VolatilityExtendedConfig()
        rng = np.random.default_rng(79)
        x = sample_market_points(config, 10, rng)
        np.testing.assert_allclose(
            EuropeanArithmeticBasketVolPricer(config)(x),
            arithmetic_basket_put_levy(
                x[:, list(config.spot_columns)],
                x[:, config.strike_column],
                x[:, config.rate_column],
                x[:, config.maturity_column],
                x[:, list(config.volatility_columns)],
                config.correlation,
                config.dividends,
            ),
            rtol=1e-12,
        )


class FactorizedTTPredictionTests(unittest.TestCase):
    def test_factorized_prediction_matches_the_corner_stencil(self):
        config = replace(PaperConfig(), physical_shape=(8, 8, 8, 8, 8, 16, 4, 4))
        grid, transform, _ = build_coordinate_grid(
            config, "moneyness_adaptive", basket_kind="geometric"
        )
        pricer = TransformedPricer(EuropeanGeometricBasketPricer(config), transform)
        model = TTPriceSurrogate(grid, pricer, seed=config.seed)
        model.fit(2_000, anova_samples=200, log=False)

        rng = np.random.default_rng(13)
        x = np.column_stack([rng.uniform(a, b, 40) for a, b in config.bounds])
        points = transform.to_model(x)
        stencil = model.predict(points)
        # the two paths sum the same terms in a different order, so agreement is
        # judged against the scale of the surface rather than each tiny price
        np.testing.assert_allclose(
            model.predict_factorized(points),
            stencil,
            rtol=1e-9,
            atol=1e-9 * float(np.max(np.abs(stencil))),
        )

    def test_factorized_prediction_is_exact_on_grid_nodes(self):
        config = replace(PaperConfig(), physical_shape=(8, 8, 8, 8, 8, 16, 4, 4))
        grid, transform, _ = build_coordinate_grid(
            config, "paper", basket_kind="geometric"
        )
        pricer = TransformedPricer(EuropeanGeometricBasketPricer(config), transform)
        model = TTPriceSurrogate(grid, pricer, seed=config.seed)
        model.fit(2_000, anova_samples=200, log=False)

        rng = np.random.default_rng(19)
        indices = grid.random_physical_indices(50, rng)
        on_grid = model.predict_on_grid(indices)
        np.testing.assert_allclose(
            model.predict_factorized(grid.indices_to_points(indices)),
            on_grid,
            rtol=1e-9,
            atol=1e-9 * float(np.max(np.abs(on_grid))),
        )

    def test_factorized_cubic_matches_the_cubic_corner_stencil(self):
        config = replace(PaperConfig(), physical_shape=(8, 8, 8, 8, 8, 16, 4, 4))
        grid, transform, _ = build_coordinate_grid(
            config, "moneyness_adaptive", basket_kind="geometric"
        )
        pricer = TransformedPricer(EuropeanGeometricBasketPricer(config), transform)
        model = TTPriceSurrogate(grid, pricer, seed=config.seed)
        model.fit(2_000, anova_samples=200, log=False)

        rng = np.random.default_rng(23)
        x = np.column_stack([rng.uniform(a, b, 25) for a, b in config.bounds])
        points = transform.to_model(x)
        for columns in ((5,), (5, 7), (0, 5, 7)):
            stencil = model.predict_hybrid_cubic(
                points, cubic_columns=columns, batch_size=4
            )
            np.testing.assert_allclose(
                model.predict_factorized(points, cubic_columns=columns),
                stencil,
                rtol=1e-9,
                atol=1e-9 * float(np.max(np.abs(stencil))),
            )

    def test_cubic_reproduces_a_cubic_function_exactly(self):
        """A cubic axis must integrate a cubic polynomial with no error."""
        grid = QTTGrid(((0.0, 1.0), (0.0, 1.0)), (16, 4))

        def cubic_in_first_axis(points):
            points = np.atleast_2d(points)
            return 2.0 * points[:, 0] ** 3 - points[:, 0] + 3.0 * points[:, 1]

        model = TTPriceSurrogate(grid, cubic_in_first_axis, seed=0)
        model.fit(400, anova_samples=200, log=False)
        rng = np.random.default_rng(29)
        points = np.column_stack((rng.uniform(0, 1, 40), rng.uniform(0, 1, 40)))
        truth = cubic_in_first_axis(points)
        cubic = model.predict_factorized(points, cubic_columns=(0,))
        linear = model.predict_factorized(points)
        self.assertLess(float(np.max(np.abs(cubic - truth))), 1e-8)
        self.assertGreater(float(np.max(np.abs(linear - truth))), 1e-4)

    def test_cubic_axis_names_resolve_to_columns(self):
        config = VolatilityExtendedConfig(layout="blocked")
        self.assertEqual(resolve_cubic_columns(config, ["none"]), ())
        self.assertEqual(
            resolve_cubic_columns(config, ["all"]), tuple(range(13))
        )
        self.assertEqual(resolve_cubic_columns(config, ["m"]), (10,))
        self.assertEqual(
            resolve_cubic_columns(config, ["vols", "m"]), (5, 6, 7, 8, 9, 10)
        )
        with self.assertRaisesRegex(ValueError, "unknown cubic axis"):
            resolve_cubic_columns(config, ["nope"])
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_cubic_columns(config, ["none", "m"])

    def test_factorized_prediction_requires_a_fit(self):
        grid = QTTGrid(((0.0, 1.0), (0.0, 1.0)), (4, 4))
        model = TTPriceSurrogate(grid, lambda x: np.zeros(len(x)))
        with self.assertRaisesRegex(RuntimeError, "fit the surrogate first"):
            model.predict_factorized(np.array([[0.5, 0.5]]))


class KernelTests(unittest.TestCase):
    def test_manhattan_laplacian(self):
        kernel = ManhattanLaplacian(length_scale=2.0, length_scale_bounds="fixed")
        x = np.array([[0.0, 0.0], [1.0, 2.0]])
        matrix = kernel(x)
        self.assertAlmostEqual(matrix[0, 1], np.exp(-1.5))
        np.testing.assert_allclose(np.diag(matrix), 1.0)


if __name__ == "__main__":
    unittest.main()
