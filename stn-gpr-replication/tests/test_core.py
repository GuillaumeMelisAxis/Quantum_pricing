import math
import unittest
from dataclasses import replace

import numpy as np

from stngpr.baselines import ManhattanLaplacian
from stngpr.config import PaperConfig
from stngpr.convergence import (
    columnwise_error_metrics,
    component_error_decomposition,
    component_residual_metrics,
    cross_gamma_sign_diagnostics,
    curve_error_diagnostics,
    finite_difference_component_arrays,
    full_hessian_error_metrics,
    gamma_matrices,
    mean_and_standard_error,
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
from stngpr.pricers import (
    AmericanArithmeticBasketLSMC,
    EuropeanArithmeticBasketQMC,
    geometric_basket_put,
)
from stngpr.risk import var_es
from stngpr.risk_grids import (
    StandardizedRiskTransform,
    build_greek_coordinate_grid,
    gamma_monitor_axis,
)
from stngpr.validation import (
    american_put_binomial,
    black_scholes_put,
    error_metrics,
    scalar_summary,
    stratified_american_points,
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

    def test_nonuniform_grid_interpolates_linear_function_exactly(self):
        grid = QTTGrid(axes=(np.array([0.0, 0.1, 0.4, 1.0]), np.array([0.0, 0.2, 1.0, 3.0])))
        points = np.array([[0.25, 0.7], [0.9, 2.4]])
        pricer = lambda x: 2.0 + 3.0 * x[:, 0] - 0.5 * x[:, 1]
        predicted = oracle_multilinear_predict(grid, pricer, points)
        np.testing.assert_allclose(predicted, pricer(points), atol=1e-14)

    def test_hybrid_cubic_interpolates_cubic_linear_function_exactly(self):
        grid = QTTGrid(axes=(
            np.array([-1.0, -0.7, -0.2, 0.1, 0.4, 0.8, 1.3, 2.0]),
            np.array([0.0, 0.3, 1.1, 3.0]),
        ))
        points = np.array([
            [-0.45, 0.7],
            [0.25, 1.8],
            [1.05, 2.4],
        ])

        def pricer(x):
            return (
                1.0
                + x[:, 0] ** 3
                + 0.5 * x[:, 0] ** 2 * x[:, 1]
                - 2.0 * x[:, 1]
            )

        predicted = oracle_hybrid_cubic_predict(
            grid,
            pricer,
            points,
            cubic_columns=(0,),
        )
        np.testing.assert_allclose(predicted, pricer(points), atol=2e-14)

    def test_cached_grid_interpolator_reuses_exact_node_values(self):
        grid = QTTGrid(axes=(
            np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
            np.array([0.0, 0.25, 0.5, 1.0]),
        ))
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
        np.testing.assert_allclose(paper.axes[config.n_assets], paper_t.axes[config.n_assets])
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
        market = np.array([
            [80.0, 90.0, 100.0, 110.0, 120.0, 105.0, 0.03, 14.0 / 365.0],
            [50.0, 65.0, 80.0, 95.0, 110.0, 75.0, 0.06, 2.0],
        ])
        recovered = transform.to_market(transform.to_model(market))
        np.testing.assert_allclose(recovered, market, rtol=2e-14, atol=2e-14)

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
            [[50] * 5, [100] * 5], [80, 80], [0.03, 0.03], [1, 1],
            config.volatilities, config.correlation,
        )
        self.assertTrue(np.all(prices >= 0.0))

    def test_arithmetic_qmc_is_reproducible(self):
        config = PaperConfig()
        parameters = np.array([[80.0, 90.0, 100.0, 110.0, 120.0, 100.0, 0.03, 0.5]])
        first = EuropeanArithmeticBasketQMC(
            config, n_paths=512, seed=17
        )(parameters)
        second = EuropeanArithmeticBasketQMC(
            config, n_paths=512, seed=17
        )(parameters)
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
        qmc_price = EuropeanArithmeticBasketQMC(
            config, n_paths=512, seed=23
        )(parameters)[0]
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
        scenarios = np.array([
            [80.0, 90.0, 100.0, 110.0, 120.0, 100.0, 0.03, 30 / 365],
            [80.0, 90.0, 100.0, 110.0, 120.0, 101.0, 0.03, 30 / 365],
        ])
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
        expected_peak = float(geometric_basket_convexity_ridge(
            maturity, basket_sigma, basket_carry
        ))
        m = np.linspace(expected_peak - 0.5, expected_peak + 0.5, 20_001)
        chi = geometric_basket_log_moneyness_convexity(
            m, maturity, rate, basket_sigma, basket_carry
        )
        self.assertTrue(np.all(chi >= 0.0))
        self.assertAlmostEqual(float(m[np.argmax(chi)]), expected_peak, places=4)

    def test_moneyness_coordinate_round_trip(self):
        transform = CoordinateTransform(5, "geometric", True)
        market = np.array([
            [80.0, 90.0, 100.0, 110.0, 120.0, 105.0, 0.03, 1.2],
            [20.0, 30.0, 40.0, 50.0, 60.0, 35.0, 0.01, 0.2],
        ])
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
        counts = {name: sum(label[0] == name for label in labels) for name in {
            label[0] for label in labels
        }}
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
            diagnostics["shape_by_component_column"][0][
                "estimate_turning_point_count"
            ],
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
            name: np.asarray(values) + 0.1
            for name, values in analytical.items()
        }
        grid_fd = {
            name: np.asarray(values) - 0.025
            for name, values in exact_fd.items()
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

    def test_residual_layers_share_the_analytical_scale(self):
        reference = {
            "price": np.array([2.0]),
            "delta": np.array([[4.0, -4.0]]),
            "gamma_diagonal": np.array([[0.5, 1.5]]),
            "cross_gamma": np.array([[0.25]]),
        }
        residual = {
            name: 0.1 * np.asarray(values)
            for name, values in reference.items()
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
        second = {
            name: np.asarray(value) + 2.0
            for name, value in first.items()
        }
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
        analytical_delta, analytical_gamma = (
            geometric_basket_put_spot_greeks(
                spots[None, :],
                [strike],
                [rate],
                [maturity],
                volatilities,
                correlation,
            )
        )

        covariance = np.outer(volatilities, volatilities) * correlation
        basket_variance = float(weights @ covariance @ weights)
        basket_sigma = np.sqrt(basket_variance)
        basket = float(np.exp(np.log(spots) @ weights))
        carry = (
            rate
            - 0.5 * float(weights @ volatilities**2)
            + 0.5 * basket_variance
        )
        std = basket_sigma * np.sqrt(maturity)
        d1 = (
            np.log(basket / strike)
            + (carry + 0.5 * basket_variance) * maturity
        ) / std
        normal_cdf_minus_d1 = 0.5 * (
            1.0 + math.erf(-d1 / np.sqrt(2.0))
        )
        normal_density_d1 = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
        discount_carry = np.exp((carry - rate) * maturity)
        basket_delta = -discount_carry * normal_cdf_minus_d1
        basket_gamma = (
            discount_carry * normal_density_d1 / (basket * std)
        )

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
            basket_second = (
                weights[i] * (weights[i] - 1.0) * basket / spots[i] ** 2
            )
            expected = (
                basket_gamma * basket_first[i] ** 2
                + basket_delta * basket_second
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
                basket_mixed = (
                    weights[i] * weights[j] * basket / (spots[i] * spots[j])
                )
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


class KernelTests(unittest.TestCase):
    def test_manhattan_laplacian(self):
        kernel = ManhattanLaplacian(length_scale=2.0, length_scale_bounds="fixed")
        x = np.array([[0.0, 0.0], [1.0, 2.0]])
        matrix = kernel(x)
        self.assertAlmostEqual(matrix[0, 1], np.exp(-1.5))
        np.testing.assert_allclose(np.diag(matrix), 1.0)


if __name__ == "__main__":
    unittest.main()
