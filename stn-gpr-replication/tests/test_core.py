import unittest

import numpy as np

from stngpr.baselines import ManhattanLaplacian
from stngpr.config import PaperConfig
from stngpr.coordinates import (
    CoordinateTransform,
    build_coordinate_grid,
    oracle_multilinear_predict,
)
from stngpr.grids import QTTGrid
from stngpr.pricers import geometric_basket_put
from stngpr.risk import var_es


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

    def test_moneyness_coordinate_round_trip(self):
        transform = CoordinateTransform(5, "geometric", True)
        market = np.array([
            [80.0, 90.0, 100.0, 110.0, 120.0, 105.0, 0.03, 1.2],
            [20.0, 30.0, 40.0, 50.0, 60.0, 35.0, 0.01, 0.2],
        ])
        np.testing.assert_allclose(
            transform.to_market(transform.to_model(market)), market, rtol=1e-14
        )


class RiskTests(unittest.TestCase):
    def test_var_es(self):
        var, es = var_es(np.arange(100.0), 0.95)
        self.assertEqual(var, 95.0)
        self.assertGreaterEqual(es, var)


class KernelTests(unittest.TestCase):
    def test_manhattan_laplacian(self):
        kernel = ManhattanLaplacian(length_scale=2.0, length_scale_bounds="fixed")
        x = np.array([[0.0, 0.0], [1.0, 2.0]])
        matrix = kernel(x)
        self.assertAlmostEqual(matrix[0, 1], np.exp(-1.5))
        np.testing.assert_allclose(np.diag(matrix), 1.0)


if __name__ == "__main__":
    unittest.main()
