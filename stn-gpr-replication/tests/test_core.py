import unittest

import numpy as np

from stngpr.baselines import ManhattanLaplacian
from stngpr.config import PaperConfig
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
