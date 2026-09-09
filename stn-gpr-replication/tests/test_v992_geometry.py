from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from scripts.validation.analyze_grid_node_redistribution_v992 import (
    GRID_MODES,
    _aggregate,
    _scenario,
)
from stngpr.config import PaperConfig
from stngpr.risk_grids import build_greek_coordinate_grid


class V992GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = replace(
            PaperConfig(),
            physical_shape=(8, 8, 8, 8, 8, 64, 8, 8),
        )
        cls.grids = {
            name: build_greek_coordinate_grid(cls.config, mode)
            for name, mode in GRID_MODES.items()
        }

    def test_scenario_matches_theoretical_metric_definitions(self):
        row = _scenario(
            self.config,
            self.grids,
            rate=0.03,
            maturity_days=30.0,
            kappas=(1.0, 2.0, 3.0),
        )
        self.assertEqual(
            set(row["node_displacement"]),
            {"D2", "D_infinity"},
        )

        pricing = np.asarray(row["physical_nodes"]["pricing_grid"])
        risk = np.asarray(row["physical_nodes"]["risk_hybrid_grid"])
        width = row["moneyness_domain"][1] - row["moneyness_domain"][0]
        expected_d2 = np.sqrt(np.mean(((risk - pricing) / width) ** 2))
        expected_d_infinity = np.max(np.abs(risk - pricing)) / width
        self.assertAlmostEqual(row["node_displacement"]["D2"], expected_d2)
        self.assertAlmostEqual(
            row["node_displacement"]["D_infinity"],
            expected_d_infinity,
        )

        for band in row["bands"].values():
            self.assertEqual(
                set(band),
                {
                    "kappa",
                    "bounds",
                    "pricing_grid",
                    "risk_hybrid_grid",
                    "coverage_gain",
                },
            )
            lower, upper = band["bounds"]
            for name, nodes in (("pricing_grid", pricing), ("risk_hybrid_grid", risk)):
                self.assertEqual(set(band[name]), {"h_fill", "P_nodes"})
                expected_concentration = np.mean(
                    (nodes >= lower) & (nodes <= upper)
                )
                self.assertAlmostEqual(
                    band[name]["P_nodes"],
                    expected_concentration,
                )
            self.assertAlmostEqual(
                band["coverage_gain"],
                band["pricing_grid"]["h_fill"]
                / band["risk_hybrid_grid"]["h_fill"],
            )

    def test_summary_contains_only_theoretical_metrics(self):
        row = _scenario(
            self.config,
            self.grids,
            rate=0.03,
            maturity_days=30.0,
            kappas=(2.0,),
        )
        summary = _aggregate([row], (2.0,))
        self.assertEqual(
            set(summary),
            {"scenario_count", "node_displacement", "bands"},
        )
        self.assertEqual(
            set(summary["node_displacement"]),
            {"D2", "D_infinity"},
        )
        self.assertEqual(
            set(summary["bands"]["kappa_2"]),
            {"h_fill", "P_nodes", "coverage_gain"},
        )


if __name__ == "__main__":
    unittest.main()
