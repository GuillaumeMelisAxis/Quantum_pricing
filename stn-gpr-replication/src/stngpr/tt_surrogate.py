from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .grids import QTTGrid


@dataclass
class FitDiagnostics:
    wall_time: float
    function_evaluations: int
    sweeps: int
    stop: str
    effective_rank: float


class TTPriceSurrogate:
    """QTT-cross price surrogate with paper-style order-2 ANOVA initialization."""

    def __init__(self, grid: QTTGrid, pricer, seed: int = 0):
        self.grid = grid
        self.pricer = pricer
        self.seed = int(seed)
        self.cores = None
        self.diagnostics = None

    @staticmethod
    def _teneva():
        try:
            import teneva
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies: pip install -e .") from exc
        return teneva

    def _evaluate_qtt(self, qtt_indices: np.ndarray) -> np.ndarray:
        return self.pricer(self.grid.qtt_indices_to_points(qtt_indices))

    def fit(
        self,
        max_evals: int,
        anova_samples: int = 2_000,
        max_sweeps: int = 20,
        rank_increment: int = 2,
        truncation: float = 1e-8,
        log: bool = True,
    ) -> FitDiagnostics:
        teneva = self._teneva()
        rng = np.random.default_rng(self.seed)
        physical = self.grid.random_physical_indices(anova_samples, rng)
        i_anova = self.grid.physical_indices_to_qtt(physical)
        y_anova = self._evaluate_qtt(i_anova)
        y0 = teneva.anova(i_anova, y_anova, r=2, order=2, seed=self.seed)

        info, cache = {}, {}
        start = perf_counter()
        self.cores = teneva.cross(
            self._evaluate_qtt,
            y0,
            m=int(max_evals),
            nswp=int(max_sweeps),
            dr_min=1,
            dr_max=int(rank_increment),
            info=info,
            cache=cache,
            log=log,
        )
        self.cores = teneva.truncate(self.cores, truncation)
        wall_time = perf_counter() - start
        self.diagnostics = FitDiagnostics(
            wall_time=wall_time,
            function_evaluations=int(info.get("m", 0)) + int(anova_samples),
            sweeps=int(info.get("nswp", 0)),
            stop=str(info.get("stop", "unknown")),
            effective_rank=float(teneva.erank(self.cores)),
        )
        return self.diagnostics

    def predict_on_grid(self, physical_indices: np.ndarray) -> np.ndarray:
        if self.cores is None:
            raise RuntimeError("fit the surrogate first")
        teneva = self._teneva()
        qtt = self.grid.physical_indices_to_qtt(physical_indices)
        return np.asarray(teneva.get_many(self.cores, qtt), dtype=float)

    def predict(self, points: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Paper's large-length-scale limit: off-grid multilinear interpolation."""
        if self.cores is None:
            raise RuntimeError("fit the surrogate first")
        points = np.atleast_2d(np.asarray(points, dtype=float))
        out = np.empty(len(points), dtype=float)
        for start in range(0, len(points), batch_size):
            p = points[start : start + batch_size]
            corners, weights = self.grid.multilinear_stencil(p)
            flat = corners.reshape(-1, corners.shape[-1])
            values = self.predict_on_grid(flat).reshape(corners.shape[:2])
            out[start : start + len(p)] = np.sum(weights * values, axis=1)
        return out

