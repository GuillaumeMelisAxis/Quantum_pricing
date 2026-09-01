from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    maximum_rank: int
    parameter_count: int
    pre_truncation_effective_rank: float
    pre_truncation_maximum_rank: int
    pre_truncation_parameter_count: int
    truncation: float | None


@dataclass
class CoreDiagnostics:
    effective_rank: float
    maximum_rank: int
    parameter_count: int
    truncation: float | None
    bond_ranks: list[int]
    core_shapes: list[list[int]]


class TTPriceSurrogate:
    """QTT-cross price surrogate with paper-style order-2 ANOVA initialization."""

    def __init__(self, grid: QTTGrid, pricer, seed: int = 0):
        self.grid = grid
        self.pricer = pricer
        self.seed = int(seed)
        self.cores = None
        self.untruncated_cores = None
        self.diagnostics = None

    @staticmethod
    def _teneva():
        try:
            import teneva
        except ImportError as exc:
            raise RuntimeError(
                "Install the project dependencies: pip install -e ."
            ) from exc
        return teneva

    def _evaluate_qtt(self, qtt_indices: np.ndarray) -> np.ndarray:
        return self.pricer(self.grid.qtt_indices_to_points(qtt_indices))

    @classmethod
    def _core_diagnostics(cls, cores, truncation=None) -> CoreDiagnostics:
        teneva = cls._teneva()
        return CoreDiagnostics(
            effective_rank=float(teneva.erank(cores)),
            maximum_rank=int(max(core.shape[-1] for core in cores[:-1])),
            parameter_count=int(sum(core.size for core in cores)),
            truncation=None if truncation is None else float(truncation),
            bond_ranks=[int(core.shape[-1]) for core in cores[:-1]],
            core_shapes=[[int(value) for value in core.shape] for core in cores],
        )

    def apply_truncation(self, truncation: float | None) -> CoreDiagnostics:
        """Reset the fitted cores from the raw cross output and truncate them."""
        if self.untruncated_cores is None:
            raise RuntimeError("fit the surrogate first")
        source = [core.copy() for core in self.untruncated_cores]
        if truncation is None or float(truncation) == 0.0:
            self.cores = source
            applied = None
        else:
            if float(truncation) < 0.0:
                raise ValueError("truncation must be non-negative")
            self.cores = self._teneva().truncate(source, float(truncation))
            applied = float(truncation)
        return self._core_diagnostics(self.cores, applied)

    def save_untruncated_cores(self, path: str | Path) -> Path:
        """Persist the raw TT-cross cores for controlled truncation studies."""
        if self.untruncated_cores is None:
            raise RuntimeError("fit the surrogate first")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            **{
                f"core_{index:03d}": np.asarray(core, dtype=float)
                for index, core in enumerate(self.untruncated_cores)
            },
        )
        return target

    def load_untruncated_cores(self, path: str | Path) -> CoreDiagnostics:
        """Restore raw cores and validate their QTT rank connectivity."""
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            names = sorted(archive.files)
            expected = [f"core_{index:03d}" for index in range(sum(self.grid.bits))]
            if names != expected:
                raise ValueError("core archive is incompatible with this QTT grid")
            cores = [np.asarray(archive[name], dtype=float) for name in names]
        for index, core in enumerate(cores):
            if core.ndim != 3 or core.shape[1] != 2:
                raise ValueError("every QTT core must have shape (r_left,2,r_right)")
            if index == 0 and core.shape[0] != 1:
                raise ValueError("the first QTT core must have left rank one")
            if index and cores[index - 1].shape[2] != core.shape[0]:
                raise ValueError("adjacent QTT core ranks are inconsistent")
        if cores[-1].shape[2] != 1:
            raise ValueError("the last QTT core must have right rank one")
        self.untruncated_cores = [core.copy() for core in cores]
        return self.apply_truncation(None)

    def fit(
        self,
        max_evals: int,
        anova_samples: int = 2_000,
        max_sweeps: int = 20,
        rank_increment: int = 2,
        truncation: float | None = 1e-8,
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
        raw_cores = teneva.cross(
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
        self.untruncated_cores = [core.copy() for core in raw_cores]
        pre_truncation = self._core_diagnostics(
            self.untruncated_cores,
            truncation=None,
        )
        post_truncation = self.apply_truncation(truncation)
        wall_time = perf_counter() - start
        self.diagnostics = FitDiagnostics(
            wall_time=wall_time,
            function_evaluations=int(info.get("m", 0)) + int(anova_samples),
            sweeps=int(info.get("nswp", 0)),
            stop=str(info.get("stop", "unknown")),
            effective_rank=post_truncation.effective_rank,
            maximum_rank=post_truncation.maximum_rank,
            parameter_count=post_truncation.parameter_count,
            pre_truncation_effective_rank=pre_truncation.effective_rank,
            pre_truncation_maximum_rank=pre_truncation.maximum_rank,
            pre_truncation_parameter_count=pre_truncation.parameter_count,
            truncation=post_truncation.truncation,
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

    def predict_hybrid_cubic(
        self,
        points: np.ndarray,
        cubic_columns,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Interpolate cubically on selected physical axes and linearly elsewhere."""
        if self.cores is None:
            raise RuntimeError("fit the surrogate first")
        points = np.atleast_2d(np.asarray(points, dtype=float))
        out = np.empty(len(points), dtype=float)
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            corners, weights = self.grid.hybrid_cubic_stencil(
                batch,
                cubic_columns=cubic_columns,
            )
            flat = corners.reshape(-1, corners.shape[-1])
            values = self.predict_on_grid(flat).reshape(corners.shape[:2])
            out[start : start + len(batch)] = np.sum(weights * values, axis=1)
        return out
