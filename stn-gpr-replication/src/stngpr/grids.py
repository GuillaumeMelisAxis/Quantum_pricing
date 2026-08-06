from __future__ import annotations

import itertools

import numpy as np


class QTTGrid:
    """Uniform physical grid with big-endian binary (QTT) index encoding."""

    def __init__(self, bounds, shape):
        self.bounds = tuple((float(a), float(b)) for a, b in bounds)
        self.shape = tuple(int(n) for n in shape)
        if len(self.bounds) != len(self.shape):
            raise ValueError("bounds and shape must have equal lengths")
        self.bits = tuple(int(np.log2(n)) for n in self.shape)
        if any(2**q != n for q, n in zip(self.bits, self.shape)):
            raise ValueError("all QTT mode sizes must be powers of two")

    @property
    def qtt_shape(self) -> tuple[int, ...]:
        return (2,) * sum(self.bits)

    def physical_indices_to_qtt(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 1:
            indices = indices[None, :]
        if indices.shape[1] != len(self.shape):
            raise ValueError("wrong physical index dimension")
        chunks = []
        for j, (n, q) in enumerate(zip(self.shape, self.bits)):
            idx = indices[:, j]
            if np.any((idx < 0) | (idx >= n)):
                raise ValueError("grid index out of bounds")
            shifts = np.arange(q - 1, -1, -1, dtype=np.int64)
            chunks.append(((idx[:, None] >> shifts) & 1).astype(np.int64))
        return np.concatenate(chunks, axis=1)

    def qtt_indices_to_physical(self, qtt_indices: np.ndarray) -> np.ndarray:
        qtt_indices = np.asarray(qtt_indices, dtype=np.int64)
        if qtt_indices.ndim == 1:
            qtt_indices = qtt_indices[None, :]
        if qtt_indices.shape[1] != sum(self.bits):
            raise ValueError("wrong QTT index dimension")
        out, start = [], 0
        for q in self.bits:
            block = qtt_indices[:, start : start + q]
            weights = 2 ** np.arange(q - 1, -1, -1)
            out.append(block @ weights)
            start += q
        return np.column_stack(out).astype(np.int64)

    def indices_to_points(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=float)
        if indices.ndim == 1:
            indices = indices[None, :]
        x = np.empty_like(indices)
        for j, ((a, b), n) in enumerate(zip(self.bounds, self.shape)):
            x[:, j] = a + (b - a) * indices[:, j] / (n - 1)
        return x

    def qtt_indices_to_points(self, qtt_indices: np.ndarray) -> np.ndarray:
        return self.indices_to_points(self.qtt_indices_to_physical(qtt_indices))

    def random_physical_indices(self, n_samples: int, rng) -> np.ndarray:
        return np.column_stack(
            [rng.integers(0, n, size=n_samples) for n in self.shape]
        )

    def multilinear_stencil(self, points: np.ndarray):
        """Return corner indices and weights for each off-grid point."""
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        m, d = points.shape
        if d != len(self.shape):
            raise ValueError("wrong point dimension")

        lo = np.empty((m, d), dtype=np.int64)
        hi = np.empty((m, d), dtype=np.int64)
        w_hi = np.empty((m, d), dtype=float)
        for j, ((a, b), n) in enumerate(zip(self.bounds, self.shape)):
            z = np.clip((points[:, j] - a) * (n - 1) / (b - a), 0, n - 1)
            lo[:, j] = np.floor(z).astype(np.int64)
            hi[:, j] = np.minimum(lo[:, j] + 1, n - 1)
            w_hi[:, j] = z - lo[:, j]
            w_hi[hi[:, j] == lo[:, j], j] = 0.0

        corners = np.empty((m, 2**d, d), dtype=np.int64)
        weights = np.empty((m, 2**d), dtype=float)
        for c, selector in enumerate(itertools.product((0, 1), repeat=d)):
            selector = np.asarray(selector, dtype=bool)
            corners[:, c, :] = np.where(selector, hi, lo)
            weights[:, c] = np.prod(np.where(selector, w_hi, 1.0 - w_hi), axis=1)
        return corners, weights

