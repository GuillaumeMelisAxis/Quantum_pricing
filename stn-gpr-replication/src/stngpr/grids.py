from __future__ import annotations

import itertools

import numpy as np


class QTTGrid:
    """Tensor-product grid with big-endian binary (QTT) index encoding."""

    def __init__(self, bounds=None, shape=None, axes=None):
        if axes is None:
            if bounds is None or shape is None:
                raise ValueError("provide either axes or both bounds and shape")
            bounds = tuple((float(a), float(b)) for a, b in bounds)
            shape = tuple(int(n) for n in shape)
            if len(bounds) != len(shape):
                raise ValueError("bounds and shape must have equal lengths")
            axes = tuple(
                np.linspace(a, b, n, dtype=float)
                for (a, b), n in zip(bounds, shape)
            )
        else:
            axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
            if not axes or any(axis.ndim != 1 or axis.size < 2 for axis in axes):
                raise ValueError("each explicit axis must be a one-dimensional array")
            if any(np.any(np.diff(axis) <= 0.0) for axis in axes):
                raise ValueError("explicit grid axes must be strictly increasing")

        self.axes = axes
        self.shape = tuple(int(axis.size) for axis in axes)
        self.bounds = tuple((float(axis[0]), float(axis[-1])) for axis in axes)
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
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 1:
            indices = indices[None, :]
        if indices.shape[1] != len(self.shape):
            raise ValueError("wrong physical index dimension")
        x = np.empty(indices.shape, dtype=float)
        for j, (axis, n) in enumerate(zip(self.axes, self.shape)):
            if np.any((indices[:, j] < 0) | (indices[:, j] >= n)):
                raise ValueError("grid index out of bounds")
            x[:, j] = axis[indices[:, j]]
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
        for j, axis in enumerate(self.axes):
            values = np.clip(points[:, j], axis[0], axis[-1])
            upper = np.searchsorted(axis, values, side="right")
            upper = np.clip(upper, 1, axis.size - 1)
            lower = upper - 1
            denominator = axis[upper] - axis[lower]
            lo[:, j] = lower
            hi[:, j] = upper
            w_hi[:, j] = (values - axis[lower]) / denominator

        corners = np.empty((m, 2**d, d), dtype=np.int64)
        weights = np.empty((m, 2**d), dtype=float)
        for c, selector in enumerate(itertools.product((0, 1), repeat=d)):
            selector = np.asarray(selector, dtype=bool)
            corners[:, c, :] = np.where(selector, hi, lo)
            weights[:, c] = np.prod(np.where(selector, w_hi, 1.0 - w_hi), axis=1)
        return corners, weights


def adaptive_axis(lower, upper, n, center=0.0, half_width=0.5, center_fraction=0.75):
    """Strictly increasing axis with most nodes in a central interval."""
    if not lower < center - half_width < center + half_width < upper:
        raise ValueError("central interval must lie strictly inside the bounds")
    n_center = int(round(n * center_fraction))
    n_center = min(max(n_center, 2), n - 2)
    n_tail = n - n_center
    n_left = n_tail // 2
    n_right = n_tail - n_left
    left = np.linspace(lower, center - half_width, n_left, endpoint=False)
    middle = np.linspace(
        center - half_width, center + half_width, n_center, endpoint=False
    )
    right = np.linspace(center + half_width, upper, n_right, endpoint=True)
    axis = np.concatenate((left, middle, right))
    if axis.size != n or np.any(np.diff(axis) <= 0.0):
        raise RuntimeError("failed to build adaptive axis")
    return axis


def sinh_centered_axis(lower, upper, n, concentration=3.0):
    """Smooth asymmetric axis concentrated near zero with exact tail coverage."""
    if not lower < 0.0 < upper:
        raise ValueError("sinh-centered bounds must straddle zero")
    if concentration <= 0.0:
        raise ValueError("concentration must be positive")
    u = np.linspace(-1.0, 1.0, n)
    scale = np.sinh(concentration)
    return np.where(
        u < 0.0,
        -abs(lower) * np.sinh(concentration * np.abs(u)) / scale,
        upper * np.sinh(concentration * u) / scale,
    )


def short_maturity_axis(lower, upper, n, power=2.0):
    """Maturity nodes concentrated near the shortest maturity."""
    u = np.linspace(0.0, 1.0, n)
    return lower + (upper - lower) * u**power
