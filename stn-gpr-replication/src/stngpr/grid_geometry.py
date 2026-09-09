from __future__ import annotations

import numpy as np


def physical_log_moneyness_nodes(
    grid,
    transform,
    n_assets: int,
    rate: float,
    maturity: float,
    reference_spot: float = 100.0,
) -> np.ndarray:
    """Map one tensor coordinate axis to physical log-moneyness nodes.

    The spots are fixed only to construct a valid market vector.  They cancel
    when ``m = log(K / G(S))`` is recovered, so the returned nodes depend on
    the coordinate transform, rate and maturity, but not on ``reference_spot``.
    """
    coordinate = np.asarray(grid.axes[int(n_assets)], dtype=float)
    if coordinate.ndim != 1 or coordinate.size < 2:
        raise ValueError("the pricing coordinate axis must contain at least two nodes")
    if not np.isfinite(rate) or maturity <= 0.0 or reference_spot <= 0.0:
        raise ValueError("rate, maturity and reference_spot are invalid")

    model = np.empty((coordinate.size, int(n_assets) + 3), dtype=float)
    model[:, :n_assets] = float(reference_spot)
    model[:, n_assets] = coordinate
    model[:, n_assets + 1] = float(rate)
    model[:, n_assets + 2] = float(maturity)
    market = transform.to_market(model)
    basket = np.exp(np.mean(np.log(market[:, :n_assets]), axis=1))
    nodes = np.log(market[:, n_assets] / basket)
    if np.any(~np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("the physical log-moneyness nodes are not strictly increasing")
    return nodes


def normalized_node_displacement(
    reference_nodes: np.ndarray,
    candidate_nodes: np.ndarray,
    domain: tuple[float, float] | None = None,
) -> dict:
    """Return normalized paired L1, L2 and Linfinity node displacements."""
    reference = np.asarray(reference_nodes, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_nodes, dtype=float).reshape(-1)
    if reference.size < 2 or reference.shape != candidate.shape:
        raise ValueError("the two grids must contain the same number of nodes")
    if np.any(np.diff(reference) <= 0.0) or np.any(np.diff(candidate) <= 0.0):
        raise ValueError("grid nodes must be strictly increasing")
    if domain is None:
        lower = min(float(reference[0]), float(candidate[0]))
        upper = max(float(reference[-1]), float(candidate[-1]))
    else:
        lower, upper = (float(value) for value in domain)
    width = upper - lower
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("the normalization domain must have positive width")

    displacement = np.abs(candidate - reference)
    return {
        "normalized_l1": float(np.mean(displacement) / width),
        "normalized_l2": float(np.sqrt(np.mean(displacement**2)) / width),
        "normalized_linf": float(np.max(displacement) / width),
        "absolute_l1": float(np.mean(displacement)),
        "absolute_l2": float(np.sqrt(np.mean(displacement**2))),
        "absolute_linf": float(np.max(displacement)),
    }


def nearest_node_distance(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Distance from each one-dimensional point to its nearest grid node."""
    x = np.asarray(nodes, dtype=float).reshape(-1)
    p = np.asarray(points, dtype=float)
    if x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("nodes must be a strictly increasing one-dimensional array")
    flat = p.reshape(-1)
    upper = np.searchsorted(x, flat, side="left")
    right = np.clip(upper, 0, x.size - 1)
    left = np.clip(upper - 1, 0, x.size - 1)
    distance = np.minimum(np.abs(flat - x[left]), np.abs(flat - x[right]))
    return distance.reshape(p.shape)


def interval_fill_distance(
    nodes: np.ndarray,
    lower: float,
    upper: float,
) -> dict:
    """Exact one-dimensional fill distance restricted to ``[lower, upper]``.

    The maximum of the nearest-node distance is attained at an interval
    endpoint or at a Voronoi midpoint between two consecutive grid nodes.
    """
    x = np.asarray(nodes, dtype=float).reshape(-1)
    lower, upper = float(lower), float(upper)
    if x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("nodes must be strictly increasing")
    if lower > upper or lower < x[0] or upper > x[-1]:
        raise ValueError("the audit interval must lie inside the grid domain")
    midpoints = 0.5 * (x[:-1] + x[1:])
    candidates = np.concatenate(
        ([lower], midpoints[(midpoints >= lower) & (midpoints <= upper)], [upper])
    )
    distances = nearest_node_distance(x, candidates)
    index = int(np.argmax(distances))
    return {
        "fill_distance": float(distances[index]),
        "location": float(candidates[index]),
        "interval": [lower, upper],
    }


def local_coverage_metrics(
    nodes: np.ndarray,
    lower: float,
    upper: float,
) -> dict:
    """Fill distance, node share and intersecting-cell gaps on one interval."""
    x = np.asarray(nodes, dtype=float).reshape(-1)
    fill = interval_fill_distance(x, lower, upper)
    inside = (x >= float(lower)) & (x <= float(upper))
    intersects = (x[:-1] < float(upper)) & (x[1:] > float(lower))
    gaps = np.diff(x)[intersects]
    return {
        **fill,
        "node_count": int(np.count_nonzero(inside)),
        "node_fraction": float(np.mean(inside)),
        "intersecting_cell_count": int(gaps.size),
        "maximum_intersecting_gap": float(np.max(gaps)) if gaps.size else 0.0,
        "median_intersecting_gap": float(np.median(gaps)) if gaps.size else 0.0,
    }


def weighted_fill_proxy(
    nodes: np.ndarray,
    evaluation_points: np.ndarray,
    weights: np.ndarray,
    distance_power: int = 1,
) -> dict:
    """Return ``max weight * distance**p`` on a dense deterministic audit set."""
    points = np.asarray(evaluation_points, dtype=float).reshape(-1)
    weight = np.asarray(weights, dtype=float).reshape(-1)
    if points.shape != weight.shape or points.size < 2:
        raise ValueError("evaluation points and weights must have matching shapes")
    if np.any(np.diff(points) <= 0.0) or np.any(weight < 0.0):
        raise ValueError("evaluation points must increase and weights be non-negative")
    if int(distance_power) != distance_power or distance_power <= 0:
        raise ValueError("distance_power must be a positive integer")
    distance = nearest_node_distance(nodes, points)
    score = weight * distance ** int(distance_power)
    index = int(np.argmax(score))
    return {
        "score": float(score[index]),
        "location": float(points[index]),
        "nearest_node_distance": float(distance[index]),
        "weight": float(weight[index]),
        "distance_power": int(distance_power),
    }


def local_cubic_lagrange(
    nodes: np.ndarray,
    values: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate local four-node Lagrange values and first two derivatives.

    The stencil selection is identical to ``QTTGrid.hybrid_cubic_stencil``.
    A centered polynomial basis is used for numerical stability.
    """
    x = np.asarray(nodes, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    query = np.asarray(points, dtype=float)
    if x.size < 4 or x.shape != y.shape or np.any(np.diff(x) <= 0.0):
        raise ValueError("local cubic interpolation requires four increasing nodes")
    flat = np.clip(query.reshape(-1), x[0], x[-1])
    upper = np.searchsorted(x, flat, side="right")
    upper = np.clip(upper, 1, x.size - 1)
    lower = upper - 1
    starts = np.clip(lower - 1, 0, x.size - 4)

    support_indices = np.arange(x.size - 3)[:, None] + np.arange(4)[None, :]
    support_x = x[support_indices]
    centers = np.mean(support_x, axis=1)
    shifted = support_x - centers[:, None]
    vandermonde = np.stack(
        (np.ones_like(shifted), shifted, shifted**2, shifted**3),
        axis=2,
    )
    coefficients = np.linalg.solve(
        vandermonde,
        y[support_indices][..., None],
    )[..., 0]
    selected = coefficients[starts]
    t = flat - centers[starts]
    estimate = (
        selected[:, 0]
        + selected[:, 1] * t
        + selected[:, 2] * t**2
        + selected[:, 3] * t**3
    )
    first = selected[:, 1] + 2.0 * selected[:, 2] * t + 3.0 * selected[:, 3] * t**2
    second = 2.0 * selected[:, 2] + 6.0 * selected[:, 3] * t
    shape = query.shape
    return estimate.reshape(shape), first.reshape(shape), second.reshape(shape)
