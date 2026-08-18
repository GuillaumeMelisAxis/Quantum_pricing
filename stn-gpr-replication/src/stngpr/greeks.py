from __future__ import annotations

import numpy as np


def _evaluate_prices(pricer, scenarios: np.ndarray) -> np.ndarray:
    values = np.asarray(pricer(scenarios), dtype=float).reshape(-1)
    if values.size != scenarios.shape[0]:
        raise ValueError("pricer must return one value per parameter vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("pricer returned non-finite values")
    return values


def finite_difference_greeks(
    pricer,
    x,
    spot_columns,
    relative_bump=1e-3,
    minimum_bump=1e-6,
):
    """Central spot Delta, Gamma and cross-Gamma from one vectorized call.

    Parameters are assumed to be expressed in market coordinates. In
    particular, the strike column must remain fixed when a spot is bumped. A
    transformed TT surrogate should therefore first be wrapped in
    MarketCoordinatePricer.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one parameter vector")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if relative_bump <= 0.0 or minimum_bump <= 0.0:
        raise ValueError("bump sizes must be positive")

    columns = tuple(int(j) for j in spot_columns)
    if not columns:
        raise ValueError("spot_columns must not be empty")
    if len(set(columns)) != len(columns):
        raise ValueError("spot_columns must be unique")
    if any(j < 0 or j >= x.size for j in columns):
        raise ValueError("spot column out of bounds")

    bumps = {
        j: max(abs(x[j]) * float(relative_bump), float(minimum_bump))
        for j in columns
    }
    if any(x[j] - bumps[j] <= 0.0 for j in columns):
        raise ValueError("central spot bumps must remain strictly positive")

    scenarios = [x.copy()]
    locations = {}
    for j in columns:
        h = bumps[j]
        up, down = x.copy(), x.copy()
        up[j] += h
        down[j] -= h
        locations[("up", j)] = len(scenarios)
        scenarios.append(up)
        locations[("down", j)] = len(scenarios)
        scenarios.append(down)

    for pos, i in enumerate(columns):
        for j in columns[pos + 1 :]:
            for si, sj, key in (
                (1, 1, "pp"),
                (1, -1, "pm"),
                (-1, 1, "mp"),
                (-1, -1, "mm"),
            ):
                z = x.copy()
                z[i] += si * bumps[i]
                z[j] += sj * bumps[j]
                locations[(key, i, j)] = len(scenarios)
                scenarios.append(z)

    values = _evaluate_prices(pricer, np.asarray(scenarios))
    base = float(values[0])
    deltas, gammas = {}, {}
    for j in columns:
        h = bumps[j]
        p_up = values[locations[("up", j)]]
        p_down = values[locations[("down", j)]]
        deltas[j] = (p_up - p_down) / (2.0 * h)
        gammas[(j, j)] = (p_up - 2.0 * base + p_down) / h**2

    for pos, i in enumerate(columns):
        for j in columns[pos + 1 :]:
            hi, hj = bumps[i], bumps[j]
            pp = values[locations[("pp", i, j)]]
            pm = values[locations[("pm", i, j)]]
            mp = values[locations[("mp", i, j)]]
            mm = values[locations[("mm", i, j)]]
            value = (pp - pm - mp + mm) / (4.0 * hi * hj)
            gammas[(i, j)] = gammas[(j, i)] = value

    return {
        "price": base,
        "delta": deltas,
        "gamma": gammas,
        "bumps": bumps,
    }
