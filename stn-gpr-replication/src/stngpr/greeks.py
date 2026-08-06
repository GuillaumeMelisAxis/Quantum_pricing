from __future__ import annotations

import numpy as np


def finite_difference_greeks(pricer, x, spot_columns, relative_bump=1e-3):
    """Central delta, gamma and cross-gamma for a vectorized price callable."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one parameter vector")
    base = float(pricer(x[None, :])[0])
    deltas, gammas = {}, {}
    bumps = {j: max(abs(x[j]) * relative_bump, 1e-6) for j in spot_columns}
    for j in spot_columns:
        h = bumps[j]
        up, dn = x.copy(), x.copy()
        up[j] += h
        dn[j] -= h
        p_up, p_dn = pricer(np.vstack((up, dn)))
        deltas[j] = (p_up - p_dn) / (2.0 * h)
        gammas[(j, j)] = (p_up - 2.0 * base + p_dn) / h**2
    for pos, i in enumerate(spot_columns):
        for j in spot_columns[pos + 1 :]:
            hi, hj = bumps[i], bumps[j]
            scenarios = []
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                z = x.copy()
                z[i] += si * hi
                z[j] += sj * hj
                scenarios.append(z)
            pp, pm, mp, mm = pricer(np.asarray(scenarios))
            value = (pp - pm - mp + mm) / (4.0 * hi * hj)
            gammas[(i, j)] = gammas[(j, i)] = value
    return {"price": base, "delta": deltas, "gamma": gammas}

