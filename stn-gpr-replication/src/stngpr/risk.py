from __future__ import annotations

import numpy as np


def var_es(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """Empirical VaR and ES using the loss-positive convention."""
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    var = float(np.quantile(losses, alpha, method="higher"))
    tail = losses[losses >= var]
    return var, float(np.mean(tail))


def revaluation_losses(pricer, current_parameters, scenario_parameters, positions=None):
    """Full-revaluation P&L/losses for one or several trades."""
    current = np.atleast_2d(np.asarray(current_parameters, dtype=float))
    scenarios = np.asarray(scenario_parameters, dtype=float)
    if scenarios.ndim == 2:
        scenarios = scenarios[:, None, :]
    n_trades = scenarios.shape[1]
    weights = np.ones(n_trades) if positions is None else np.asarray(positions, dtype=float)
    v0 = np.asarray(pricer(current), dtype=float)
    vt = np.column_stack([pricer(scenarios[:, j, :]) for j in range(n_trades)])
    pnl = (vt - v0[None, :]) @ weights
    return -pnl

