"""
hybrid_validation.py
======================

Part 5 - Hybrid Validation.

Compares V_Hybrid (PennyLane-simulated quantum propagator + classical
barrier monitoring) against V_Classical (sparse matrix-exponential
propagator) for N = 16, 32, 64 interior grid points, reporting prices,
relative error and timing for each method.

Author: Guillaume Melis
"""

from __future__ import annotations

import time

import pandas as pd

from hamiltonian_pricer import HamiltonianPricer
from hybrid_pricer import HybridHamiltonianPricer

MARKET = dict(S0=100.0, K=100.0, B=80.0, sigma=0.20, r=0.03, T=1.0)
TAB_DIR = "tables"


def hybrid_validation(
    dims=(16, 32, 64),
    frequency: str = "monthly",
    n_trotter_steps: int = 8,
) -> pd.DataFrame:
    rows = []
    for dim in dims:
        N = dim + 1

        t0 = time.time()
        hp = HamiltonianPricer(N=N, n_std=10, **MARKET)
        V_classical = hp.price(frequency=frequency)
        t_classical = time.time() - t0

        t0 = time.time()
        hyb = HybridHamiltonianPricer(N=N, n_trotter_steps=n_trotter_steps, **MARKET)
        V_hybrid = hyb.price(frequency=frequency)
        t_hybrid = time.time() - t0

        rel_err = abs(V_hybrid - V_classical) / abs(V_classical)
        rows.append(
            {
                "N": dim,
                "n_qubits": hyb.n_qubits,
                "Classical": V_classical,
                "Hybrid": V_hybrid,
                "RelativeError": rel_err,
                "Time_Classical_s": t_classical,
                "Time_Hybrid_s": t_hybrid,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/part5_hybrid_validation.csv", index=False)
    return df


if __name__ == "__main__":
    print(hybrid_validation())
