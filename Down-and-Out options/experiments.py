"""
experiments.py
================

Part 3 - Numerical Validation.

Experiment 1: Hamiltonian vs analytical Black-Scholes (no-barrier limit)
             and vs the closed-form continuously-monitored Down-and-Out
             Call price.
Experiment 2: Hamiltonian vs Monte Carlo for Down-and-Out Calls across
             barrier levels B/S0 = 0.70, 0.80, 0.90, 0.95.
Experiment 3: Spatial convergence study (N = 32,...,512), reference at
             N = 2048, log-log convergence plot, estimated order p.
Experiment 4: Impact of monitoring frequency (monthly/weekly/daily) and
             convergence toward the continuously monitored limit.

Author: Guillaume Melis
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hamiltonian_pricer import HamiltonianPricer
from monte_carlo_pricer import MonteCarloPricer
from analytic_pricing import bs_call_price, down_and_out_call_price

# ----------------------------------------------------------------------
# Common market parameters used throughout the numerical study
# ----------------------------------------------------------------------
MARKET = dict(S0=100.0, K=100.0, sigma=0.20, r=0.03, T=1.0)
FIG_DIR = "figures"
TAB_DIR = "tables"


def _savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
# Experiment 1
# ----------------------------------------------------------------------
def experiment_1(N: int = 1000, n_std: float = 10.0) -> pd.DataFrame:
    """Validate the Hamiltonian framework against analytical benchmarks.

    Row 1: barrier pushed to ~0 (no-barrier limit) vs vanilla BS call.
    Row 2: barrier at B=80 vs the closed-form continuously-monitored
           Down-and-Out Call (Hamiltonian run with daily monitoring as
           a practical proxy for continuous monitoring).
    """
    S0, K, sigma, r, T = MARKET["S0"], MARKET["K"], MARKET["sigma"], MARKET["r"], MARKET["T"]

    rows = []

    # --- no-barrier limit vs vanilla Black-Scholes ---
    hp = HamiltonianPricer(S0=S0, K=K, B=1e-6, sigma=sigma, r=r, T=T, N=N, n_std=n_std)
    v_h = hp.price(frequency="daily")
    v_bs = bs_call_price(S0, K, r, sigma, T)
    rows.append(
        {
            "case": "Vanilla call (B -> 0)",
            "V_Hamiltonian": v_h,
            "V_Benchmark": v_bs,
            "abs_error": abs(v_h - v_bs),
            "rel_error": abs(v_h - v_bs) / v_bs,
        }
    )

    # --- Down-and-Out vs closed-form continuous-monitoring benchmark ---
    B = 80.0
    hp_do = HamiltonianPricer(S0=S0, K=K, B=B, sigma=sigma, r=r, T=T, N=N, n_std=n_std)
    v_h_do = hp_do.price(frequency="daily")
    v_bs_do = down_and_out_call_price(S0, K, B, r, sigma, T)
    rows.append(
        {
            "case": "Down-and-Out (B=80, daily proxy for continuous)",
            "V_Hamiltonian": v_h_do,
            "V_Benchmark": v_bs_do,
            "abs_error": abs(v_h_do - v_bs_do),
            "rel_error": abs(v_h_do - v_bs_do) / v_bs_do,
        }
    )

    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/experiment1_bs_validation.csv", index=False)
    return df


# ----------------------------------------------------------------------
# Experiment 2
# ----------------------------------------------------------------------
def experiment_2(
    barrier_ratios=(0.70, 0.80, 0.90, 0.95),
    N: int = 800,
    n_mc_paths: int = 200_000,
    frequency: str = "daily",
    seed: int = 7,
) -> pd.DataFrame:
    """Compare V_Hamiltonian and V_MonteCarlo for Down-and-Out Calls
    across several barrier levels."""
    S0, K, sigma, r, T = MARKET["S0"], MARKET["K"], MARKET["sigma"], MARKET["r"], MARKET["T"]
    rows = []
    for ratio in barrier_ratios:
        B = ratio * S0
        hp = HamiltonianPricer(S0=S0, K=K, B=B, sigma=sigma, r=r, T=T, N=N, n_std=10)
        v_h = hp.price(frequency=frequency)

        mc = MonteCarloPricer(S0=S0, K=K, B=B, sigma=sigma, r=r, T=T, seed=seed)
        res = mc.price(n_paths=n_mc_paths, frequency=frequency)

        rel_err = abs(v_h - res.price) / res.price
        rows.append(
            {
                "B/S0": ratio,
                "Hamiltonian": v_h,
                "MonteCarlo": res.price,
                "MC_StdError": res.std_error,
                "MC_CI_low": res.ci_low,
                "MC_CI_high": res.ci_high,
                "RelativeError": rel_err,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(f"{TAB_DIR}/experiment2_mc_validation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["B/S0"], df["Hamiltonian"], "o-", label="Hamiltonian")
    ax.errorbar(
        df["B/S0"],
        df["MonteCarlo"],
        yerr=1.96 * df["MC_StdError"],
        fmt="s--",
        label="Monte Carlo (95% CI)",
        capsize=4,
    )
    ax.set_xlabel("B / S0")
    ax.set_ylabel("Down-and-Out Call price")
    ax.set_title("Hamiltonian vs Monte Carlo across barrier levels")
    ax.legend()
    _savefig(fig, "experiment2_hamiltonian_vs_mc")

    return df


# ----------------------------------------------------------------------
# Experiment 3
# ----------------------------------------------------------------------
def experiment_3(
    Ns=(32, 64, 128, 256, 512, 1024),
    N_ref: int = 2048,
    B_ratio: float = 0.8,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Spatial convergence study. Computes |V_N - V_ref| for a sequence
    of grid sizes N, estimates the convergence order p via a log-log
    least-squares fit, and produces the convergence table/plot."""
    S0, K, sigma, r, T = MARKET["S0"], MARKET["K"], MARKET["sigma"], MARKET["r"], MARKET["T"]
    B = B_ratio * S0

    hp_ref = HamiltonianPricer(S0=S0, K=K, B=B, sigma=sigma, r=r, T=T, N=N_ref, n_std=10)
    V_ref = hp_ref.price(frequency=frequency)

    rows = []
    for N in Ns:
        hp = HamiltonianPricer(S0=S0, K=K, B=B, sigma=sigma, r=r, T=T, N=N, n_std=10)
        V_N = hp.price(frequency=frequency)
        h = hp.h
        err = abs(V_N - V_ref)
        rows.append({"N": N, "h": h, "V_N": V_N, "error": err})

    df = pd.DataFrame(rows)
    # estimate convergence order p via error ~ C h^p  =>  log(err) = log(C) + p log(h)
    log_h = np.log(df["h"].values)
    log_err = np.log(df["error"].values)
    p, logC = np.polyfit(log_h, log_err, 1)

    df.attrs["V_ref"] = V_ref
    df.attrs["N_ref"] = N_ref
    df.attrs["p_estimate"] = p
    df.to_csv(f"{TAB_DIR}/experiment3_convergence.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(df["h"], df["error"], "o-", label="numerical error")
    fit_h = np.linspace(df["h"].min(), df["h"].max(), 50)
    ax.loglog(fit_h, np.exp(logC) * fit_h ** p, "--", label=f"fit: O(h^{p:.2f})")
    ax.set_xlabel("mesh size h")
    ax.set_ylabel(r"$|V_N - V_{ref}|$")
    ax.set_title("Spatial convergence of the discrete Hamiltonian scheme")
    ax.legend()
    ax.invert_xaxis()
    _savefig(fig, "experiment3_convergence_loglog")

    with open(f"{TAB_DIR}/experiment3_summary.txt", "w") as f:
        f.write(f"V_ref (N={N_ref}) = {V_ref:.6f}\n")
        f.write(f"Estimated convergence order p = {p:.4f} (expected p ~ 2)\n")

    return df


# ----------------------------------------------------------------------
# Experiment 4
# ----------------------------------------------------------------------
def experiment_4(
    B_ratio: float = 0.8,
    N: int = 800,
    n_std: float = 10.0,
) -> pd.DataFrame:
    """Study the impact of monitoring frequency: monthly / weekly /
    daily, plus a high-frequency proxy for continuous monitoring, and
    compare to the closed-form continuously-monitored benchmark."""
    S0, K, sigma, r, T = MARKET["S0"], MARKET["K"], MARKET["sigma"], MARKET["r"], MARKET["T"]
    B = B_ratio * S0

    hp = HamiltonianPricer(S0=S0, K=K, B=B, sigma=sigma, r=r, T=T, N=N, n_std=n_std)

    freqs = [("monthly", None), ("weekly", None), ("daily", None), ("continuous(2520)", 2520)]
    rows = []
    for label, n_mon in freqs:
        freq_arg = "continuous" if n_mon is not None else label
        V = hp.price(frequency=freq_arg, n_monitoring=n_mon)
        M = n_mon if n_mon is not None else {"monthly": 12, "weekly": 52, "daily": 252}[label]
        rows.append({"schedule": label, "M": M, "price": V})

    V_continuous_benchmark = down_and_out_call_price(S0, K, B, r, sigma, T)
    df = pd.DataFrame(rows)
    df.attrs["V_continuous_benchmark"] = V_continuous_benchmark
    df.to_csv(f"{TAB_DIR}/experiment4_monitoring_frequency.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["M"], df["price"], "o-", label="Discretely monitored Hamiltonian price")
    ax.axhline(
        V_continuous_benchmark, color="red", linestyle="--", label="Continuous-monitoring closed form"
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of monitoring dates M (log scale)")
    ax.set_ylabel("Down-and-Out Call price")
    ax.set_title("Convergence toward the continuously-monitored barrier price")
    ax.legend()
    _savefig(fig, "experiment4_monitoring_frequency")

    return df


if __name__ == "__main__":
    print("Running Experiment 1...")
    print(experiment_1())
    print("\nRunning Experiment 2...")
    print(experiment_2())
    print("\nRunning Experiment 3...")
    print(experiment_3())
    print("\nRunning Experiment 4...")
    print(experiment_4())


# ----------------------------------------------------------------------
# Extra check (requested follow-up): |V_Hamiltonian - V_MonteCarlo| vs N
# ----------------------------------------------------------------------
def hamiltonian_vs_mc_stabilization(
    Ns=(32, 64, 128, 256, 512),
    B_ratio: float = 0.8,
    frequency: str = "daily",
    n_mc_paths: int = 1_000_000,
    mc_seed: int = 2026,
) -> pd.DataFrame:
    """For a fixed (single, large-sample) Monte Carlo benchmark, compute
    |V_Hamiltonian(N) - V_MonteCarlo| as the spatial grid is refined.

    This isolates the *residual* gap to the Monte Carlo benchmark once
    the Hamiltonian scheme has spatially converged: that residual is
    expected to stabilize around the Monte Carlo standard error (it
    reflects MC sampling noise / discrete-monitoring vs scheme
    differences, not further reducible Hamiltonian discretization
    error).
    """
    S0, K, sigma, r, T = MARKET["S0"], MARKET["K"], MARKET["sigma"], MARKET["r"], MARKET["T"]
    B = B_ratio * S0

    # one single, large, fixed Monte Carlo benchmark
    mc = MonteCarloPricer(S0=S0, K=K, B=B, sigma=sigma, r=r, T=T, seed=mc_seed)
    mc_res = mc.price(n_paths=n_mc_paths, frequency=frequency)
    V_MC = mc_res.price

    rows = []
    for N in Ns:
        hp = HamiltonianPricer(S0=S0, K=K, B=B, sigma=sigma, r=r, T=T, N=N, n_std=10)
        V_H = hp.price(frequency=frequency)
        rows.append(
            {
                "N": N,
                "h": hp.h,
                "V_Hamiltonian": V_H,
                "V_MonteCarlo": V_MC,
                "abs_gap": abs(V_H - V_MC),
                "MC_std_error": mc_res.std_error,
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["V_MonteCarlo"] = V_MC
    df.attrs["MC_std_error"] = mc_res.std_error
    df.attrs["MC_n_paths"] = n_mc_paths
    df.to_csv(f"{TAB_DIR}/extra_hamiltonian_vs_mc_stabilization.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["N"], df["abs_gap"], "o-", label=r"$|V_H(N) - V_{MC}|$")
    ax.axhline(
        1.96 * mc_res.std_error,
        color="red",
        linestyle="--",
        label="MC 95% half-width (1.96 SE)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (spatial grid size, log scale)")
    ax.set_ylabel(r"$|V_H - V_{MC}|$")
    ax.set_title("Hamiltonian-MonteCarlo gap vs spatial resolution")
    ax.legend()
    _savefig(fig, "extra_hamiltonian_vs_mc_stabilization")

    return df