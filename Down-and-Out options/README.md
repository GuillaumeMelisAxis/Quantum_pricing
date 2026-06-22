# Discrete Hamiltonian Models for Barrier Option Pricing — Numerical Study

Numerical companion to the paper *"Construction of a discrete Hamiltonian for
Down-and-Out option pricing and study of its quantum implementation"* (G. Melis, 2026).

## Layout

```
src/
  hamiltonian_pricer.py    Part 1 - HamiltonianPricer (classical engine)
  monte_carlo_pricer.py    Part 2 - MonteCarloPricer (GBM benchmark)
  analytic_pricing.py      Closed-form Black-Scholes / continuous Down-and-Out benchmarks
  experiments.py           Part 3 - Experiments 1-4
  hybrid_pricer.py         Part 4 - HybridHamiltonianPricer (PennyLane)
  hybrid_validation.py     Part 5 - Hybrid vs classical validation
  run_all.py               Single entry point reproducing every table/figure
notebooks/
  numerical_study.ipynb    Reproducible, executed notebook walking through all parts
tables/                    CSV tables (Experiments 1-4, Part 5, timing comparison)
figures/                   PNG figures (Experiments 2, 3, 4)
```

Reproduce everything with:
```
cd src && python run_all.py
```
or open `notebooks/numerical_study.ipynb`.

## What was implemented

**Part 1 (`HamiltonianPricer`)** — builds the log-price grid, the sparse tridiagonal
discrete Hamiltonian `H_h`, the diagonal barrier projector `P_B`, and runs the
recursion `U^{k+1} = P_B exp(-dt_k H_h) U^k` via `scipy.sparse.linalg.expm_multiply`,
for arbitrary (daily/weekly/monthly/continuous) monitoring schedules.

**Part 2 (`MonteCarloPricer`)** — exact lognormal GBM path simulation at the
monitoring dates, Down-and-Out Call payoff, price/std-error/CI, scalable to
100k–1M paths.

**Part 3 (Experiments 1–4)** — see `tables/` and `figures/` for full numerical
output; headline results:
- Experiment 1: Hamiltonian price matches vanilla Black-Scholes and the
  closed-form continuously-monitored Down-and-Out price to ≈0.15–0.18% relative
  error.
- Experiment 2: Hamiltonian vs Monte Carlo across `B/S0 ∈ {0.70, 0.80, 0.90, 0.95}`
  agree at the sub-2% level (within MC confidence intervals).
- Experiment 3: spatial convergence is close to, but slightly below, the
  theoretical second order (the projector's interaction with grid resolution adds
  some non-monotonic noise on top of the underlying O(h²) Hamiltonian truncation
  error — see the discussion cell in the notebook).
- Experiment 4: the discretely-monitored price converges monotonically toward
  the continuous-monitoring closed form as monitoring frequency increases.

**Part 4 (`HybridHamiltonianPricer`)** — Pauli-decomposes `H_h` into a PennyLane
`Hamiltonian`, amplitude-encodes the state into `n = log2(N-1)` qubits, and
simulates `exp(-dt H_h)` via `qml.ApproxTimeEvolution` using a Wick rotation
(`t = -i·dt`) to turn the framework's native unitary evolution into the real
(imaginary-time) decay needed for pricing. The barrier projector is applied
classically between monitoring dates.

**Part 5 (hybrid validation)** — for `N = 16, 32, 64` (4–6 qubits), the hybrid
pricer matches the classical sparse-matrix pricer to within ≈0.5% relative
error; runtime on the classical quantum simulator grows rapidly with qubit
count, consistent with the paper's framing (demonstrating a natural mapping
to Hamiltonian-simulation primitives, not a quantum-advantage claim).

## Two implementation pitfalls worth flagging for the paper

1. **Similarity transform.** `H_h` as defined in the paper (`-σ²/2 Δh + σ²/2 β I`)
   is the *symmetrized* form of the Black-Scholes Hamiltonian; it omits the
   first-derivative drift term. Pricing with it directly (without correction)
   is biased by several percent. The correct procedure is to work with
   `C(x,τ) = e^{αx} φ(x,τ)`, `α = (σ²/2 - r)/σ²`: weight the terminal payoff by
   `e^{-αx}` before evolving under `H_h`, and reweight by `e^{αx}` after. This is
   implemented in `HamiltonianPricer._terminal_payoff` / `_interpolate_price` and
   is worth stating explicitly in Section 3 of the paper, since as written the
   reader could plausibly apply `H_h` directly to the unweighted payoff.
2. **PennyLane drops the identity term.** `ApproxTimeEvolution` discards the
   Pauli-identity component of the Hamiltonian during Trotterization, since it is
   an unobservable global phase under genuine unitary evolution. For the
   real-decay (imaginary-time) use case here, that component carries a genuine
   multiplicative normalization and must be reapplied by hand — see the comment
   block at the top of `hybrid_pricer.py`.

## Dependencies
`numpy`, `scipy`, `pandas`, `matplotlib`, `pennylane` (tested with PennyLane 0.45).
