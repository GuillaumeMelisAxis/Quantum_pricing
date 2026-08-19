# Experimental protocol

## Stage 0 - Reproducibility contract

Every run records the seed, grid, pricing assumptions, TT-cross budget, number of
actual black-box evaluations, TT rank, wall time and inference time. Results from
the paper cannot be considered exactly reproduced until the authors' omitted
volatility/correlation and algorithm settings are known. Until then, the target is
to reproduce the qualitative scaling and approximate figure shapes.

## Stage 1 - Paper experiments

The paper grid remains the strict reproduction baseline. Two extensions keep the
same 37-core QTT shape: a uniform log-moneyness coordinate and a sinh-adaptive
log-moneyness coordinate concentrated at ATM, with a quadratic maturity grid.
Every European run reports an oracle grid interpolator so TT-cross error can be
separated from irreducible off-grid interpolation error.

### European geometric basket put

- Inputs: `(S1, S2, S3, S4, S5, K, r, T)`.
- Grid: `32^5 x 64 x 8 x 8`; QTT shape `[2] * 37`.
- Labels: closed-form geometric-basket price.
- Test set: 1,000 continuous points, uniformly sampled over paper ranges.
- TT: order-2, rank-2 TT-ANOVA initialization; adaptive TT-cross.
- Baseline: exact noise-free GPR with `exp(-||x-x'||_1/L)`.
- Metrics: MAE, RMSE, p95 absolute error, maximum error, build time and query time.

### American arithmetic basket put

- Labels: LSMC with 10,000 paths and 30 time steps.
- Primary run: common random numbers (CRN) to expose interpolation error rather
  than discontinuous Monte Carlo noise.
- Robustness run: independent seeds and repeated labels to measure noise floor.
- Compare TT and GPR at equal black-box call budgets as well as equal wall time.

### Final American grid robustness gate

- Compare `moneyness_adaptive_uniform_maturity` and `moneyness_adaptive` on the
  same 200 stratified test points.
- Repeat every budget (7,500, 9,000 and 12,000 evaluations) with five paired TT
  initialization seeds.
- Keep the LSMC oracle, test points and independent reference identical across
  the paired runs.
- Report mean, standard deviation, median, minimum and maximum MAE across seeds,
  together with ATM MAE, rank and paired adaptive-grid win rate.
- Accept the adaptive maturity refinement as robust only if it wins for at least
  four seeds out of five at 9,000 and 12,000 evaluations, has positive mean
  paired improvement, and its worst-seed MAE is no more than twice its median.

## Stage 2 - Greeks

The trusted and surrogate pricers are bumped with identical central-difference
stencils. Bump convergence is checked at relative spot bumps `2e-2`, `1e-2`,
`5e-3`, `2e-3` and `1e-3`.

Outputs:

- five deltas;
- five diagonal gammas;
- ten distinct cross-gammas;
- error conditional on moneyness and maturity;
- error versus distance to the exercise boundary;
- no-arbitrage/homogeneity checks where applicable.

For the European geometric basket, analytic or automatic reference Greeks should
eventually replace finite differences. Finite differences remain useful because
they test the deployed surrogate exactly as a risk engine would use it.

### Short-maturity ATM resolution gate

Before fitting another TT, isolate the deterministic grid floor on a panel with
log-moneyness `[-0.05, -0.025, 0, 0.025, 0.05]` and maturities
`[3, 7, 14, 30, 90]` days. Cross the physical resolutions
`n_m in [64, 128, 256]` and `n_T in [8, 16, 32]` while leaving every other
axis, market point and finite-difference bump unchanged.

The experiment reports errors by exact maturity and moneyness level, with
special summaries for the seven-day ATM layer. It also records
`Delta m_ATM / (sigma_B sqrt(T))`. This distinguishes insufficient temporal
resolution from the structural inability of a maturity-independent moneyness
grid to resolve the shrinking short-dated convexity layer.

Every estimated Gamma matrix is also projected onto the positive-semidefinite
cone by clipping its negative eigenvalues. Raw and projected errors are both
reported, together with negative diagonal counts, non-PSD matrix counts,
minimum eigenvalues and Frobenius projection errors. For a contract whose true
price is convex in the spot vector, the PSD projection is the nearest admissible
Hessian in Frobenius norm and cannot increase its Frobenius error.

Once the deterministic grid gate is passed, fit the TT directly on the selected
`512 x 64` moneyness-maturity resolution. The paired convergence test compares
budgets 20,000, 50,000 and 100,000 against both the analytical Greeks and the
exact-grid oracle. It reports reconstruction-only errors separately from the
irreducible interpolation floor and applies the same PSD projection to both
oracle and TT Hessians.

```bash
python scripts/validation/validate_refined_tt_greeks.py \
    --moneyness-nodes 512 \
    --maturity-nodes 64 \
    --budgets 20000 50000 100000 \
    --anova-samples 2000 \
    --replicates 3 \
    --relative-bump 0.002
```

```bash
python scripts/validation/validate_short_maturity_greeks.py \
    --replicates 3 \
    --moneyness-nodes 64 128 256 \
    --maturity-nodes 8 16 32 \
    --relative-bump 0.002
```

## Stage 3 - Exercise region

For an American put, define the exercise premium

`EP(x) = V_american(x) - intrinsic(x)`.

The estimated boundary is the level set `EP(x) = 0`, with a numerical tolerance
tied to the LSMC standard error. Tests are performed on:

1. the diagonal slice `S1 = ... = S5`;
2. one-asset shocks with the other spots fixed;
3. principal-component basket shocks;
4. random two-dimensional slices through the 5D spot space.

Metrics are boundary-location error, exercise/continue classification error and
false-exercise rate. Adaptive TT-cross sampling will then be compared with the
uniform grid by allocating extra calls where `|EP|` is small.

## Stage 4 - VaR and Expected Shortfall

Construct portfolios of heterogeneous strikes/maturities and generate correlated
risk-factor scenarios. Compare trusted full revaluation with TT revaluation at
95%, 97.5% and 99%:

- VaR and ES absolute/relative error;
- P&L distribution Wasserstein distance;
- Spearman rank correlation of scenario losses;
- overlap of the worst 1% scenario sets;
- runtime and break-even number of revaluations including surrogate build cost.

Two portfolio constructions are tested separately:

- one surrogate per trade followed by TT/price aggregation;
- one portfolio-level surrogate built directly from the aggregate black-box
  pricing functional.

The second construction must be rebuilt after position changes, so the operational
break-even point is part of the result rather than assumed.
