# Experimental protocol

## Stage 0 - Reproducibility contract

Every run records the seed, grid, pricing assumptions, TT-cross budget, number of
actual black-box evaluations, TT rank, wall time and inference time. Results from
the paper cannot be considered exactly reproduced until the authors' omitted
volatility/correlation and algorithm settings are known. Until then, the target is
to reproduce the qualitative scaling and approximate figure shapes.

## Stage 1 - Paper experiments

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

