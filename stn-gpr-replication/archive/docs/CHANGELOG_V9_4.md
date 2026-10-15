# Version 9.4

Version 9.4 reintroduces QTT compression above the validated full-Hessian
interpolation experiment.

## Added

- `scripts/validation/validate_tt_greek_convergence.py`
  - sweeps requested TT-cross budgets from 20k to 200k;
  - repeats the retained 150k budget over five independent seeds;
  - stores every price, Delta and Hessian component;
  - decomposes total error into finite-difference, grid-interpolation and TT
    reconstruction layers with an exact closure check;
  - records effective/maximum rank, parameter count, actual oracle calls,
    fitting time, sign fidelity and PSD diagnostics;
  - checkpoints after every fit and resumes compatible result files.
- `scripts/figures/plot_tt_greek_convergence.py`
  - plots the error/rank/oracle-budget curve;
  - plots the four-layer deterministic error decomposition;
  - compares the retained Greek profiles with analytical and grid controls;
  - summarizes accuracy and cost dispersion across TT seeds.
- Unit tests for the four-layer TT error identity and the bounded coordinate.

## Tensor-domain correction

The v9.3 standardized-risk coordinate was designed and validated locally.  A
global TT tensor combines every coordinate node with every maturity, exposing
an otherwise hidden Cartesian-domain pathology.  V9.4 adds
`BoundedStandardizedRiskTransform`: the maturity-dependent standardized range
is normalized to `[-1,1]`, whose endpoints map to the original fixed
log-moneyness bounds for every rate and maturity.  This prevents artificial
extreme strikes while preserving standardized-risk concentration.

## Frozen paper protocol

- tensor shape `64^5 x 512 x 8 x 64` (48 QTT cores);
- bounded standardized-risk coordinate;
- risk-hybrid cubic interpolation;
- 0.2% fixed-strike spot bump;
- 60 deterministic out-of-sample market points;
- requested budgets 20k, 50k, 100k, 150k and 200k;
- retained budget 150k;
- seeds 20260401--20260405;
- no Monte Carlo noise.
