# Version 9.3

Version 9.3 promotes the selected v9.2 risk-adaptive interpolation rule from a
two-factor diagnostic to the complete five-asset spot risk tensor.

## Added

- `scripts/validation/validate_full_spot_hessian.py`
  - computes all five fixed-strike spot Deltas;
  - computes all five diagonal Gammas and ten cross-Gammas;
  - compares the 5x5 Hessian with analytical and exact-price finite-difference
    controls;
  - reports per-component normalized errors, aggregate Frobenius error,
    cross-Gamma sign agreement and material PSD defects;
  - uses 51 price evaluations per market point.
- `scripts/figures/plot_full_spot_hessian.py`
  - plots Delta and Hessian component errors;
  - compares analytical and interpolated ATM Hessian matrices;
  - plots the minimum Hessian eigenvalue across moneyness.
- Unit tests for columnwise, sign-fidelity and full-Hessian diagnostics.

## Frozen numerical choices

- 64 nodes on every spot axis;
- 512 standardized-risk nodes;
- 64 maturity nodes;
- risk-hybrid cubic interpolation;
- 0.2% relative fixed-strike bump;
- no TT compression and no Monte Carlo noise.

TT compression is deliberately deferred to v9.4 so that any subsequent error
can be attributed to reconstruction rather than to an incompletely validated
grid or Greek stencil.
