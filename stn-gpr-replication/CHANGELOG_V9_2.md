# Version 9.2 — finite-difference and interpolation error floor

## Added

- `decompose_geometric_greek_error_floor.py`, a grid-only analytical gate with
  fixed `standardized_risk` coordinates and 64 nodes on each spot axis;
- independent spot panels not used by the v9.1 resolution ablation;
- relative-bump ablations for exact-price and interpolated finite differences;
- multilinear versus risk-hybrid cubic interpolation;
- a common-scale additive decomposition for price, Delta, diagonal Gamma and
  cross-Gamma;
- conditional diagnostics by maturity, central moneyness region and spot panel;
- pointwise arrays and four publication-oriented figure families;
- unit tests for decomposition closure and common analytical normalization.

## Scientific contract

All spot bumps hold strike fixed.  The standardized coordinate is recomputed
after every bump, so neither log-moneyness nor standardized risk is held fixed.
TT compression and Monte Carlo noise are deliberately absent.  The experiment
therefore decides whether the v9.1 Hessian plateau is caused by the bump,
interpolation, or cancellation between the two before a TT budget is spent.
