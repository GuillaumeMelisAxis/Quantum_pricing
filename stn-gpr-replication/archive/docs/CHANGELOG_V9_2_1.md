# Version 9.2.1 — dense Delta/Gamma visual validation

## Added

- dense log-moneyness profiles with 41, 101 or 201 points per curve;
- exact 3-, 7-, 30- and 90-day market maturities in the paper profile;
- pointwise Delta, diagonal-Gamma and cross-Gamma arrays;
- analytical and exact-price finite-difference controls;
- curve-level MAE, RMSE, 95%/99% quantiles and maximum-error locations;
- total-variation ratios and turning-point counts to flag cubic oscillations;
- raw Hessian shape diagnostics on every `(panel, maturity)` curve;
- two-row figures showing Greek levels and log-scale absolute errors;
- an automatic hybrid-only zoom whose axes are not distorted by the
  multilinear control's large oscillations;
- a unit test confirming that the shape diagnostic detects an artificial
  oscillation.

## Scope

The physical grid remains `64^5 x 512 x 8 x 64` in standardized-risk
coordinates.  Spot bumps hold strike fixed.  TT compression and Monte Carlo
noise remain disabled so the figures isolate the interpolation layer.
