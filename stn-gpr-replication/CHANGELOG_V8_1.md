# Version 8.1 - Reference convergence and Greek-noise ablations

## Added

- `converge_european_arithmetic_reference.py`: joint Sobol path-count,
  finite-difference bump and control-variate-coefficient convergence study.
- `converge_american_reference.py`: joint LSMC path-count, exercise-step, bump
  and exercise-policy convergence study on a reduced nine-price Greek stencil.
- `plot_reference_convergence.py`: publication-ready normalized-standard-error
  curves for the European and American reference experiments.
- `run_greeks_v81_smoke.bat`: one-command Windows smoke campaign.
- `stngpr.convergence`: shared deterministic panel, selected-risk stencil,
  component uncertainty, Hessian Frobenius uncertainty and PSD diagnostics.

## European arithmetic correction

`EuropeanArithmeticBasketQMC` now accepts
`control_variate_beta="estimated"`, `"unit"` or a finite scalar. The `unit`
mode freezes the coefficient across every bumped scenario and isolates the
Greek noise introduced when the sample-optimal coefficient is re-estimated.

## American arithmetic ablations

`AmericanArithmeticBasketLSMC` now exposes three explicit policy contracts:

- `refit`: regress continuation values again after every bump;
- `frozen`: fit on the base scenario and reuse the policy on the same paths;
- `frozen_oos`: fit on the base scenario and evaluate on an independent path
  sample.

Frozen-policy outputs are diagnostic variance ablations. They are not declared
unbiased replacements for the standard refit estimator. The scripts report
their bias against the highest-resolution refit anchor before any selection.

## Reproducibility

- Every run retains component means and standard errors across seeds.
- Successive path-count differences, bump stability and policy/beta-mode
  comparisons are recorded in the JSON.
- Hessian uncertainty uses the Frobenius norm of componentwise standard errors
  and is reported alongside negative diagonals and non-PSD matrix counts.
- The smoke profiles are intentionally small plumbing gates; use the
  intermediate profiles for the first scientific comparison.
