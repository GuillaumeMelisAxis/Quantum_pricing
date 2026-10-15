# Version 8 - Cross-product Greeks experiments

## Added

- `EuropeanArithmeticBasketQMC`: deterministic scrambled Sobol-QMC pricing with
  common random numbers and an exact geometric-basket control variate.
- `validate_european_arithmetic_greeks.py`: independent multi-scramble
  reference, training-label diagnostics, exact-grid floor, TT reconstruction,
  raw and PSD-projected Hessians.
- `validate_american_greeks.py`: independent multi-seed LSMC Greek reference,
  same-seed reconstruction error, total error, confidence-band coverage and
  product-specific TT budget sweeps.
- `plot_greeks_product_comparison.py`: component curves, `(m,T)` error heat maps
  and accuracy/rank/evaluation summaries in PNG and PDF.
- `run_greeks_v8_smoke.bat`: one-command Windows x64 Native Tools smoke gate.

## Preserved

- Market coordinate order `(S1,S2,S3,S4,S5,K,r,T)`.
- Fixed-strike spot bumps with `m = log(K/B(S))` recomputed after each bump.
- Four-node local cubic interpolation on bumped spot axes and moneyness.
- Raw Hessians retained before optional nearest-PSD projection.
- Frozen paper configuration `32^5 x 512 x 8 x 64` and 0.2% relative bump.

## Important interpretation

The European geometric result remains the only analytical Greek benchmark.
European arithmetic results carry QMC uncertainty, while American arithmetic
results additionally carry LSMC discretization and exercise-policy noise. The
three products must therefore be compared with their reference uncertainty and
oracle cost, not by normalized MAE alone.
