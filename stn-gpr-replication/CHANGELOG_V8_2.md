# Version 8.2

Version 8.2 freezes the validated American reference settings and introduces a
paper-oriented error decomposition for fixed-strike spot Greeks.

## American reference

The intermediate and paper profiles use the validated reference configuration:

- 50,000 paths
- 80 exercise dates
- eight independent seeds
- fully refitted LSMC policy after every spot bump
- 3% relative bump
- 2.5%-3.5% retained as the empirical robustness interval

The supporting bump-plateau and exercise-step convergence JSON files are kept
under `results/reference_validation/`.

## Additive error decomposition

`scripts/validation/decompose_american_tt_errors.py` separates

1. LSMC training-label error relative to the independent reference
2. adaptive-grid interpolation error relative to direct training labels
3. TT reconstruction error relative to the exact grid interpolant
4. total TT error relative to the independent reference

The signed arrays satisfy, up to floating-point precision,

```text
TT - reference
  = training - reference
  + grid - training
  + TT - grid
```

The identity is checked for price, Delta, diagonal Gamma and cross-Gamma.

## Exact grid layer

`CachedGridInterpolator` evaluates every required physical grid node with the
same deterministic training oracle used by TT-cross. Repeated stencil nodes are
cached, so the grid layer contains interpolation error but no TT error.

## Compression diagnostics

TT outputs now include the effective rank, maximum rank, number of TT
parameters, oracle evaluations and compression ratios relative to the logical
tensor size.

## Figures

`plot_american_error_decomposition.py` produces a four-panel figure containing
the error layers, TT budget convergence, compression cost and PSD diagnostics.
