# Version 9 — derivative-aware coordinate ablation

## Scientific objective

The v8.2 budget extension showed that increasing TT-cross effort on the
price-adaptive grid does not produce monotone or acceptable American Gamma
convergence. Version 9 therefore freezes TT compression and tests the earlier
layer where the structural error can originate: the physical coordinate grid.

The first v9 gate is deliberately analytical. It uses the closed-form
five-asset European geometric-basket put, exact fixed-strike spot Greeks and an
exact grid-node pricing oracle. This removes LSMC noise and TT reconstruction
error before any new coordinate is accepted.

## Equal-cost candidates

Every candidate uses the same physical tensor shape, spot axes, rate axis,
maturity axis, test points, cubic interpolation rule and 0.2% central spot
bump. Only the sixth physical coordinate changes.

1. `m_uniform`: uniform log-moneyness.
2. `price_adaptive`: the existing sinh concentration around `m=0`.
3. `gamma_monitor`: equidistribution of
   `floor + max_T normalized sqrt(abs(d4u/dm4))`. For cubic interpolation,
   second-derivative error is proportional to `h^2 |d4u/dm4|`, which motivates
   the square-root node-density monitor. Its design maturities are declared in
   advance and differ from the validation maturities.
4. `standardized_risk`: a uniform grid in

   `z = asinh((m - m_ridge(r,T)) / (sigma_B sqrt(T)))`.

   This follows both the maturity-dependent location and width of the
   convexity ridge. The inverse map is applied after each market-space bump, so
   all reported spot Greeks hold strike fixed.

All candidates retain a rectangular tensor-product domain. The standardized
map therefore has a maturity-dependent implied log-moneyness range; this is
reported explicitly in the JSON and must be considered before QTT compression.

## Error separation

The output contains three distinct layers:

- analytical price, Delta and Hessian;
- the same finite-difference stencil applied directly to exact prices;
- the stencil applied to cubic grid interpolation.

The second layer measures bump truncation. The difference between the third and
second layers isolates grid interpolation under the selected stencil. The
validator also records conditional errors by log-moneyness and maturity,
pointwise errors, local implied `m` spacing, cache cost, Hessian Frobenius error
and convexity/PSD violations.

## Included smoke and intermediate checks

The bundled results are implementation checks, not final paper estimates. In
the smoke profile (64 coordinate nodes), the Gamma monitor substantially
improves the joint Greek score. In the intermediate profile (256 coordinate
nodes), the standardized-risk coordinate ranks first on the joint score and on
diagonal and cross-Gamma accuracy. This resolution-dependent crossover is a
useful result to validate with the paper profile and out-of-sample designs.

TT compression should be reintroduced only after the grid-only paper profile,
a bump check and out-of-sample maturity/product tests confirm the selected
coordinate geometry.
