# Version 9.8 — final pricing-grid versus risk-grid comparison

The final experiment is restricted to the analytically controlled European
geometric-basket put. It compares two grids:

1. the paper-I price-adaptive sinh log-moneyness grid;
2. the bounded standardized-risk grid selected for coherent Greeks.

The first stage gives both coordinates identical physical mode sizes and uses
exact grid-node prices. This isolates node placement from TT reconstruction
and architecture size. The second stage restores the native configurations
`32^5 x 64 x 8 x 8` and `64^5 x 512 x 8 x 64`, fits raw TT-cross cores with
paired seeds and derives price, Delta and Hessian from one scalar cubic surface.

Report the matched and native stages separately. The former supports a causal
coordinate statement; the latter reports the actual accuracy--complexity
trade-off. No arithmetic or American result is inferred from this benchmark.

See `V9_8_RUN_COMMANDS.md` for the resumable Windows commands.

# Version 9.7 — local Greek-aware truncation selection

V9.6 established that the bounded standardized-risk grid and a single cubic
interpolant can produce accurate price, Delta and full spot Hessian estimates.
It also showed that the `1e-8` price-norm truncation passes aggregate error
criteria while failing locally at seven days.  V9.7 locates the admissible
compression boundary.

The paper profile fits one raw 200k TT for each seed 20260401--20260405 and
applies

```text
raw, 1e-14, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8
```

to the same raw cores.  A two-stage protocol controls runtime without changing
the selection rule:

1. all seed–tolerance pairs are screened with the previously validated
   component-hybrid operator;
2. robust candidates are considered from most to least compressed and tested
   across every seed with the unified scalar cubic interpolant;
3. the first candidate passing the unified global and local gates is selected;
4. the immediately more aggressive rejected tolerance is evaluated as a
   boundary control.

The local gate requires worst-curve diagonal- and cross-Gamma errors below
2.5% and worst-point errors below 10%.  Global price, Delta, Gamma, Hessian,
PSD and material cross-Gamma sign criteria from v9.6 are retained.  Raw core
archives and JSON checkpoints make every phase resumable.

# Version 9.6 — scalar interpolant consistency and local Greek gate

V9.6 tests whether the Greek engine differentiates one well-defined price
surface.  This distinction matters because the v9.2--v9.5 component-hybrid
rule chose different cubic axes for price, Delta, diagonal Gamma and cross-
Gamma.  Its individual components can be accurate without forming the
gradient and Hessian of a single scalar interpolant.

The experiment freezes the bounded standardized-risk grid and seed 20260401,
the worst compressed seed in v9.5.  Four interpolation operators are applied
to identical exact-grid or TT nodal values:

- multilinear on every physical coordinate;
- cubic only on the bounded risk-coordinate axis;
- the former component-dependent hybrid control;
- one unified interpolant, cubic on all five spot axes and the bounded risk
  coordinate, and linear in rate and maturity.

The paper profile evaluates the exact-grid interpolation floor, an untruncated
200k TT and the same cores truncated at `1e-8`.  Global normalized MAEs,
full-Hessian error, PSD and material-sign diagnostics are retained.  The new
local gate additionally requires worst-curve Gamma errors below 2.5% and
worst-point errors below 10% across the selected `(m,T)` panel.  Among fixed
scalar interpolants that pass every gate, the code selects the fastest.

# Version 9.5 — Greek-aware multi-seed truncation

V9.5 tests whether the untruncated 200k TT-cross approximation is robust across
the five frozen seeds and determines how much compression can be applied
without losing derivative information.  A single raw fit is performed for
each seed.  All truncation variants therefore share the same underlying TT
cores and differ only through the post-processing tolerance.

The paper profile uses

- budget: `200000`;
- seeds: `20260401` to `20260405`;
- truncations: raw, `1e-14`, `1e-12`, `1e-10`, `1e-8`;
- the v9.4 bounded standardized-risk coordinate;
- the same 60-point analytical five-asset Greek panel;
- material cross-Gamma sign tolerance: `5e-4`.

The JSON reports mean, standard deviation, range and worst-seed errors for each
tolerance.  A tolerance is robustly accepted only if every requested seed
passes every Greek-aware criterion.  Among robust candidates, the code selects
the one with the smallest mean retained parameter fraction.  See
`V9_5_RUN_COMMANDS.md`.

# Version 9.4.1 — budget-seed and truncation stability

The v9.4 paper campaign identified a stable 150k candidate but did not establish
monotone budget convergence: for seed 20260401 the post-truncation effective
rank fell between 150k and 200k while the price error remained unchanged and
the Hessian error increased.  V9.4.1 tests whether this is a seed-specific
trajectory or a systematic derivative-sensitivity effect.

The first experiment completes the Cartesian matrix

```text
budgets = {100k, 150k, 200k}
seeds   = {20260401, ..., 20260405}.
```

It retains the same 60-point paper panel, bounded standardized-risk grid,
0.2% fixed-strike bump and risk-hybrid cubic interpolation.  Existing v9.4
runs are reused, leaving only eight missing `(budget,seed)` pairs.

The second experiment separates TT-cross construction from SVD-style
truncation.  For each requested fit, raw cross cores are frozen and the Greek
audit is repeated after truncations `none`, `1e-10`, `1e-8` and `1e-6`.  Ranks,
parameter retention, price/Greek errors, full-Hessian error, PSD diagnostics
and errors relative to the untruncated TT are stored.  This directly tests the
hypothesis that modes negligible in the price norm remain material after two
spot differentiations.

Dense 101--201 point moneyness curves are deliberately deferred until this gate
selects a robust budget and truncation rule.  See `V9_4_1_RUN_COMMANDS.md`.

# Version 9.4 — QTT budget and seed convergence

V9.4 is the first TT-compressed experiment after the grid-only v9.0--v9.3
sequence.  It uses the European geometric-basket put, the full 51-price
fixed-strike stencil, three out-of-sample spot panels, four maturities and five
log-moneyness levels.  Requested QTT-cross budgets are 20k, 50k, 100k, 150k and
200k; the actual oracle count, including ANOVA initialization and the final
cross batch, is always reported.

The initial implementation revealed a domain pathology before any paper run:
the unbounded coordinate

```text
z = asinh((m - ridge(r,T)) / (sigma_B sqrt(T)))
```

is well behaved locally but its global extrema cannot be crossed with every
maturity in one Cartesian tensor.  Extreme short-maturity `z` values combined
with long maturities produced economically meaningless log-moneyness and
destabilized TT-ANOVA.  V9.4 maps the maturity-dependent standardized interval
to `[-1,1]`, so both endpoints recover the fixed global moneyness bounds at
every `(r,T)`.  The coordinate remains concentrated in standardized risk while
the complete tensor stays inside the pricing domain.

For every TT run, the stored residuals satisfy componentwise

```text
TT FD - analytical
  = (exact-price FD - analytical)
  + (grid FD - exact-price FD)
  + (TT FD - grid FD).
```

The paper campaign contains one full budget sweep and five independent seeds
at the retained 150k budget.  Acceptance is based on total price/Greek error,
full-Hessian Frobenius error, material cross-Gamma sign agreement and the
relative norm of the negative Hessian part.  See `V9_4_RUN_COMMANDS.md`.

# Version 9.3 — complete five-asset Hessian gate

The v9.3 experiment is the last grid-only gate before TT compression is
reintroduced.  It keeps the v9.2 numerical choices fixed and expands the
finite-difference stencil from two selected spot factors to all five factors.
Each market point therefore requires 51 fixed-strike prices and yields five
Deltas plus the fifteen independent entries of the symmetric 5x5 Hessian.

The experiment does not impose non-negative cross-Gammas.  For a geometric
basket, the analytical cross-Gamma can be negative because the basket map is
concave in the individual spots.  Instead, v9.3 measures sign agreement away
from near-zero tails and tests positive semidefiniteness of the full Hessian
with both absolute and scale-aware diagnostics.

The paper profile uses 101 moneyness values in `[-0.35, 0.35]`, maturities of
3, 7, 30 and 90 days and all three out-of-sample spot panels.  No TT or Monte
Carlo layer is present.  See `V9_3_RUN_COMMANDS.md`.

# Version 9.2.1 — dense functional audit

The v9.2.1 experiment is not another aggregate benchmark.  It samples 201
uniform log-moneyness values in `[-0.35, 0.35]` on each paper-profile curve and
stores analytical, exact-price finite-difference, multilinear and risk-hybrid
cubic Greeks.  The requested 3-, 7-, 30- and 90-day maturities are evaluated as
market values rather than snapped to maturity-grid nodes.  Its purpose is to
detect hidden overshoots or oscillations between the sparse v9.2 control
points.  See `V9_2_1_RUN_COMMANDS.md`.

# Version 9.2 — residual Greek error floor

The v9.2 gate freezes the `standardized_risk` coordinate and the v9.1 choice
of 64 nodes on every spot axis.  It then varies only the relative finite-
difference bump and the interpolation rule on three off-grid spot panels not
used by v9.1.  The European geometric basket provides exact prices and exact
fixed-strike Greeks, so every reported residual satisfies

```text
grid FD - analytical
  = (exact-price FD - analytical) + (grid FD - exact-price FD).
```

The first term is the numerical differentiation layer and the second is the
grid-interpolation layer.  TT reconstruction and Monte Carlo noise remain
disabled.  The paper gate covers five bumps, six maturity buckets, seven
log-moneyness levels and both multilinear and risk-hybrid cubic interpolation.
See `V9_2_RUN_COMMANDS.md` for the exact commands.

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

## Stage 1b - Volatility as a sampled dimension

An extension beyond the paper. Each underlying gets its own volatility
coordinate, so the European geometric basket is learned over 13 physical inputs
instead of 8.

- Inputs: `(S1, sigma1, S2, sigma2, S3, sigma3, S4, sigma4, S5, sigma5, K, r, T)`.
- Interleaved on purpose: a tensor train only compresses along the mode order it
  is given, and the strongest local coupling is between an asset and its own
  volatility.
- `sigma_i in [0.05, 0.80]`, gridded uniformly in `u_i = log(sigma_i)`.
- Grid: `(32 x 16)^5 x 64 x 8 x 8`; QTT shape `[2] * 57` (up from 37 cores).
- Correlation, dividends and every other bound are inherited from `PaperConfig`.
- Test points are drawn log-uniformly in sigma by default, matching the grid
  coordinate; `--vol-sampling uniform` draws them uniformly in sigma instead.
- Reported per model: effective rank, price MAE and seconds per query, in the
  JSON `headline` block, plus a `by_basket_volatility` error breakdown keyed on
  `sigma_B = sqrt(w' (sigma sigma' o C) w)`.

This script's profiles are sized for the 13-dimensional domain, not shared
with the 8-dimensional runs: `intermediate` uses TT budgets 20,000-200,000 with
GPR up to 5,000, and `paper` runs 50,000-1,000,000 with GPR up to 10,000.
`smoke` stays tiny on purpose - it is a wiring check, not a result.

```bash
python scripts/reproduce_european_vol.py --profile intermediate
python scripts/reproduce_european_vol.py --profile intermediate --grid-mode paper
python scripts/reproduce_european_vol.py --profile paper --skip-gpr
```

### Smoke profile, 500 test points, direct strike axis

| model | effective rank | price MAE | normalized MAE | s/query |
| --- | --- | --- | --- | --- |
| oracle interpolation floor | - | 0.116 | 0.0024 | 9.9e-3 |
| TT, 2,489 evaluations | 5.03 | 7.351 | 0.152 | 6.8e-6 |
| TT, 5,463 evaluations | 6.85 | 3.427 | 0.071 | 1.0e-5 |
| GPR, n = 200 | - | 6.403 | 0.132 | 1.7e-5 |
| GPR, n = 500 | - | 4.670 | 0.096 | 4.2e-5 |

The smoke budgets are far too small at 13 dimensions: TT-cross exhausts the
budget (`stop: m`) after one or two sweeps, and 5,000 evaluations spread over 57
cores leave the surrogate two orders of magnitude above its own interpolation
floor. For reference, the same smoke budgets at 8 dimensions reach MAE 0.094
against a floor of 0.079. Continuing the budget on the same grid does converge:

| evaluations | effective rank | price MAE | normalized MAE | s/query |
| --- | --- | --- | --- | --- |
| 10,444 | 9.23 | 2.620 | 0.054 | 1.2e-5 |
| 20,485 | 12.12 | 1.715 | 0.035 | 1.6e-5 |
| 50,479 | 17.38 | 1.297 | 0.027 | 7.7e-5 |
| 100,476 | 22.57 | 0.862 | 0.018 | 1.6e-4 |
| 200,088 | 28.77 | 0.669 | 0.014 | 2.1e-4 |
| 499,755 | 39.40 | 0.397 | 0.0082 | 3.5e-4 |

### Fix: fit P / basket_spot on moneyness grids

The moneyness plateau documented below is repaired by changing the fit target,
not the grid. `--target price_over_basket` (the automatic choice on moneyness
grids) makes TT-cross, the oracle and the GPR baseline learn `P / G(S)` and
multiplies predictions back by the exact basket spot of the query point. In
moneyness coordinates the geometric closed form divided by `G` contains no spot
dependence at all, so the five spot modes collapse to rank one and the huge
out-of-box corners lose their `G` amplification. Validation on
`moneyness_adaptive` (smoke test set, `--budgets 5000 20000 50000 --skip-gpr`,
stored in `results/european_vol_scaled_target_check_moneyness_adaptive.json`):

| evaluations | effective rank | price MAE | raw-price target at same grid |
| --- | --- | --- | --- |
| oracle floor | - | 0.114 | 0.103 |
| 5,445 | 6.65 | 1.609 | 8.674 |
| 20,323 | 9.88 | 0.182 | 7.320 |
| 50,263 | 15.44 | 0.130 | 7.181 |

At 50,000 evaluations the scaled target sits on the interpolation floor and is
three times better than the direct strike grid reaches at 500,000. The scaled
oracle floor differs slightly from the raw one because interpolating `P / G`
and multiplying by the exact `G` is itself a better interpolant.

### The moneyness coordinate stops paying once volatility is a dimension (raw price target)

`log(K / basket_spot)` is the better coordinate at 8 dimensions but is actively
harmful at 13:

| grid mode | evaluations | effective rank | price MAE | bias |
| --- | --- | --- | --- | --- |
| `paper` | 50,479 | 17.38 | 1.297 | +1.070 |
| `moneyness_uniform` | 50,294 | 16.97 | 15.691 | +15.690 |
| `moneyness_adaptive` | 50,446 | 16.67 | 7.181 | +7.171 |

The moneyness runs plateau: their MAE is flat from 20,000 to 500,000
evaluations while the rank climbs from 11.6 to 32.7. The cause is the grid box,
not the cross. On the moneyness grid a corner is `K = G(S) exp(m)` with
`m <= log(K_max / S_min) = 3.69`, so grid strikes reach roughly 6,000 and grid
prices reach several thousand, far outside the market box. At 8 dimensions the
deep-ITM price is the rank-one product `G(S) f(m, r, T)` and that unused region
costs nothing. Adding volatility replaces `f` with a factor containing
`exp(T sigma_B^2 / 2)`, whose full pairwise coupling across the five volatility
modes is not low rank, so the cross spends its rank on the unused region. The
error is almost pure positive bias and concentrates deep ITM.

### The log-volatility axis costs a little accuracy at the interpolation floor

Measured with `--oracle-only`, so this isolates the grid from TT-cross error:

| axis | 8 nodes | 16 nodes | 32 nodes |
| --- | --- | --- | --- |
| `u = log(sigma)` | 0.234 | 0.116 | 0.091 |
| `sigma` | 0.136 | 0.095 | 0.086 |

A uniform sigma axis interpolates slightly better at every resolution, and the
ordering survives a uniform-sigma test law (0.137 log against 0.089 linear at 16
nodes). The put price is close to linear in sigma over most of `[0.05, 0.80]`,
so the log axis buys resolution where it is not needed and gives it up near
`sigma = 0.80`, where the node spacing widens to 0.148 against 0.05 uniform. The
gap is small in absolute terms - both floors are under 0.25% of the mean
absolute price, and an order of magnitude below the TT-cross error at any budget
tested - so it does not drive the headline numbers, but the log transform should
be understood as a domain-shape choice rather than an accuracy win.

### Prediction cost

Off-grid multilinear interpolation over `2^d` corners is 8,192 tensor lookups
per query at `d = 13`, which costs 46-71 ms per query and would dominate any
timing comparison against the 8-dimensional runs. Because the interpolation
weight is a product over physical dimensions and the TT value is a product of
core slices, the corner sum factorizes: each dimension's two candidate index
chains can be blended in place. `TTPriceSurrogate.predict_factorized` does this
in 57 small matrix products, agrees with the corner stencil to machine
precision, and is what the reported seconds per query measure. The scripts still
time the stencil on a subset and record the agreement in `stencil_inference`.

## Stage 1c - Basket type, mode ordering and a Monte-Carlo reference

Two questions the volatility extension raised, settled empirically. Both run in
about nine minutes together.

```bash
python scripts/validation/compare_basket_layouts.py --budgets 50000 100000
python scripts/validation/compare_monte_carlo.py
```

### Which mode ordering compresses better depends on the basket

A coupling costs rank on every bond it has to cross, so the better ordering is
the one that keeps the dominant coupling contiguous. Which coupling dominates is
a property of the payoff, not of the framework.

At 100,000 evaluations, truncation `1e-4`, 2,000 test points:

| basket | layout | eff. rank | storage | MAE | floor |
| --- | --- | --- | --- | --- | --- |
| arithmetic | **interleaved** | 12.61 | **137.0 KiB** | **0.1924** | 0.0920 |
| arithmetic | blocked | 14.35 | 177.4 KiB | 0.3305 | 0.0920 |
| geometric | interleaved | 6.40 | 35.4 KiB | 0.1047 | 0.1047 |
| geometric | **blocked** | 5.59 | **27.0 KiB** | 0.1080 | 0.1047 |

The two baskets disagree, and the mechanism explains why. Under the
`P / B(s)` target the geometric price is exactly spot-free, so its spot modes
carry nothing: blocking lets them sit at rank 1 (measured: exactly 1 at every
truncation), while interleaving forces those 25 QTT cores to relay the
volatility aggregate at rank 6 to 24. The arithmetic price keeps a genuine
dependence on basket composition and couples `S_i` to `sigma_i` through
`exp(sigma_i sigma_j C_ij T)`. That is a local pairwise coupling, which is
exactly what interleaving is for: it wins on both accuracy (1.7x at 100,000
evaluations, 2.8x at 50,000) and storage. Its spot-block rank stays at 16 to 21
under either ordering, confirming the composition dependence is real rather than
an artifact.

The arithmetic surface is also genuinely harder: it is still a factor of two
above its interpolation floor at 100,000 evaluations, where the geometric
surface has already reached its own.

Arithmetic labels use the Levy lognormal moment match, which is what makes
100,000 TT-cross calls affordable. It carries the correct coupling structure but
is an approximation: measured against 100,000-path Monte Carlo it is 0.08% at
total volatility below 0.10, 1.9% in [0.20, 0.35) and 5.0% in [0.35, 0.60), 1.6%
overall. Reported errors are surrogate-versus-Levy, not surrogate-versus-truth.

### Truncation, not budget, sets the storage

The default `1e-8` truncation is far tighter than the interpolation floor and
inflates storage several-fold at unchanged accuracy. Geometric, blocked,
100,000 evaluations:

| truncation | eff. rank | storage | MAE |
| --- | --- | --- | --- |
| 1e-8 | 15.62 | 210.2 KiB | 0.1065 |
| 1e-6 | 13.74 | 162.6 KiB | 0.1065 |
| 1e-5 | 10.11 | 88.2 KiB | 0.1065 |
| **1e-4** | **5.59** | **27.0 KiB** | **0.1080** |
| 1e-3 | 3.26 | 9.2 KiB | 0.1507 |

Nearly 8x the storage for a 1.4% MAE change. Set the truncation so the TT error
lands just under the interpolation floor.

### Monte-Carlo reference on the geometric basket

Both methods scored against the closed form, 2,000 test points, mean absolute
price 44.67. Monte Carlo simulates all five correlated assets to maturity in one
step, which is what a general engine pays for a European basket.

| method | MAE | RMSE | bias | s/query | speed vs MC-100k |
| --- | --- | --- | --- | --- | --- |
| QTT surrogate | 0.1065 | 0.1531 | +0.1053 | 4.27e-05 | 343x |
| (interpolation floor) | 0.1035 | 0.1509 | +0.1033 | - | - |
| Monte Carlo, 10,000 paths | 0.0683 | 0.1126 | -0.0019 | 1.05e-03 | 13.9x |
| Monte Carlo, 100,000 paths | 0.0220 | 0.0366 | +0.0014 | 1.46e-02 | 1.0x |

Read carefully, this is not a clean surrogate win. Monte Carlo at 10,000 paths
is already *more accurate* than the surrogate, because the surrogate is pinned
at its interpolation floor while Monte Carlo error keeps falling as
`1/sqrt(N)`. The surrogate's advantage is speed at a fixed accuracy target and
determinism, not accuracy:

- at matched accuracy, reaching the surrogate's RMSE needs about 5,400 paths,
  costing 5.7e-04 s/query, still **13x slower** than the surrogate;
- the 8.6 s build amortizes after 8,512 queries against 10,000-path Monte Carlo,
  or 589 queries against 100,000-path Monte Carlo;
- the surrogate returns the *same* price every call. Monte Carlo error is fresh
  noise on every revaluation (repeat-to-repeat standard deviation 0.041 at
  10,000 paths), which is what damages finite-difference Greeks and paired
  scenario comparisons.

### The floor is convexity bias, and it is reducible

The surrogate's error is almost entirely bias (+0.1053 out of 0.1065 MAE), and
so is the floor's (+0.1033 out of 0.1035). Multilinear interpolation of a convex
price surface always overestimates, so this is a systematic discretization
effect rather than noise. Replacing linear with cubic interpolation on the
high-curvature axes shrinks it substantially (oracle floor, 500 test points):

| interpolation | floor MAE | bias |
| --- | --- | --- |
| multilinear | 0.1145 | +0.1143 |
| cubic on `m` | 0.0408 | +0.0402 |
| cubic on `m` and `T` | 0.0364 | +0.0361 |

### Factorized cubic interpolation

`predict_factorized` now takes `cubic_columns`. The interpolation weight is a
product over modes whether an axis is linear or cubic, so the factorization is
unchanged: a cubic axis blends four candidate index chains instead of two. Cost
goes from `prod_j k_j` corner lookups to `sum_j k_j q_j` small matrix products,
so making every axis cubic at most doubles the query, against a factor of two
per cubic axis for the corner stencil. Measured against
`predict_hybrid_cubic` the two agree to `1.1e-15` relative, and the factorized
path is **16,629x faster** (2.85e-05 against 4.74e-01 seconds per query with a
cubic moneyness axis).

Which axes to make cubic, on the same fitted train (2,000 test points, blocked
layout, 100,000 evaluations, truncation `1e-6`, rank 13.74):

| cubic axes | MAE | bias | s/query |
| --- | --- | --- | --- |
| none (multilinear) | 0.1065 | +0.1053 | 4.56e-05 |
| `m` | 0.0417 | +0.0396 | 5.98e-05 |
| `m`, `T` | 0.0394 | +0.0375 | 5.83e-05 |
| `m`, `T`, `r` | 0.0365 | +0.0345 | 5.99e-05 |
| `m`, `T`, vols | 0.0102 | +0.0047 | 8.13e-05 |
| **all 13** | **0.0091** | **+0.0017** | 8.56e-05 |

The volatility axes matter more than the moneyness axis once the latter is
fixed: sixteen nodes spanning a factor of sixteen in sigma is a coarse grid, and
the price is curved in sigma. Making every axis cubic is the default
(`--cubic-axes all`); it cuts MAE by 11.7x and bias by 62x for twice the query
cost, and the tails improve as much as the mean:

| | MAE | bias | p95 | max |
| --- | --- | --- | --- | --- |
| multilinear | 0.1065 | +0.1053 | 0.3205 | 0.8988 |
| cubic, all axes | 0.0091 | +0.0017 | 0.0299 | 0.1190 |

Nothing about the representation changes: identical cores, identical rank and
identical 163 KiB of storage. Cubic is a read-time choice, and the query stays
`O(sum_j k_j q_j)` - linear in the number of modes, never exponential.

### Monte Carlo revisited with cubic interpolation

The comparison reverses. Same test set and same fitted train as above:

| method | MAE | p95 | max | s/query | speed |
| --- | --- | --- | --- | --- | --- |
| **QTT surrogate, cubic** | **0.0091** | **0.0299** | **0.1190** | 8.65e-05 | **167x** |
| same TT, multilinear | 0.1065 | 0.3205 | 0.8988 | 4.26e-05 | 338x |
| Monte Carlo, 10,000 paths | 0.0683 | 0.2499 | 0.7654 | 9.95e-04 | 14.5x |
| Monte Carlo, 100,000 paths | 0.0220 | 0.0830 | 0.2746 | 1.44e-02 | 1.0x |

The surrogate is now **2.4x more accurate than 100,000-path Monte Carlo and 167x
faster**, on the mean and on both tail measures. At matched accuracy Monte Carlo
would need about 671,000 paths, costing 9.7e-02 seconds per query, roughly
**1,100x** the surrogate. The 8.2 s build amortizes after 571 queries against
100,000-path Monte Carlo.

One caveat: both interpolants return small negative prices on deep
out-of-the-money points whose true value is under 0.064 (worst case -0.072
cubic, -0.029 linear, on 7% of points). A `maximum(price, 0)` clip is exact for
a put and can only reduce the error; it is deliberately not applied inside the
surrogate, which reports the raw interpolant.

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

### V8 product-transferability gate

The geometric European experiment remains the analytical control. Two product
extensions are then evaluated without changing the market assumptions, fixed-
strike bump contract or adaptive coordinate design.

For the European arithmetic basket, the low-cost TT label oracle is a fixed
scrambled Sobol design with a geometric-basket control variate. The validation
reference uses independent Sobol scramblings and a larger path count. The JSON
reports the standard error across scramblings, the direct training-oracle error,
the exact-grid interpolation floor and the final TT error separately.

For the American arithmetic basket, reference Greeks are computed seed by seed
with common random numbers inside each bump stencil. The reported reference is
the multi-seed mean; its component-level standard errors are retained. Three
errors must not be conflated:

1. training LSMC versus the independent reference;
2. TT versus the same-seed training LSMC;
3. TT versus the independent reference.

The `smoke` and `intermediate` profiles are mandatory gates before the `paper`
profile. The final paper grid is `32^5 x 512 x 8 x 64`, but the American TT
budget is selected independently because the cost and noise of an LSMC label
are product-specific. Every result file stores raw and PSD-projected Hessians.
The comparison figures display actual Greek profiles, normalized error heat
maps, effective ranks and oracle evaluation counts.

### V8.1 reference-stabilization gate

The V8 smoke campaign showed that the European arithmetic price and Delta can
be benchmarked accurately, whereas its Hessian remains sensitive to QMC sample
size. It also showed that the American LSMC Hessian is not a usable reference at
2,000 paths. TT budgets are therefore frozen until the two reference studies
below are completed.

#### European arithmetic QMC

Cross the number of paths per Sobol scrambling, the fixed-strike relative bump
and the geometric control-variate coefficient. Compare a sample-estimated
coefficient with `beta=1` fixed across the entire stencil. Record:

- component means and standard errors across independent scramblings;
- normalized standard error for price, Delta and both Gamma families;
- Frobenius norm of the componentwise Hessian standard errors;
- differences from the highest-path anchor and from the previous path count;
- negative diagonal and non-PSD counts.

The intermediate gate uses paths `[4096, 16384, 65536]`, bumps
`[0.2%, 0.5%, 1%]` and five scramblings on the seven- and thirty-day ATM
points.

#### American arithmetic LSMC

Cross path counts, exercise-step counts, bump sizes and three exercise-policy
contracts:

1. `refit`: regress the continuation policy again for every bumped scenario;
2. `frozen`: fit on the base scenario and reuse the policy on common paths;
3. `frozen_oos`: train the base policy and evaluate it on independent paths.

The diagnostic stencil bumps two selected assets. It contains one price, two
Deltas, two diagonal Gammas and one cross-Gamma, requiring nine prices per
market point instead of the 51 prices of the full five-asset Hessian. This is a
computational screening device only; the final selected estimator must be
retested on every asset pair.

The intermediate gate uses paths `[2000, 10000]`, steps `[20, 40]`, bumps
`[0.5%, 1%]`, five seeds and the thirty- and ninety-day ATM points. A frozen
policy is retained only if its lower dispersion is accompanied by a small and
stable discrepancy from the converged refit estimator.

Reference acceptance targets are normalized standard errors below `0.5%` for
price and `2%` for Delta, together with Hessian Frobenius uncertainty below
`15%`. These thresholds do not replace bump stability or successive-path
convergence. Componentwise relative Gamma errors are not used alone when the
reference component is close to zero.

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

## V9.9.2 - Geometric node redistribution

The final coordinate audit compares the pricing and risk-hybrid grids before
any TT reconstruction is introduced.  Both candidates use the same physical
mode sizes and the same log-moneyness endpoints.  For every `(r,T)` slice, the
experiment computes the normalized root-mean-square displacement `D2`, the
normalized maximum displacement `D_infinity`, and the local fill distance in

`R_kappa = [m_star-kappa sigma_G sqrt(T), m_star+kappa sigma_G sqrt(T)]`.

The two node fractions `P_kappa^(pricing)` and `P_kappa^(risk)` identify the
concentration mechanism, while
`G_fill=h_kappa^(pricing)/h_kappa^(risk)` verifies whether this redistribution
actually reduces the largest local hole.  A value above one indicates improved
risk-hybrid coverage.

Formally, the implementation records
`P_kappa^(g)=n^(-1) sum_j 1{m_j^(g) in R_kappa}` as `P_nodes`.  The experiment
contains no interpolation error, price derivative, curvature proxy or TT-cross
fit; these quantities are intentionally excluded to preserve a one-to-one
correspondence with the theoretical section.

The per-trade construction is implemented:

```bash
python scripts/validation/validate_portfolio_var.py --profile smoke --grid-mode paper
python scripts/validation/validate_portfolio_var.py \
    --profile intermediate --grid-mode moneyness_adaptive
```

Scenarios are drawn once, in a single exact GBM step to the risk horizon, and
reused by the closed-form pricer, the grid interpolation oracle and the TT
surrogate, so every comparison is paired. The engine ages each trade by the
horizon at fixed contractual strike and refuses positions expiring inside it;
`Portfolio.split_at_horizon` excludes them explicitly and the excluded set is
recorded in the JSON, since it changes with the horizon and changes the book.

The horizon defaults to one day. This is the demanding case rather than the mild
one: the P&L shrinks with the horizon while the pricing error does not, so the
surrogate error is measured against a smaller signal and a tighter Monte-Carlo
interval. The paper maturity axis is the visible cause, its first cell spanning
`[1/365, 0.43]`; short-dated positions are interpolated linearly across five
months, so their error does not cancel between the base state and the aged one.

The JSON reports, per confidence level, the VaR order-statistic interval and the
ES standard error alongside the surrogate error, so a surrogate discrepancy can
be read against the Monte-Carlo noise it must beat. `--skip-tt` isolates the
interpolation floor of the selected grid in risk space, which is the Stage-1
`oracle_interpolation` measurement transported into a risk metric. The scenario
domain coverage records how often a scenario state leaves the grid box, since an
interpolant clamps silently and the tail is where that happens.
