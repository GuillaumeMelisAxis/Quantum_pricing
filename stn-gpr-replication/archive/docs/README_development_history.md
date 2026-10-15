# STN-GPR replication

## Version 9.8: final pricing-grid versus risk-grid comparison

Version 9.8 closes the controlled European geometric-basket Greek programme.
It compares the paper-I price-adaptive log-moneyness grid with the final bounded
standardized-risk grid in two complementary experiments:

- an exact-grid, matched-resolution coordinate ablation;
- a native-architecture raw-TT comparison over common seeds and oracle budgets.

The production comparison derives price, all five spot Deltas and the complete
`5 x 5` fixed-strike Hessian from one unified scalar cubic surface. It retains
the v9.7 decision not to apply post-fit price-norm truncation. A dense 30-day
profile makes `Delta_1`, `Gamma_1,1` and `Gamma_1,2` visually auditable.

European arithmetic and American products are explicitly outside the v9.8
scope. See `V9_8_RUN_COMMANDS.md` and `CHANGELOG_V9_8.md`.

## Version 9.7: Greek-aware truncation selection

Version 9.7 determines the largest TT truncation that remains admissible for a
risk engine.  Every tolerance is applied to identical raw 200k TT-cross cores
and screened over five seeds with both global and local maturity–moneyness
criteria.  The most compressed robust candidates are then certified with the
v9.6 unified scalar cubic price surface.

The implementation persists raw TT cores and checkpoints every completed
variant.  Long paper runs can therefore resume without refitting completed
seeds.  The first rejected tolerance is retained as a boundary control, so the
selection demonstrates both acceptance and failure rather than reporting only
the winning configuration.  See `V9_7_RUN_COMMANDS.md`.

## Version 9.6: one price surface for prices and Greeks

Version 9.6 audits a requirement that aggregate error tables alone cannot
establish: price, Delta and the complete spot Hessian should be finite-
difference derivatives of the same interpolated scalar TT price surface.  It
compares multilinear interpolation, cubic interpolation along the bounded
risk coordinate, the previous component-dependent hybrid rule and one fixed
cubic rule on all five spot coordinates plus the bounded risk coordinate.

The exact-grid oracle isolates the interpolation floor before TT-cross is
introduced.  The raw 200k fit is retained as the accuracy reference and the
`1e-8` truncation as the compressed candidate.  Acceptance now combines the
existing global price/Greek/Hessian tests with worst-slice and worst-point
criteria in the `(m,T)` plane.  See `V9_6_RUN_COMMANDS.md`.

## Version 9.5: Greek-aware multi-seed truncation

Version 9.5 closes the price-norm truncation investigation on the selected
bounded standardized-risk grid.  For each of five TT-cross seeds, it fits one
raw 200k surrogate and reapplies the tolerances `0`, `1e-14`, `1e-12`, `1e-10`
and `1e-8` to exactly the same raw cores.  The output records the complete QTT
bond-rank path, price and full-Hessian errors, material cross-Gamma signs and
the fraction of raw parameters retained.

The automatic selection is intentionally conservative: it chooses the most
compressed tolerance that passes every price, Delta, Gamma, Hessian, PSD and
material-sign criterion for every requested seed.  Dedicated figures show the
seed range, compression response, pass/fail matrix and accuracy-compression
frontier.  See `V9_5_RUN_COMMANDS.md`.

## Version 9.4.1: budget-seed and truncation stability

Version 9.4.1 follows the non-monotone v9.4 Hessian result.  It extends the
100k, 150k and 200k budgets to the complete five-seed matrix while resuming the
existing v9.4 JSON, so only eight missing fits are computed.  Every new fit
records QTT rank and parameter count both before and after price-norm
truncation.

A separate controlled ablation fits raw TT-cross cores once per `(seed,budget)`
pair, then reapplies the untruncated control and the `1e-10`, `1e-8` and `1e-6`
truncations to those identical cores.  Any resulting Greek variation is
therefore attributable to truncation rather than to a new stochastic fit.  A
scale-sensitivity audit for near-zero cross-Gamma signs is also stored.  See
`V9_4_1_RUN_COMMANDS.md`.

## Version 9.4: QTT reconstruction convergence for Greeks

Version 9.4 reintroduces QTT-cross only after the v9.3 five-asset Greek gate.
It measures the 20k--200k reconstruction-budget curve, retains 150k as the
candidate operating point, and repeats that point over five independent TT
seeds.  Every result separates finite-difference, grid-interpolation and TT
reconstruction errors for price, all five Deltas and the full 5x5 spot
Hessian.

The first compression smoke test exposed that the original unbounded
standardized coordinate is local rather than globally tensor-compatible: its
global coordinate extrema combined with unrelated maturities can generate
log-moneyness outside the economic domain.  V9.4 therefore uses a bounded
standardized-risk coordinate whose endpoints map exactly to the original
log-moneyness bounds for every rate and maturity.  The JSON includes a tensor-
domain audit, actual oracle evaluations, ranks, fit times, sign diagnostics and
scale-aware PSD diagnostics.  See `V9_4_RUN_COMMANDS.md`.

## Version 9.3: full five-asset spot Hessian

Version 9.3 freezes the v9.2 selection at 64 spot nodes, 512 standardized-risk
nodes, 64 maturity nodes and a 0.2% fixed-strike spot bump.  It validates all
five spot Deltas, five diagonal Gammas and ten cross-Gammas of the European
geometric-basket put.  The complete 5x5 Hessian is compared with the analytical
oracle and exact-price finite differences on dense moneyness curves and three
heterogeneous spot panels.  Componentwise errors, cross-Gamma sign fidelity,
Frobenius error and scale-aware PSD diagnostics are reported.  See
`V9_3_RUN_COMMANDS.md`.

## Version 9.2.1: dense Greek profile audit

Version 9.2.1 closes the visual-validation gap left by the sparse v9.2 panel.
It evaluates continuous fixed-strike Delta, diagonal Gamma and cross-Gamma
profiles on up to 201 uniformly spaced log-moneyness values per curve.  Exact
analytical, exact-price finite-difference, multilinear and risk-hybrid cubic
curves are stored pointwise, together with MAE, tail/max errors, total
variation, turning-point and Hessian-shape diagnostics.  See
`V9_2_1_RUN_COMMANDS.md`.

## Version 9.2: Greek error-floor decomposition

Version 9.2 freezes the v9.1 selection (`standardized_risk`, 64 nodes on
every spot axis) and determines why the remaining second-order Greek error
does not vanish monotonically under spot refinement.  On spot panels not used
in v9.1, it ablates the relative bump and interpolation rule and reports the
machine-checkable identity

`grid FD - analytical = finite-difference error + grid-interpolation error`.

No TT or Monte Carlo approximation is present in this diagnostic gate.  See
`V9_2_RUN_COMMANDS.md` and `CHANGELOG_V9_2.md`.

## Version 9.1: standardized-risk spot-resolution ablation

The v9 paper profile selects the maturity-standardized risk coordinate, but its
remaining diagonal-Gamma error is concentrated outside the central moneyness
region and appears common to otherwise different moneyness grids. Version 9.1
tests the resulting spot-resolution hypothesis without TT compression.

The experiment holds the standardized coordinate at 512 nodes, maturity at 64
nodes, the rate axis, bump and market points fixed, then varies every spot axis
through 16, 32, 64 and 128 nodes. Three fixed off-grid basket panels prevent
the 32-node reference architecture from receiving an exact-node advantage.
See `V9_1_RUN_COMMANDS.md`.

## Version 9: derivative-aware coordinate ablation

Version 9 starts the constructive experiment that follows the negative v8.2
result. It does not spend another TT-cross budget on the existing pricing grid.
Instead, it compares four equal-size coordinate systems against analytical
fixed-strike Greeks of a European geometric-basket put:

- uniform log-moneyness;
- the existing price-adaptive sinh grid;
- a static Gamma-error monitor based on the fourth derivative of normalized
  price;
- a maturity-standardized risk coordinate centred on the analytical
  convexity ridge.

The validator separates the exact-price finite-difference truncation error from
the grid interpolation error, uses common spot/rate/maturity nodes, and reports
price, Delta, diagonal Gamma, cross-Gamma, full Hessian and PSD diagnostics.
No TT or Monte Carlo layer is present in this first gate. See
`V9_RUN_COMMANDS.md` and `CHANGELOG_V9.md`.

## Version 8.2: American Greek error decomposition

Version 8.2 adds a controlled decomposition of American fixed-strike Greek
errors into LSMC training-label, adaptive-grid interpolation and TT
reconstruction layers. The intermediate and paper profiles use the validated
50,000-path, 80-exercise-date, eight-seed LSMC reference with a 3% bump.

See `V8_2_RUN_COMMANDS.md` for the Windows Native Tools commands and
`CHANGELOG_V8_2.md` for the mathematical definition of each error layer.

Independent implementation of *STN-GPR: A Singularity Tensor Network Framework
for Efficient Option Pricing* (Gribben et al., 2026), followed by experiments on
VaR/ES, Greeks and the American exercise region.

## What is reproduced

- five-asset geometric European basket put;
- five-asset arithmetic American basket put using LSMC;
- 8 physical inputs: five spots, strike, rate and time to maturity;
- paper grid `32^5 x 64 x 8 x 8`, encoded as a 37-core QTT;
- order-2 TT-ANOVA initialization and rank-adaptive TT-cross;
- off-grid multilinear interpolation;
- exact GPR baseline with a Laplacian/Matérn-1/2 kernel;
- MAE versus training budget and wall-clock time.

The paper does not publish code and omits the volatility vector, correlation
matrix, dividend yields, precise TT-cross settings, LSMC randomization protocol
and the numerical data behind its figures. Defaults in `PaperConfig` are therefore
explicit reconstruction assumptions, not claimed author settings.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python scripts/reproduce_european.py --profile smoke
python scripts/reproduce_european.py --profile intermediate
python scripts/reproduce_european.py --profile paper
python scripts/reproduce_american.py --profile intermediate
python -m pytest
```

Grid experiments preserve the same QTT shape and are selected independently:

```bash
python scripts/reproduce_european.py --profile intermediate --grid-mode paper
python scripts/reproduce_european.py --profile intermediate --grid-mode moneyness_uniform
python scripts/reproduce_european.py --profile intermediate --grid-mode moneyness_adaptive
```

The interpolation floor can be compared in a few seconds, without TT/GPR fits:

```bash
python scripts/reproduce_european.py --profile intermediate --grid-mode paper --oracle-only
python scripts/reproduce_european.py --profile intermediate --grid-mode moneyness_uniform --oracle-only
python scripts/reproduce_european.py --profile intermediate --grid-mode moneyness_adaptive --oracle-only
```

`moneyness_adaptive` uses an asymmetric hyperbolic-sine map calibrated to
concentrate nodes around `log(K / basket_spot) = 0` while retaining adequate
tail resolution, and a quadratic maturity grid near expiry. The European JSON
also reports `oracle_interpolation`: the irreducible
multilinear interpolation error of the selected grid before TT-cross error.

## Volatility-extended European experiment

An extension beyond the paper: each underlying carries its own volatility
coordinate, so the surrogate covers
`(S_1, sigma_1, S_2, sigma_2, ..., S_5, sigma_5, K, r, T)` instead of pricing a
single fixed-volatility slice. Spots and volatilities are **interleaved** so that
an asset sits next to its own volatility in the tensor-train mode ordering, and
volatilities are gridded on `u_i = log(sigma_i)` over `sigma in [0.05, 0.80]`.
The domain grows from 8 physical modes and 37 QTT cores to 13 modes and 57
cores; the grid is `(32 x 16)^5 x 64 x 8 x 8`.

```bash
python scripts/reproduce_european_vol.py --profile intermediate
python scripts/reproduce_european_vol.py --profile intermediate --grid-mode paper
python scripts/reproduce_european_vol.py --profile smoke --vol-scale linear --oracle-only
python scripts/reproduce_european_vol.py --profile paper --skip-gpr
```

`--vol-nodes` sets the resolution of each log-volatility axis, `--vol-scale`
switches between the `log(sigma)` and plain `sigma` axis, `--vol-sampling`
chooses the test law for sigma, and `--budgets` overrides the profile budgets.
Each JSON adds a `headline` block with the effective rank, price MAE and
seconds per query per model, and a `by_basket_volatility` error breakdown
alongside the usual moneyness and maturity buckets.

Profiles here are sized for 13 dimensions (`intermediate`: TT budgets
20,000-200,000; `paper`: up to 1,000,000); the shared `smoke` budgets are a
wiring check only. Under `log(K / basket_spot)` the grid corners reach strikes
far outside `[K_min, K_max]`; that region is harmless when the deep-ITM price is
a rank-one product of the basket spot and a `(m, r, T)` factor, but once the
five volatility modes couple through `exp(T sigma_B^2 / 2)`, fitting the raw
price plateaus. Moneyness grids therefore default to `--target
price_over_basket`: TT, oracle and GPR learn `P / G(S)` - spot-free for the
geometric closed form in these coordinates - and predictions are re-multiplied
by the query point's exact basket spot, which restores convergence to the grid
floor (MAE 0.130 at 50,000 evaluations against a 0.114 floor, versus a 7.2
plateau on the raw price).

Prediction uses `TTPriceSurrogate.predict_factorized`, which contracts each
mode's candidate index chains in place rather than enumerating the product
stencil. At `d = 13` that replaces 8,192 tensor lookups per query with 57 small
matrix products; the scripts still time the corner stencil on a subset and
record their agreement.

Passing `cubic_columns` makes selected axes cubic instead of linear - four
candidate chains on those axes rather than two. The weights still factorize per
mode, so the cost stays `O(sum_j k_j q_j)`: making *every* axis cubic at most
doubles the query, where the corner stencil pays a factor of two per cubic axis
(measured 16,629x apart). Scripts expose this as `--cubic-axes`, taking any of
`spots vols m r T` plus `all` and `none`, defaulting to `all`.

Cubic interpolation is what removes the convexity bias of the multilinear
scheme. On the volatility-extended geometric basket it cuts MAE from 0.1065 to
0.0091 and bias from +0.1053 to +0.0017 for twice the query cost, with the
tensor train, its rank and its storage completely unchanged - making the
surrogate 2.4x more accurate than 100,000-path Monte Carlo and 167x faster.

### Basket type, mode ordering and a Monte-Carlo reference

Two validation scripts extend the experiment beyond the geometric closed form.

```bash
# grid rank and storage: arithmetic vs geometric, interleaved vs blocked  (~7 min)
python scripts/validation/compare_basket_layouts.py --budgets 50000 100000

# surrogate vs Monte Carlo at 10k and 100k paths, geometric basket  (~2 min)
python scripts/validation/compare_monte_carlo.py
```

`VolatilityExtendedConfig(layout=...)` selects the mode ordering. `interleaved`
pairs each spot with its own volatility; `blocked` places all spots first, then
all volatilities. A coupling costs rank on every bond it must cross, so the
better ordering is the one that keeps the dominant coupling contiguous - and
which coupling dominates depends on the basket. Both scripts sweep the TT
truncation tolerance, because the default `1e-8` is far tighter than the
interpolation floor and inflates storage several-fold at unchanged accuracy.

Arithmetic-basket labels come from `arithmetic_basket_put_levy`, a lognormal
moment match. It is fast enough for a hundred thousand TT-cross calls and
carries the true `exp(sigma_i sigma_j C_ij T)` cross terms, but it is an
approximation: about 1.6% from Monte Carlo across this domain, under 0.4% below
total volatility 0.2 and around 5% above 0.35. Surrogate error against it is
therefore surrogate-versus-Levy, not surrogate-versus-truth.

`basket_put_monte_carlo` simulates the five correlated assets to maturity in a
single step and returns prices with their standard errors, for either basket.

## American validation sequence

The four scripts below are intended to be run in order. Start with `smoke` and
inspect its JSON before increasing the profile:

```bash
python scripts/validation/validate_lsmc_1d.py --profile smoke
python scripts/validation/lsmc_convergence.py --profile smoke
python scripts/validation/validate_american_independent.py --profile smoke
python scripts/validation/compare_american_grids.py --profile smoke
```

The first test benchmarks the one-dimensional LSMC implementation against a
CRR tree. The second measures path/step convergence. The third evaluates the TT
against both its deterministic common-random-number oracle and a higher-fidelity
multi-seed reference. The fourth performs a controlled grid ablation:

- `paper`: uniform strike and maturity;
- `paper_adaptive_maturity`: uniform strike, short-maturity refinement;
- `moneyness_adaptive_uniform_maturity`: ATM refinement, uniform maturity;
- `moneyness_adaptive`: ATM and short-maturity refinement.

This factorization distinguishes gains from the moneyness coordinate from gains
caused by maturity refinement. All grid runs share the same test points, LSMC
seed and independent reference labels.

The final robustness gate repeats the two moneyness grids with paired TT seeds.
The intermediate profile uses 200 stratified points, five TT seeds and budgets
7,500, 9,000 and 12,000:

```bash
python scripts/validation/validate_american_grid_robustness.py --profile intermediate
```

This is a long experiment (30 TT fits). It writes the JSON checkpoint after
every fit and caches the expensive independent labels. Running the same command
again resumes missing fits. Use `--restart` only when a full recomputation is
intended. The output reports the adaptive grid's paired win rate and the
distribution of MAE, RMSE, bias, rank, ATM error and timing across TT seeds.

The `smoke` profile validates the complete pipeline with a small pricing budget.
The `paper` profile uses the paper grid and should be run on a machine with ample
CPU time. TT-cross never materializes the full 137.4-billion-entry tensor.

The complete validation design is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## V8 - Greek transferability across products

Version 8 keeps the validated five-asset geometric benchmark and adds two
explicit out-of-sample product classes:

- a European arithmetic-basket put priced by scrambled Sobol-QMC with an exact
  geometric-basket control variate;
- an American arithmetic-basket put whose TT error is separated from an
  independent multi-seed LSMC reference.

The three validators store component-level arrays so that Delta, diagonal Gamma
and cross-Gamma curves can be compared rather than summarized only by one MAE.
Start with the following commands from the project root in the Windows x64
Native Tools Command Prompt:

```bat
python -m pytest

python scripts\validation\validate_refined_tt_greeks.py ^
  --moneyness-nodes 64 --maturity-nodes 8 --budgets 2000 ^
  --anova-samples 300 --replicates 1 --relative-bump 0.002 ^
  --output results\greeks_v8_geometric_smoke.json

python scripts\validation\validate_european_arithmetic_greeks.py ^
  --profile smoke --stage all ^
  --output results\greeks_v8_european_arithmetic_smoke.json

python scripts\validation\validate_american_greeks.py ^
  --profile smoke --stage tt ^
  --output results\greeks_v8_american_arithmetic_smoke.json

python scripts\figures\plot_greeks_product_comparison.py ^
  --geometric results\greeks_v8_geometric_smoke.json ^
  --arithmetic-european results\greeks_v8_european_arithmetic_smoke.json ^
  --arithmetic-american results\greeks_v8_american_arithmetic_smoke.json ^
  --maturity-days 30 ^
  --output-dir figures\greeks_v8_smoke
```

The plotting command produces PNG and PDF versions of three figures: Greek
profiles in log-moneyness, error heat maps in the `(m,T)` plane, and the joint
accuracy/compression-cost summary. By default the displayed Hessian components
are PSD-projected while raw estimates remain in every JSON; add `--raw` to plot
the unconstrained Hessians.

After the smoke gate passes, use the `intermediate` profiles. The `paper`
profiles restore the frozen `32^5 x 512 x 8 x 64` grid. The European geometric
run then uses the validated 150k TT-cross budget; the American script instead
tests product-specific budgets because each LSMC label is much more expensive
and noisy.

## V8.1 - Reference convergence before TT compression

Version 8.1 implements the reference-stabilization gate motivated by the V8
smoke results. It must be run before increasing a TT-cross budget. From the
project root in the Windows x64 Native Tools Command Prompt, the complete smoke
campaign is:

```bat
scripts\run_greeks_v81_smoke.bat
```

The command runs the unit tests, both reference experiments and the comparison
figures. The equivalent individual commands are:

```bat
python scripts\validation\converge_european_arithmetic_reference.py ^
  --profile smoke ^
  --output results\greeks_v81_european_reference_smoke.json

python scripts\validation\converge_american_reference.py ^
  --profile smoke ^
  --output results\greeks_v81_american_reference_smoke.json

python scripts\figures\plot_reference_convergence.py ^
  --european results\greeks_v81_european_reference_smoke.json ^
  --american results\greeks_v81_american_reference_smoke.json ^
  --output-dir figures\greeks_v81_reference_convergence
```

The European experiment crosses Sobol path counts, spot bumps and two
control-variate contracts: a sample-estimated coefficient and a coefficient
fixed to one across the stencil. The American experiment crosses LSMC paths,
exercise dates, bumps and three policy treatments: full refit, frozen policy on
the same paths and frozen policy evaluated out of sample.

Only two spot factors are bumped in the American convergence gate. This retains
one diagonal and one cross-Gamma while reducing a five-asset full-Hessian
stencil from 51 prices to 9 prices per market point. The selected configuration
must later be validated on the complete five-asset Hessian.

After the smoke gate succeeds, run the first scientific comparison with:

```bat
python scripts\validation\converge_european_arithmetic_reference.py ^
  --profile intermediate ^
  --output results\greeks_v81_european_reference_intermediate.json

python scripts\validation\converge_american_reference.py ^
  --profile intermediate ^
  --output results\greeks_v81_american_reference_intermediate.json
```

Do not interpret a frozen-policy result as a corrected American Greek merely
because its variance is smaller. The JSON reports its discrepancy from the
highest-resolution refit anchor so that variance reduction and estimator bias
remain separate.

## Market-coordinate Greeks

Spot Greeks must be computed at fixed contractual strike. A surrogate trained
in log-moneyness coordinates should therefore be exposed through
MarketCoordinatePricer before applying finite differences:

    from stngpr.coordinates import MarketCoordinatePricer
    from stngpr.greeks import finite_difference_greeks

    market_surrogate = MarketCoordinatePricer(model.predict, transform)
    greeks = finite_difference_greeks(
        market_surrogate,
        market_parameters,
        spot_columns=range(config.n_assets),
        relative_bump=1e-3,
    )

The adapter recomputes m = log(K / B(S)) after each spot bump while keeping K
fixed. The finite-difference routine evaluates the complete Delta, Gamma and
cross-Gamma stencil in one vectorized pricing call and returns the actual bump
used for every spot.

Run the analytical bump-convergence test first:

    python scripts/validation/validate_european_greeks.py \
        --profile smoke \
        --stage exact

Measure next the interpolation floor using exact values at every grid corner:

    python scripts/validation/validate_european_greeks.py \
        --profile smoke \
        --stage grid \
        --grid-mode moneyness_adaptive

Finally, run the same stencil on the adaptive TT surrogate:

    python scripts/validation/validate_european_greeks.py \
        --profile smoke \
        --stage tt \
        --grid-mode moneyness_adaptive

The commands write JSON diagnostics under the results directory. The output
contains global and regional errors for Delta, diagonal Gamma and cross-Gamma,
together with grid-cell crossing rates and the TT fit diagnostics when
applicable. The default relative bumps are 10%, 5%, 3%, 2%, 1% and 0.5%.

Test the hybrid cubic correction with local bumps:

    python scripts/validation/validate_european_greeks.py \
        --profile smoke \
        --stage grid_cubic \
        --grid-mode moneyness_adaptive \
        --relative-bumps 0.05 0.02 0.01 0.005 0.002 0.001

    python scripts/validation/validate_european_greeks.py \
        --profile smoke \
        --stage tt_cubic \
        --grid-mode moneyness_adaptive \
        --relative-bumps 0.05 0.02 0.01 0.005 0.002 0.001

For a diagonal Gamma, the relevant spot and log-moneyness axes use local
four-node Lagrange interpolation. For a cross-Gamma, both spot axes and
log-moneyness are cubic. All remaining axes stay multilinear.

Run the short-maturity ATM grid ablation before increasing the TT budget:

    python scripts/validation/validate_short_maturity_greeks.py \
        --replicates 3 \
        --moneyness-nodes 64 128 256 \
        --maturity-nodes 8 16 32 \
        --relative-bump 0.002

This is an exact-grid experiment: it does not fit a TT. The nine paired
configurations determine whether short-dated Gamma error is controlled by the
maturity resolution, the moneyness resolution, or their interaction. The JSON
contains global, conditional and pointwise metrics, together with the local
ratio `Delta m_ATM / (sigma_B sqrt(T))`.

The same output compares the unconstrained Gamma matrix with its orthogonal
projection onto the positive-semidefinite cone. The projection clips negative
eigenvalues, reports all convexity violations before and after correction, and
retains the raw results for auditability. Use this correction only when product
convexity in the selected market factors is theoretically justified.

After selecting the refined grid, run the paired TT budget convergence test:

    python scripts/validation/validate_refined_tt_greeks.py \
        --moneyness-nodes 512 \
        --maturity-nodes 64 \
        --budgets 20000 50000 100000 \
        --anova-samples 2000 \
        --replicates 3 \
        --relative-bump 0.002

The script computes the exact-grid oracle once, then fits an independent TT for
every budget. It separates analytical error from TT-versus-grid reconstruction
error and records raw/projected Hessian accuracy, PSD violations, TT rank,
function evaluations, sweeps and timings. The JSON is checkpointed after the
oracle and after every completed budget.

## Monte-Carlo VaR by full grid revaluation

The risk engine diffuses the five spots to the risk horizon under a correlated
GBM and reprices every trade from scratch in every scenario. The pricer is any
callable mapping market rows `(S_1, ..., S_5, K, r, T)` to prices, so the
closed-form pricer, the interpolation oracle and a TT surrogate wrapped in
`MarketCoordinatePricer` are interchangeable:

    from stngpr.portfolio import Portfolio
    from stngpr.risk import MonteCarloVaREngine
    from stngpr.scenarios import GBMScenarioGenerator

    portfolio = Portfolio.from_arrays(
        strikes=[90.0, 100.0, 110.0],
        maturities=[0.5, 1.0, 2.0],
        quantities=[100.0, -75.0, 50.0],
    )
    engine = MonteCarloVaREngine(
        portfolio,
        GBMScenarioGenerator.from_config(config),
        horizon=1.0 / 252.0,
        levels=(0.95, 0.975, 0.99),
    )
    result = engine.run(pricer, spots, rate=0.03, n_scenarios=50_000, seed=config.seed)
    print(result.var(0.99), result.expected_shortfall(0.99))

Three modelling choices are explicit rather than implicit. Scenarios are drawn
in one exact GBM step, which is exact for any payoff depending only on the state
at the horizon. Every trade is aged by the horizon at fixed contractual strike.
The default drift is risk-neutral; pass `drift` to `GBMScenarioGenerator` for a
real-world measure.

A position expiring inside the horizon has no live-option value to interpolate,
and its payoff depends on the spot at its own expiry rather than at the horizon,
which a one-step diffusion does not produce. `market_parameters` therefore
refuses such a trade instead of clipping its maturity onto the grid floor.
`Portfolio.split_at_horizon` performs the exclusion explicitly:

    live, expiring = portfolio.split_at_horizon(horizon)

It returns the split rather than applying it, because dropping a position
changes the book: the resulting VaR is the VaR of `live`, not of `portfolio`,
and the validation script records the excluded trades in its JSON.

`simulate_spots` is separate from `run` so that a trusted pricer and a surrogate
can be compared on identical scenarios. `compare_loss_distributions` then
reports paired VaR/ES errors, Wasserstein distance, Spearman rank correlation
and worst-tail set overlap; `var_es_uncertainty` reports the order-statistic VaR
interval and the ES standard error, so surrogate error can be read against
Monte-Carlo error; `domain_coverage` reports the fraction of scenario states
falling outside the grid box, which is where an interpolant silently clamps.

Run the paired experiment. The horizon defaults to one day:

```bash
python scripts/validation/validate_portfolio_var.py --profile smoke --grid-mode paper
python scripts/validation/validate_portfolio_var.py \
    --profile intermediate --grid-mode moneyness_adaptive
python scripts/validation/validate_portfolio_var.py \
    --profile intermediate --grid-mode paper --horizon-days 10
```

Shortening the horizon shrinks the P&L but not the pricing error, so the
surrogate error matters more, not less, at one day than at ten. On the paper
grid the oracle 95% VaR error grows from 0.5% at ten days to 4.4% at one day
while the Monte-Carlo interval tightens, which makes the short horizon the
binding accuracy test rather than the easy one.

Use `--skip-tt` to measure only the interpolation floor of the selected grid in
risk space. The JSON also contains the per-trade Euler decomposition of the
expected shortfall and the break-even number of revaluations that amortizes the
surrogate build cost. Against a closed-form pricer that break-even does not
exist; it becomes the relevant number when the black box is LSMC.

## Publication figures

Generate the analytical geometric-basket convexity heat maps with the actual
uniform and adaptive QTT nodes overlaid:

```bash
python scripts/figures/plot_convexity_heatmaps.py --output-dir figures
```

The command writes both `figures/convexity-grid-comparison.png` and
`figures/convexity-grid-comparison.pdf`. The color field is the scale-invariant
strike-convexity indicator `(K^2 / B0) * d2V/dK2`; the dashed curve is its
analytical maximum as a function of maturity.

Generate the computational-coordinate map and the local cell widths of the
hyperbolic-sine log-moneyness discretization:

```bash
python scripts/figures/plot_sinh_discretization.py --output-dir figures
```

The command writes `figures/sinh-moneyness-discretization.png` and
`figures/sinh-moneyness-discretization.pdf`. By default it uses the same
64-node axis and concentration parameter `gamma=3` as the experiments.

## Experimental roadmap

1. Match the European error/training-budget curves.
2. Reproduce the American arithmetic-basket experiment with controlled LSMC noise.
3. Compare repriced portfolio VaR and ES, including tail-ranking stability.
4. Compare delta, gamma and cross-gamma surfaces to trusted finite differences.
5. Oversample and diagnose the region around the American exercise boundary.

## V9.9.2 physical node-redistribution audit

Version 9.9.2 isolates the geometric effect of the paper-I pricing coordinate
and the risk-hybrid coordinate at identical node count.  Its output is aligned
term by term with the mathematical framework: normalized root-mean-square
displacement `D2`, normalized maximum displacement `D_infinity`, coverage gain
`G_fill=h_pricing/h_risk`, and the two node concentrations
`P_kappa^(pricing)` and `P_kappa^(risk)`.

No interpolation error, price derivative, curvature proxy or TT diagnostic is
reported by this experiment.

Run the fast smoke profile with:

```bat
python scripts\validation\analyze_grid_node_redistribution_v992.py ^
  --profile smoke ^
  --output results\greeks_v992_grid_geometry_smoke.json
```

The complete commands and interpretation contract are documented in
`V9_9_2_RUN_COMMANDS.md` and `CHANGELOG_V9_9_2.md`.
