# STN-GPR replication

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
