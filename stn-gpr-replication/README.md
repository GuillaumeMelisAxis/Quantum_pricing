# STN-GPR replication

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
