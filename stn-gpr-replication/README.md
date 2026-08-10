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
