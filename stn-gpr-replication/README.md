# From Pricing to Portfolio Risk: Adaptive Tensor-Train Surrogates for Multi-Asset Options

Code, results and manuscript for the paper by G. Melis, A. Decouvelaere and
I. Koutchoukali (`paper/rapport_final.tex`). The library `stngpr` is an
independent implementation of the STN-GPR tensor-train pricer (Gribben et al.)
with the paper's extensions: log-moneyness and hyperbolic-sine grids,
short-maturity refinement, the risk-hybrid coordinate for Greeks, volatility
as a tensor coordinate, and full-revaluation VaR / Expected Shortfall.

## Layout

```
stn-gpr-replication/
├── reproduce_paper.py        one command to rerun every experiment of the paper
├── paper/
│   ├── rapport_final.tex     the manuscript
│   ├── reproductibilite.md   detailed settings (axes, TT-cross, LSMC, MC, threads), in French
│   └── figures/              every \includegraphics target, written by the pipeline
├── experiments/              one folder per experiment, in the order of the paper
│   ├── 00_method_illustrations/   convexity heat map, sinh grid (Methodology, App. E)
│   ├── 01_node_placement/         D2, D_inf, coverage gain, concentration  (Sec. 5.1)
│   ├── 02_european_pricing/       original vs adaptive grid, European put  (Sec. 5.2.1)
│   ├── 03_american_pricing/       LSMC checks + maturity refinement        (Sec. 5.2.2)
│   ├── 04_greeks/                 pricing vs risk-hybrid grid, Hessians    (Sec. 5.3.1)
│   ├── 05_volatility_cost/        cost of the five volatility axes         (Sec. 5.3.2)
│   ├── 06_query_benchmark/        13-D surrogate vs Monte Carlo per query  (Sec. 5.4.1)
│   └── 07_var_es/                 one-day VaR and ES of the option book    (Sec. 5.4.2)
├── results/                  data written by experiments/, same folder names
├── src/stngpr/               the library (grids, TT-cross surrogate, pricers, risk)
├── tests/                    unit tests (pytest)
├── tools/environment.py      prints CPU, BLAS and package versions
├── requirements-lock.txt     exact package versions used for the paper
└── archive/                  earlier exploratory work, not needed for the paper
```

## Setup

Python 3.10 or later; the paper was produced with Python 3.12.9.

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps
.venv\Scripts\python.exe -m pytest -q
```

Linux / macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest -q
```

The tests take about a minute. `teneva` is pinned to 0.14.11: its API changes
between releases.

## Reproducing the paper

Run every command from this folder (the one containing `reproduce_paper.py`);
all paths are relative to it. Below, `python` means the virtual-environment
interpreter.

```bash
python reproduce_paper.py --list           # experiments, outputs, expected durations
python reproduce_paper.py --smoke          # whole pipeline with tiny settings
python reproduce_paper.py                  # everything with the paper's settings
python reproduce_paper.py --only greeks    # one or several experiments
python reproduce_paper.py --figures-only   # redraw figures from existing results
python reproduce_paper.py --dry-run        # print the exact commands
```

- Start with `--smoke` on a new machine. It writes to `results_smoke/` (ignored
  by git) and checks the whole pipeline in minutes. Its numbers are not meaningful.
- The full run takes many hours; the volatility-cost experiment alone took
  2.6 h on the authors' laptop. Run experiments one at a time with `--only` if
  needed. The American experiment saves a checkpoint after every fit and resumes
  if restarted.
- `--dry-run` prints each step as a plain command, so any step can be rerun by
  hand. The settings live in `reproduce_paper.py`. Note that several scripts
  have a profile called `paper`, but most published runs used `intermediate`.
  The runner always passes the settings that were actually used.
- The run records the package versions, machine and git commit in
  `results/environment.json`.

Then compile the manuscript from `paper/`:

```bash
cd paper
pdflatex rapport_final && bibtex rapport_final && pdflatex rapport_final && pdflatex rapport_final
```

`paper/references.bib` is not in the repository yet. Add it from the Overleaf
project before compiling.

## Where each result comes from

| Paper element | Experiment folder | Main script | Output |
|---|---|---|---|
| Fig. convexity-grid-comparison, sinh-moneyness-discretization | `00_method_illustrations` | `plot_convexity_heatmaps.py`, `plot_sinh_discretization.py` | `paper/figures/` |
| Sec. 5.1, three node-placement figures | `01_node_placement` | `analyze_grid_node_redistribution_v992.py` (profile `paper`) | `results/01_node_placement/grid_geometry.json` |
| Table 4, Fig. results-european-grid | `02_european_pricing` | `reproduce_european.py` (profile `intermediate`, grids `paper` and `moneyness_adaptive`) | `results/02_european_pricing/european_*.json` |
| Sec. 5.2.2 oracle checks (binomial bias, training-oracle MAE) | `03_american_pricing` | `validate_lsmc_1d.py`, `lsmc_convergence.py` (`intermediate`) | `results/03_american_pricing/lsmc_*.json` |
| Table 5, Fig. results-american-grid | `03_american_pricing` | `validate_american_grid_robustness.py` (`intermediate`) | `results/03_american_pricing/american_grid_robustness.json` |
| Tables 6-7, Fig. greeks_v98_dense_profiles | `04_greeks` | `compare_pricing_risk_grids.py` (`paper`) | `results/04_greeks/pricing_vs_risk_grid.json` |
| Sec. 5.3.2, Fig. figure1_statistical_cost_d0_d5 | `05_volatility_cost` | `phase4_dsigma_sweep.py`, `merge_budget_tiers.py`, `phase5_figures.py` | `results/05_volatility_cost/phase4_extended_*` |
| Sec. 5.4.1, Fig. e3_accuracy_vs_time | `06_query_benchmark` | `benchmark_vol_risk_grids.py` (`intermediate`, E3-E5) | `results/06_query_benchmark/vol_risk_grid_benchmark_intermediate.json` |
| Tables 8-9 and the cumulative-cost table | `07_var_es` | `benchmark_var_qtt_basket_book.py` (`full`, tag `bench_es`) | `results/07_var_es/var_qtt_benchmark_bench_es.md` |

Inside `05_volatility_cost/`, `common.py` holds every frozen parameter of that
protocol and `measure.py` defines what each fit records. `phase1` to `phase3`
(choice of the volatility resolution, node-coincidence gate, dry run) and
`phase6` (controls) are the preparatory and control phases. Their outputs are
kept in `results/05_volatility_cost/`. `build_report.py` assembles them into
`rapport_experiences.md`.

### Results already in the repository

| Experiment | Stored results |
|---|---|
| 05 volatility cost | complete (all phases), figure in `paper/figures/` |
| 06 query benchmark | complete, figure in `paper/figures/` |
| 07 VaR / ES | complete (`bench_es` run) |
| 03 American | LSMC reference-label cache only (saves the most expensive part of a rerun) |
| 00, 01, 02, 03, 04 | outputs and figures not stored: run `reproduce_paper.py --only ...` |

## Seeds, threads and timings

- All random draws use NumPy's PCG64 generator. Every seed is set in the scripts
  and listed in the paper's Appendix C. With the same seeds, accuracies reproduce
  exactly, up to floating-point summation order.
- The query benchmark and the VaR/ES experiment fix every BLAS thread count to
  one before importing NumPy. The other experiments use the BLAS default.
- Wall-clock times depend on the machine and are only comparable within one
  run. On hybrid CPUs (performance and efficiency cores) the same computation
  varied by a factor of up to 3. See `paper/reproductibilite.md`, section 6.

## The library

| Module | Content |
|---|---|
| `config.py` | `PaperConfig` (8-D setting), `VolatilityExtendedConfig` (13-D) |
| `grids.py`, `coordinates.py` | QTT grids, sinh / quadratic maps, log-moneyness transform |
| `risk_grids.py`, `grid_geometry.py` | risk-hybrid coordinate, node-placement metrics |
| `tt_surrogate.py` | TT-cross fit (teneva), multilinear and factorized cubic queries, rank diagnostics |
| `pricers.py`, `baselines.py` | closed-form, Levy, Monte Carlo, LSMC, QMC pricers; GPR baseline |
| `greeks.py`, `convergence.py` | analytical and finite-difference Greeks at fixed strike, PSD projection, Greek error metrics |
| `vol_extension.py`, `sigma_axes.py`, `sigma_surrogate.py` | volatility as tensor coordinate (13-D, and 0 to 5 volatility axes) |
| `portfolio.py`, `risk.py`, `scenarios.py` | option books, GBM scenarios, VaR / ES |
| `validation.py` | error metrics, binomial tree, 1-D LSMC, stratified American test points |
| `diagnostics.py` | geometric-basket convexity and ridge formulas |

## Archive

`archive/` keeps the development history: version notes (V8 to V9.9), earlier
reports, the previous README, the earlier manuscript `paper_risk.tex`, and the
exploratory scripts and results that did not make it into the paper. See
`archive/README.md`.
