# stn-gpr-replication

Independent implementation and extension of the STN-GPR tensor-train option-pricing
framework (Gribben et al. 2026). Lives inside the `Quantum_pricing` monorepo but is
fully self-contained: its own `pyproject.toml`, `.venv`, `src/`, `tests/`. Treat it as
the project root, not `Quantum_pricing/`.

Shared repo: branch `adding_vol` carries the volatility extension (this file's focus);
a colleague's Greeks work (`STN-GPR-Greeks`, `risk_grids.py`, `convergence.py`,
`EuropeanArithmeticBasketQMC`) is merged in from `main`. Expect both areas to keep
moving — re-check `git log` before assuming a symbol's shape from this file alone.

## Environment

```bash
.venv/Scripts/python.exe -m pytest -q        # 127 tests, ~30s
.venv/Scripts/python.exe -m ruff check src scripts tests
```

Windows + Git Bash. Always call the venv interpreter explicitly
(`.venv/Scripts/python.exe`), not `python`. `pip install -e ".[dev]"` after a fresh
clone. Core dependency is `teneva==0.14.11` (TT-cross); pin matters, its API moves.

## Architecture

- **`config.py`**: `PaperConfig` (paper's 8-D setup, fixed volatility) and
  `VolatilityExtendedConfig` (13-D: `(S_1, sigma_1, ..., S_5, sigma_5, K, r, T)`).
  The latter has a `layout` field (`"interleaved"` or `"blocked"`) that controls
  QTT mode ordering — see gotcha below, it is not a cosmetic choice.
- **`pricers.py`**: `geometric_basket_put` (closed form) and
  `arithmetic_basket_put_levy` (Levy lognormal moment-match — an *approximation*,
  see gotcha) both take per-row volatility vectors. `basket_put_monte_carlo` is the
  ground-truth reference for either basket kind. `EuropeanArithmeticBasketQMC`
  (colleague's, deterministic Sobol QMC + geometric control variate) is a better
  arithmetic label source than Levy when its cost is affordable.
- **`vol_extension.py`**: `InterleavedVolTransform` (market ↔ model coordinates,
  `u = log(sigma)`), `build_vol_extended_grid`, `BasketScaledModelPricer` (fits
  `P / basket_spot` instead of raw price — required at 13-D, see gotcha),
  `resolve_cubic_columns` / `CUBIC_AXIS_NAMES` for cubic-axis selection. Also
  carries the colleague's Greek coordinate families lifted to 13-D:
  `build_vol_extended_risk_grid` (`VOL_RISK_GRID_MODES`),
  `VolStandardizedRiskTransform` / `BoundedVolStandardizedRiskTransform` (σ_B is
  now per-row, not a config constant), and `build_vol_grid`, which dispatches to
  either family by mode name. `VolExtendedPriceSurrogate` is the end-to-end
  path — grid, transform, `P/G` scaling, TT-cross, cubic prediction — so
  `.fit(budget)` then `.price(market_points)` prices on any of them.
- **`tt_surrogate.py`**: `TTPriceSurrogate.predict_factorized(points,
  cubic_columns=...)` is the fast prediction path — contracts each mode's 2 (linear)
  or 4 (cubic) candidate index chains in place, `O(sum_j k_j q_j)` instead of the
  corner stencil's `O(prod_j k_j)`. Always prefer it over `predict`/
  `predict_hybrid_cubic` except when explicitly cross-checking correctness.

- **`sigma_axes.py` / `sigma_surrogate.py`**: the *additive* volatility axes, added
  for the dimensional-cost experiment. `SigmaAugmentedConfig` takes an independent
  `n_sigma_axes` (`d_sigma`, 0..n_assets) instead of pinning one sigma per asset,
  and its `sigma_block="tail"` layout appends the sigma axes *after* the shared
  `(S_1..S_n, K, r, T)`, so the `d_sigma=0` QTT chain is exactly the `d_sigma=k`
  chain minus the sigma cores. That is what `vol_extension` cannot do: neither of
  its layouts puts the sigma block at the chain tail, so its rank profiles are not
  comparable core by core across dimension counts. `SigmaAugmentedSurrogate`
  always fits `P/G` and *refuses* post-fit SVD truncation rather than defaulting
  it off. `spot_collapsed_interpolation_floor` computes the cubic-on-every-axis
  interpolation floor at 13-D, which the `vol_extension` path cannot (gotcha 8):
  `P/G` is spot-free on these coordinates and Lagrange weights sum to one per
  axis, so the spot stencil collapses exactly, `4**13` -> `4**8`. The identity is
  re-measured at run time on a subsample, not assumed.

Detailed experiment protocol and results: `archive/docs/EXPERIMENTS.md`.
Version-by-version narrative and exact commands:
`archive/docs/README_development_history.md`. Both are large — grep for a
stage/version heading rather than reading end to end. Their paths predate the
October 2026 reorganization (see "Where things live").

## Non-obvious gotchas (learned the hard way this session)

1. **On a moneyness grid, fit `P / basket_spot`, not the raw price, once volatility
   is a dimension.** `log(K / basket_spot)` grid corners reach strikes far outside
   the market box; harmless at 8-D (deep-ITM price is rank-1 in spot), but at 13-D
   the volatility coupling through `exp(T sigma_B^2/2)` forces the cross to spend
   rank on that unused region and MAE plateaus around 7 instead of converging to
   the ~0.1 interpolation floor. `BasketScaledModelPricer` fixes this — it's the
   default (`--target auto`) in `reproduce_european_vol.py`.
2. **Cubic interpolation removes most of the residual error, near-free.** Linear
   interpolation of a convex price surface is a large systematic *positive bias*
   (~0.10 of a ~0.11 MAE). Cubic on all axes cuts MAE ~12x and bias ~60x for about
   2x query cost (`O(sum k_j q_j)` scales gently — corner stencil would pay 2x
   *per* cubic axis instead). Default `--cubic-axes all` in the vol scripts.
3. **Layout (interleaved vs blocked) depends on the basket *and on the grid
   family*, not a universal rule.** On the **pricing** grids, geometric +
   `P/basket_spot`: price is exactly spot-free, so blocked wins (spot QTT cores
   collapse to rank 1, confirmed by truncation sweep). Arithmetic: genuine
   `S_i`-`sigma_i` coupling via `exp(sigma_i sigma_j C_ij T)`, so interleaved wins
   (1.7-2.8x better MAE, less storage). **On the 13-D `bounded_standardized_risk`
   grid the geometric rule inverts**: interleaved gave 0.06% one-day VaR error
   against blocked's 0.60% at budget 150k and equal effective rank. That grid's
   strike coordinate is itself a function of sigma, so the sigma-strike coupling
   no longer sits where blocked puts it. Measure per (basket, grid family).
4. **TT-cross truncation should target the interpolation floor, not `1e-8`.**
   Default truncation gives ~8x more storage than truncating near the actual
   accuracy floor, for a <2% MAE change. Always sweep truncation when reporting
   storage numbers.
5. **Levy is an approximation, not truth, for arithmetic-basket labels.** ~1.6%
   off a 10^6-path Monte Carlo reference overall, worse (~5%) above total
   volatility 0.35. Any arithmetic-basket error reported against Levy is
   surrogate-vs-Levy, not surrogate-vs-truth — decompose
   `(surrogate - Levy) + (Levy - reference)` before concluding the surrogate is
   bad; often the Levy term dominates. `EuropeanArithmeticBasketQMC` is the
   candidate fix, not yet substituted into the main comparison scripts.
6. **The standardized-risk grids need a finer maturity axis than the config
   default.** Their strike coordinate is a function of `(r, T, sigma)`, so a
   fixed coordinate names a different strike at every maturity and the surface
   moves along `T` far faster than on a moneyness grid — where the config only
   spends 8 nodes. At a 150k budget on the default domain,
   `bounded_standardized_risk` stalls at MAE 0.35 with `maturity_nodes=8` and
   hits 0.0086 with 32, while `price_adaptive` is 0.0054 either way. The stall
   is the grid's interpolation floor, not TT: effective rank keeps climbing
   (11 → 23) while MAE sits still, and the multilinear oracle floor on that grid
   is 0.30. Raise `maturity_nodes` before comparing coordinate families.
7. **The unbounded `standardized_risk` mode cannot be built at 13-D.** One
   global `z` interval must cover every `(r, T, sigma)` the other axes offer;
   with sigma spanning 0.05–0.80 the envelope reaches `|z| ≈ 8.7`, and the
   opposite corner of the box maps to `|log(K/G)| ≈ 2.6e3`, i.e. a strike that
   is not a float. `build_vol_extended_risk_grid` raises with those numbers
   rather than handing back a grid full of `inf`. It still builds on a narrow
   volatility box; `bounded_standardized_risk` is the general answer.
8. **The oracle interpolation floor cannot go fully cubic at 13-D.** It
   enumerates the product stencil, so `2**13 = 8192` corners multilinear against
   `4**13 = 67M` fully cubic (52 GiB). Only `TTPriceSurrogate.predict_factorized`
   contracts mode by mode; `VolExtendedPriceSurrogate.price_interpolation_floor`
   therefore defaults to multilinear and refuses stencils above `max_stencil`.
   Do not compare a multilinear floor against a cubic TT error and call the gap
   TT error.
10. **`d_sigma` and mode order are not free in `vol_extension`, and that is a
    protocol trap.** `VolatilityExtendedConfig` is pinned at `2n+3` dimensions,
    and both its layouts put the sigma block *before* `(K, r, T)`. Any experiment
    that varies the number of volatility factors, or that wants to align rank
    profiles core by core against the 8-D chain, needs `sigma_axes.py` instead.
    Related: the 8-D path in `compare_pricing_risk_grids.py` fits the raw price
    `P` while `VolExtendedPriceSurrogate` fits `P/G` - comparing the two directly
    mixes a dimension effect with a target effect (see gotcha 1 for how large
    that is). `sigma_surrogate.py` fits `P/G` at every `d_sigma` for this reason.

9. **`teneva.cross` is called without an `e`/`e_vld` tolerance** in `fit()` — it
   always runs to the requested budget (`stop: "m"` in diagnostics), so rank grows
   long after MAE has stopped improving. Not yet fixed; budget past ~100k
   evaluations on the 13-D problem is likely wasted, check the `effective_rank`
   vs `mae` curve before spending more. On the 8-D geometric problem the VaR
   error plateaus at 35k and is flat to 400k, and it is *not monotone* on the
   way there: 20k measured worse (6.4%) than 10k (2.5%). Never pick a budget
   from one point. See gotcha 14 for why more budget can *lower* a risk metric
   even though the fit itself never degrades.

13. **A TT fit at a fixed budget is a draw, exactly like a Monte Carlo run.**
    Single-seed 13-D budget sweeps returned 0.06% at 150k where the 3-seed rms
    is 1.81% - the lucky seed, and it was nearly published as "the" accuracy.
    Both approximate arms in `benchmark_var_qtt_basket_book.py` therefore sweep
    several seeds and report the quadratic mean plus the range. A different
    ANOVA seed picks different pivots, hence a different error texture - same
    mechanism as gotcha 14.

14. **A bigger TT budget can give a *worse* VaR while the fit strictly
    improves. Do not tune a risk metric through the price error.** On the 13-D
    risk grid the portfolio price MAE falls monotonically across the whole
    sweep - 3.69e-1 / 1.54e-1 / 4.49e-2 / 1.66e-2 / 6.31e-3 / 3.50e-3 /
    2.04e-3 at 25k-400k, a factor 180, with 400k the *best* of all - while the
    3-seed VaR error runs 3.45 / 11.12 / 4.29 / 1.81 / 0.33 / **0.11** / 0.44 %.
    What oscillates is the survival rate `MAE(PnL)/MAE(price)`: 0.261, 0.409,
    0.331, 0.329, 0.162, 0.221, 0.589. A VaR is a quantile of `V(s) - V(0)`
    over a one-day displacement, so only the part of the error that *varies*
    over ~1% in spot survives; the smooth part cancels. TT-cross is an
    interpolant, exact at its pivots and wandering between them, and
    `maxvol_rect` reselects those pivots every sweep (`dr_max=2` rows added per
    pass). More budget therefore does not refine the same approximation - it
    reshuffles *where* the error vanishes. Amplitude falls monotonically;
    spatial texture is redrawn, and texture is what the cancellation sees.
    Monte Carlo under CRN is the mirror image, not merely immune: its error is
    a bias common to the whole scenario cloud that the P&L difference removes
    (survival 0.028 / 0.062 / 0.073 / 0.012 at 1k-64k paths). Its *price*
    error is not even monotone - the 3-seed mean rose from 0.55% to 0.69% of
    the book between 16k and 64k paths, being the size of a random bias - while
    its *risk* error falls cleanly (VaR 95 0.82% -> 0.18%). QTT: price
    converges, risk oscillates; MC: price oscillates, risk converges. Neither
    price error predicts the risk error. Consequence: the cheap, reference-free stopping criterion
    (global fit norm) does not flag the bad budgets, so the budget must be swept
    against the risk metric itself. `archive/docs/rapport_var_qtt_panier_otm13d.md`
    section 4.3 and `archive/results/figures/var_qtt_versus_monte_carlo.pdf`.

11. **A put-only surrogate prices geometric-basket calls for free, exactly.**
    `C - P = e^{-rT}(F - K)` with `F = G_0 exp(carry T)` is model-exact and
    costs no pricing work, so the call error *equals* the put error — measured
    residual `5.2e-14` on a 6-option book. `geometric_basket_parity_offset` in
    `pricers.py`. There is nothing to gain by also training on calls; it halves
    the surface to learn. Does not transfer to the arithmetic basket, whose
    forward has no such closed form.

12. **Wall-clock numbers from this machine carry a ~2.7x dispersion and cannot
    be compared across runs.** The nested-MC arm, byte-identical work, measured
    2.74 s and 7.39 s in two runs, and `[3.80, 7.50, 7.39]` across three
    repeats inside one run; the same TT fit measured 9.14 s in one section and
    3.35 s in another section of the *same process*. Any cost claim must be an
    intra-run ratio, and preferably a machine-free one (oracle calls, path
    counts). `benchmark_var_qtt_basket_book.py` takes `--timing-repeats` and
    keeps min/max in the JSON for this reason — read the spread, not the
    median alone.

## Where things live

Reorganized in October 2026 around the final manuscript `paper/rapport_final.tex`:
only what the paper uses is at the top level, everything else is in `archive/`.

- `reproduce_paper.py`: single entry point; holds the exact settings of every
  published run (`--list`, `--smoke`, `--only NAME`, `--dry-run`). Most
  published runs used profile `intermediate`, not the scripts' `paper` profile.
- `experiments/NN_<name>/`: one folder per paper experiment (00 method
  illustrations, 01 node placement, 02 European, 03 American, 04 Greeks,
  05 volatility cost, 06 query benchmark, 07 VaR/ES). Scripts are run as files
  from the project root; sibling imports (e.g. the Greek helpers in
  `04_greeks/`, `common.py` in `05_volatility_cost/`) rely on the script's own
  folder being on `sys.path`. Folder names start with digits, so tests load
  them with `importlib` (see `tests/test_v992_geometry.py`).
- `results/NN_<name>/`: outputs, same names. The local `.gitignore` re-includes
  `results/**/*.json` and `*.csv` against the repo-root ignore rules; caches
  (`results/**/cache/`, `*.npz`) stay ignored, except the already-tracked
  American LSMC label cache.
- `paper/figures/`: figures at the exact paths `\includegraphics` uses; the
  plotting scripts default to writing there.
- `experiments/07_var_es/benchmark_var_qtt_basket_book.py` is the one-day VaR
  protocol on a 6-option call/put book over five synthetic stocks: closed form
  vs nested Monte Carlo vs put-only QTT lifted by parity, all on shared
  scenarios. `experiments/06_query_benchmark/benchmark_vol_risk_grids.py` is
  the accuracy/query-cost protocol for the vol-extended grids (profiles
  `smoke` ~3 min, `intermediate` ~20 min, `paper` ~2 h).
- `experiments/05_volatility_cost/`: the dimensional-cost protocol, phase by
  phase (`common.py` holds every frozen parameter and the run manifest;
  `measure.py` defines what is recorded per fit).
- `archive/scripts/`: exploratory scripts (13-D driver `reproduce_european_vol.py`,
  layout / Monte Carlo / arithmetic comparisons, Greek ablations). Some need
  `PYTHONPATH=experiments/04_greeks`.
- `tests/test_core.py`: core tests, all classes. `tests/test_sigma_axes.py`: the
  additivity and reduced-floor-equivalence tests for `sigma_axes.py`.
