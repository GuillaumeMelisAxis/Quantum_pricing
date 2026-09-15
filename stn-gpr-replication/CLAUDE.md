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
.venv/Scripts/python.exe -m pytest -q        # 83 tests, ~10s
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

Detailed experiment protocol and results: `EXPERIMENTS.md`. Version-by-version
narrative and exact commands: `README.md`. Both are large — grep for a stage/version
heading rather than reading end to end.

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
3. **Layout (interleaved vs blocked) depends on the basket, not a universal rule.**
   Geometric + `P/basket_spot`: price is exactly spot-free, so blocked wins (spot
   QTT cores collapse to rank 1, confirmed by truncation sweep). Arithmetic: genuine
   `S_i`-`sigma_i` coupling via `exp(sigma_i sigma_j C_ij T)`, so interleaved wins
   (1.7-2.8x better MAE, less storage). Don't default one for both baskets.
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
9. **`teneva.cross` is called without an `e`/`e_vld` tolerance** in `fit()` — it
   always runs to the requested budget (`stop: "m"` in diagnostics), so rank grows
   long after MAE has stopped improving. Not yet fixed; budget past ~100k
   evaluations on the 13-D problem is likely wasted, check the `effective_rank`
   vs `mae` curve before spending more.

## Where things live

- `scripts/reproduce_european.py` / `reproduce_european_vol.py`: main experiment
  entry points (8-D paper reproduction / 13-D volatility extension).
- `scripts/validation/`: targeted comparisons — `compare_basket_layouts.py`,
  `compare_monte_carlo.py`, `compare_arithmetic_reference.py`,
  `validate_portfolio_var.py`, plus the colleague's Greeks validators.
  `benchmark_vol_risk_grids.py` is the paper-grade accuracy/query-cost protocol
  for the vol-extended risk grids (surrogate vs Monte Carlo vs the closed-form
  oracle; profiles `smoke` ~3 min, `intermediate` ~20 min, `paper` ~2 h).
  Protocol and threats to validity: `EXPERIMENTS.md` Stage 1d.
- `results/*.json`: gitignored (`**/*.json`), not committed — regenerate from the
  scripts rather than expecting them to be present after a fresh clone.
- `tests/test_core.py`: single test file, 83 tests, all classes.
