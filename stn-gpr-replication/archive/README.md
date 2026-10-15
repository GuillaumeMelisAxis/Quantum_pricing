# Archive

Development history of the project. None of it is needed to reproduce the
paper: see `../README.md` and `../reproduce_paper.py` for that.

| Folder | Content |
|---|---|
| `docs/` | version notes `CHANGELOG_V*.md` and `V*_RUN_COMMANDS.md`, the previous README (`README_development_history.md`, version-by-version narrative V8 to V9.9.2), `EXPERIMENTS.md` (full experiment protocols), the VaR reports `rapport_var_qtt_panier*.md`, `phase0_inventaire.md`, and the earlier manuscript `paper_risk.tex` |
| `scripts/` | exploratory scripts: Greek ablations and truncation studies (V9 to V9.7), arithmetic-basket and Monte Carlo comparisons, earlier VaR benchmarks, the 13-D European driver `reproduce_european_vol.py`, `reproduce_american.py`, and the `run_greeks_v*_smoke.bat` launchers |
| `results/` | outputs of those scripts (JSON outputs are git-ignored and exist only on the machine that produced them) |

The scripts are kept as they were. Some import helpers that now live in
`experiments/04_greeks/` (`validate_dense_greek_profiles`,
`validate_full_spot_hessian`, `validate_tt_greek_convergence`,
`audit_price_greek_interpolant_consistency`). To run one of them, put that
folder on the path first, from the project root:

```bash
PYTHONPATH=experiments/04_greeks python archive/scripts/validation/ablate_tt_greek_truncation.py --profile smoke
```

Their default output paths (`results/...`, `figures/...`) and the commands in
the archived docs refer to the layout before the reorganization of
October 2026.
