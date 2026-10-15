# Version 9.4 run commands

Run these commands from the repository root in the x64 Native Tools Command
Prompt after activating the virtual environment.

## Installation and smoke gate

```bat
python -m pip install -e ".[dev]"
scripts\run_greeks_v94_smoke.bat
```

The smoke gate checks the complete budget/seed/checkpoint/figure pipeline on a
six-point market panel.  Its deliberately small 2k and 5k budgets validate the
code path, not paper-level Gamma accuracy.

## Complete paper campaign

```bat
python scripts\validation\validate_tt_greek_convergence.py ^
  --profile paper ^
  --campaign all ^
  --cross-log ^
  --output results\greeks_v94_tt_convergence_paper.json
```

The command fits seed `20260401` at requested budgets 20k, 50k, 100k, 150k and
200k, then fits seeds `20260402`--`20260405` at the retained 150k budget.  The
JSON is checkpointed after every fit.  Rerunning the same command resumes and
skips completed `(seed,budget)` pairs.  Use `--overwrite` only to restart the
entire campaign.

Actual oracle evaluations can exceed the requested cross budget slightly
because the ANOVA initialization and final TT-cross batch are also counted and
reported.

## Figures

```bat
python scripts\figures\plot_tt_greek_convergence.py ^
  --input results\greeks_v94_tt_convergence_paper.json ^
  --panel oos_mid ^
  --maturity-days 30 ^
  --output-dir figures\greeks_v94_tt_convergence_paper
```

The command writes PNG and PDF versions of the budget convergence, deterministic
error decomposition, retained Greek profiles and five-seed robustness figures.

## Optional split campaigns

The budget and seed studies may be run separately with `--campaign budget` and
`--campaign seeds`.  Use different output filenames when splitting them; the
`all` campaign is preferred because it produces one self-contained result file.
