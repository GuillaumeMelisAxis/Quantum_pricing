# Version 8.2 run commands

The commands below target the **x64 Native Tools Command Prompt**, not
PowerShell. Run them from the repository root after activating the existing
virtual environment.

## Installation and tests

```bat
call .venv\Scripts\activate.bat
python -m pip install -e .
python -m unittest discover -s tests -v
```

## 1. Smoke test

```bat
python scripts\validation\decompose_american_tt_errors.py ^
  --profile smoke ^
  --output results\greeks_v82_american_error_decomposition_smoke.json
```

Generate the diagnostic figure:

```bat
python scripts\figures\plot_american_error_decomposition.py ^
  --input results\greeks_v82_american_error_decomposition_smoke.json ^
  --output-dir figures\greeks_v82_american_decomposition_smoke
```

The smoke profile checks the pipeline only. Its low-path reference and small TT
budget are not suitable for scientific conclusions.

## 2. Intermediate experiment

```bat
python scripts\validation\decompose_american_tt_errors.py ^
  --profile intermediate ^
  --cross-log ^
  --output results\greeks_v82_american_error_decomposition_intermediate.json
```

```bat
python scripts\figures\plot_american_error_decomposition.py ^
  --input results\greeks_v82_american_error_decomposition_intermediate.json ^
  --output-dir figures\greeks_v82_american_decomposition_intermediate
```

This is the next experiment to run. It uses the validated 50,000-path,
80-exercise-date, eight-seed reference and budgets of 5,000 and 10,000 oracle
evaluations.

## 3. Paper profile

Run this only after the intermediate JSON has been inspected:

```bat
python scripts\validation\decompose_american_tt_errors.py ^
  --profile paper ^
  --cross-log ^
  --output results\greeks_v82_american_error_decomposition_paper.json
```

```bat
python scripts\figures\plot_american_error_decomposition.py ^
  --input results\greeks_v82_american_error_decomposition_paper.json ^
  --output-dir figures\greeks_v82_american_decomposition_paper
```

The paper profile is deliberately expensive. It spans 20 moneyness-maturity
points and uses TT budgets of 10,000, 20,000 and 50,000 evaluations.

## Optional staged diagnostics

Stop after the independent reference:

```bat
python scripts\validation\decompose_american_tt_errors.py ^
  --profile intermediate ^
  --stage reference ^
  --output results\greeks_v82_american_reference_only.json
```

Stop after the exact grid layer:

```bat
python scripts\validation\decompose_american_tt_errors.py ^
  --profile intermediate ^
  --stage grid ^
  --output results\greeks_v82_american_grid_only.json
```
