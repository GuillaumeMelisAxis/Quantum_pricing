@echo off
python scripts\validation\validate_tt_greek_truncation_multiseed.py ^
  --profile smoke ^
  --budgets 5000 ^
  --tt-seeds 20260401 20260402 ^
  --truncations 0 1e-14 1e-12 1e-10 1e-8 ^
  --output results\greeks_v95_tt_truncation_multiseed_smoke.json
if errorlevel 1 exit /b %errorlevel%

python scripts\figures\plot_tt_greek_truncation_multiseed.py ^
  --input results\greeks_v95_tt_truncation_multiseed_smoke.json ^
  --output-dir figures\greeks_v95_tt_truncation_multiseed_smoke
