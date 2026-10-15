@echo off
setlocal

python -m pytest
if errorlevel 1 exit /b %errorlevel%

python scripts\validation\validate_tt_greek_convergence.py ^
  --profile smoke ^
  --campaign matrix ^
  --output results\greeks_v941_budget_seed_matrix_smoke.json ^
  --overwrite
if errorlevel 1 exit /b %errorlevel%

python scripts\validation\ablate_tt_greek_truncation.py ^
  --profile smoke ^
  --output results\greeks_v941_tt_truncation_smoke.json ^
  --overwrite
if errorlevel 1 exit /b %errorlevel%

python scripts\figures\plot_tt_greek_stability.py ^
  --matrix-input results\greeks_v941_budget_seed_matrix_smoke.json ^
  --truncation-input results\greeks_v941_tt_truncation_smoke.json ^
  --output-dir figures\greeks_v941_tt_stability_smoke
if errorlevel 1 exit /b %errorlevel%

echo Version 9.4.1 smoke campaign complete.
endlocal
