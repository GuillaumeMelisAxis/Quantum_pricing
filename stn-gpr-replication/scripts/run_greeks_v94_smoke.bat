@echo off
setlocal

python -m pytest
if errorlevel 1 exit /b %errorlevel%

python scripts\validation\validate_tt_greek_convergence.py ^
  --profile smoke ^
  --campaign all ^
  --output results\greeks_v94_tt_convergence_smoke.json ^
  --overwrite
if errorlevel 1 exit /b %errorlevel%

python scripts\figures\plot_tt_greek_convergence.py ^
  --input results\greeks_v94_tt_convergence_smoke.json ^
  --panel oos_mid ^
  --maturity-days 30 ^
  --output-dir figures\greeks_v94_tt_convergence_smoke
if errorlevel 1 exit /b %errorlevel%

echo Version 9.4 smoke campaign complete.
endlocal
