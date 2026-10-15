@echo off
setlocal

python -m pytest
if errorlevel 1 exit /b %errorlevel%

python scripts\validation\validate_full_spot_hessian.py ^
  --profile smoke ^
  --output results\greeks_v93_full_hessian_smoke.json
if errorlevel 1 exit /b %errorlevel%

python scripts\figures\plot_full_spot_hessian.py ^
  --input results\greeks_v93_full_hessian_smoke.json ^
  --panel oos_mid ^
  --maturity-days 30 ^
  --output-dir figures\greeks_v93_full_hessian_smoke
if errorlevel 1 exit /b %errorlevel%

echo Version 9.3 smoke campaign complete.
endlocal
