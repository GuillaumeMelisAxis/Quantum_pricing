@echo off
setlocal

python -m pytest
if errorlevel 1 exit /b %errorlevel%

python scripts\validation\ablate_standardized_spot_resolution.py ^
  --profile smoke ^
  --output results\greeks_v91_spot_resolution_smoke.json
if errorlevel 1 exit /b %errorlevel%

python scripts\figures\plot_spot_resolution_ablation.py ^
  --input results\greeks_v91_spot_resolution_smoke.json ^
  --output-dir figures\greeks_v91_spot_resolution_smoke
if errorlevel 1 exit /b %errorlevel%

echo Version 9.1 smoke campaign complete.
endlocal
