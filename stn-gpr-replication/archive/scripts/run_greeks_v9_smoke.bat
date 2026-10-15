@echo off
setlocal

python -m pytest
if errorlevel 1 exit /b %errorlevel%

python scripts\validation\compare_greek_coordinate_grids.py ^
  --profile smoke ^
  --output results\greeks_v9_coordinate_ablation_smoke.json
if errorlevel 1 exit /b %errorlevel%

python scripts\figures\plot_greek_coordinate_ablation.py ^
  --input results\greeks_v9_coordinate_ablation_smoke.json ^
  --output-dir figures\greeks_v9_coordinate_ablation_smoke
if errorlevel 1 exit /b %errorlevel%

echo Version 9 smoke campaign complete.
endlocal
