@echo off
setlocal

python -m pytest
if errorlevel 1 exit /b %errorlevel%

python scripts\validation\decompose_geometric_greek_error_floor.py ^
  --profile smoke ^
  --output results\greeks_v92_error_floor_smoke.json
if errorlevel 1 exit /b %errorlevel%

python scripts\figures\plot_greek_error_floor.py ^
  --input results\greeks_v92_error_floor_smoke.json ^
  --output-dir figures\greeks_v92_error_floor_smoke
if errorlevel 1 exit /b %errorlevel%

echo Version 9.2 smoke campaign complete.
endlocal
