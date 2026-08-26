@echo off
setlocal

python -m pytest
if errorlevel 1 exit /b %errorlevel%

python scripts\validation\validate_dense_greek_profiles.py ^
  --profile smoke ^
  --output results\greeks_v921_dense_profiles_smoke.json
if errorlevel 1 exit /b %errorlevel%

python scripts\figures\plot_dense_greek_profiles.py ^
  --input results\greeks_v921_dense_profiles_smoke.json ^
  --risk-column 0 ^
  --all-bumps ^
  --output-dir figures\greeks_v921_dense_profiles_smoke
if errorlevel 1 exit /b %errorlevel%

echo Version 9.2.1 smoke campaign complete.
endlocal
