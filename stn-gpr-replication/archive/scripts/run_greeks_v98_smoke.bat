@echo off
setlocal
python -m pytest -q
if errorlevel 1 exit /b %errorlevel%
python scripts\validation\compare_pricing_risk_grids.py ^
  --profile smoke ^
  --cross-log ^
  --core-cache-dir results\greeks_v98_smoke_cores ^
  --output results\greeks_v98_pricing_vs_risk_grid_smoke.json
if errorlevel 1 exit /b %errorlevel%
python scripts\figures\plot_pricing_risk_grid_comparison.py ^
  --input results\greeks_v98_pricing_vs_risk_grid_smoke.json ^
  --output-dir figures\greeks_v98_pricing_vs_risk_grid_smoke
exit /b %errorlevel%
