@echo off
setlocal

python scripts\validation\analyze_grid_node_redistribution_v992.py ^
  --profile smoke ^
  --output results\greeks_v992_grid_geometry_smoke.json
if errorlevel 1 exit /b %errorlevel%

python scripts\figures\plot_grid_node_redistribution_v992.py ^
  --input results\greeks_v992_grid_geometry_smoke.json ^
  --output-dir figures\greeks_v992_grid_geometry_smoke
if errorlevel 1 exit /b %errorlevel%

endlocal
