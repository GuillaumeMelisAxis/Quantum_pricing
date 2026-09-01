@echo off
setlocal
python scripts\validation\audit_price_greek_interpolant_consistency.py ^
  --profile smoke ^
  --cross-log ^
  --output results\greeks_v96_interpolant_consistency_smoke.json
if errorlevel 1 exit /b %errorlevel%
python scripts\figures\plot_interpolant_consistency.py ^
  --input results\greeks_v96_interpolant_consistency_smoke.json ^
  --output-dir figures\greeks_v96_interpolant_consistency_smoke
endlocal
