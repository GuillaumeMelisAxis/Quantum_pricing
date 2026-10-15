@echo off
setlocal
python scripts\validation\select_greek_aware_truncation.py ^
  --profile smoke ^
  --cross-log ^
  --core-cache-dir results\greeks_v97_smoke_cores ^
  --output results\greeks_v97_greek_aware_truncation_smoke.json
if errorlevel 1 exit /b %errorlevel%
python scripts\figures\plot_greek_aware_truncation.py ^
  --input results\greeks_v97_greek_aware_truncation_smoke.json ^
  --output-dir figures\greeks_v97_greek_aware_truncation_smoke
endlocal
