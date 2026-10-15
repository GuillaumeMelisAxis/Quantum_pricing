@echo off
setlocal

if not exist results mkdir results
if not exist figures\greeks_v8_smoke mkdir figures\greeks_v8_smoke

echo [1/5] Running unit tests...
python -m pytest
if errorlevel 1 exit /b 1

echo [2/5] European geometric control...
python scripts\validation\validate_refined_tt_greeks.py ^
  --moneyness-nodes 64 ^
  --maturity-nodes 8 ^
  --budgets 2000 ^
  --anova-samples 300 ^
  --replicates 1 ^
  --relative-bump 0.002 ^
  --output results\greeks_v8_geometric_smoke.json
if errorlevel 1 exit /b 1

echo [3/5] European arithmetic extension...
python scripts\validation\validate_european_arithmetic_greeks.py ^
  --profile smoke ^
  --stage all ^
  --output results\greeks_v8_european_arithmetic_smoke.json
if errorlevel 1 exit /b 1

echo [4/5] American arithmetic extension...
python scripts\validation\validate_american_greeks.py ^
  --profile smoke ^
  --stage tt ^
  --output results\greeks_v8_american_arithmetic_smoke.json
if errorlevel 1 exit /b 1

echo [5/5] Comparison figures...
python scripts\figures\plot_greeks_product_comparison.py ^
  --geometric results\greeks_v8_geometric_smoke.json ^
  --arithmetic-european results\greeks_v8_european_arithmetic_smoke.json ^
  --arithmetic-american results\greeks_v8_american_arithmetic_smoke.json ^
  --maturity-days 30 ^
  --output-dir figures\greeks_v8_smoke
if errorlevel 1 exit /b 1

echo V8 smoke campaign completed successfully.
endlocal
