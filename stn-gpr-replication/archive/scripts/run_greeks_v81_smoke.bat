@echo off
setlocal

if not exist results mkdir results
if not exist figures\greeks_v81_reference_convergence mkdir figures\greeks_v81_reference_convergence

echo [1/4] Running unit tests...
python -m pytest
if errorlevel 1 exit /b 1

echo [2/4] European arithmetic QMC reference convergence...
python scripts\validation\converge_european_arithmetic_reference.py ^
  --profile smoke ^
  --output results\greeks_v81_european_reference_smoke.json
if errorlevel 1 exit /b 1

echo [3/4] American arithmetic LSMC reference convergence...
python scripts\validation\converge_american_reference.py ^
  --profile smoke ^
  --output results\greeks_v81_american_reference_smoke.json
if errorlevel 1 exit /b 1

echo [4/4] Plotting reference-convergence diagnostics...
python scripts\figures\plot_reference_convergence.py ^
  --european results\greeks_v81_european_reference_smoke.json ^
  --american results\greeks_v81_american_reference_smoke.json ^
  --output-dir figures\greeks_v81_reference_convergence
if errorlevel 1 exit /b 1

echo V8.1 reference-convergence smoke campaign completed successfully.
endlocal
