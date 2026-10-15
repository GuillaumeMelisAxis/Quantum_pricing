# V8.1 local execution commands

Run every command from the project root in the Windows x64 Native Tools Command
Prompt, after activating the existing virtual environment.

## 1. Install the updated package

```bat
python -m pip install -e .
```

## 2. Complete smoke gate

```bat
scripts\run_greeks_v81_smoke.bat
```

Expected result files:

- `results\greeks_v81_european_reference_smoke.json`
- `results\greeks_v81_american_reference_smoke.json`
- `figures\greeks_v81_reference_convergence\european_reference_convergence.png`
- `figures\greeks_v81_reference_convergence\american_reference_convergence.png`

The batch stops immediately if a unit test or experiment fails.

## 3. Intermediate European QMC experiment

```bat
python scripts\validation\converge_european_arithmetic_reference.py ^
  --profile intermediate ^
  --output results\greeks_v81_european_reference_intermediate.json
```

This crosses 4,096, 16,384 and 65,536 Sobol paths; 0.2%, 0.5% and 1%
fixed-strike bumps; five independent scramblings; and estimated versus unit
control-variate coefficients.

## 4. Intermediate American LSMC experiment

```bat
python scripts\validation\converge_american_reference.py ^
  --profile intermediate ^
  --output results\greeks_v81_american_reference_intermediate.json
```

This crosses 2,000 and 10,000 paths; 20 and 40 exercise steps; 0.5% and 1%
bumps; five seeds; and refit, frozen and frozen-out-of-sample policies. It uses
the nine-price two-factor diagnostic stencil.

## 5. Plot the intermediate results

```bat
python scripts\figures\plot_reference_convergence.py ^
  --european results\greeks_v81_european_reference_intermediate.json ^
  --american results\greeks_v81_american_reference_intermediate.json ^
  --american-steps 40 ^
  --output-dir figures\greeks_v81_reference_convergence_intermediate
```

Send back both JSON files. The PNG files are useful for visual inspection, but
the estimator selection will be based on the component standard errors,
successive-path differences, bump stability, Frobenius Hessian uncertainty and
policy/beta-mode bias stored in the JSON.
