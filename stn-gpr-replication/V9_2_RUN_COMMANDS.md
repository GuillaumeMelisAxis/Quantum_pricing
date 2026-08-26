# Version 9.2 commands — Windows x64 Native Tools Command Prompt

Run every command from the project root with the virtual environment active.

## Smoke gate

```bat
python -m pip install -e ".[dev]"
scripts\run_greeks_v92_smoke.bat
```

Equivalent individual commands:

```bat
python -m pytest

python scripts\validation\decompose_geometric_greek_error_floor.py ^
  --profile smoke ^
  --output results\greeks_v92_error_floor_smoke.json

python scripts\figures\plot_greek_error_floor.py ^
  --input results\greeks_v92_error_floor_smoke.json ^
  --maturity-days 30 ^
  --risk-column 0 ^
  --output-dir figures\greeks_v92_error_floor_smoke
```

## Intermediate gate

```bat
python scripts\validation\decompose_geometric_greek_error_floor.py ^
  --profile intermediate ^
  --output results\greeks_v92_error_floor_intermediate.json

python scripts\figures\plot_greek_error_floor.py ^
  --input results\greeks_v92_error_floor_intermediate.json ^
  --maturity-days 30 ^
  --risk-column 0 ^
  --output-dir figures\greeks_v92_error_floor_intermediate
```

## Paper experiment

```bat
python scripts\validation\decompose_geometric_greek_error_floor.py ^
  --profile paper ^
  --output results\greeks_v92_error_floor_paper.json

python scripts\figures\plot_greek_error_floor.py ^
  --input results\greeks_v92_error_floor_paper.json ^
  --maturity-days 30 ^
  --risk-column 0 ^
  --output-dir figures\greeks_v92_error_floor_paper
```

The paper profile fixes `N_S=64`, uses the `standardized_risk` coordinate and
compares 0.05%, 0.1%, 0.2%, 0.5% and 1% relative bumps under multilinear and
risk-hybrid cubic interpolation.  Its 126 market points are built from three
spot panels that were not used to select the v9.1 resolution.

The JSON retains analytical components, exact-price finite differences,
interpolated components and pointwise residuals for every method/bump pair.
The plotter writes four PNG/PDF figure pairs: bump convergence, additive error
decomposition, Delta/Gamma profiles and `(m,T)` error heatmaps.
