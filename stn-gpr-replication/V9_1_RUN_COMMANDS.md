# Version 9.1 commands — Windows x64 Native Tools Command Prompt

Run these commands from the project root with the virtual environment active.

## Smoke gate

```bat
python -m pip install -e ".[dev]"
python -m pytest

python scripts\validation\ablate_standardized_spot_resolution.py ^
  --profile smoke ^
  --output results\greeks_v91_spot_resolution_smoke.json

python scripts\figures\plot_spot_resolution_ablation.py ^
  --input results\greeks_v91_spot_resolution_smoke.json ^
  --output-dir figures\greeks_v91_spot_resolution_smoke
```

## Controlled intermediate run

```bat
python scripts\validation\ablate_standardized_spot_resolution.py ^
  --profile intermediate ^
  --output results\greeks_v91_spot_resolution_intermediate.json

python scripts\figures\plot_spot_resolution_ablation.py ^
  --input results\greeks_v91_spot_resolution_intermediate.json ^
  --output-dir figures\greeks_v91_spot_resolution_intermediate
```

## Paper experiment

```bat
python scripts\validation\ablate_standardized_spot_resolution.py ^
  --profile paper ^
  --output results\greeks_v91_spot_resolution_paper.json

python scripts\figures\plot_spot_resolution_ablation.py ^
  --input results\greeks_v91_spot_resolution_paper.json ^
  --maturity-days 30 ^
  --risk-column 0 ^
  --output-dir figures\greeks_v91_spot_resolution_paper
```

The paper profile contains 189 fixed market points from three distinct spot
panels. It changes only the five spot-axis resolutions. The main diagnostics
are `spot_grids.<N>.error_against_analytical`, the region-specific errors under
`regional_error_against_analytical`, and the consecutive observed orders under
`convergence`.

The plotting command also creates direct and pointwise-error profiles for
Delta and diagonal Gamma. `--maturity-days` selects the nearest maturity in the
fixed validation panel; `--risk-column` is the zero-based underlying index.
