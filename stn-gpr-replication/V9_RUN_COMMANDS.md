# Version 9 commands — Windows x64 Native Tools Command Prompt

Run every command from the project root after activating the existing virtual
environment.

## 1. Installation and unit tests

```bat
python -m pip install -e ".[dev]"
python -m pytest
```

## 2. Fast implementation gate

```bat
python scripts\validation\compare_greek_coordinate_grids.py ^
  --profile smoke ^
  --output results\greeks_v9_coordinate_ablation_smoke.json

python scripts\figures\plot_greek_coordinate_ablation.py ^
  --input results\greeks_v9_coordinate_ablation_smoke.json ^
  --output-dir figures\greeks_v9_coordinate_ablation_smoke
```

## 3. First scientific run

```bat
python scripts\validation\compare_greek_coordinate_grids.py ^
  --profile intermediate ^
  --output results\greeks_v9_coordinate_ablation_intermediate.json

python scripts\figures\plot_greek_coordinate_ablation.py ^
  --input results\greeks_v9_coordinate_ablation_intermediate.json ^
  --output-dir figures\greeks_v9_coordinate_ablation_intermediate
```

## 4. High-resolution confirmation

Run this only after inspecting the intermediate JSON and figures.

```bat
python scripts\validation\compare_greek_coordinate_grids.py ^
  --profile paper ^
  --output results\greeks_v9_coordinate_ablation_paper.json

python scripts\figures\plot_greek_coordinate_ablation.py ^
  --input results\greeks_v9_coordinate_ablation_paper.json ^
  --output-dir figures\greeks_v9_coordinate_ablation_paper
```

## Optional controlled overrides

The number of nodes must remain a power of two for QTT compatibility.

```bat
python scripts\validation\compare_greek_coordinate_grids.py ^
  --profile intermediate ^
  --moneyness-nodes 512 ^
  --maturity-nodes 64 ^
  --risk-columns 0 1 ^
  --relative-bump 0.002 ^
  --output results\greeks_v9_coordinate_ablation_512.json
```

The principal JSON comparison is
`grids.<mode>.error_against_analytical`. The field
`error_against_exact_price_finite_difference` isolates interpolation from bump
truncation. Do not select a grid from price MAE alone: inspect diagonal Gamma,
cross-Gamma, the joint score, conditional errors and Hessian shape together.
