# Version 9.3 run commands

Run these commands from the repository root in the x64 Native Tools Command
Prompt after activating the virtual environment.

## Smoke gate

```bat
python -m pip install -e ".[dev]"
scripts\run_greeks_v93_smoke.bat
```

The smoke profile uses 31 moneyness values, the 7- and 30-day maturities and
the `oos_mid` spot panel.

## Paper profile

```bat
python scripts\validation\validate_full_spot_hessian.py ^
  --profile paper ^
  --output results\greeks_v93_full_hessian_paper.json

python scripts\figures\plot_full_spot_hessian.py ^
  --input results\greeks_v93_full_hessian_paper.json ^
  --panel oos_mid ^
  --maturity-days 30 ^
  --output-dir figures\greeks_v93_full_hessian_paper
```

The paper profile evaluates 101 moneyness values, 3-, 7-, 30- and 90-day
maturities, and the `oos_low`, `oos_mid` and `oos_dispersed` panels.  It stores
1,212 market points and applies the full 51-price stencil at every point.

The figure command writes PNG and PDF versions of:

- the five Delta errors and the symmetric 5x5 Hessian error map;
- the analytical, risk-hybrid and signed-error ATM Hessian matrices;
- the minimum Hessian eigenvalue along every stored maturity curve.
