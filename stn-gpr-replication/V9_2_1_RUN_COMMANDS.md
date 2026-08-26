# Version 9.2.1 commands — Windows x64 Native Tools Command Prompt

Run these commands from the project root with the virtual environment active.

## Smoke gate — 41 moneyness values per curve

```bat
python -m pip install -e ".[dev]"
scripts\run_greeks_v921_smoke.bat
```

## Intermediate visual audit — 101 values per curve

```bat
python scripts\validation\validate_dense_greek_profiles.py ^
  --profile intermediate ^
  --output results\greeks_v921_dense_profiles_intermediate.json

python scripts\figures\plot_dense_greek_profiles.py ^
  --input results\greeks_v921_dense_profiles_intermediate.json ^
  --relative-bump 0.002 ^
  --risk-column 0 ^
  --all-bumps ^
  --output-dir figures\greeks_v921_dense_profiles_intermediate
```

## Paper visual audit — 201 values per curve

```bat
python scripts\validation\validate_dense_greek_profiles.py ^
  --profile paper ^
  --output results\greeks_v921_dense_profiles_paper.json

python scripts\figures\plot_dense_greek_profiles.py ^
  --input results\greeks_v921_dense_profiles_paper.json ^
  --relative-bump 0.002 ^
  --risk-column 0 ^
  --all-bumps ^
  --output-dir figures\greeks_v921_dense_profiles_paper
```

The paper profile evaluates 804 market points: 201 moneyness values at each of
3, 7, 30 and 90 days on the `oos_mid` spot panel.  Both 0.1% and 0.2% bumps are
stored.  `--all-bumps` creates sixteen PNG/PDF figure pairs: two views for each
`(maturity, bump)` combination, comprising a complete-method comparison and a
zoom containing only the analytical, exact-price FD and risk-hybrid cubic
curves.  Every figure shows Delta, diagonal Gamma and cross-Gamma on its first
row and their pointwise absolute errors on its second row.
