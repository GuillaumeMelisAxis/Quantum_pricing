# STN-GPR replication

Independent implementation of *STN-GPR: A Singularity Tensor Network Framework
for Efficient Option Pricing* (Gribben et al., 2026), followed by experiments on
VaR/ES, Greeks and the American exercise region.

## What is reproduced

- five-asset geometric European basket put;
- five-asset arithmetic American basket put using LSMC;
- 8 physical inputs: five spots, strike, rate and time to maturity;
- paper grid `32^5 x 64 x 8 x 8`, encoded as a 37-core QTT;
- order-2 TT-ANOVA initialization and rank-adaptive TT-cross;
- off-grid multilinear interpolation;
- exact GPR baseline with a Laplacian/Matérn-1/2 kernel;
- MAE versus training budget and wall-clock time.

The paper does not publish code and omits the volatility vector, correlation
matrix, dividend yields, precise TT-cross settings, LSMC randomization protocol
and the numerical data behind its figures. Defaults in `PaperConfig` are therefore
explicit reconstruction assumptions, not claimed author settings.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python scripts/reproduce_european.py --profile smoke
python scripts/reproduce_european.py --profile paper
python -m pytest
```

The `smoke` profile validates the complete pipeline with a small pricing budget.
The `paper` profile uses the paper grid and should be run on a machine with ample
CPU time. TT-cross never materializes the full 137.4-billion-entry tensor.

The complete validation design is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Experimental roadmap

1. Match the European error/training-budget curves.
2. Reproduce the American arithmetic-basket experiment with controlled LSMC noise.
3. Compare repriced portfolio VaR and ES, including tail-ranking stability.
4. Compare delta, gamma and cross-gamma surfaces to trusted finite differences.
5. Oversample and diagnose the region around the American exercise boundary.
