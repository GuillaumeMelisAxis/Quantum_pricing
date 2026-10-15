r"""Phase 2 - blocking check that the two grids place the same physical nodes.

The risk-hybrid strike coordinate is a function of ``(r, T, sigma)`` through the
convexity ridge ``m*(r, T, sigma)`` and the scale ``a(T, sigma) = sigma_G(sigma)
sqrt(T)``. If the 13-D grid evaluated at ``sigma_1 = ... = sigma_5 = sigma_ref``
did not place its log-moneyness nodes where the 8-D grid places them, a
difference in NMAE between the two would mix a dimension effect with a residual
coordinate effect and the comparison would be void.

Two grid constructions are compared:

* the frozen-volatility production path, ``risk_grids.build_greek_coordinate_grid``
  on a ``PaperConfig`` of the risk-hybrid shape - what the existing papers used;
* the sigma-augmented path of this experiment at each ``d_sigma`` in the sweep,
  evaluated on the common slice.

Both are swept over a grid of ``(r, T)`` covering the whole domain and reported
as normalized displacements ``D_2`` and ``D_inf``. The gate is machine
precision, ``< 1e-12``. Failing it stops the experiment.

    .venv/Scripts/python.exe experiments/05_volatility_cost/phase2_node_coincidence.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    D_SIGMA_SWEEP,
    OUTPUT_DIR,
    PHASE2_GATE,
    RISK_GRID,
    SIGMA_BOUNDS,
    SIGMA_REF,
    run_manifest,
    write_csv,
    write_json,
)

from stngpr.config import PaperConfig
from stngpr.grid_geometry import (
    normalized_node_displacement,
    physical_log_moneyness_nodes,
)
from stngpr.risk_grids import build_greek_coordinate_grid
from stngpr.sigma_axes import (
    SigmaAugmentedConfig,
    build_sigma_augmented_grid,
)


def paper_config() -> PaperConfig:
    shape = (
        *(RISK_GRID["spot_nodes"],) * 5,
        RISK_GRID["strike_nodes"],
        RISK_GRID["rate_nodes"],
        RISK_GRID["maturity_nodes"],
    )
    return replace(PaperConfig(), physical_shape=shape)


def augmented_config(n_sigma_axes: int, n_sigma_nodes: int) -> SigmaAugmentedConfig:
    return SigmaAugmentedConfig(
        n_sigma_axes=n_sigma_axes,
        sigma_block="tail",
        sigma_reference=SIGMA_REF,
        volatility_bounds=SIGMA_BOUNDS,
        volatility_nodes=n_sigma_nodes,
        spot_nodes=RISK_GRID["spot_nodes"],
        strike_nodes=RISK_GRID["strike_nodes"],
        rate_nodes=RISK_GRID["rate_nodes"],
        maturity_nodes=RISK_GRID["maturity_nodes"],
    )


def augmented_log_moneyness_nodes(
    config, grid, transform, rate: float, maturity: float, sigma: float
) -> np.ndarray:
    """Physical ``m = log(K / G(S))`` of the strike axis at one market regime.

    The spots cancel in the ratio, so the reference spot below only serves to
    build a valid market vector - the same argument as
    ``grid_geometry.physical_log_moneyness_nodes``.
    """
    coordinate = np.asarray(grid.axes[config.strike_column], dtype=float)
    model = np.empty((coordinate.size, config.n_dimensions), dtype=float)
    model[:, list(config.spot_columns)] = 100.0
    if config.sigma_columns:
        model[:, list(config.sigma_columns)] = float(sigma)
    model[:, config.strike_column] = coordinate
    model[:, config.rate_column] = float(rate)
    model[:, config.maturity_column] = float(maturity)
    market = transform.to_market(model)
    basket = np.exp(np.mean(np.log(market[:, list(config.spot_columns)]), axis=1))
    nodes = np.log(market[:, config.strike_column] / basket)
    if np.any(~np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("the physical log-moneyness nodes are not strictly increasing")
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sigma", type=int, required=True)
    parser.add_argument("--rate-points", type=int, default=9)
    parser.add_argument("--maturity-points", type=int, default=9)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    paper = paper_config()
    grid8, transform8, description8 = build_greek_coordinate_grid(
        paper, RISK_GRID["mode"]
    )

    rates = np.linspace(*paper.rate_bounds, int(args.rate_points))
    # Quadratic in the maturity sweep too, so the short end - where the scale
    # sqrt(T) is smallest and any coordinate mismatch is most amplified - is
    # actually probed rather than skipped over.
    maturities = paper.maturity_bounds[0] + (
        paper.maturity_bounds[1] - paper.maturity_bounds[0]
    ) * np.linspace(0.0, 1.0, int(args.maturity_points)) ** 2

    domain = tuple(description8["coordinate_bounds"])
    print(
        f"phase 2: {len(rates)} rates x {len(maturities)} maturities, "
        f"sigma = {SIGMA_REF}, gate D < {PHASE2_GATE:.0e}"
    )

    rows, worst = [], {}
    for d_sigma in D_SIGMA_SWEEP:
        config = augmented_config(d_sigma, int(args.n_sigma))
        grid, transform, _ = build_sigma_augmented_grid(config, RISK_GRID["mode"])
        per_config = []
        for rate in rates:
            for maturity in maturities:
                reference = physical_log_moneyness_nodes(
                    grid8, transform8, paper.n_assets, float(rate), float(maturity)
                )
                candidate = augmented_log_moneyness_nodes(
                    config, grid, transform, float(rate), float(maturity), SIGMA_REF
                )
                displacement = normalized_node_displacement(
                    reference,
                    candidate,
                    domain=(
                        min(float(reference[0]), float(candidate[0])),
                        max(float(reference[-1]), float(candidate[-1])),
                    ),
                )
                row = {
                    "d_sigma": int(d_sigma),
                    "n_sigma": int(args.n_sigma),
                    "rate": float(rate),
                    "maturity": float(maturity),
                    "D_2": displacement["normalized_l2"],
                    "D_inf": displacement["normalized_linf"],
                    "absolute_l2": displacement["absolute_l2"],
                    "absolute_linf": displacement["absolute_linf"],
                }
                rows.append(row)
                per_config.append(row)
        worst[d_sigma] = {
            "max_D_2": max(row["D_2"] for row in per_config),
            "max_D_inf": max(row["D_inf"] for row in per_config),
            "argmax_D_2": max(per_config, key=lambda row: row["D_2"]),
        }
        print(
            f"  d_sigma={d_sigma}: max D_2 = {worst[d_sigma]['max_D_2']:.3e}  "
            f"max D_inf = {worst[d_sigma]['max_D_inf']:.3e}"
        )

    overall_d2 = max(entry["max_D_2"] for entry in worst.values())
    overall_dinf = max(entry["max_D_inf"] for entry in worst.values())
    passed = bool(overall_d2 < PHASE2_GATE and overall_dinf < PHASE2_GATE)

    conclusion = (
        f"Over {len(rates) * len(maturities)} (r, T) regimes and every d_sigma in "
        f"{list(D_SIGMA_SWEEP)}, the sigma-augmented grid evaluated at "
        f"sigma = {SIGMA_REF} places its log-moneyness nodes at "
        f"max D_2 = {overall_d2:.3e} and max D_inf = {overall_dinf:.3e} from the "
        f"frozen-volatility production grid: "
        + ("PASS" if passed else "FAIL")
        + f" against the {PHASE2_GATE:.0e} machine-precision gate."
    )

    write_csv(Path(args.output_dir) / "phase2_coincidence_noeuds.csv", rows)
    write_json(
        Path(args.output_dir) / "phase2_coincidence_noeuds.json",
        {
            "manifest": run_manifest(),
            "n_sigma": int(args.n_sigma),
            "rates": rates.tolist(),
            "maturities": maturities.tolist(),
            "coordinate_bounds": list(domain),
            "worst_by_d_sigma": worst,
            "max_D_2": overall_d2,
            "max_D_inf": overall_dinf,
            "gate": PHASE2_GATE,
            "passed": passed,
            "conclusion": conclusion,
        },
    )
    print(f"\n{conclusion}")

    if not passed:
        print(
            "\nSTOP: the two grid constructions do not agree on the common slice. "
            "No production experiment is run until this is fixed."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
