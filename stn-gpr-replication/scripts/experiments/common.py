"""Shared, frozen protocol constants for the dimensional-cost experiment.

Everything a run needs in order to be reproducible from the manifest alone lives
here, and nothing here is a tunable. It is imported by every phase script so
that a value cannot drift between phases.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---- frozen protocol parameters -------------------------------------------

SIGMA_REF = 0.20
SIGMA_BOUNDS = (0.05, 0.60)
SIGMA_OFF_NODE = 0.235

RISK_GRID = {
    "mode": "bounded_standardized_risk",
    "spot_nodes": 64,
    "strike_nodes": 512,
    "rate_nodes": 8,
    "maturity_nodes": 64,
}
PRICING_GRID = {
    "mode": "price_adaptive",
    "spot_nodes": 32,
    "strike_nodes": 64,
    "rate_nodes": 8,
    "maturity_nodes": 8,
}

D_SIGMA_SWEEP = (0, 1, 2, 3, 5)
TT_BUDGETS = (20_000, 60_000, 150_000, 300_000)
SEEDS = (20260401, 20260402, 20260403, 20260404, 20260405)

ANOVA_SAMPLES = 2_000
MAX_SWEEPS = 20
RANK_INCREMENT = 2
TRUNCATION = None

# Test-set seeds. Distinct from the fit seeds so a test set is never a subset of
# anything the cross saw, and frozen here so every phase draws the same points.
TEST_SEED_SHARED = 20260327
TEST_SEED_SIGMA_BOX = 20260328
N_TEST_POINTS = 2_000
N_LATENCY_POINTS = 150

UNRESOLVED_FACTOR = 3.0
PHASE1_TOLERANCE = 1.10
PHASE2_GATE = 1e-12

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "experiments"


# ---- run manifest ---------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_manifest() -> dict:
    """Everything a third party needs to reproduce the numbers."""
    import scipy
    import teneva

    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": _git("status", "--porcelain"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "teneva": teneva.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "frozen": {
            "sigma_reference": SIGMA_REF,
            "sigma_bounds": list(SIGMA_BOUNDS),
            "sigma_off_node": SIGMA_OFF_NODE,
            "risk_grid": RISK_GRID,
            "pricing_grid": PRICING_GRID,
            "d_sigma_sweep": list(D_SIGMA_SWEEP),
            "tt_budgets": list(TT_BUDGETS),
            "seeds": list(SEEDS),
            "anova_samples": ANOVA_SAMPLES,
            "max_sweeps": MAX_SWEEPS,
            "rank_increment": RANK_INCREMENT,
            "post_fit_svd_truncation": TRUNCATION,
            "correlation": "equicorrelation(5, 0.30)",
            "dividends": "zeros(5)",
            "fit_target": "P / G(S) in every configuration",
            "interpolant": "mixed-order unified interpolant, cubic on every axis",
            "sigma_axis_scale": "raw sigma, uniform nodes",
            "promoted_assets": "the first d_sigma assets, fixed order",
            "test_seed_shared": TEST_SEED_SHARED,
            "test_seed_sigma_box": TEST_SEED_SIGMA_BOX,
            "n_test_points": N_TEST_POINTS,
            "n_latency_points": N_LATENCY_POINTS,
            "unresolved_factor": UNRESOLVED_FACTOR,
        },
    }


# ---- test-point construction ---------------------------------------------


def shared_axis_points(n_samples: int, config, seed: int) -> np.ndarray:
    """Uniform draws on the eight shared axes, in their canonical order.

    Returned as ``(n, n_assets + 3)`` — spots, strike, rate, maturity — and
    identical for every configuration, which is what makes the NMAE on the
    common slice comparable across ``d_sigma``.
    """
    rng = np.random.default_rng(int(seed))
    return np.column_stack(
        [
            *(
                rng.uniform(*config.spot_bounds, n_samples)
                for _ in range(config.n_assets)
            ),
            rng.uniform(*config.strike_bounds, n_samples),
            rng.uniform(*config.rate_bounds, n_samples),
            rng.uniform(*config.maturity_bounds, n_samples),
        ]
    )


def sigma_box_draws(n_samples: int, n_assets: int, seed: int) -> np.ndarray:
    """``(n, n_assets)`` volatilities, uniform and independent per asset."""
    rng = np.random.default_rng(int(seed))
    return rng.uniform(*SIGMA_BOUNDS, (n_samples, n_assets))


def assemble_market_points(config, shared: np.ndarray, sigma: np.ndarray):
    """Place shared-axis values and volatilities at this config's columns.

    ``sigma`` is always a full ``(n, n_assets)`` matrix. Only the promoted
    columns are written into the parameter vector; the rest of the matrix is
    what the oracle would have used anyway, and is dropped because the
    configuration freezes those assets at ``sigma_reference``.
    """
    shared = np.atleast_2d(np.asarray(shared, dtype=float))
    sigma = np.atleast_2d(np.asarray(sigma, dtype=float))
    out = np.empty((shared.shape[0], config.n_dimensions), dtype=float)
    for position, column in enumerate(config.shared_columns):
        out[:, column] = shared[:, position]
    if config.sigma_columns:
        out[:, list(config.sigma_columns)] = sigma[:, list(config.promoted_assets)]
    return out


def sigma_node_placement(n_sigma_nodes: int) -> dict:
    """Where ``sigma_reference`` falls relative to the sigma axis nodes."""
    axis = np.linspace(*SIGMA_BOUNDS, int(n_sigma_nodes))
    step = float(axis[1] - axis[0])
    upper = int(np.searchsorted(axis, SIGMA_REF, side="right"))
    upper = min(max(upper, 1), axis.size - 1)
    lower = upper - 1
    return {
        "n_sigma": int(n_sigma_nodes),
        "sigma_axis_step": step,
        "bracketing_node_low": float(axis[lower]),
        "bracketing_node_high": float(axis[upper]),
        "cell_fraction_of_sigma_ref": float(
            (SIGMA_REF - axis[lower]) / (axis[upper] - axis[lower])
        ),
        "distance_to_nearest_node": float(
            min(SIGMA_REF - axis[lower], axis[upper] - SIGMA_REF)
        ),
        "sigma_ref_on_node": bool(np.any(np.abs(axis - SIGMA_REF) < 1e-15)),
    }


# ---- output helpers -------------------------------------------------------


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(fieldnames or rows[0].keys())
    # A fit record carries more than the table shows - the full bond-rank
    # profile, the core index lists - which belong in the JSON, not in a CSV
    # cell. Dropping them here rather than raising keeps the caller's column
    # list the single definition of the table.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
