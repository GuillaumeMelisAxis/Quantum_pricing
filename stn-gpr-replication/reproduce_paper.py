r"""Reproduce every figure and table of ``paper/rapport_final.tex``.

Each experiment of the paper is a folder in ``experiments/`` (numbered in the
order of the paper), writes its data to the matching folder in ``results/`` and
its figures to ``paper/figures/``, at the exact paths ``\includegraphics`` uses.
This script runs them in order with the settings used for the publication.

    python reproduce_paper.py --list                 # what will run, and how long
    python reproduce_paper.py --smoke                # whole pipeline, tiny settings
    python reproduce_paper.py                        # everything, paper settings
    python reproduce_paper.py --only greeks var_es   # selected experiments
    python reproduce_paper.py --figures-only         # redraw from existing results
    python reproduce_paper.py --dry-run              # print the commands only

``--smoke`` writes to ``results_smoke/`` and never touches ``results/`` or
``paper/figures/``: it checks that the pipeline runs on a new machine in
minutes, its numbers are meaningless.

Every step is a plain command; ``--dry-run`` prints them so any single step can
be rerun by hand.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = Path("experiments")


@dataclass
class Step:
    script: str
    paper: list[str]
    smoke: list[str]
    figure: bool = False


@dataclass
class Experiment:
    name: str
    section: str
    produces: str
    duration: str
    steps: list[Step] = field(default_factory=list)


def experiments(out: str, fig: str) -> list[Experiment]:
    """The pipeline. ``out`` is the results root, ``fig`` the figure folder."""
    node = f"{out}/01_node_placement"
    eur = f"{out}/02_european_pricing"
    ame = f"{out}/03_american_pricing"
    grk = f"{out}/04_greeks"
    vol = f"{out}/05_volatility_cost"
    qry = f"{out}/06_query_benchmark"
    var = f"{out}/07_var_es"
    return [
        Experiment(
            "method_illustrations",
            "Methodology / Appendix E",
            "convexity-grid-comparison, sinh-moneyness-discretization",
            "seconds",
            [
                Step(
                    "00_method_illustrations/plot_convexity_heatmaps.py",
                    ["--output-dir", fig],
                    ["--output-dir", fig],
                    figure=True,
                ),
                Step(
                    "00_method_illustrations/plot_sinh_discretization.py",
                    ["--output-dir", fig],
                    ["--output-dir", fig],
                    figure=True,
                ),
            ],
        ),
        Experiment(
            "node_placement",
            "Results 5.1",
            "greeks_v992_node_displacement / coverage_gain / node_concentration",
            "minutes",
            [
                Step(
                    "01_node_placement/analyze_grid_node_redistribution_v992.py",
                    ["--profile", "paper", "--output", f"{node}/grid_geometry.json"],
                    ["--profile", "smoke", "--output", f"{node}/grid_geometry.json"],
                ),
                Step(
                    "01_node_placement/plot_grid_node_redistribution_v992.py",
                    ["--input", f"{node}/grid_geometry.json", "--output-dir", fig],
                    ["--input", f"{node}/grid_geometry.json", "--output-dir", fig],
                    figure=True,
                ),
            ],
        ),
        Experiment(
            "european_pricing",
            "Results 5.2.1, Table 4",
            "results-european-grid, European table",
            "not recorded",
            [
                Step(
                    "02_european_pricing/reproduce_european.py",
                    [
                        "--profile", "intermediate", "--grid-mode", "paper",
                        "--output", f"{eur}/european_paper.json",
                    ],
                    [
                        "--profile", "smoke", "--grid-mode", "paper",
                        "--output", f"{eur}/european_paper.json",
                    ],
                ),
                Step(
                    "02_european_pricing/reproduce_european.py",
                    [
                        "--profile", "intermediate",
                        "--grid-mode", "moneyness_adaptive",
                        "--output", f"{eur}/european_moneyness_adaptive.json",
                    ],
                    [
                        "--profile", "smoke", "--grid-mode", "moneyness_adaptive",
                        "--output", f"{eur}/european_moneyness_adaptive.json",
                    ],
                ),
                Step(
                    "02_european_pricing/plot_european_grid.py",
                    [
                        "--original", f"{eur}/european_paper.json",
                        "--adaptive", f"{eur}/european_moneyness_adaptive.json",
                        "--output-dir", fig,
                    ],
                    [
                        "--original", f"{eur}/european_paper.json",
                        "--adaptive", f"{eur}/european_moneyness_adaptive.json",
                        "--output-dir", fig,
                    ],
                    figure=True,
                ),
            ],
        ),
        Experiment(
            "american_pricing",
            "Results 5.2.2, Table 5",
            "results-american-grid, American table, LSMC oracle checks",
            "long, not recorded",
            [
                # 60 one-dimensional puts against a 3,000-step binomial tree.
                Step(
                    "03_american_pricing/validate_lsmc_1d.py",
                    ["--profile", "intermediate", "--output", f"{ame}/lsmc_1d.json"],
                    ["--profile", "smoke", "--output", f"{ame}/lsmc_1d.json"],
                ),
                # Training oracle (2,000 paths, 15 dates) against the reference.
                Step(
                    "03_american_pricing/lsmc_convergence.py",
                    [
                        "--profile", "intermediate",
                        "--output", f"{ame}/lsmc_convergence.json",
                    ],
                    [
                        "--profile", "smoke",
                        "--output", f"{ame}/lsmc_convergence.json",
                    ],
                ),
                # 2 grids x 3 budgets x 5 seeds = 30 fits; resumes if interrupted.
                Step(
                    "03_american_pricing/validate_american_grid_robustness.py",
                    [
                        "--profile", "intermediate",
                        "--output", f"{ame}/american_grid_robustness.json",
                    ],
                    [
                        "--profile", "smoke",
                        "--output", f"{ame}/american_grid_robustness.json",
                    ],
                ),
                Step(
                    "03_american_pricing/plot_american_grid.py",
                    [
                        "--input", f"{ame}/american_grid_robustness.json",
                        "--output-dir", fig,
                    ],
                    [
                        "--input", f"{ame}/american_grid_robustness.json",
                        "--output-dir", fig,
                    ],
                    figure=True,
                ),
            ],
        ),
        Experiment(
            "greeks",
            "Results 5.3.1, Tables 6-7",
            "greeks_v98_dense_profiles, Greek and dense-audit tables",
            "long, not recorded",
            [
                Step(
                    "04_greeks/compare_pricing_risk_grids.py",
                    [
                        "--profile", "paper",
                        "--core-cache-dir", f"{grk}/cache",
                        "--output", f"{grk}/pricing_vs_risk_grid.json",
                    ],
                    [
                        "--profile", "smoke",
                        "--core-cache-dir", f"{grk}/cache",
                        "--output", f"{grk}/pricing_vs_risk_grid.json",
                    ],
                ),
                Step(
                    "04_greeks/plot_pricing_risk_grid_comparison.py",
                    [
                        "--input", f"{grk}/pricing_vs_risk_grid.json",
                        "--output-dir", fig,
                    ],
                    [
                        "--input", f"{grk}/pricing_vs_risk_grid.json",
                        "--output-dir", fig,
                    ],
                    figure=True,
                ),
            ],
        ),
        Experiment(
            "volatility_cost",
            "Results 5.3.2",
            "volatility_dimension/figure1_statistical_cost_d0_d5, cost numbers",
            "~2.6 h (recorded run)",
            [
                # Pre-registered ladder: 5 d_sigma x 4 budgets x 5 seeds.
                Step(
                    "05_volatility_cost/phase4_dsigma_sweep.py",
                    ["--n-sigma", "16", "--output-dir", vol, "--tag", "phase4"],
                    [
                        "--n-sigma", "16", "--d-sigma", "0", "5",
                        "--budgets", "5000", "--seeds", "20260401",
                        "--points", "200", "--skip-cell-sweep",
                        "--output-dir", vol, "--tag", "phase4",
                    ],
                ),
                # Post-hoc 600k tier, compared against the same floors.
                Step(
                    "05_volatility_cost/phase4_dsigma_sweep.py",
                    [
                        "--n-sigma", "16", "--budgets", "600000",
                        "--skip-cell-sweep",
                        "--floors-from", f"{vol}/phase4_run.json",
                        "--output-dir", vol, "--tag", "phase4b",
                    ],
                    [
                        "--n-sigma", "16", "--d-sigma", "0", "5",
                        "--budgets", "10000", "--seeds", "20260401",
                        "--points", "200", "--skip-cell-sweep",
                        "--floors-from", f"{vol}/phase4_run.json",
                        "--output-dir", vol, "--tag", "phase4b",
                    ],
                ),
                Step(
                    "05_volatility_cost/merge_budget_tiers.py",
                    [
                        "--base", "phase4", "--extension", "phase4b",
                        "--tag", "phase4_extended", "--dir", vol,
                    ],
                    [
                        "--base", "phase4", "--extension", "phase4b",
                        "--tag", "phase4_extended", "--dir", vol,
                    ],
                ),
                Step(
                    "05_volatility_cost/phase5_figures.py",
                    [
                        "--tag", "phase4_extended", "--figure1-only", "0", "5",
                        "--input-dir", vol,
                        "--output-dir", f"{fig}/volatility_dimension",
                    ],
                    [
                        "--tag", "phase4_extended", "--figure1-only", "0", "5",
                        "--input-dir", vol,
                        "--output-dir", f"{fig}/volatility_dimension",
                    ],
                    figure=True,
                ),
            ],
        ),
        Experiment(
            "query_benchmark",
            "Results 5.4.1",
            "VARfigures/e3_accuracy_vs_time",
            "~20 min",
            [
                Step(
                    "06_query_benchmark/benchmark_vol_risk_grids.py",
                    [
                        "--profile", "intermediate",
                        "--experiments", "e3", "e4", "e5",
                        "--output", f"{qry}/vol_risk_grid_benchmark_intermediate.json",
                    ],
                    [
                        "--profile", "smoke", "--experiments", "e3",
                        "--output", f"{qry}/vol_risk_grid_benchmark_intermediate.json",
                    ],
                ),
                Step(
                    "06_query_benchmark/make_fig_e3_accuracy_vs_time.py",
                    [
                        "--input", f"{qry}/vol_risk_grid_benchmark_intermediate.json",
                        "--output-dir", f"{fig}/VARfigures",
                    ],
                    [
                        "--input", f"{qry}/vol_risk_grid_benchmark_intermediate.json",
                        "--output-dir", f"{fig}/VARfigures",
                    ],
                    figure=True,
                ),
            ],
        ),
        Experiment(
            "var_es",
            "Results 5.4.2, Tables 8-9 and E",
            "VaR/ES tables (results/07_var_es/var_qtt_benchmark_bench_es.md)",
            "~30 min",
            [
                Step(
                    "07_var_es/benchmark_var_qtt_basket_book.py",
                    [
                        "--profile", "full", "--budget", "300000",
                        "--qtt-budget-sweep",
                        "50000", "100000", "150000", "200000", "300000", "400000",
                        "--mc-path-sweep", "1024", "4096", "16384", "65536",
                        "--mc-sweep-seeds", "4242", "911", "20260921",
                        "--output-dir", var, "--tag", "bench_es",
                    ],
                    ["--profile", "smoke", "--output-dir", var, "--tag", "bench_es"],
                ),
            ],
        ),
    ]


def record_environment(path: Path) -> None:
    """Versions and machine, next to the results they produced."""
    packages = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "teneva"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    payload = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    every = [experiment.name for experiment in experiments("", "")]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", choices=every, metavar="NAME")
    parser.add_argument("--skip", nargs="+", choices=every, default=[], metavar="NAME")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue with the next experiment when a step fails",
    )
    args = parser.parse_args()

    out = "results_smoke" if args.smoke else "results"
    fig = "results_smoke/paper_figures" if args.smoke else "paper/figures"
    plan = [
        experiment
        for experiment in experiments(out, fig)
        if (args.only is None or experiment.name in args.only)
        and experiment.name not in args.skip
    ]

    if args.list:
        for index, experiment in enumerate(experiments(out, fig)):
            print(f"{index:02d} {experiment.name:<22} {experiment.section}")
            print(f"   produces: {experiment.produces}")
            print(f"   duration: {experiment.duration} (paper settings)")
        return 0

    if not args.dry_run:
        record_environment(ROOT / out / "environment.json")

    failed = []
    for experiment in plan:
        print(f"\n=== {experiment.name} ({experiment.section})", flush=True)
        for step in experiment.steps:
            if args.figures_only and not step.figure:
                continue
            command = [
                sys.executable,
                str(EXP / step.script),
                *(step.smoke if args.smoke else step.paper),
            ]
            print("$ " + " ".join(command), flush=True)
            if args.dry_run:
                continue
            start = time.perf_counter()
            result = subprocess.run(command, cwd=ROOT, check=False)
            elapsed = time.perf_counter() - start
            if result.returncode != 0:
                print(f"!!! failed after {elapsed:.0f} s", flush=True)
                failed.append(experiment.name)
                if not args.keep_going:
                    return result.returncode
                break
            print(f"--- done in {elapsed:.0f} s", flush=True)

    if failed:
        print("\nfailed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
