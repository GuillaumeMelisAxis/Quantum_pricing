from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _group_matrix_runs(payload):
    grouped = {}
    for run in payload["runs"].values():
        grouped.setdefault(int(run["budget"]), []).append(run)
    return {
        budget: sorted(records, key=lambda run: int(run["tt_seed"]))
        for budget, records in sorted(grouped.items())
    }


def _mean_range(records, getter):
    values = np.asarray([getter(record) for record in records], dtype=float)
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def _plot_with_range(axis, x, summary, label, color):
    mean = np.asarray([record[0] for record in summary])
    lower = np.asarray([record[1] for record in summary])
    upper = np.asarray([record[2] for record in summary])
    axis.plot(x, mean, color=color, marker="o", label=label)
    axis.fill_between(x, lower, upper, color=color, alpha=0.15)


def plot_budget_seed_accuracy(payload, output_dir):
    grouped = _group_matrix_runs(payload)
    budgets = np.asarray(list(grouped), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    components = (
        ("price", "Price", "C0", axes[0]),
        ("delta", r"$\Delta$", "C1", axes[0]),
        ("gamma_diagonal", r"$\Gamma_{ii}$", "C2", axes[1]),
        ("cross_gamma", r"$\Gamma_{ij}$", "C3", axes[1]),
    )
    for name, label, color, axis in components:
        summary = [
            _mean_range(
                grouped[int(budget)],
                lambda run, component=name: 100.0
                * run["global_error_against_analytical"][component][
                    "normalized_mae"
                ],
            )
            for budget in budgets
        ]
        _plot_with_range(axis, budgets, summary, label, color)
    hessian = [
        _mean_range(
            grouped[int(budget)],
            lambda run: 100.0
            * run["full_hessian_error_against_analytical"][
                "aggregate_relative_frobenius_error"
            ],
        )
        for budget in budgets
    ]
    _plot_with_range(axes[1], budgets, hessian, "Full Hessian", "C4")
    axes[0].set_title("Price and first derivative")
    axes[1].set_title("Second derivatives")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Requested TT-cross budget")
        axis.set_ylabel("Normalized error (%)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("v9.4.1 budget × seed stability")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v941_budget_seed_accuracy")


def plot_budget_seed_structure(payload, output_dir):
    grouped = _group_matrix_runs(payload)
    budgets = np.asarray(list(grouped), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    rank = [
        _mean_range(
            grouped[int(budget)],
            lambda run: run["fit"]["effective_rank"],
        )
        for budget in budgets
    ]
    _plot_with_range(axes[0], budgets, rank, "Post-truncation rank", "C0")
    for budget, records in grouped.items():
        axes[0].scatter(
            np.full(len(records), budget),
            [record["fit"]["effective_rank"] for record in records],
            color="C0",
            s=18,
            alpha=0.65,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Requested TT-cross budget")
    axes[0].set_ylabel("Effective QTT rank")
    axes[0].set_title("Rank bifurcation across seeds")
    axes[0].grid(alpha=0.25)

    minimum_sign_agreement = 100.0
    for color, tolerance in zip(
        ("C0", "C1", "C2"),
        ("1e-04", "5e-04", "1e-03"),
        strict=True,
    ):
        summary = [
            _mean_range(
                grouped[int(budget)],
                lambda run, key=tolerance: 100.0
                * run["cross_gamma_sign_sensitivity"][key][
                    "sign_agreement_fraction"
                ],
            )
            for budget in budgets
        ]
        _plot_with_range(
            axes[1],
            budgets,
            summary,
            rf"zero tolerance ${tolerance}$",
            color,
        )
        minimum_sign_agreement = min(
            minimum_sign_agreement,
            *(record[1] for record in summary),
        )
    axes[1].axhline(99.0, color="0.4", linestyle=":", linewidth=1.0)
    axes[1].set_xscale("log")
    axes[1].set_ylim(max(0.0, minimum_sign_agreement - 5.0), 100.5)
    axes[1].set_xlabel("Requested TT-cross budget")
    axes[1].set_ylabel("Cross-Gamma sign agreement (%)")
    axes[1].set_title("Sign materiality sensitivity")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("v9.4.1 QTT structural stability")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v941_budget_seed_structure")


def _group_truncation_variants(payload):
    order = [
        "untruncated" if float(value) == 0.0 else f"{float(value):.0e}"
        for value in payload["settings"]["truncations"]
    ]
    grouped = {}
    for run in payload["runs"].values():
        budget = int(run["budget"])
        for label, variant in run["variants"].items():
            grouped.setdefault((budget, label), []).append(variant)
    return order, grouped


def plot_truncation_accuracy(payload, output_dir):
    order, grouped = _group_truncation_variants(payload)
    budgets = sorted({budget for budget, _ in grouped})
    fig, axes = plt.subplots(
        1,
        len(budgets),
        figsize=(6.0 * len(budgets), 4.6),
        squeeze=False,
        sharey=True,
    )
    components = (
        ("gamma_diagonal", r"$\Gamma_{ii}$"),
        ("cross_gamma", r"$\Gamma_{ij}$"),
    )
    positions = np.arange(len(order), dtype=float)
    for axis, budget in zip(axes.flat, budgets, strict=True):
        for name, label in components:
            summary = [
                _mean_range(
                    grouped[(budget, truncation)],
                    lambda variant, component=name: 100.0
                    * variant["global_error_against_analytical"][component][
                        "normalized_mae"
                    ],
                )
                for truncation in order
            ]
            _plot_with_range(axis, positions, summary, label, f"C{len(axis.lines)}")
        hessian = [
            _mean_range(
                grouped[(budget, truncation)],
                lambda variant: 100.0
                * variant["full_hessian_error_against_analytical"][
                    "aggregate_relative_frobenius_error"
                ],
            )
            for truncation in order
        ]
        _plot_with_range(axis, positions, hessian, "Full Hessian", "C2")
        axis.set_xticks(positions, ["raw" if item == "untruncated" else item for item in order])
        axis.set_yscale("log")
        axis.set_xlabel("TT truncation tolerance")
        axis.set_title(f"Budget {budget:,}")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0, 0].set_ylabel("Normalized error (%)")
    fig.suptitle("v9.4.1 Greek sensitivity to price-norm truncation")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v941_truncation_accuracy")


def plot_truncation_retention(payload, output_dir):
    order, grouped = _group_truncation_variants(payload)
    budgets = sorted({budget for budget, _ in grouped})
    positions = np.arange(len(order), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for color_index, budget in enumerate(budgets):
        rank = [
            _mean_range(
                grouped[(budget, truncation)],
                lambda variant: variant["core_diagnostics"]["effective_rank"],
            )
            for truncation in order
        ]
        retention = [
            _mean_range(
                grouped[(budget, truncation)],
                lambda variant: 100.0
                * variant["parameter_retention_fraction"],
            )
            for truncation in order
        ]
        _plot_with_range(
            axes[0],
            positions,
            rank,
            f"Budget {budget:,}",
            f"C{color_index}",
        )
        _plot_with_range(
            axes[1],
            positions,
            retention,
            f"Budget {budget:,}",
            f"C{color_index}",
        )
    labels = ["raw" if item == "untruncated" else item for item in order]
    axes[0].set_ylabel("Effective QTT rank")
    axes[1].set_ylabel("Parameters retained (%)")
    axes[0].set_title("Post-truncation rank")
    axes[1].set_title("Compression imposed by truncation")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.set_xlabel("TT truncation tolerance")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("v9.4.1 truncation mechanism audit")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v941_truncation_retention")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-input", type=Path, required=True)
    parser.add_argument("--truncation-input", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v941_tt_stability"),
    )
    args = parser.parse_args()

    matrix = json.loads(args.matrix_input.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != "greeks-tt-convergence-v9.4":
        raise ValueError("matrix-input is not a v9.4 TT-convergence result")
    plot_budget_seed_accuracy(matrix, args.output_dir)
    plot_budget_seed_structure(matrix, args.output_dir)

    if args.truncation_input is not None:
        truncation = json.loads(
            args.truncation_input.read_text(encoding="utf-8")
        )
        if truncation.get("schema_version") != "greeks-tt-truncation-v9.4.1":
            raise ValueError(
                "truncation-input is not a v9.4.1 truncation result"
            )
        plot_truncation_accuracy(truncation, args.output_dir)
        plot_truncation_retention(truncation, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
