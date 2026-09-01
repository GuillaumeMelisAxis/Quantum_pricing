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


def _order(payload):
    return [
        "untruncated" if float(value) == 0.0 else f"{float(value):.0e}"
        for value in payload["settings"]["truncations"]
    ]


def _records(payload, label):
    return [
        run["variants"][label]
        for run in sorted(
            payload["runs"].values(),
            key=lambda record: int(record["tt_seed"]),
        )
        if label in run["variants"]
    ]


def _mean_range(records, getter):
    values = np.asarray([getter(record) for record in records], dtype=float)
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def _plot_range(axis, positions, summaries, label, color):
    mean = np.asarray([item[0] for item in summaries])
    lower = np.asarray([item[1] for item in summaries])
    upper = np.asarray([item[2] for item in summaries])
    axis.plot(positions, mean, color=color, marker="o", label=label)
    axis.fill_between(positions, lower, upper, color=color, alpha=0.16)


def plot_accuracy(payload, output_dir):
    order = _order(payload)
    positions = np.arange(len(order), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    first = (("price", "Price"), ("delta", r"$\Delta$"))
    second = (
        ("gamma_diagonal", r"$\Gamma_{ii}$"),
        ("cross_gamma", r"$\Gamma_{ij}$"),
    )
    for axis, components in zip(axes, (first, second), strict=True):
        for color_index, (component, label) in enumerate(components):
            summaries = [
                _mean_range(
                    _records(payload, truncation),
                    lambda variant, key=component: 100.0
                    * variant["global_error_against_analytical"][key][
                        "normalized_mae"
                    ],
                )
                for truncation in order
            ]
            _plot_range(axis, positions, summaries, label, f"C{color_index}")
    hessian = [
        _mean_range(
            _records(payload, truncation),
            lambda variant: 100.0
            * variant["full_hessian_error_against_analytical"][
                "aggregate_relative_frobenius_error"
            ],
        )
        for truncation in order
    ]
    _plot_range(axes[1], positions, hessian, "Full Hessian", "C2")
    labels = ["raw" if label == "untruncated" else label for label in order]
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.set_yscale("log")
        axis.set_xlabel("TT truncation tolerance")
        axis.set_ylabel("Normalized error (%)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_title("Price and first derivative")
    axes[1].set_title("Second derivatives")
    fig.suptitle("v9.5 multi-seed truncation accuracy — mean and seed range")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v95_truncation_accuracy")


def plot_compression(payload, output_dir):
    order = _order(payload)
    positions = np.arange(len(order), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    rank = [
        _mean_range(
            _records(payload, label),
            lambda variant: variant["core_diagnostics"]["effective_rank"],
        )
        for label in order
    ]
    retention = [
        _mean_range(
            _records(payload, label),
            lambda variant: 100.0 * variant["parameter_retention_fraction"],
        )
        for label in order
    ]
    _plot_range(axes[0], positions, rank, "Effective rank", "C0")
    _plot_range(axes[1], positions, retention, "Parameter retention", "C1")
    labels = ["raw" if label == "untruncated" else label for label in order]
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.set_xlabel("TT truncation tolerance")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Effective QTT rank")
    axes[1].set_ylabel("Raw parameters retained (%)")
    axes[0].set_title("Rank response")
    axes[1].set_title("Compression response")
    fig.suptitle("v9.5 compression path on identical raw TT cores")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v95_truncation_compression")


def plot_seed_robustness(payload, output_dir):
    order = _order(payload)
    runs = sorted(
        payload["runs"].values(), key=lambda record: int(record["tt_seed"])
    )
    seeds = [int(run["tt_seed"]) for run in runs]
    accepted = np.asarray([
        [
            float(run["variants"][label]["greek_aware_acceptance"]["passed"])
            for label in order
        ]
        for run in runs
    ])
    hessian = np.asarray([
        [
            100.0
            * run["variants"][label]["full_hessian_error_against_analytical"][
                "aggregate_relative_frobenius_error"
            ]
            for label in order
        ]
        for run in runs
    ])
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    image = axis.imshow(accepted, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for row in range(len(seeds)):
        for column in range(len(order)):
            axis.text(
                column,
                row,
                f"{hessian[row, column]:.2f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    axis.set_xticks(
        np.arange(len(order)),
        ["raw" if label == "untruncated" else label for label in order],
    )
    axis.set_yticks(np.arange(len(seeds)), [str(seed) for seed in seeds])
    axis.set_xlabel("TT truncation tolerance")
    axis.set_ylabel("TT-cross seed")
    axis.set_title("Greek-aware acceptance; cells show Hessian error")
    fig.colorbar(image, ax=axis, ticks=(0, 1), label="Fail / pass")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v95_seed_robustness")


def plot_pareto(payload, output_dir):
    order = _order(payload)
    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    for color_index, label in enumerate(order):
        records = _records(payload, label)
        retention = np.asarray(
            [100.0 * item["parameter_retention_fraction"] for item in records]
        )
        hessian = np.asarray([
            100.0
            * item["full_hessian_error_against_analytical"][
                "aggregate_relative_frobenius_error"
            ]
            for item in records
        ])
        display = "raw" if label == "untruncated" else label
        axis.scatter(retention, hessian, color=f"C{color_index}", alpha=0.55)
        axis.scatter(
            np.mean(retention),
            np.max(hessian),
            color=f"C{color_index}",
            marker="D",
            s=70,
            label=f"{display}: mean compression / worst error",
        )
    axis.axhline(
        100.0
        * payload["greek_aware_thresholds"][
            "hessian_relative_frobenius_error"
        ],
        color="0.4",
        linestyle=":",
        linewidth=1.2,
        label="Hessian acceptance threshold",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Raw TT parameters retained (%)")
    axis.set_ylabel("Hessian relative error (%)")
    axis.set_title("v9.5 accuracy–compression frontier")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v95_accuracy_compression_pareto")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v95_tt_truncation"),
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-tt-truncation-v9.5":
        raise ValueError("input is not a v9.5 multi-seed truncation result")
    plot_accuracy(payload, args.output_dir)
    plot_compression(payload, args.output_dir)
    plot_seed_robustness(payload, args.output_dir)
    plot_pareto(payload, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
