from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _label(value):
    return "raw" if float(value) == 0.0 else f"{float(value):.0e}"


def _key(value):
    return "untruncated" if float(value) == 0.0 else f"{float(value):.0e}"


def _save(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_screening_frontier(payload, output_dir):
    truncations = payload["settings"]["truncations"]
    keys = [_key(value) for value in truncations]
    labels = [_label(value) for value in truncations]
    candidates = payload["screening_selection"]["candidates"]
    positions = np.arange(len(keys), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    metrics = (
        ("gamma_diagonal_normalized_mae", r"$\Gamma_{ii}$ global", "C0"),
        ("cross_gamma_normalized_mae", r"$\Gamma_{ij}$ global", "C1"),
        ("hessian_relative_frobenius_error", "Full Hessian", "C2"),
    )
    for metric, label, color in metrics:
        axes[0].plot(
            positions,
            [100.0 * candidates[key]["metrics"][metric]["maximum"] for key in keys],
            marker="o",
            label=label,
            color=color,
        )
    axes[0].axhline(2.5, color="0.4", linestyle=":", label="Gamma threshold")
    axes[0].axhline(3.0, color="0.4", linestyle="--", label="Hessian threshold")
    local_metrics = (
        ("gamma_curve_normalized_mae", r"$\Gamma_{ii}$ curve", "C0"),
        ("cross_gamma_curve_normalized_mae", r"$\Gamma_{ij}$ curve", "C1"),
        ("gamma_point_normalized_max_error", r"$\Gamma_{ii}$ point max", "C2"),
        ("cross_gamma_point_normalized_max_error", r"$\Gamma_{ij}$ point max", "C3"),
    )
    for metric, label, color in local_metrics:
        axes[1].plot(
            positions,
            [100.0 * candidates[key]["metrics"][metric]["maximum"] for key in keys],
            marker="o",
            label=label,
            color=color,
        )
    axes[1].axhline(2.5, color="0.4", linestyle=":", label="Curve threshold")
    axes[1].axhline(10.0, color="0.4", linestyle="--", label="Point threshold")
    for axis, title in zip(
        axes,
        ("Worst-seed global errors", "Worst-seed local errors"),
        strict=True,
    ):
        axis.set_yscale("log")
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.set_xlabel("TT truncation tolerance")
        axis.set_ylabel("Normalized error (%)")
        axis.set_title(title)
        axis.grid(alpha=0.23)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("v9.7 Greek-aware truncation frontier")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v97_accuracy_frontier")


def plot_compression(payload, output_dir):
    truncations = payload["settings"]["truncations"]
    keys = [_key(value) for value in truncations]
    labels = [_label(value) for value in truncations]
    candidates = payload["screening_selection"]["candidates"]
    positions = np.arange(len(keys), dtype=float)
    retention = np.asarray(
        [
            100.0 * candidates[key]["metrics"]["parameter_retention_fraction"]["mean"]
            for key in keys
        ]
    )
    ranks = np.asarray(
        [candidates[key]["metrics"]["effective_rank"]["mean"] for key in keys]
    )
    times = np.asarray(
        [candidates[key]["metrics"]["wall_time_seconds"]["mean"] for key in keys]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    axes[0].bar(positions - 0.18, retention, 0.36, color="C0", label="Parameters")
    rank_axis = axes[0].twinx()
    rank_axis.bar(positions + 0.18, ranks, 0.36, color="C1", label="Effective rank")
    axes[0].set_ylabel("Raw parameters retained (%)")
    rank_axis.set_ylabel("Effective QTT rank")
    axes[0].set_title("Compression response")
    handles = axes[0].patches[:1] + rank_axis.patches[:1]
    axes[0].legend(handles, ["Parameters", "Effective rank"], frameon=False)
    axes[1].plot(positions, times, marker="o", color="C2")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Screening evaluation time (s)")
    axes[1].set_title("Evaluation cost")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.set_xlabel("TT truncation tolerance")
        axis.grid(alpha=0.23)
    fig.suptitle("v9.7 compression and screening cost")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v97_compression")


def plot_seed_gate(payload, output_dir):
    truncations = payload["settings"]["truncations"]
    keys = [_key(value) for value in truncations]
    labels = [_label(value) for value in truncations]
    seeds = payload["settings"]["tt_seeds"]
    accepted = np.asarray(
        [
            [
                float(
                    payload["screening_runs"][str(seed)]["variants"][key][
                        "greek_aware_local_acceptance"
                    ]["passed"]
                )
                for key in keys
            ]
            for seed in seeds
        ]
    )
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    image = axis.imshow(accepted, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for row in range(len(seeds)):
        for column in range(len(keys)):
            record = payload["screening_runs"][str(seeds[row])]["variants"][
                keys[column]
            ]
            hessian = (
                100.0
                * record["full_hessian_error_against_analytical"][
                    "aggregate_relative_frobenius_error"
                ]
            )
            axis.text(
                column,
                row,
                f"{hessian:.2f}%",
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set_xticks(np.arange(len(keys)), labels)
    axis.set_yticks(np.arange(len(seeds)), [str(seed) for seed in seeds])
    axis.set_xlabel("TT truncation tolerance")
    axis.set_ylabel("TT-cross seed")
    axis.set_title("Local Greek gate; cells show global Hessian error")
    fig.colorbar(image, ax=axis, ticks=(0, 1), label="Fail / pass")
    fig.tight_layout()
    _save(fig, output_dir, "greeks_v97_seed_gate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v97_greek_aware_truncation"),
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "greeks-greek-aware-truncation-v9.7":
        raise ValueError("input is not a v9.7 Greek-aware truncation result")
    plot_screening_frontier(payload, args.output_dir)
    plot_compression(payload, args.output_dir)
    plot_seed_gate(payload, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
