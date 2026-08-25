from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = (
    ("price", "Price"),
    ("delta", r"$\Delta$"),
    ("gamma_diagonal", r"$\Gamma_{ii}$"),
    ("cross_gamma", r"$\Gamma_{ij}$"),
)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _positive(value):
    if value is None or not np.isfinite(value):
        return np.nan
    return max(100.0 * float(value), 1e-8)


def _finish(fig, output_dir, stem):
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_european(data, output_dir):
    paths = [int(value) for value in data["settings"]["path_counts"]]
    bumps = [float(value) for value in data["settings"]["relative_bumps"]]
    modes = list(data["settings"]["control_variate_beta_modes"])
    colors = {"estimated": "#555555", "unit": "#ed7d31"}
    linestyles = ["-", "--", ":", "-."]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)
    for axis, (component, title) in zip(axes.flat, COMPONENTS):
        for mode in modes:
            for index, bump in enumerate(bumps):
                bump_key = f"{bump:.10g}"
                values = [
                    _positive(
                        data["runs"][mode][str(path)][bump_key]["uncertainty"]
                        [component]["normalized_mean_standard_error"]
                    )
                    for path in paths
                ]
                axis.plot(
                    paths,
                    values,
                    color=colors.get(mode),
                    linestyle=linestyles[index % len(linestyles)],
                    marker="o" if mode == "estimated" else "s",
                    label=f"{mode}, h={100*bump:g}%",
                )
        axis.set_title(title)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.set_ylabel("Normalized standard error (%)")
    for axis in axes[-1]:
        axis.set_xlabel("Sobol paths per scramble")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=min(4, len(labels)),
    )
    fig.suptitle(
        "European arithmetic reference convergence",
        y=0.99,
        fontsize=14,
    )
    _finish(fig, output_dir, "european_reference_convergence")


def plot_american(data, output_dir, selected_steps=None):
    paths = [int(value) for value in data["settings"]["path_counts"]]
    steps = [int(value) for value in data["settings"]["exercise_steps"]]
    bumps = [float(value) for value in data["settings"]["relative_bumps"]]
    modes = list(data["settings"]["policy_modes"])
    selected_steps = max(steps) if selected_steps is None else int(selected_steps)
    if selected_steps not in steps:
        raise ValueError("selected American exercise-step count is absent from the JSON")
    colors = {
        "refit": "#555555",
        "frozen": "#ed7d31",
        "frozen_oos": "#4e79a7",
    }
    linestyles = ["-", "--", ":", "-."]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex=True)
    for axis, (component, title) in zip(axes.flat, COMPONENTS):
        for mode in modes:
            for index, bump in enumerate(bumps):
                bump_key = f"{bump:.10g}"
                values = [
                    _positive(
                        data["runs"][mode][str(path)][str(selected_steps)]
                        [bump_key]["uncertainty"][component]
                        ["normalized_mean_standard_error"]
                    )
                    for path in paths
                ]
                axis.plot(
                    paths,
                    values,
                    color=colors.get(mode),
                    linestyle=linestyles[index % len(linestyles)],
                    marker={"refit": "o", "frozen": "s", "frozen_oos": "^"}.get(mode),
                    label=f"{mode}, h={100*bump:g}%",
                )
        axis.set_title(title)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.set_ylabel("Normalized standard error (%)")
    for axis in axes[-1]:
        axis.set_xlabel("LSMC paths")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=min(3, len(labels)),
    )
    fig.suptitle(
        f"American arithmetic reference convergence ({selected_steps} exercise steps)",
        y=0.99,
        fontsize=14,
    )
    _finish(fig, output_dir, "american_reference_convergence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--european", type=Path, required=True)
    parser.add_argument("--american", type=Path, required=True)
    parser.add_argument("--american-steps", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/greeks_v81_reference_convergence"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_european(_load(args.european), args.output_dir)
    plot_american(_load(args.american), args.output_dir, args.american_steps)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
