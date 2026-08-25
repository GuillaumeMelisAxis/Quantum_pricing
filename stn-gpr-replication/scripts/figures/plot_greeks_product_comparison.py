from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


COMPONENTS = ("price", "delta", "gamma_diagonal", "cross_gamma")
CURVE_COMPONENTS = ("delta", "gamma_diagonal", "cross_gamma")
LABELS = {
    "price": "Price",
    "delta": r"$\Delta$",
    "gamma_diagonal": r"$\Gamma_{ii}$",
    "cross_gamma": r"$\Gamma_{ij}$",
}
PRODUCT_LABELS = {
    "european_geometric_basket_put": "European geometric",
    "european_arithmetic_basket_put": "European arithmetic",
    "american_arithmetic_basket_put": "American arithmetic",
}
REFERENCE_COLOR = "#4D4D4D"
TT_COLOR = "#E87722"


def _arrays(payload):
    return {name: np.asarray(payload[name], dtype=float) for name in COMPONENTS}


def _latest_numeric_key(mapping):
    keys = [key for key in mapping if str(key).isdigit()]
    if not keys:
        raise ValueError("no numeric TT budget found in result file")
    return max(keys, key=lambda value: int(value))


def load_result(path: Path, raw: bool):
    data = json.loads(path.read_text(encoding="utf-8"))
    product = data.get("product")
    if product == "european_geometric_basket_put":
        reference_block = data.get("analytical_reference", {})
        run_key = _latest_numeric_key(data.get("tt", {}))
        run = data["tt"][run_key]
    elif product == "european_arithmetic_basket_put":
        reference_block = data.get("reference", {})
        run = data.get("runs", {}).get("tt")
        run_key = str(run.get("fit", {}).get("budget")) if run else "tt"
    elif product == "american_arithmetic_basket_put":
        reference_block = data.get("independent_reference", {})
        run_key = _latest_numeric_key(data.get("runs", {}))
        run = data["runs"][run_key]
    else:
        raise ValueError(f"unsupported or missing product field in {path}")
    if "components" not in reference_block or not run:
        raise ValueError(
            f"{path} does not contain v8 component arrays; rerun its v8 validator"
        )
    estimate_key = "components" if raw else "components_psd_projected"
    if estimate_key not in run:
        estimate_key = "components"
    design = data["test_design"]
    reference_se = reference_block.get("component_standard_errors")
    fit = run.get("fit", {})
    return {
        "product": product,
        "label": PRODUCT_LABELS[product],
        "moneyness": np.asarray(design["log_moneyness"], dtype=float),
        "maturity_days": np.asarray(design["maturity_days"], dtype=float),
        "reference": _arrays(reference_block["components"]),
        "reference_se": _arrays(reference_se) if reference_se else None,
        "estimate": _arrays(run[estimate_key]),
        "rank": fit.get("effective_rank"),
        "evaluations": fit.get("function_evaluations"),
        "budget": fit.get("budget", int(run_key) if str(run_key).isdigit() else None),
    }


def _point_scalar(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values
    return np.mean(values, axis=tuple(range(1, values.ndim)))


def _curve(result, component, maturity):
    maturities = result["maturity_days"]
    available = np.unique(maturities)
    selected = float(available[np.argmin(np.abs(available - maturity))])
    mask_t = np.isclose(maturities, selected)
    m = result["moneyness"][mask_t]
    ref = _point_scalar(result["reference"][component])[mask_t]
    est = _point_scalar(result["estimate"][component])[mask_t]
    se = None
    if result["reference_se"] is not None:
        se = _point_scalar(result["reference_se"][component])[mask_t]
    x, ref_curve, est_curve, se_curve = [], [], [], []
    for value in np.unique(m):
        local = np.isclose(m, value)
        x.append(value)
        ref_curve.append(np.mean(ref[local]))
        est_curve.append(np.mean(est[local]))
        if se is not None:
            se_curve.append(np.mean(np.abs(se[local])))
    return (
        np.asarray(x),
        np.asarray(ref_curve),
        np.asarray(est_curve),
        np.asarray(se_curve) if se is not None else None,
        selected,
    )


def plot_curves(results, output_dir, maturity):
    fig, axes = plt.subplots(
        len(results),
        len(CURVE_COMPONENTS),
        figsize=(12.5, 3.25 * len(results)),
        squeeze=False,
    )
    for row, result in enumerate(results):
        selected = None
        for column, component in enumerate(CURVE_COMPONENTS):
            ax = axes[row, column]
            x, reference, estimate, se, selected = _curve(
                result,
                component,
                maturity,
            )
            ax.plot(x, reference, color=REFERENCE_COLOR, marker="o", label="Reference")
            ax.plot(x, estimate, color=TT_COLOR, marker="s", label="TT + interpolation")
            if se is not None:
                ax.fill_between(
                    x,
                    reference - 1.96 * se,
                    reference + 1.96 * se,
                    color=REFERENCE_COLOR,
                    alpha=0.15,
                    label="Reference uncertainty",
                )
            ax.axvline(0.0, color="#BDBDBD", linewidth=0.8)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Log-moneyness m")
            ax.set_title(LABELS[component])
            if column == 0:
                ax.set_ylabel(f"{result['label']}\ncomponent mean")
        axes[row, -1].text(
            0.98,
            0.04,
            f"T = {selected:g}d",
            ha="right",
            va="bottom",
            transform=axes[row, -1].transAxes,
            color="#666666",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Fixed-strike spot Greeks across basket products", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"greeks_product_curves.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def _pointwise_normalized_error(result, component):
    reference = np.asarray(result["reference"][component], dtype=float)
    estimate = np.asarray(result["estimate"][component], dtype=float)
    absolute = np.abs(estimate - reference)
    if absolute.ndim > 1:
        absolute = np.mean(absolute, axis=tuple(range(1, absolute.ndim)))
    scale = float(np.mean(np.abs(reference)))
    return 100.0 * absolute / max(scale, 1e-14)


def plot_heatmaps(results, output_dir):
    all_values = []
    matrices = {}
    for row, result in enumerate(results):
        m_values = np.unique(result["moneyness"])
        t_values = np.unique(result["maturity_days"])
        for column, component in enumerate(CURVE_COMPONENTS):
            error = _pointwise_normalized_error(result, component)
            matrix = np.full((len(t_values), len(m_values)), np.nan)
            for i, maturity in enumerate(t_values):
                for j, money in enumerate(m_values):
                    mask = (
                        np.isclose(result["maturity_days"], maturity)
                        & np.isclose(result["moneyness"], money)
                    )
                    if np.any(mask):
                        matrix[i, j] = np.mean(error[mask])
            matrices[(row, column)] = (matrix, m_values, t_values)
            all_values.extend(matrix[np.isfinite(matrix) & (matrix > 0.0)].tolist())
    positive = np.asarray(all_values)
    vmin = max(float(np.quantile(positive, 0.02)), 1e-4) if positive.size else 1e-4
    vmax = max(float(np.quantile(positive, 0.98)), 10.0 * vmin) if positive.size else 1.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(
        len(results),
        len(CURVE_COMPONENTS),
        figsize=(12.5, 3.1 * len(results)),
        squeeze=False,
    )
    image = None
    for row, result in enumerate(results):
        for column, component in enumerate(CURVE_COMPONENTS):
            matrix, m_values, t_values = matrices[(row, column)]
            ax = axes[row, column]
            image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma", norm=norm)
            ax.set_xticks(range(len(m_values)), [f"{value:+.3g}" for value in m_values])
            ax.set_yticks(range(len(t_values)), [f"{value:g}" for value in t_values])
            ax.set_xlabel("m")
            if column == 0:
                ax.set_ylabel(f"{result['label']}\nT (days)")
            ax.set_title(LABELS[component])
    fig.colorbar(image, ax=axes, fraction=0.018, pad=0.02, label="Normalized absolute error (%)")
    fig.suptitle("Greek error localization in the (m, T) plane", y=0.995)
    fig.subplots_adjust(top=0.94, right=0.90, hspace=0.35, wspace=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"greeks_error_heatmaps.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def _normalized_mae(result, component):
    reference = np.asarray(result["reference"][component], dtype=float)
    estimate = np.asarray(result["estimate"][component], dtype=float)
    scale = float(np.mean(np.abs(reference)))
    return 100.0 * float(np.mean(np.abs(estimate - reference))) / max(scale, 1e-14)


def plot_summary(results, output_dir):
    fig, (ax_error, ax_cost) = plt.subplots(1, 2, figsize=(12.0, 4.2))
    x = np.arange(len(COMPONENTS))
    width = 0.8 / len(results)
    colors = ("#555555", "#E87722", "#8C6D46")
    for index, result in enumerate(results):
        values = [_normalized_mae(result, component) for component in COMPONENTS]
        ax_error.bar(
            x + (index - (len(results) - 1) / 2) * width,
            values,
            width,
            label=result["label"],
            color=colors[index % len(colors)],
        )
    ax_error.set_yscale("log")
    ax_error.set_xticks(x, [LABELS[name] for name in COMPONENTS])
    ax_error.set_ylabel("Normalized MAE (%)")
    ax_error.grid(axis="y", alpha=0.2)
    ax_error.legend(frameon=False)
    ax_error.set_title("Accuracy")

    labels = [result["label"] for result in results]
    ranks = [np.nan if result["rank"] is None else result["rank"] for result in results]
    evaluations = [
        np.nan if result["evaluations"] is None else result["evaluations"]
        for result in results
    ]
    positions = np.arange(len(results))
    ax_cost.bar(positions - 0.18, ranks, 0.36, color="#555555", label="Effective rank")
    ax_eval = ax_cost.twinx()
    ax_eval.bar(positions + 0.18, evaluations, 0.36, color=TT_COLOR, label="Oracle evaluations")
    ax_cost.set_xticks(positions, labels, rotation=15, ha="right")
    ax_cost.set_ylabel("Effective QTT rank")
    ax_eval.set_ylabel("Oracle evaluations")
    ax_cost.set_title("Compression cost")
    handles_1, labels_1 = ax_cost.get_legend_handles_labels()
    handles_2, labels_2 = ax_eval.get_legend_handles_labels()
    ax_cost.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.12),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"greeks_accuracy_and_cost.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometric", type=Path, required=True)
    parser.add_argument("--arithmetic-european", type=Path, required=True)
    parser.add_argument("--arithmetic-american", type=Path, required=True)
    parser.add_argument("--maturity-days", type=float, default=30.0)
    parser.add_argument("--raw", action="store_true", help="plot raw rather than PSD-projected Hessians")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/greeks_v8"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        load_result(args.geometric, args.raw),
        load_result(args.arithmetic_european, args.raw),
        load_result(args.arithmetic_american, args.raw),
    ]
    plot_curves(results, args.output_dir, args.maturity_days)
    plot_heatmaps(results, args.output_dir)
    plot_summary(results, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
