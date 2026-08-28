from __future__ import annotations

import numpy as np

from .greeks import finite_difference_greeks, finite_difference_hybrid_greeks

COMPONENTS = ("price", "delta", "gamma_diagonal", "cross_gamma")


def deterministic_moneyness_panel(
    config,
    log_moneyness,
    maturity_days,
    rate=0.03,
    spots=None,
):
    """Return a transparent fixed-spot panel for Greek convergence tests."""
    if spots is None:
        spots = np.linspace(80.0, 120.0, config.n_assets)
    spots = np.asarray(spots, dtype=float)
    if spots.shape != (config.n_assets,):
        raise ValueError("spots must contain one value per asset")
    basket = float(np.mean(spots))
    points, m_values, t_values = [], [], []
    for days in maturity_days:
        for moneyness in log_moneyness:
            strike = basket * np.exp(float(moneyness))
            points.append(
                np.concatenate((spots, [strike, float(rate), float(days) / 365.0]))
            )
            m_values.append(float(moneyness))
            t_values.append(float(days))
    return (
        np.asarray(points, dtype=float),
        np.asarray(m_values, dtype=float),
        np.asarray(t_values, dtype=float),
    )


def finite_difference_component_arrays(
    pricer,
    points,
    risk_columns,
    relative_bump,
):
    """Evaluate a selected fixed-strike Greek stencil on every market point."""
    columns = tuple(int(column) for column in risk_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("risk_columns must be non-empty and unique")
    records = [
        finite_difference_greeks(
            pricer,
            point,
            spot_columns=columns,
            relative_bump=relative_bump,
        )
        for point in np.asarray(points, dtype=float)
    ]
    pairs = [
        (left, right)
        for position, left in enumerate(columns)
        for right in columns[position + 1 :]
    ]
    return {
        "price": np.asarray([record["price"] for record in records]),
        "delta": np.asarray([
            [record["delta"][column] for column in columns]
            for record in records
        ]),
        "gamma_diagonal": np.asarray([
            [record["gamma"][(column, column)] for column in columns]
            for record in records
        ]),
        "cross_gamma": np.asarray([
            [record["gamma"][pair] for pair in pairs]
            for record in records
        ]),
        "risk_columns": columns,
        "cross_gamma_pairs": pairs,
    }


def finite_difference_hybrid_component_arrays(
    interpolator,
    transform,
    points,
    risk_columns,
    relative_bump,
):
    """Selected fixed-strike Greeks from a hybrid grid interpolator."""
    columns = tuple(int(column) for column in risk_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("risk_columns must be non-empty and unique")
    records = [
        finite_difference_hybrid_greeks(
            interpolator,
            transform,
            point,
            spot_columns=columns,
            relative_bump=relative_bump,
        )
        for point in np.asarray(points, dtype=float)
    ]
    pairs = [
        (left, right)
        for position, left in enumerate(columns)
        for right in columns[position + 1 :]
    ]
    return {
        "price": np.asarray([record["price"] for record in records]),
        "delta": np.asarray([
            [record["delta"][column] for column in columns]
            for record in records
        ]),
        "gamma_diagonal": np.asarray([
            [record["gamma"][(column, column)] for column in columns]
            for record in records
        ]),
        "cross_gamma": np.asarray([
            [record["gamma"][pair] for pair in pairs]
            for record in records
        ]),
        "risk_columns": columns,
        "cross_gamma_pairs": pairs,
    }


def mean_and_standard_error(replications):
    mean, standard_error = {}, {}
    for name in COMPONENTS:
        values = np.stack(
            [np.asarray(run[name], dtype=float) for run in replications],
            axis=0,
        )
        mean[name] = np.mean(values, axis=0)
        standard_error[name] = (
            np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
            if values.shape[0] > 1
            else np.zeros_like(mean[name])
        )
    return mean, standard_error


def component_metrics(reference, estimate):
    output = {}
    for name in COMPONENTS:
        truth = np.asarray(reference[name], dtype=float).reshape(-1)
        value = np.asarray(estimate[name], dtype=float).reshape(-1)
        error = value - truth
        scale = float(np.mean(np.abs(truth)))
        mae = float(np.mean(np.abs(error)))
        output[name] = {
            "count": int(truth.size),
            "mae": mae,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "max_absolute_error": float(np.max(np.abs(error))),
            "mean_error_bias": float(np.mean(error)),
            "mean_absolute_reference": scale,
            "normalized_mae": mae / scale if scale > 1e-14 else None,
        }
    return output


def component_residual_metrics(reference, residual):
    """Summarize component residuals on a common reference scale.

    Unlike :func:`component_metrics`, ``residual`` is already an error array.
    This helper is useful for additive error decompositions because every
    layer is normalized by the same analytical reference rather than by the
    intermediate approximation used to construct that layer.
    """
    output = {}
    for name in COMPONENTS:
        truth = np.asarray(reference[name], dtype=float).reshape(-1)
        error = np.asarray(residual[name], dtype=float).reshape(-1)
        if error.shape != truth.shape:
            raise ValueError(f"residual component {name!r} has the wrong shape")
        if not np.all(np.isfinite(error)):
            raise ValueError(f"residual component {name!r} contains non-finite values")
        scale = float(np.mean(np.abs(truth)))
        mae = float(np.mean(np.abs(error)))
        output[name] = {
            "count": int(truth.size),
            "mae": mae,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "max_absolute_error": float(np.max(np.abs(error))),
            "mean_error_bias": float(np.mean(error)),
            "mean_absolute_reference": scale,
            "normalized_mae": mae / scale if scale > 1e-14 else None,
        }
    return output


def component_error_decomposition(analytical, exact_price_fd, grid_price_fd):
    r"""Return the additive finite-difference/grid error decomposition.

    For every stored component the identity is

    ``grid_price_fd - analytical =``
    ``(exact_price_fd - analytical) + (grid_price_fd - exact_price_fd)``.

    The four returned residual arrays and their metrics use the analytical
    component as a common normalization scale.  ``closure`` should therefore
    be zero up to floating-point round-off and acts as a machine-checkable
    audit of the reported decomposition.
    """
    layers = {}
    definitions = {
        "finite_difference": (exact_price_fd, analytical),
        "grid_interpolation": (grid_price_fd, exact_price_fd),
        "total": (grid_price_fd, analytical),
    }
    for layer, (left, right) in definitions.items():
        layers[layer] = {
            name: (
                np.asarray(left[name], dtype=float)
                - np.asarray(right[name], dtype=float)
            )
            for name in COMPONENTS
        }
    layers["closure"] = {
        name: (
            layers["total"][name]
            - layers["finite_difference"][name]
            - layers["grid_interpolation"][name]
        )
        for name in COMPONENTS
    }
    return {
        layer: {
            "residuals": {
                name: np.asarray(values[name], dtype=float)
                for name in COMPONENTS
            },
            "metrics": component_residual_metrics(analytical, values),
        }
        for layer, values in layers.items()
    }


def tt_component_error_decomposition(
    analytical,
    exact_price_fd,
    grid_price_fd,
    tt_price_fd,
):
    r"""Decompose total TT Greek error into differentiation, grid and TT layers.

    For each stored component, the machine-checkable identity is

    ``tt - analytical = (exact_fd - analytical)``
    ``+ (grid_fd - exact_fd) + (tt - grid_fd)``.

    Every layer is normalized by the analytical component scale.
    """
    layers = {}
    definitions = {
        "finite_difference": (exact_price_fd, analytical),
        "grid_interpolation": (grid_price_fd, exact_price_fd),
        "tt_reconstruction": (tt_price_fd, grid_price_fd),
        "total": (tt_price_fd, analytical),
    }
    for layer, (left, right) in definitions.items():
        layers[layer] = {
            name: (
                np.asarray(left[name], dtype=float)
                - np.asarray(right[name], dtype=float)
            )
            for name in COMPONENTS
        }
    layers["closure"] = {
        name: (
            layers["total"][name]
            - layers["finite_difference"][name]
            - layers["grid_interpolation"][name]
            - layers["tt_reconstruction"][name]
        )
        for name in COMPONENTS
    }
    return {
        layer: {
            "residuals": {
                name: np.asarray(values[name], dtype=float)
                for name in COMPONENTS
            },
            "metrics": component_residual_metrics(analytical, values),
        }
        for layer, values in layers.items()
    }


def curve_error_diagnostics(moneyness, reference, estimate):
    """Measure pointwise error and shape distortion along one moneyness curve."""
    moneyness = np.asarray(moneyness, dtype=float).reshape(-1)
    truth = np.asarray(reference, dtype=float)
    value = np.asarray(estimate, dtype=float)
    if truth.ndim == 1:
        truth = truth[:, None]
    if value.ndim == 1:
        value = value[:, None]
    if truth.shape != value.shape or truth.shape[0] != moneyness.size:
        raise ValueError("curve arrays and moneyness have incompatible shapes")
    if moneyness.size < 3 or np.any(np.diff(moneyness) <= 0.0):
        raise ValueError("moneyness must contain at least three increasing values")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(value)):
        raise ValueError("curve arrays must contain only finite values")

    absolute_error = np.abs(value - truth)
    scale = float(np.mean(np.abs(truth)))
    flat_maximum = int(np.argmax(absolute_error))
    row_at_maximum, column_at_maximum = np.unravel_index(
        flat_maximum,
        absolute_error.shape,
    )

    def turning_point_count(values):
        differences = np.diff(values)
        numerical_threshold = 100.0 * np.finfo(float).eps * max(
            1.0, float(np.max(np.abs(values)))
        )
        material_threshold = 1.0e-3 * max(
            float(np.ptp(values)),
            np.finfo(float).tiny,
        )
        threshold = max(numerical_threshold, material_threshold)
        signs = np.sign(differences[np.abs(differences) > threshold])
        return int(np.count_nonzero(signs[1:] != signs[:-1]))

    by_column = []
    for column in range(truth.shape[1]):
        reference_variation = float(np.sum(np.abs(np.diff(truth[:, column]))))
        estimate_variation = float(np.sum(np.abs(np.diff(value[:, column]))))
        by_column.append({
            "column": int(column),
            "reference_total_variation": reference_variation,
            "estimate_total_variation": estimate_variation,
            "total_variation_ratio": (
                estimate_variation / reference_variation
                if reference_variation > 1e-14
                else None
            ),
            "reference_turning_point_count": turning_point_count(
                truth[:, column]
            ),
            "estimate_turning_point_count": turning_point_count(
                value[:, column]
            ),
            "turning_point_relative_slope_tolerance": 1.0e-3,
        })

    mae = float(np.mean(absolute_error))
    maximum = float(absolute_error[row_at_maximum, column_at_maximum])
    return {
        "point_count": int(moneyness.size),
        "component_count": int(truth.shape[1]),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean((value - truth) ** 2))),
        "q95_absolute_error": float(np.quantile(absolute_error, 0.95)),
        "q99_absolute_error": float(np.quantile(absolute_error, 0.99)),
        "max_absolute_error": maximum,
        "mean_absolute_reference": scale,
        "normalized_mae": mae / scale if scale > 1e-14 else None,
        "normalized_max_absolute_error": (
            maximum / scale if scale > 1e-14 else None
        ),
        "moneyness_at_maximum_error": float(moneyness[row_at_maximum]),
        "component_column_at_maximum_error": int(column_at_maximum),
        "shape_by_component_column": by_column,
    }


def uncertainty_summary(reference, standard_error):
    output = {}
    for name in COMPONENTS:
        truth = np.asarray(reference[name], dtype=float)
        se = np.asarray(standard_error[name], dtype=float)
        scale = float(np.mean(np.abs(truth)))
        mean_se = float(np.mean(np.abs(se)))
        output[name] = {
            "mean_standard_error": mean_se,
            "mean_absolute_reference": scale,
            "normalized_mean_standard_error": (
                mean_se / scale if scale > 1e-14 else None
            ),
        }
    return output


def gamma_matrices(components):
    diagonal = np.asarray(components["gamma_diagonal"], dtype=float)
    cross = np.asarray(components["cross_gamma"], dtype=float)
    if diagonal.ndim != 2:
        raise ValueError("gamma_diagonal must be a two-dimensional array")
    n_points, n_risk = diagonal.shape
    expected_cross = n_risk * (n_risk - 1) // 2
    if cross.shape != (n_points, expected_cross):
        raise ValueError("cross_gamma has an incompatible shape")
    matrices = np.zeros((n_points, n_risk, n_risk), dtype=float)
    for index in range(n_risk):
        matrices[:, index, index] = diagonal[:, index]
    column = 0
    for left in range(n_risk):
        for right in range(left + 1, n_risk):
            matrices[:, left, right] = cross[:, column]
            matrices[:, right, left] = cross[:, column]
            column += 1
    return matrices


def hessian_shape_diagnostics(matrices, tolerance=1e-12):
    matrices = np.asarray(matrices, dtype=float)
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    )
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    negative_diagonal = diagonal < -float(tolerance)
    negative_eigenvalues = eigenvalues < -float(tolerance)
    return {
        "tolerance": float(tolerance),
        "matrix_count": len(matrices),
        "component_count": int(diagonal.size),
        "negative_diagonal_count": int(np.count_nonzero(negative_diagonal)),
        "negative_diagonal_fraction": float(np.mean(negative_diagonal)),
        "matrices_with_negative_diagonal": int(
            np.count_nonzero(np.any(negative_diagonal, axis=1))
        ),
        "negative_eigenvalue_count": int(np.count_nonzero(negative_eigenvalues)),
        "matrices_outside_psd_cone": int(
            np.count_nonzero(np.any(negative_eigenvalues, axis=1))
        ),
        "minimum_diagonal": float(np.min(diagonal)),
        "minimum_eigenvalue": float(np.min(eigenvalues)),
    }


def columnwise_error_metrics(reference, estimate, labels=None):
    """Return an error summary for every column of a component array."""
    truth = np.asarray(reference, dtype=float)
    value = np.asarray(estimate, dtype=float)
    if truth.ndim == 1:
        truth = truth[:, None]
    if value.ndim == 1:
        value = value[:, None]
    if truth.shape != value.shape:
        raise ValueError("reference and estimate must have the same shape")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(value)):
        raise ValueError("component arrays must contain only finite values")
    if labels is None:
        labels = [str(column) for column in range(truth.shape[1])]
    if len(labels) != truth.shape[1]:
        raise ValueError("labels must contain one entry per component column")

    output = []
    for column, label in enumerate(labels):
        reference_column = truth[:, column]
        error = value[:, column] - reference_column
        scale = float(np.mean(np.abs(reference_column)))
        mae = float(np.mean(np.abs(error)))
        output.append({
            "column": int(column),
            "label": str(label),
            "count": int(reference_column.size),
            "mae": mae,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "max_absolute_error": float(np.max(np.abs(error))),
            "mean_error_bias": float(np.mean(error)),
            "mean_absolute_reference": scale,
            "normalized_mae": mae / scale if scale > 1e-14 else None,
        })
    return output


def full_hessian_error_metrics(reference, estimate, relative_psd_tolerance=1e-4):
    """Measure aggregate Hessian fidelity and material PSD defects.

    The PSD threshold is scaled by the mean analytical Frobenius norm.  This
    keeps round-off eigenvalues in nearly flat tails separate from economically
    material violations.
    """
    truth = gamma_matrices(reference)
    value = gamma_matrices(estimate)
    if truth.shape != value.shape:
        raise ValueError("reference and estimate Hessians must have the same shape")
    error = value - truth
    truth_norms = np.linalg.norm(truth, axis=(1, 2))
    error_norms = np.linalg.norm(error, axis=(1, 2))
    aggregate_truth_norm = float(np.linalg.norm(truth))
    aggregate_error_norm = float(np.linalg.norm(error))
    mean_truth_norm = float(np.mean(truth_norms))
    material_tolerance = float(relative_psd_tolerance) * mean_truth_norm
    eigenvalues = np.linalg.eigvalsh(0.5 * (value + np.swapaxes(value, 1, 2)))
    minimum_eigenvalues = np.min(eigenvalues, axis=1)
    negative_eigenvalues = np.minimum(eigenvalues, 0.0)
    negative_part_norm = float(np.linalg.norm(negative_eigenvalues))
    return {
        "matrix_count": len(truth),
        "mean_reference_frobenius_norm": mean_truth_norm,
        "mean_error_frobenius_norm": float(np.mean(error_norms)),
        "aggregate_relative_frobenius_error": (
            aggregate_error_norm / aggregate_truth_norm
            if aggregate_truth_norm > 1e-14
            else None
        ),
        "q95_pointwise_frobenius_error": float(np.quantile(error_norms, 0.95)),
        "max_pointwise_frobenius_error": float(np.max(error_norms)),
        "relative_psd_tolerance": float(relative_psd_tolerance),
        "material_psd_absolute_tolerance": material_tolerance,
        "material_psd_violation_count": int(
            np.count_nonzero(minimum_eigenvalues < -material_tolerance)
        ),
        "minimum_eigenvalue": float(np.min(minimum_eigenvalues)),
        "negative_part_relative_frobenius_norm": (
            negative_part_norm / aggregate_truth_norm
            if aggregate_truth_norm > 1e-14
            else None
        ),
    }


def cross_gamma_sign_diagnostics(
    reference,
    estimate,
    labels=None,
    relative_zero_tolerance=1e-4,
):
    """Compare cross-Gamma signs while excluding immaterial near-zero tails."""
    truth = np.asarray(reference, dtype=float)
    value = np.asarray(estimate, dtype=float)
    if truth.ndim == 1:
        truth = truth[:, None]
    if value.ndim == 1:
        value = value[:, None]
    if truth.shape != value.shape:
        raise ValueError("reference and estimate cross-Gammas must match")
    if labels is None:
        labels = [str(column) for column in range(truth.shape[1])]
    if len(labels) != truth.shape[1]:
        raise ValueError("labels must contain one entry per cross-Gamma column")

    by_component = []
    total_material = 0
    total_agreeing = 0
    for column, label in enumerate(labels):
        reference_column = truth[:, column]
        estimate_column = value[:, column]
        scale = float(np.max(np.abs(reference_column)))
        tolerance = float(relative_zero_tolerance) * scale
        material = np.abs(reference_column) > tolerance
        agreement = np.sign(reference_column[material]) == np.sign(
            estimate_column[material]
        )
        material_count = int(np.count_nonzero(material))
        agreeing_count = int(np.count_nonzero(agreement))
        total_material += material_count
        total_agreeing += agreeing_count
        by_component.append({
            "column": int(column),
            "label": str(label),
            "scale": scale,
            "absolute_zero_tolerance": tolerance,
            "reference_negative_count": int(
                np.count_nonzero(reference_column < -tolerance)
            ),
            "estimate_negative_count": int(
                np.count_nonzero(estimate_column < -tolerance)
            ),
            "material_count": material_count,
            "sign_agreement_fraction": (
                agreeing_count / material_count if material_count else None
            ),
        })
    return {
        "relative_zero_tolerance": float(relative_zero_tolerance),
        "material_count": int(total_material),
        "sign_agreement_fraction": (
            total_agreeing / total_material if total_material else None
        ),
        "by_component": by_component,
    }


def hessian_uncertainty_summary(reference, standard_error):
    reference_matrices = gamma_matrices(reference)
    standard_error_matrices = gamma_matrices(standard_error)
    reference_norm = np.linalg.norm(reference_matrices, axis=(1, 2))
    se_norm = np.linalg.norm(standard_error_matrices, axis=(1, 2))
    scale = float(np.mean(reference_norm))
    mean_se = float(np.mean(se_norm))
    return {
        "definition": "Frobenius norm of componentwise standard errors",
        "mean_standard_error_frobenius": mean_se,
        "mean_reference_frobenius_norm": scale,
        "normalized_mean_standard_error_frobenius": (
            mean_se / scale if scale > 1e-14 else None
        ),
    }


def components_to_json(components):
    return {
        name: np.asarray(components[name], dtype=float).tolist()
        for name in COMPONENTS
    }
