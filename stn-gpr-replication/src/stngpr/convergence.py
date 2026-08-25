from __future__ import annotations

import numpy as np

from .greeks import finite_difference_greeks


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
        "matrix_count": int(len(matrices)),
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
