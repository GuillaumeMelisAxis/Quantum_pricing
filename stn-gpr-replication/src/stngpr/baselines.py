from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel
from sklearn.preprocessing import MinMaxScaler


class ManhattanLaplacian(Kernel):
    """Paper kernel exp(-||x-y||_1 / L), with an optimizable scalar L."""

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-3, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, x, y=None, eval_gradient=False):
        x = np.atleast_2d(x)
        y_is_x = y is None
        y = x if y_is_x else np.atleast_2d(y)
        distance = np.abs(x[:, None, :] - y[None, :, :]).sum(axis=2)
        length = float(self.length_scale)
        kernel = np.exp(-distance / length)
        if eval_gradient:
            if not y_is_x:
                raise ValueError("gradient can only be evaluated when y is None")
            if self.hyperparameter_length_scale.fixed:
                return kernel, np.empty((*kernel.shape, 0))
            # derivative with respect to log(L), as expected by sklearn
            return kernel, (kernel * distance / length)[:, :, None]
        return kernel

    def diag(self, x):
        return np.ones(np.atleast_2d(x).shape[0])

    def is_stationary(self):
        return True

    def __repr__(self):
        return f"ManhattanLaplacian(length_scale={float(self.length_scale):.6g})"


class ExactLaplacianGPR:
    """Exact noise-free GPR baseline; Matern(nu=0.5) is the Laplacian kernel."""

    def __init__(self, optimize=True, seed=0):
        self.scaler = MinMaxScaler()
        length_bounds = (1e-3, 1e5) if optimize else "fixed"
        kernel = ManhattanLaplacian(1.0, length_bounds)
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            optimizer="fmin_l_bfgs_b" if optimize else None,
            n_restarts_optimizer=0,
            random_state=seed,
        )

    def fit(self, x, y):
        self.model.fit(self.scaler.fit_transform(x), y)
        return self

    def predict(self, x):
        return self.model.predict(self.scaler.transform(x))
