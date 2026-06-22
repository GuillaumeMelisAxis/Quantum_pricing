"""
hamiltonian_pricer.py
======================

Classical implementation of the discrete Hamiltonian pricing scheme for
discretely monitored Down-and-Out barrier options, following the
discretization introduced in Section 3 of the paper:

    H_h = -(sigma^2 / 2) * Delta_h + (sigma^2 / 2) * beta * I

    (P_B U)_i = U_i  if x_i > B,  0 otherwise

    U^{k+1} = P_B exp(-dt_k H_h) U^k

The pricer works in the log-price variable x = ln(S), builds a uniform
grid, assembles the tridiagonal discrete Hamiltonian as a sparse matrix,
evaluates the matrix exponential acting on the state vector with
scipy.sparse.linalg.expm_multiply, and applies the barrier projector at
each monitoring date. The price at S0 is recovered by linear
interpolation of the terminal state on the grid.

Author: Guillaume Melis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

MonitoringFrequency = Literal["daily", "weekly", "monthly", "continuous"]

# Trading-convention number of monitoring dates per year.
_MONITORING_DATES_PER_YEAR: dict[str, int] = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
}


def build_monitoring_schedule(
    T: float,
    frequency: MonitoringFrequency,
    n_steps: int | None = None,
) -> np.ndarray:
    """Build an array of monitoring dates t_0=0 < t_1 < ... < t_M=T.

    Parameters
    ----------
    T : float
        Maturity (in years).
    frequency : {"daily", "weekly", "monthly", "continuous"}
        Monitoring frequency. "continuous" is approximated with n_steps
        equally spaced dates (n_steps must then be supplied).
    n_steps : int, optional
        Number of monitoring dates to use when frequency == "continuous"
        or to override the default trading-convention count.

    Returns
    -------
    np.ndarray
        Array of monitoring dates including t_0 = 0 and t_M = T.
    """
    if frequency == "continuous":
        if n_steps is None:
            raise ValueError("n_steps must be provided for continuous monitoring")
        M = n_steps
    elif n_steps is not None:
        M = n_steps
    else:
        M = max(1, round(_MONITORING_DATES_PER_YEAR[frequency] * T))
    return np.linspace(0.0, T, M + 1)


@dataclass
class HamiltonianPricer:
    """Discrete Hamiltonian pricer for Down-and-Out Call options.

    Parameters
    ----------
    S0 : float
        Current spot price.
    K : float
        Strike.
    B : float
        Barrier level (Down-and-Out).
    sigma : float
        Volatility.
    r : float
        Risk-free rate.
    T : float
        Maturity (years).
    N : int
        Number of grid intervals (grid has N+1 points, N-1 interior dof).
    x_min, x_max : float, optional
        Bounds of the log-price computational domain. If not provided,
        a default domain is built around ln(S0) using `n_std` standard
        deviations of log-returns, clipped so the barrier is included
        and (if possible) lies on a grid point.
    n_std : float
        Number of log-volatility standard deviations used to size the
        default domain when x_min/x_max are not provided.
    """

    S0: float
    K: float
    B: float
    sigma: float
    r: float
    T: float
    N: int = 256
    x_min: float | None = None
    x_max: float | None = None
    n_std: float = 6.0

    # populated by build_grid / build_hamiltonian / build_projector
    grid: np.ndarray = field(init=False, repr=False, default=None)
    h: float = field(init=False, repr=False, default=None)
    beta: float = field(init=False, repr=False, default=None)
    H_h: sparse.spmatrix = field(init=False, repr=False, default=None)
    P_B: sparse.spmatrix = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.beta = ((self.sigma ** 2 / 2 + self.r) / self.sigma ** 2) ** 2
        # Similarity-transform exponent that symmetrizes H_BS into H_h:
        # C(x,tau) = e^{alpha x} phi(x,tau), with phi evolving under H_h.
        # See Section 2.2-2.3 / 3.1 of the paper: alpha = (sigma^2/2 - r)/sigma^2.
        self.alpha = (self.sigma ** 2 / 2 - self.r) / self.sigma ** 2
        self.build_grid()
        self.build_hamiltonian()
        self.build_projector()

    # ------------------------------------------------------------------
    # Part 1 required methods
    # ------------------------------------------------------------------
    def build_grid(self) -> np.ndarray:
        """Build the uniform log-price grid x_i = x_min + i h, i=0..N.

        The domain is sized so that it comfortably contains both ln(S0)
        and ln(B), and (when possible) snaps the barrier exactly onto a
        grid point, as required for the projector to be consistent with
        the continuous monitoring operator (see Lemma 3.6.2 in the
        paper).
        """
        x0 = np.log(self.S0)
        xB = np.log(self.B)
        if self.x_min is None or self.x_max is None:
            spread = self.n_std * self.sigma * np.sqrt(self.T)
            x_min = min(x0, xB) - spread
            x_max = max(x0, xB) + spread
        else:
            x_min, x_max = self.x_min, self.x_max

        h = (x_max - x_min) / self.N
        # Snap the barrier to the nearest grid point so that PB exactly
        # reproduces the indicator function 1{x > B} (Lemma 3.6.2).
        j_B = round((xB - x_min) / h)
        shift = xB - (x_min + j_B * h)
        x_min += shift
        x_max += shift

        self.x_min, self.x_max = x_min, x_max
        self.h = h
        self.grid = x_min + np.arange(self.N + 1) * h
        return self.grid

    def build_hamiltonian(self) -> sparse.spmatrix:
        """Assemble the tridiagonal discrete Hamiltonian H_h on the
        interior grid points (homogeneous Dirichlet boundary conditions).

            H_h = -(sigma^2/2) * Delta_h + (sigma^2/2) * beta * I_{N-1}

        Delta_h is the centered second-difference operator.
        """
        n = self.N - 1  # interior degrees of freedom
        h2 = self.h ** 2
        sig2 = self.sigma ** 2

        main = (sig2 / h2 + sig2 / 2 * self.beta) * np.ones(n)
        off = (-sig2 / (2 * h2)) * np.ones(n - 1)

        H_h = sparse.diags([off, main, off], offsets=[-1, 0, 1], format="csr")
        self.H_h = H_h
        return H_h

    def build_projector(self) -> sparse.spmatrix:
        """Build the diagonal barrier projector P_B on interior nodes."""
        interior_x = self.grid[1:-1]
        xB = np.log(self.B)
        eta = (interior_x > xB).astype(float)
        self.P_B = sparse.diags(eta, format="csr")
        return self.P_B

    def _terminal_payoff(self) -> np.ndarray:
        """Down-and-Out Call terminal payoff sampled on interior nodes,
        expressed in the phi-representation phi(x,0) = e^{-alpha x} h(x)
        so that it evolves under the symmetric Hamiltonian H_h (see
        the similarity transform C = e^{alpha x} phi discussed in
        Sections 2.2-3.1 of the paper).
        """
        interior_x = self.grid[1:-1]
        S_grid = np.exp(interior_x)
        payoff = np.maximum(S_grid - self.K, 0.0)
        payoff[S_grid <= self.B] = 0.0
        phi0 = np.exp(-self.alpha * interior_x) * payoff
        return phi0

    def price(
        self,
        frequency: MonitoringFrequency = "daily",
        n_monitoring: int | None = None,
        return_curve: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        """Run the discrete Hamiltonian pricing recursion and return the
        discounted option price interpolated at S0.

        U^{k+1} = P_B exp(-dt_k H_h) U^k,   k = 0, ..., M-1

        Parameters
        ----------
        frequency : {"daily", "weekly", "monthly", "continuous"}
            Monitoring schedule.
        n_monitoring : int, optional
            Override for the number of monitoring dates (used directly
            when frequency == "continuous").
        return_curve : bool
            If True, also return the full terminal price curve U^M on
            the interior grid (useful for plotting/diagnostics).
        """
        schedule = build_monitoring_schedule(self.T, frequency, n_monitoring)
        dts = np.diff(schedule)

        U = self._terminal_payoff()
        for dt in dts:
            U = expm_multiply(-dt * self.H_h, U)
            U = self.P_B @ U

        price0 = self._interpolate_price(U)
        if return_curve:
            interior_x = self.grid[1:-1]
            C_curve = np.exp(self.alpha * interior_x) * U
            return price0, C_curve
        return price0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _interpolate_price(self, U: np.ndarray) -> float:
        """Linear interpolation of the priced curve U (phi-representation,
        on interior nodes) at x0 = ln(S0), converted back to price space
        via C(x0) = e^{alpha x0} phi(x0)."""
        x0 = np.log(self.S0)
        interior_x = self.grid[1:-1]
        if x0 <= interior_x[0] or x0 >= interior_x[-1]:
            raise ValueError("S0 lies outside the computational domain interior")
        phi0 = np.interp(x0, interior_x, U)
        return float(np.exp(self.alpha * x0) * phi0)
