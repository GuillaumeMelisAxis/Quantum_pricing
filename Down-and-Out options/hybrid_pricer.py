"""
hybrid_pricer.py
==================

Part 4 - Hybrid Quantum-Classical Architecture.

The discrete Hamiltonian propagator e^{-dt H_h} is evaluated on a
simulated quantum register using PennyLane's `ApproxTimeEvolution`
(first-order Trotter-Suzuki), while the barrier monitoring projector
P_B is applied classically between monitoring dates, exactly reproducing
the scheme

    U^{k+1} = P_B e^{-dt_k H_h} U^k.

Implementation notes
---------------------
* N = 2**n_qubits grid points are encoded into the amplitudes of an
  n_qubits-qubit register via `qml.StatePrep` (amplitude embedding).
* `ApproxTimeEvolution(H, t, n_trotter)` natively implements the
  *unitary* evolution exp(-i H t). To obtain the *real* (imaginary-time)
  decay exp(-dt H) required for option pricing, we apply the standard
  Wick rotation t = -i*dt: exp(-i H (-i dt)) = exp(-H dt).
* PennyLane drops the global-phase contribution of the Pauli identity
  component of H when Trotterizing (it is unobservable for a genuine
  *unitary* evolution). For our real-decay use case this identity
  component carries a genuine multiplicative normalization factor
  exp(-dt * c_I), which we therefore re-applied by hand after reading
  out the simulated state.
* Because StatePrep requires a normalized vector, the working state is
  renormalized before encoding and the norm is restored on read-out.

Author: Guillaume Melis
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pennylane as qml

from hamiltonian_pricer import HamiltonianPricer, MonitoringFrequency, build_monitoring_schedule


def _identity_coefficient(H: qml.Hamiltonian, n_qubits: int) -> float:
    """Extract the coefficient of the pure-identity Pauli term I^{otimes n}."""
    coeffs, ops = H.terms()
    target = "I" * n_qubits
    c_I = 0.0
    for c, o in zip(coeffs, ops):
        if qml.pauli.pauli_word_to_string(o) == target:
            c_I += float(np.real(c))
    return c_I


@dataclass
class HybridHamiltonianPricer:
    """Hybrid quantum-classical pricer reproducing the discrete
    Hamiltonian barrier scheme of `HamiltonianPricer`, but delegating
    the propagator e^{-dt H_h} to a (simulated) quantum circuit.

    Parameters mirror `HamiltonianPricer`. `N` must be such that the
    number of interior degrees of freedom (N-1) is a power of two; the
    class will raise otherwise. In practice we instantiate with
    N = dim + 1 so that dim = N - 1 = 2**n_qubits.
    """

    S0: float
    K: float
    B: float
    sigma: float
    r: float
    T: float
    N: int
    n_trotter_steps: int = 8
    x_min: float | None = None
    x_max: float | None = None
    n_std: float = 10.0

    classical: HamiltonianPricer = field(init=False, repr=False, default=None)
    n_qubits: int = field(init=False, repr=False, default=0)
    H_pl: qml.Hamiltonian = field(init=False, repr=False, default=None)
    c_identity: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        # Reuse the classical class to build the grid, H_h, P_B and the
        # initial payoff so that both pricers operate on *identical*
        # discretizations -- the only thing that differs is how the
        # propagator e^{-dt H_h} is evaluated.
        self.classical = HamiltonianPricer(
            S0=self.S0, K=self.K, B=self.B, sigma=self.sigma, r=self.r, T=self.T,
            N=self.N, x_min=self.x_min, x_max=self.x_max, n_std=self.n_std,
        )
        dim = self.N - 1
        n_qubits = int(round(np.log2(dim)))
        if 2 ** n_qubits != dim:
            raise ValueError(
                f"N-1={dim} interior points must be a power of 2 for amplitude "
                f"encoding (choose N = 2**n + 1)."
            )
        self.n_qubits = n_qubits
        self.build_hamiltonian_pl()

    def build_hamiltonian_pl(self) -> qml.Hamiltonian:
        """Pauli-decompose the dense discrete Hamiltonian matrix into a
        PennyLane Hamiltonian acting on n_qubits."""
        H_dense = self.classical.H_h.toarray()
        self.H_pl = qml.pauli_decompose(H_dense, wire_order=range(self.n_qubits))
        self.c_identity = _identity_coefficient(self.H_pl, self.n_qubits)
        return self.H_pl

    def _propagate_step(self, U: np.ndarray, dt: float) -> np.ndarray:
        """Evaluate exp(-dt H_h) U via simulated Hamiltonian evolution
        on a quantum register (Wick-rotated ApproxTimeEvolution)."""
        norm = np.linalg.norm(U)
        if norm == 0.0:
            return U
        psi0 = U / norm

        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(psi0.astype(complex), wires=range(self.n_qubits))
            qml.ApproxTimeEvolution(self.H_pl, -1j * dt, self.n_trotter_steps)
            return qml.state()

        out = np.real(circuit())
        # restore norm and re-apply the identity-term decay factor that
        # PennyLane drops as an unobservable global phase
        return out * norm * np.exp(-dt * self.c_identity)

    def price(
        self,
        frequency: MonitoringFrequency = "daily",
        n_monitoring: int | None = None,
        return_curve: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        """Run the hybrid quantum-classical pricing recursion:

            U^{k+1} = P_B [quantum-simulated exp(-dt_k H_h)] U^k
        """
        schedule = build_monitoring_schedule(self.T, frequency, n_monitoring)
        dts = np.diff(schedule)

        U = self.classical._terminal_payoff()
        for dt in dts:
            U = self._propagate_step(U, dt)
            U = self.classical.P_B @ U

        price0 = self.classical._interpolate_price(U)
        if return_curve:
            interior_x = self.classical.grid[1:-1]
            C_curve = np.exp(self.classical.alpha * interior_x) * U
            return price0, C_curve
        return price0
