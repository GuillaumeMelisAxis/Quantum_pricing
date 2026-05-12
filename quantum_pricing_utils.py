import numpy as np

from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit.primitives import StatevectorSampler
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit_finance.applications.estimation import EuropeanCallPricing, EuropeanCallDelta


def price_Eurocall(S0 : float, K : float,  r : float, vol : float, T : float, n_qubits : int,  c_approx : float, epsilon : float=0.01) -> float :
    """
    Function to call to price an european call using Qiskit framework 
    """

    mu = (r - 0.5*vol**2) * T + np.log(S0)
    sigma = vol * np.sqrt(T)
    mean = np.exp(mu + sigma**2 /2)
    variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
    stddev = np.sqrt(variance)

    low = np.maximum(0, mean - 3* stddev)
    high = mean + 3 * stddev

    uncertainty_model = LogNormalDistribution(
        n_qubits, mu=mu, sigma = sigma**2, bounds=(low, high)
    )

    european_call_pricing = EuropeanCallPricing(
        num_state_qubits=n_qubits,
        strike_price=K,
        rescaling_factor=c_approx,
        bounds=(low, high),
        uncertainty_model=uncertainty_model,
    )

    alpha = 0.05

    sampler = StatevectorSampler()

    problem = european_call_pricing.to_estimation_problem()

    ae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon,
        alpha=alpha,
        sampler=sampler
    )
    result = ae.estimate(problem)

    return european_call_pricing.interpret(result)


def price_Europut(S0 : float, K : float, r : float, vol : float, T : float, n_qubits : int, c_approx : float, epsilon : float=0.01) -> float :
    """
    Wrapper to price an European Put with Quantum framework (Qiskit)
    """

    mu = (r - 0.5*vol**2) * T + np.log(S0)
    sigma = vol * np.sqrt(T)
    mean = np.exp(mu + sigma**2 /2)
    variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
    stddev = np.sqrt(variance)

    low = np.maximum(0, mean - 3* stddev)
    high = mean + 3 * stddev

    uncertainty_model = LogNormalDistribution(
        n_qubits, mu=mu, sigma = sigma**2, bounds=(low, high)
    )

    rescaling_factor = 0.25

    # setup piecewise linear objective fcuntion
    breakpoints = [low, K]
    slopes = [-1, 0]
    offsets = [K - low, 0]
    f_min = 0
    f_max = K - low
    european_put_objective = LinearAmplitudeFunction(
        n_qubits,
        slopes,
        offsets,
        domain=(low, high),
        image=(f_min, f_max),
        breakpoints=breakpoints,
        rescaling_factor=rescaling_factor,
    )

    european_put = european_put_objective.compose(uncertainty_model, front=True)

    alpha = 0.05

    problem = EstimationProblem(
        state_preparation=european_put,
        objective_qubits=[n_qubits],
        post_processing=european_put_objective.post_processing,
    )

    sampler = StatevectorSampler()
    # construct amplitude estimation
    ae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon, alpha=alpha, sampler=sampler
    )

    result = ae.estimate(problem)

    return result.estimation_processed