import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
from scipy.linalg import eigh


def bs_closed_form_call(S0, K, r, sigma, T):
    """Prix Black-Scholes exact d'un call européen, pour vérification."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def price_option_hamiltonian_bs(
    S0,
    K,
    r,
    sigma,
    T,
    option_type="call",
    S_max=None,
    N=400,
    Mt=800,
):
    """
    Pricing d'une option européenne via la formulation Hamiltonienne de Black-Scholes.

    H = -1/2 * sigma^2 * S^2 * d2/dS2 - r * S * d/dS + r
    et on propage V vers le passé dans le temps.

    Parameters
    ----------
    S0 : float
        Spot actuel
    K : float
        Strike
    r : float
        Taux sans risque
    sigma : float
        Volatilité
    T : float
        Maturité
    option_type : str
        "call" ou "put"
    S_max : float
        Bord supérieur du domaine en spot
    N : int
        Nombre de points en prix
    Mt : int
        Nombre de pas de temps

    Returns
    -------
    price : float
        Prix interpolé en S0
    grid_S : np.ndarray
        Grille des spots
    V : np.ndarray
        Valeur de l'option sur la grille à t=0
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    if S_max is None:
        S_max = max(4 * K, 4 * S0)

    S = np.linspace(0.0, S_max, N)
    dS = S[1] - S[0]
    dt = T / Mt

    # Dérivées finies sur toute la grille
    upper1 = np.ones(N - 1) / (2 * dS)
    lower1 = -np.ones(N - 1) / (2 * dS)
    D1 = sparse.diags([lower1, upper1], offsets=[-1, 1], shape=(N, N), format="lil")

    lower2 = np.ones(N - 1) / (dS**2)
    diag2 = -2.0 * np.ones(N) / (dS**2)
    upper2 = np.ones(N - 1) / (dS**2)
    D2 = sparse.diags([lower2, diag2, upper2], offsets=[-1, 0, 1], shape=(N, N), format="lil")

    # On neutralise les lignes de bord
    for mat in (D1, D2):
        mat[0, :] = 0.0
        mat[-1, :] = 0.0

    D1 = D1.tocsr()
    D2 = D2.tocsr()

    S_diag = sparse.diags(S, format="csr")
    S2_diag = sparse.diags(S**2, format="csr")
    I = sparse.identity(N, format="csr")

    # Hamiltonien BS
    H = -0.5 * sigma**2 * (S2_diag @ D2) - r * (S_diag @ D1) + r * I

    # Crank-Nicolson : (I + dt/2 H) V_n = (I - dt/2 H) V_{n+1}
    A = (I + 0.5 * dt * H).tolil()
    B = (I - 0.5 * dt * H).tolil()

    # Conditions de Dirichlet aux frontières
    for mat in (A, B):
        mat[0, :] = 0.0
        mat[0, 0] = 1.0
        mat[-1, :] = 0.0
        mat[-1, -1] = 1.0

    A = A.tocsr()
    B = B.tocsr()

    # Payoff à maturité
    if option_type == "call":
        V = np.maximum(S - K, 0.0)
    else:
        V = np.maximum(K - S, 0.0)

    history = [V.copy()]
    for n in range(Mt):
        tau_next = (n + 1) * dt
        rhs = B @ V

        if option_type == "call":
            rhs[0] = 0.0
            rhs[-1] = S_max - K * np.exp(-r * tau_next)
        else:
            rhs[0] = K * np.exp(-r * tau_next)
            rhs[-1] = 0.0

        V = spsolve(A, rhs)
        history.append(V.copy())

    history = np.array(history)
    price = np.interp(S0, S, V)
    return price, S, V, history

def quantum_down_and_out_call(
    S0,
    K,
    B,
    T,
    r,
    sigma,
    N=500,
    xmax_mult=5.0
):
    """
    Down-and-Out European Call using the Hermitian Hamiltonian approach.

    Parameters
    ----------
    S0 : float
        Spot price
    K : float
        Strike
    B : float
        Barrier (must satisfy B < S0)
    T : float
        Maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    N : int
        Number of spatial grid points
    xmax_mult : float
        Upper truncation of log-space domain

    Returns
    -------
    float
        Option price
    """

    if B >= S0:
        return 0.0

    # ----------------------------------------------------
    # Log-space domain
    # ----------------------------------------------------

    xmin = np.log(B)

    xmax = np.log(max(xmax_mult * K, xmax_mult * S0))

    x = np.linspace(xmin, xmax, N)

    dx = x[1] - x[0]

    # ----------------------------------------------------
    # Gauge transform
    # ----------------------------------------------------

    mu = r - 0.5 * sigma**2

    alpha = -mu / sigma**2

    Veff = r + mu**2 / (2.0 * sigma**2)

    # ----------------------------------------------------
    # Hermitian Hamiltonian
    #
    # H = -σ²/2 d²/dx² + Veff
    # ----------------------------------------------------

    main_diag = (
        sigma**2 / dx**2
        + Veff
    ) * np.ones(N)

    off_diag = (
        -sigma**2 / (2.0 * dx**2)
    ) * np.ones(N - 1)

    H = (
        np.diag(main_diag)
        + np.diag(off_diag, 1)
        + np.diag(off_diag, -1)
    )

    # ----------------------------------------------------
    # Infinite barrier at x = xmin
    #
    # ψ(B)=0
    # ----------------------------------------------------

    H[0, :] = 0.0
    H[:, 0] = 0.0
    H[0, 0] = 1e12

    # ----------------------------------------------------
    # Terminal payoff
    # ----------------------------------------------------

    payoff = np.maximum(np.exp(x) - K, 0.0)

    psi_T = np.exp(-alpha * x) * payoff

    # ----------------------------------------------------
    # Spectral decomposition
    # ----------------------------------------------------

    eigvals, eigvecs = eigh(H)

    coeffs = eigvecs.T @ psi_T

    coeffs *= np.exp(-eigvals * T)

    psi_0 = eigvecs @ coeffs

    # ----------------------------------------------------
    # Back to original variables
    # ----------------------------------------------------

    V0_grid = np.exp(alpha * x) * psi_0

    # ----------------------------------------------------
    # Interpolate
    # ----------------------------------------------------

    price = np.interp(np.log(S0), x, V0_grid)

    return float(price)

def mc_down_and_out_call(
    S0,
    K,
    B,
    T,
    r,
    sigma,
    n_paths=200000,
    n_steps=252
):

    dt = T / n_steps

    S = np.full(n_paths, S0, dtype=float)

    alive = np.ones(n_paths, dtype=bool)

    for _ in range(n_steps):

        z = np.random.normal(size=n_paths)

        S *= np.exp(
            (r - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * z
        )

        alive &= (S > B)

    payoff = np.where(
        alive,
        np.maximum(S - K, 0.0),
        0.0
    )

    disc_payoff = np.exp(-r * T) * payoff

    price = disc_payoff.mean()

    error = disc_payoff.std(ddof=1) / np.sqrt(n_paths)

    return price, error


if __name__ == "__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    B = 80

    price, grid, values, H = price_option_hamiltonian_bs(
        S0, K, r, sigma, T, option_type="call", S_max=400, N=300, Mt=600
    )

    exact = bs_closed_form_call(S0, K, r, sigma, T)

    print(f"Prix Hamiltonien : {price:.6f}")
    print(f"Black-Scholes exact : {exact:.6f}")
    print(f"Erreur absolue : {abs(price - exact):.6f}")

    ham_price = quantum_down_and_out_call(
        S0, K, B, T, r, sigma
    )

    mc_price, mc_err = mc_down_and_out_call(
        S0, K, B, T, r, sigma
    )

    print(f"Hamiltonian : {ham_price:.6f}")
    print(f"Monte Carlo : {mc_price:.6f} ± {1.96*mc_err:.6f}")