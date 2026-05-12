import numpy as np
from scipy.stats import norm


def bs_call(S0 : float, K : float, sigma : float, r : float, T : float) -> float:
    """Prix analytique Black-Scholes d'un Call européen."""

    if T <= 0:
        return max(S0 - K, 0.0)
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S0 : float, K : float, sigma : float, r : float, T : float) -> float:
    """Prix analytique Black-Scholes d'un Put européen."""

    if T <= 0:
        return max(S0 - K, 0.0)
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def monte_carlo_european_options(S0 : float, K : float, r : float, T : float, sigma : float, num_paths : int, n_steps : int, seed=None) -> dict:
    """
    Prix Monte Carlo d'un call et d'un put européens sous Black-Scholes.

    Paramètres
    ----------
    S0 : float
        Prix initial du sous-jacent
    K : float
        Strike
    r : float
        Taux sans risque annuel
    T : float
        Maturité en années
    sigma : float
        Volatilité annuelle
    num_paths : int
        Nombre de trajectoires Monte Carlo
    n_steps : int
        Nombre de pas de temps
    seed : int or None
        Graine aléatoire pour reproductibilité

    Retour
    ------
    dict avec les prix estimés et les erreurs standards
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    # Simulation des trajectoires
    S = np.full(num_paths, S0, dtype=float)

    for _ in range(n_steps):
        Z = np.random.normal(size=num_paths)
        S *= np.exp(drift + vol * Z)

    # Payoffs à maturité
    call_payoffs = np.maximum(S - K, 0.0)
    put_payoffs = np.maximum(K - S, 0.0)

    # Actualisation
    discount_factor = np.exp(-r * T)
    call_price = discount_factor * np.mean(call_payoffs)
    put_price = discount_factor * np.mean(put_payoffs)

    # Erreur standard de l'estimateur MC
    call_std_error = discount_factor * np.std(call_payoffs, ddof=1) / np.sqrt(num_paths)
    put_std_error = discount_factor * np.std(put_payoffs, ddof=1) / np.sqrt(num_paths)

    return {
        "call_price": call_price,
        "put_price": put_price,
        "call_std_error": call_std_error,
        "put_std_error": put_std_error
    }