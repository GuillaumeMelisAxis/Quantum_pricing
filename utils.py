import numpy as np
from scipy.stats import norm


def bs_call(S0, K, sigma, r, T):
    """Prix analytique Black-Scholes d'un Call européen."""

    if T <= 0:
        return max(S0 - K, 0.0)
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S0, K, sigma, r, T):
    """Prix analytique Black-Scholes d'un Put européen."""

    if T <= 0:
        return max(S0 - K, 0.0)
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)