r"""VaR d'une option sur panier geometrique: grille hybride, Monte Carlo, forme fermee.

Le portefeuille est une position unique sur un put panier geometrique a maturite
fixee. On en calcule la VaR de quatre facons, en faisant varier *deux* axes
independants:

  * la facon de produire les scenarios  - historique (HS) ou Monte Carlo (MC)
  * la facon de revaloriser un scenario - forme fermee, grille hybride QTT, MC

La forme fermee est exacte a la precision machine, donc pour un jeu de scenarios
donne elle fournit la VaR de reference: tout ecart mesure ici est une erreur de
revalorisation, jamais une erreur d'echantillonnage des scenarios. C'est
pourquoi les trois revalorisations partagent toujours le meme jeu de scenarios.

La question posee est celle de la compensation des erreurs. Une VaR est un
quantile d'une distribution de P&L, et le P&L est une *difference* de prix:

    PnL(s) = V(s) - V(0)

Si l'erreur de revalorisation e(s) = Vhat(s) - V*(s) varie lentement avec les
facteurs de risque, elle se retranche entre s et 0 et disparait largement du
P&L. Si elle est un bruit independant d'un scenario a l'autre, elle ne se
retranche pas, elle s'ajoute en variance et elle repousse le quantile vers
l'exterieur. Le script mesure les deux regimes plutot que de les supposer:
l'erreur de prix e(s), l'erreur de P&L e(s) - e(0), leur rapport, et l'erreur
finale sur la VaR.

Le cout d'initialisation est compte. La grille hybride paie un apprentissage
TT-cross une fois, puis des requetes quasi gratuites; Monte Carlo ne paie rien
d'avance et tout a chaque requete. Le script reporte les deux et le nombre de
revalorisations au-dela duquel la grille devient la moins chere.

    .venv/Scripts/python.exe scripts/validation/benchmark_var_geometric_basket.py --profile full
"""

from __future__ import annotations

import os

# Comme dans benchmark_vol_risk_grids.py: le nombre de threads BLAS est lu a
# l'import de numpy, donc il doit etre fixe avant. Un cout par requete qui
# depend silencieusement du nombre de coeurs n'est pas un nombre reproductible.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _variable in _THREAD_VARIABLES:
    os.environ.setdefault(_variable, "1")

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import VolatilityExtendedConfig
from stngpr.pricers import basket_put_monte_carlo, geometric_basket_put
from stngpr.vol_extension import VolExtendedPriceSurrogate


# ==========================================================================
# 1. Le portefeuille
# ==========================================================================


@dataclass(frozen=True)
class Portfolio:
    """Position unique: ``n_contracts`` puts sur panier geometrique equipondere.

    Le point de marche initial est choisi au centre de la boite de la grille,
    loin de ses bords: la grille couvre S dans [5, 150], sigma dans [0.05, 0.80],
    K dans [1, 200], r dans [0.005, 0.08] et T dans [1/365, 3]. Un scenario qui
    sortirait de cette boite serait extrapole par la grille et l'ecart mesure ne
    serait plus une erreur d'interpolation mais une erreur d'extrapolation; la
    generation des chocs borne donc explicitement les facteurs (cf. clipping).
    """

    n_assets: int = 5
    n_contracts: float = 1000.0
    spot: float = 100.0          # S_i(0), identique pour les cinq sous-jacents
    volatility: float = 0.20     # sigma_i(0)
    strike: float = 100.0        # a la monnaie: K = G(S(0)) = 100
    rate: float = 0.03
    maturity: float = 1.0        # "une maturite donnee"
    correlation_rho: float = 0.30  # equicorrelation, celle du modele de pricing

    @property
    def correlation(self) -> np.ndarray:
        c = np.full((self.n_assets, self.n_assets), self.correlation_rho)
        np.fill_diagonal(c, 1.0)
        return c

    @property
    def dividends(self) -> np.ndarray:
        return np.zeros(self.n_assets)

    def base_market_point(self) -> np.ndarray:
        """Point de marche initial, dans le layout interleaved de la grille.

        Colonnes: (S_1, sigma_1, ..., S_5, sigma_5, K, r, T).
        """
        row = np.empty(2 * self.n_assets + 3)
        row[0::2][: self.n_assets] = self.spot
        for i in range(self.n_assets):
            row[2 * i] = self.spot
            row[2 * i + 1] = self.volatility
        row[2 * self.n_assets] = self.strike
        row[2 * self.n_assets + 1] = self.rate
        row[2 * self.n_assets + 2] = self.maturity
        return row


# ==========================================================================
# 2. Le generateur de donnees
# ==========================================================================


@dataclass(frozen=True)
class ShockModelParameters:
    """Parametres du processus generateur de l'historique de chocs.

    Tout est ici: il n'y a aucune constante numerique cachee dans le code du
    generateur. Ce bloc est serialise tel quel dans le JSON de sortie.
    """

    n_days: int = 2000               # ~8 ans a 252 jours ouvres
    trading_days: float = 252.0
    n_assets: int = 5

    # -- innovations de rendement -----------------------------------------
    return_correlation: float = 0.30  # equicorrelation des z^S, = rho du pricer
    student_df: float = 5.0           # degres de liberte de la loi de Student
    real_world_drift: float = 0.0     # mu_i, cf. docstring de generate_history

    # -- processus de volatilite ------------------------------------------
    vol_mean_reversion: float = 3.0   # kappa, par an, sur log(sigma)
    vol_long_run: float = 0.20        # sigma_bar_i
    vol_stationary_sd: float = 0.35   # ecart-type stationnaire de log(sigma)
    vol_correlation: float = 0.50     # correlation des innovations de vol
    leverage: float = -0.60           # corr(z^S_i, xi_i), effet de levier

    # -- bornes (boite de la grille) --------------------------------------
    vol_floor: float = 0.06
    vol_cap: float = 0.75
    spot_floor: float = 6.0
    spot_cap: float = 145.0

    seed: int = 20260918


@dataclass
class ShockHistory:
    """Historique de chocs journaliers et son journal de generation."""

    log_return: np.ndarray        # (n_days, n_assets), x^S
    log_vol_change: np.ndarray    # (n_days, n_assets), x^sigma
    spot_path: np.ndarray         # (n_days + 1, n_assets), trajectoire simulee
    vol_path: np.ndarray          # (n_days + 1, n_assets)
    parameters: ShockModelParameters
    diagnostics: dict = field(default_factory=dict)


def generate_history(params: ShockModelParameters) -> ShockHistory:
    r"""Simule un historique journalier de chocs de spot et de volatilite.

    Le processus est entierement specifie ci-dessous; c'est ce paragraphe qui
    est destine a la section *data* du rapport.

    **Horizon.** ``n_days`` pas journaliers, dt = 1 / ``trading_days``.

    **Structure de dependance.** A chaque date t on tire un vecteur gaussien de
    dimension 2n par construction a facteurs, donc definie positive par
    construction plutot que par verification a posteriori:

        f, g  ~ N(0, 1)           facteur marche, facteur volatilite (communs)
        e_i, h_i ~ N(0, 1)        idiosyncratiques, independants

        z^S_i = sqrt(rho_S) f + sqrt(1 - rho_S) e_i
        xi_i  = lam z^S_i + sqrt(1 - lam^2) [ sqrt(rho_V') g
                                              + sqrt(1 - rho_V') h_i ]

    avec lam = ``leverage``. Cela donne exactement

        corr(z^S_i, z^S_j) = rho_S                       (= return_correlation)
        corr(z^S_i, xi_i)  = lam                         (= leverage)
        corr(xi_i, xi_j)   = lam^2 rho_S + (1 - lam^2) rho_V'
        corr(z^S_i, xi_j)  = lam rho_S        pour i != j (levier croise)

    et ``rho_V'`` est resolu pour que corr(xi_i, xi_j) vaille exactement
    ``vol_correlation``. La construction echoue explicitement si la cible n'est
    pas atteignable, plutot que de renvoyer une matrice non-PSD.

    **Queues epaisses.** Le vecteur de dimension 2n est ensuite multiplie par
    une variable de melange commune W = sqrt((nu - 2) / chi2_nu), ce qui en fait
    un Student multivarie a nu = ``student_df`` degres de liberte, standardise a
    variance unite. Le melange est commun aux rendements et aux innovations de
    volatilite: il produit donc simultanement des queues epaisses et de
    l'agregation de volatilite, et il preserve exactement la matrice de
    correlation ci-dessus.

    **Volatilite.** log sigma suit un AR(1), discretisation exacte d'un
    Ornstein-Uhlenbeck de vitesse kappa = ``vol_mean_reversion``:

        log sigma_{i,t} = log sigma_bar + phi (log sigma_{i,t-1} - log sigma_bar)
                          + eta xi_{i,t}
        phi = exp(-kappa dt)
        eta = s_inf sqrt(1 - phi^2)

    ou s_inf = ``vol_stationary_sd`` est l'ecart-type stationnaire de log sigma.
    Parametrer par l'ecart-type stationnaire plutot que par eta rend le calibrage
    lisible: s_inf = 0.35 signifie que sigma fluctue d'environ +/-35% autour de
    sigma_bar en termes logarithmiques.

    **Spot.** Le spot est integre avec la volatilite *de la veille*, ce qui rend
    le rendement du jour t previsible en variance connaissant t-1, comme dans un
    modele a volatilite stochastique discretise:

        S_{i,t} = S_{i,t-1} exp( (mu_i - 0.5 sigma_{i,t-1}^2) dt
                                 + sigma_{i,t-1} sqrt(dt) z^S_{i,t} )

    Le drift reel mu_i vaut zero par defaut. Ce n'est pas une hypothese de
    marche efficient: c'est le refus d'injecter une vue de rendement espere dans
    un historique qui ne sert qu'a produire des chocs. Sur un horizon d'un jour
    le drift est de toute facon negligeable devant le terme de diffusion, et le
    laisser a zero garde l'historique neutre.

    **Chocs.** Ce que l'historique exporte n'est pas la trajectoire mais ses
    variations relatives, convention RiskMetrics:

        x^S_{i,t}     = log(S_{i,t} / S_{i,t-1})
        x^sigma_{i,t} = log(sigma_{i,t} / sigma_{i,t-1})

    **Bornes.** sigma est ecrete a [``vol_floor``, ``vol_cap``] et S a
    [``spot_floor``, ``spot_cap``], strictement a l'interieur de la boite de la
    grille. Le nombre de jours ecretes est reporte dans ``diagnostics``: s'il
    n'est pas nul, l'historique touche le bord du domaine de la grille et la
    comparaison mesurerait de l'extrapolation. On veut le voir, pas le subir.
    """
    n, days = params.n_assets, params.n_days
    dt = 1.0 / params.trading_days
    rng = np.random.default_rng(params.seed)

    rho_s = params.return_correlation
    lam = params.leverage
    if not -1.0 < lam < 1.0:
        raise ValueError("leverage doit etre dans ]-1, 1[")
    # corr(xi_i, xi_j) = lam^2 rho_S + (1 - lam^2) rho_V'  ->  on resout rho_V'
    rho_v_prime = (params.vol_correlation - lam**2 * rho_s) / (1.0 - lam**2)
    if not 0.0 <= rho_v_prime <= 1.0:
        raise ValueError(
            f"vol_correlation={params.vol_correlation} n'est pas atteignable "
            f"avec leverage={lam} et return_correlation={rho_s}: "
            f"rho_V'={rho_v_prime:.3f} sort de [0, 1]"
        )

    # -- innovations gaussiennes par facteurs ------------------------------
    f = rng.standard_normal((days, 1))
    g = rng.standard_normal((days, 1))
    e = rng.standard_normal((days, n))
    h = rng.standard_normal((days, n))

    z_s = np.sqrt(rho_s) * f + np.sqrt(1.0 - rho_s) * e
    xi = lam * z_s + np.sqrt(1.0 - lam**2) * (
        np.sqrt(rho_v_prime) * g + np.sqrt(1.0 - rho_v_prime) * h
    )

    # -- melange de Student, commun aux 2n composantes ---------------------
    nu = params.student_df
    if nu <= 2.0:
        raise ValueError("student_df doit etre > 2 pour une variance finie")
    chi2 = rng.chisquare(nu, size=(days, 1))
    mixing = np.sqrt((nu - 2.0) / chi2)  # E[W^2] = 1 -> innovations standardisees
    z_s = z_s * mixing
    xi = xi * mixing

    # -- volatilite: AR(1) sur log sigma -----------------------------------
    phi = float(np.exp(-params.vol_mean_reversion * dt))
    eta = params.vol_stationary_sd * np.sqrt(1.0 - phi**2)
    log_vol = np.empty((days + 1, n))
    log_vol[0] = np.log(params.vol_long_run)
    log_bar = np.log(params.vol_long_run)
    for t in range(1, days + 1):
        log_vol[t] = log_bar + phi * (log_vol[t - 1] - log_bar) + eta * xi[t - 1]
    # Pas d'ecretage non plus sur la trajectoire de volatilite. log sigma est un
    # AR(1) gaussien, donc sigma est lognormale et ne peut pas devenir negative:
    # un sigma a 0.06 est un regime de basse volatilite, pas une valeur illegale.
    # Le seul ecretage qui a un sens est celui du scenario, dans apply_shocks.
    vol_path = np.exp(log_vol)
    vol_clipped = int(
        np.sum((vol_path < params.vol_floor) | (vol_path > params.vol_cap))
    )

    # -- spot: integre avec la volatilite de la veille ---------------------
    spot_path = np.empty((days + 1, n))
    spot_path[0] = 100.0  # niveau arbitraire: seuls les rendements sont exportes
    for t in range(1, days + 1):
        sigma_prev = vol_path[t - 1]
        increment = (
            params.real_world_drift - 0.5 * sigma_prev**2
        ) * dt + sigma_prev * np.sqrt(dt) * z_s[t - 1]
        spot_path[t] = spot_path[t - 1] * np.exp(increment)
    # Le niveau de la trajectoire n'est jamais borne: seuls ses rendements
    # journaliers sont exportes, et c'est au niveau du *scenario* que la boite
    # de la grille doit etre respectee (cf. apply_shocks). Borner la trajectoire
    # ici deformerait les rendements du jour ou elle touche la borne, ce qui
    # fabriquerait des chocs qui ne viennent pas du processus.

    log_return = np.diff(np.log(spot_path), axis=0)
    log_vol_change = np.diff(np.log(vol_path), axis=0)

    annualized = float(np.std(log_return) * np.sqrt(params.trading_days))
    diagnostics = {
        "phi": phi,
        "eta": float(eta),
        "rho_v_prime": float(rho_v_prime),
        "vol_days_outside_reporting_band": vol_clipped,
        "realized_annualized_vol": annualized,
        "daily_return_sd": float(np.std(log_return)),
        "daily_return_kurtosis": float(_kurtosis(log_return.ravel())),
        "daily_log_vol_change_sd": float(np.std(log_vol_change)),
        "empirical_return_correlation": float(
            _mean_offdiag(np.corrcoef(log_return.T))
        ),
        "empirical_vol_correlation": float(
            _mean_offdiag(np.corrcoef(log_vol_change.T))
        ),
        "empirical_leverage": float(
            np.mean([
                np.corrcoef(log_return[:, i], log_vol_change[:, i])[0, 1]
                for i in range(n)
            ])
        ),
        "vol_min": float(vol_path.min()),
        "vol_max": float(vol_path.max()),
        "worst_1d_return": float(log_return.min()),
        "best_1d_return": float(log_return.max()),
    }
    return ShockHistory(
        log_return=log_return,
        log_vol_change=log_vol_change,
        spot_path=spot_path,
        vol_path=vol_path,
        parameters=params,
        diagnostics=diagnostics,
    )


def _kurtosis(x: np.ndarray) -> float:
    centered = x - x.mean()
    return float(np.mean(centered**4) / np.mean(centered**2) ** 2)


def _mean_offdiag(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(matrix[mask].mean())


# ==========================================================================
# 3. Construction des scenarios
# ==========================================================================


def apply_shocks(
    portfolio: Portfolio,
    log_return: np.ndarray,
    log_vol_change: np.ndarray,
    params: ShockModelParameters,
) -> tuple[np.ndarray, dict]:
    """Applique les chocs au point de marche courant -> points de marche.

    Convention de re-valorisation: choc relatif (log) sur le spot et sur la
    volatilite, strike, taux et maturite inchanges.

    Maturite gelee: T reste a T_0 dans tous les scenarios. Le P&L mesure alors
    exactement le risque porte par les facteurs qui sont choques, sans y melanger
    la decroissance temporelle deterministe d'un jour. C'est une convention, elle
    doit etre annoncee: avec T roule a T_0 - 1/252 toute la distribution de P&L
    se decale du theta, ce qui deplace la VaR sans rien changer a la question des
    erreurs de revalorisation.

    C'est ici, et seulement ici, que la boite de la grille est imposee. Le
    dictionnaire renvoye compte les coordonnees ecretees: il doit valoir zero,
    sinon la comparaison mesure de l'extrapolation de la grille et non son
    erreur d'interpolation, et le resultat n'est pas celui qu'on croit lire.
    """
    n = portfolio.n_assets
    m = log_return.shape[0]
    points = np.empty((m, 2 * n + 3))
    raw_spots = portfolio.spot * np.exp(log_return)
    raw_vols = portfolio.volatility * np.exp(log_vol_change)
    spots = np.clip(raw_spots, params.spot_floor, params.spot_cap)
    vols = np.clip(raw_vols, params.vol_floor, params.vol_cap)
    clipping = {
        "spot_coordinates_clipped": int(np.sum(raw_spots != spots)),
        "vol_coordinates_clipped": int(np.sum(raw_vols != vols)),
        "spot_min": float(raw_spots.min()),
        "spot_max": float(raw_spots.max()),
        "vol_min": float(raw_vols.min()),
        "vol_max": float(raw_vols.max()),
        "grid_box_spot": [params.spot_floor, params.spot_cap],
        "grid_box_vol": [params.vol_floor, params.vol_cap],
    }
    for i in range(n):
        points[:, 2 * i] = spots[:, i]
        points[:, 2 * i + 1] = vols[:, i]
    points[:, 2 * n] = portfolio.strike
    points[:, 2 * n + 1] = portfolio.rate
    points[:, 2 * n + 2] = portfolio.maturity
    return points, clipping


def fit_gaussian_scenarios(
    history: ShockHistory, n_scenarios: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Scenarios Monte Carlo: gaussienne multivariee ajustee sur l'historique.

    C'est la VaR Monte Carlo au sens usuel: on ajuste une loi parametrique aux
    chocs observes puis on en tire des scenarios. Le vecteur ajuste est celui de
    dimension 2n empile (rendements puis variations de log-vol), avec sa moyenne
    et sa covariance empiriques completes - donc la correlation rendement/vol et
    l'effet de levier sont conserves, mais les queues de Student ne le sont pas.

    Ce mis-specification est deliberee et sans effet sur ce qu'on mesure: les
    trois revalorisations partagent le meme tirage de scenarios, donc l'erreur de
    modele se retranche exactement dans les comparaisons. Elle n'apparait que
    dans l'ecart entre VaR historique et VaR Monte Carlo, qui est rapporte a part.
    """
    stacked = np.hstack([history.log_return, history.log_vol_change])
    mean = stacked.mean(axis=0)
    cov = np.cov(stacked, rowvar=False)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean, cov, size=n_scenarios, method="cholesky")
    n = history.parameters.n_assets
    info = {
        "distribution": "gaussienne multivariee, moyenne et covariance empiriques",
        "dimension": int(stacked.shape[1]),
        "n_scenarios": int(n_scenarios),
        "seed": int(seed),
        "fitted_on_days": int(stacked.shape[0]),
        "note": (
            "les queues de Student de l'historique ne sont pas reproduites; "
            "l'ecart HS/MC qui en resulte est rapporte separement"
        ),
    }
    return draws[:, :n], draws[:, n:], info


# ==========================================================================
# 4. Revalorisation
# ==========================================================================


def price_analytic(portfolio: Portfolio, points: np.ndarray) -> np.ndarray:
    n = portfolio.n_assets
    return geometric_basket_put(
        points[:, 0 : 2 * n : 2],
        points[:, 2 * n],
        points[:, 2 * n + 1],
        points[:, 2 * n + 2],
        points[:, 1 : 2 * n : 2],
        portfolio.correlation,
        portfolio.dividends,
    )


def price_monte_carlo(
    portfolio: Portfolio,
    points: np.ndarray,
    n_paths: int,
    seed: int,
    common_random_numbers: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Revalorisation Monte Carlo, avec ou sans nombres aleatoires communs.

    ``common_random_numbers=False`` est ce que fait un moteur de risque naif:
    chaque scenario recoit un tirage frais, donc l'erreur de revalorisation est
    un bruit independant d'un scenario a l'autre. Elle ne se compense pas dans
    le P&L et elle gonfle la queue.

    ``common_random_numbers=True`` rejoue exactement les memes normales
    standardisees pour chaque scenario. L'erreur devient une fonction lisse des
    facteurs de risque et se retranche en grande partie dans le P&L. C'est la
    contrepartie Monte Carlo de ce que fait structurellement un surrogate
    deterministe, et c'est pour cela que les deux sont compares ici.
    """
    n = portfolio.n_assets
    spots = points[:, 0 : 2 * n : 2]
    vols = points[:, 1 : 2 * n : 2]
    strikes = points[:, 2 * n]
    rates = points[:, 2 * n + 1]
    maturities = points[:, 2 * n + 2]
    if not common_random_numbers:
        return basket_put_monte_carlo(
            spots, strikes, rates, maturities, vols,
            portfolio.correlation, n_paths,
            basket_kind="geometric", dividends=portfolio.dividends, seed=seed,
        )
    prices = np.empty(points.shape[0])
    errors = np.empty(points.shape[0])
    for i in range(points.shape[0]):
        p, e = basket_put_monte_carlo(
            spots[i : i + 1], strikes[i : i + 1], rates[i : i + 1],
            maturities[i : i + 1], vols[i : i + 1],
            portfolio.correlation, n_paths,
            basket_kind="geometric", dividends=portfolio.dividends,
            rng=np.random.default_rng(seed),
        )
        prices[i], errors[i] = p[0], e[0]
    return prices, errors


# ==========================================================================
# 5. VaR, ES et analyse de compensation
# ==========================================================================


def var_es(pnl: np.ndarray, level: float) -> tuple[float, float]:
    """VaR et ES a ``level`` (0.99 = 99%), en perte positive.

    Convention: quantile empirique lineaire (numpy ``method='linear'``) du P&L a
    la probabilite 1 - level, renvoye en signe de perte. L'ES est la moyenne des
    P&L au-dela de ce quantile, borne inferieurement a une observation.
    """
    threshold = float(np.quantile(pnl, 1.0 - level, method="linear"))
    tail = pnl[pnl <= threshold]
    if tail.size == 0:
        tail = np.array([threshold])
    return -threshold, -float(tail.mean())


def bootstrap_var_interval(pnl, level, resamples=4000, seed=991):
    """IC bootstrap de la VaR de reference, par reechantillonnage des scenarios.

    Sans cette borne on ne sait pas lire les erreurs des autres bras. Un
    estimateur historique a 2000 jours place sa VaR 99% sur la 20e pire
    observation: son propre bruit d'echantillonnage se chiffre en pourcents, et
    une erreur de revalorisation plus petite que cet intervalle n'est pas
    distinguable de zero, quelle que soit la methode qui l'a produite.
    """
    rng = np.random.default_rng(seed)
    n = pnl.size
    draws = np.empty(resamples)
    for b in range(resamples):
        draws[b] = var_es(pnl[rng.integers(0, n, n)], level)[0]
    return {
        "level": 0.95,
        "resamples": int(resamples),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "standard_error": float(np.std(draws, ddof=1)),
    }


def cancellation_report(
    approx_prices: np.ndarray,
    exact_prices: np.ndarray,
    approx_base: float,
    exact_base: float,
) -> dict:
    """Erreur de prix contre erreur de P&L: la compensation, mesuree.

    e(s) est l'erreur de prix par scenario, e(0) celle au point de base. Comme
    PnL = V(s) - V(0), l'erreur de P&L vaut exactement e(s) - e(0): ce qui est
    commun aux deux se retranche. Le ``cancellation_ratio`` est le rapport des
    erreurs absolues moyennes; en dessous de 1 la compensation opere, au-dessus
    de 1 l'approximation ajoute du bruit au lieu d'en retirer.
    """
    price_error = approx_prices - exact_prices
    base_error = approx_base - exact_base
    pnl_error = price_error - base_error
    mean_abs_price = float(np.mean(np.abs(price_error)))
    mean_abs_pnl = float(np.mean(np.abs(pnl_error)))
    return {
        "price_error_mean": float(np.mean(price_error)),
        "price_error_sd": float(np.std(price_error, ddof=1)),
        "price_error_mean_abs": mean_abs_price,
        "base_point_error": float(base_error),
        "pnl_error_mean": float(np.mean(pnl_error)),
        "pnl_error_sd": float(np.std(pnl_error, ddof=1)),
        "pnl_error_mean_abs": mean_abs_pnl,
        "cancellation_ratio": (
            mean_abs_pnl / mean_abs_price if mean_abs_price > 0 else float("nan")
        ),
        "systematic_share": (
            abs(float(np.mean(price_error))) / mean_abs_price
            if mean_abs_price > 0
            else float("nan")
        ),
    }


# ==========================================================================
# 6. Orchestration
# ==========================================================================

PROFILES = {
    "smoke": {
        "n_days": 250,
        "mc_scenarios": 500,
        "budgets": (20_000,),
        "mc_paths": (4_096, 16_384),
        "crn_paths": (16_384,),
        "outer_mc_paths": (4_096,),
        "anova_samples": 500,
    },
    "full": {
        "n_days": 2000,
        "mc_scenarios": 5000,
        "budgets": (20_000, 60_000, 150_000),
        "mc_paths": (4_096, 16_384, 65_536, 262_144),
        "crn_paths": (16_384, 262_144),
        "outer_mc_paths": (4_096, 16_384, 65_536),
        "anova_samples": 1000,
    },
}

LEVELS = (0.975, 0.99)


def timed_call(function):
    start = perf_counter()
    result = function()
    return result, perf_counter() - start


def arm_result(name, prices, base_price, exact_prices, exact_base, portfolio,
               seconds, setup_seconds, extra=None):
    """Assemble le resultat d'un bras: VaR, ES, erreurs, couts."""
    pnl = portfolio.n_contracts * (prices - base_price)
    exact_pnl = portfolio.n_contracts * (exact_prices - exact_base)
    row = {
        "arm": name,
        "base_price": float(base_price),
        "revaluation_seconds": float(seconds),
        "setup_seconds": float(setup_seconds),
        "total_seconds": float(seconds + setup_seconds),
        "seconds_per_revaluation": float(seconds / prices.size),
        "n_scenarios": int(prices.size),
    }
    for level in LEVELS:
        var, es = var_es(pnl, level)
        var_ref, es_ref = var_es(exact_pnl, level)
        tag = f"{level * 100:g}".replace(".", "_")
        row[f"var_{tag}"] = var
        row[f"es_{tag}"] = es
        row[f"var_{tag}_error"] = var - var_ref
        row[f"var_{tag}_relative_error"] = (
            (var - var_ref) / var_ref if var_ref != 0 else float("nan")
        )
        row[f"es_{tag}_error"] = es - es_ref
    row.update(
        cancellation_report(prices, exact_prices, base_price, exact_base)
    )
    if extra:
        row.update(extra)
    return row


def run_scenario_set(label, points, portfolio, surrogates, args,
                     mc_paths, crn_paths, log):
    """Execute les trois revalorisations sur un meme jeu de scenarios."""
    base_point = portfolio.base_market_point().reshape(1, -1)

    exact, exact_seconds = timed_call(lambda: price_analytic(portfolio, points))
    exact_base = float(price_analytic(portfolio, base_point)[0])
    exact_pnl = portfolio.n_contracts * (exact - exact_base)
    reference = {"family": "closed_form"}
    for level in LEVELS:
        tag = f"{level * 100:g}".replace(".", "_")
        reference[f"var_{tag}_bootstrap_ci"] = bootstrap_var_interval(
            exact_pnl, level
        )
    rows = [
        arm_result(
            "closed_form", exact, exact_base, exact, exact_base, portfolio,
            exact_seconds, 0.0, reference,
        )
    ]
    log(f"  [{label}] forme fermee: {exact_seconds:.3f} s")

    for budget, entry in surrogates.items():
        model = entry["model"]
        prices, seconds = timed_call(lambda m=model: m.price(points))
        base = float(model.price(base_point)[0])
        rows.append(arm_result(
            f"hybrid_grid_{budget}", prices, base, exact, exact_base, portfolio,
            seconds, entry["training_seconds"],
            {
                "family": "hybrid_grid",
                "budget": int(budget),
                "effective_rank": entry["effective_rank"],
                "parameter_count": entry["parameter_count"],
                "oracle_calls": int(budget),
            },
        ))
        log(f"  [{label}] grille {budget:>7,}: {seconds:.3f} s")

    for paths in mc_paths:
        prices, seconds = timed_call(
            lambda p=paths: price_monte_carlo(
                portfolio, points, p, args.mc_seed, False
            )[0]
        )
        base = float(price_monte_carlo(
            portfolio, base_point, paths, args.mc_seed + 1, False
        )[0][0])
        rows.append(arm_result(
            f"monte_carlo_{paths}", prices, base, exact, exact_base, portfolio,
            seconds, 0.0,
            {"family": "monte_carlo", "paths": int(paths),
             "common_random_numbers": False},
        ))
        log(f"  [{label}] MC {paths:>7,} (independant): {seconds:.1f} s")

    for paths in crn_paths:
        prices, seconds = timed_call(
            lambda p=paths: price_monte_carlo(
                portfolio, points, p, args.mc_seed, True
            )[0]
        )
        base = float(price_monte_carlo(
            portfolio, base_point, paths, args.mc_seed, True
        )[0][0])
        rows.append(arm_result(
            f"monte_carlo_crn_{paths}", prices, base, exact, exact_base,
            portfolio, seconds, 0.0,
            {"family": "monte_carlo_crn", "paths": int(paths),
             "common_random_numbers": True},
        ))
        log(f"  [{label}] MC {paths:>7,} (CRN): {seconds:.1f} s")

    return rows


def break_even(rows):
    """Nombre de revalorisations au-dela duquel la grille est la moins chere."""
    grids = [r for r in rows if r["family"] == "hybrid_grid"]
    mcs = [r for r in rows if r["family"] in ("monte_carlo", "monte_carlo_crn")]
    out = []
    for grid in grids:
        for mc in mcs:
            gap = mc["seconds_per_revaluation"] - grid["seconds_per_revaluation"]
            out.append({
                "grid": grid["arm"],
                "monte_carlo": mc["arm"],
                "grid_setup_seconds": grid["setup_seconds"],
                "seconds_saved_per_revaluation": float(gap),
                "break_even_revaluations": (
                    float(grid["setup_seconds"] / gap) if gap > 0 else float("inf")
                ),
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--mode", default="price_adaptive")
    parser.add_argument("--maturity-nodes", type=int, default=32)
    parser.add_argument("--history-seed", type=int, default=20260918)
    parser.add_argument("--scenario-seed", type=int, default=771)
    parser.add_argument("--mc-seed", type=int, default=4242)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    profile = PROFILES[args.profile]

    def log(message):
        if not args.quiet:
            print(message, flush=True)

    portfolio = Portfolio()
    config = VolatilityExtendedConfig(maturity_nodes=args.maturity_nodes)
    if not np.allclose(config.correlation, portfolio.correlation):
        raise SystemExit(
            "la correlation du portefeuille et celle de la grille different"
        )

    log(f"profil {args.profile}")
    log("1. generation de l'historique de chocs")
    params = ShockModelParameters(
        n_days=profile["n_days"], seed=args.history_seed
    )
    history = generate_history(params)
    for key, value in history.diagnostics.items():
        log(f"   {key:32s} {value}")

    log("2. scenarios")
    hs_points, hs_clipping = apply_shocks(
        portfolio, history.log_return, history.log_vol_change, params
    )
    mc_returns, mc_vols, mc_info = fit_gaussian_scenarios(
        history, profile["mc_scenarios"], args.scenario_seed
    )
    mc_points, mc_clipping = apply_shocks(portfolio, mc_returns, mc_vols, params)
    log(f"   historique : {hs_points.shape[0]} scenarios, "
        f"{hs_clipping['spot_coordinates_clipped']} spots et "
        f"{hs_clipping['vol_coordinates_clipped']} vols ecretes")
    log(f"   monte carlo: {mc_points.shape[0]} scenarios, "
        f"{mc_clipping['spot_coordinates_clipped']} spots et "
        f"{mc_clipping['vol_coordinates_clipped']} vols ecretes")

    log("3. apprentissage des grilles hybrides (cout d'initialisation)")
    surrogates = {}
    for budget in profile["budgets"]:
        model = VolExtendedPriceSurrogate(config, args.mode, seed=config.seed)
        start = perf_counter()
        diagnostics = model.fit(
            budget, anova_samples=profile["anova_samples"], log=False
        )
        training = perf_counter() - start
        surrogates[budget] = {
            "model": model,
            "training_seconds": training,
            "effective_rank": float(diagnostics.effective_rank),
            "parameter_count": int(diagnostics.parameter_count),
        }
        log(f"   budget {budget:>7,}: {training:6.1f} s, "
            f"rang {diagnostics.effective_rank:5.2f}, "
            f"{diagnostics.parameter_count:,} parametres")

    log("4. VaR historique (full valuation)")
    hs_rows = run_scenario_set(
        "HS", hs_points, portfolio, surrogates, args,
        profile["mc_paths"], profile["crn_paths"], log,
    )
    log("5. VaR Monte Carlo")
    mc_rows = run_scenario_set(
        "MC", mc_points, portfolio, surrogates, args,
        profile["outer_mc_paths"], (), log,
    )

    report = {
        "experiment": (
            "VaR d'un put sur panier geometrique: grille hybride vs Monte Carlo"
        ),
        "profile": args.profile,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "threads": {v: os.environ.get(v) for v in _THREAD_VARIABLES},
        },
        "portfolio": asdict(portfolio),
        "surrogate_configuration": {
            "mode": args.mode,
            "maturity_nodes": int(args.maturity_nodes),
            "physical_shape": list(config.physical_shape),
            "qtt_cores": len(config.qtt_shape),
            "seed": int(config.seed),
        },
        "data_generation": {
            "model_parameters": asdict(params),
            "diagnostics": history.diagnostics,
            "shock_convention": "log-relatif sur S et sigma, K/r/T geles",
            "maturity_convention": "T gelee a T_0, le P&L exclut le theta",
            "historical_scenario_box": hs_clipping,
            "monte_carlo_scenario_box": mc_clipping,
        },
        "monte_carlo_scenarios": mc_info,
        "historical_var": hs_rows,
        "monte_carlo_var": mc_rows,
        "break_even_historical": break_even(hs_rows),
        "levels": list(LEVELS),
        "quantile_convention": (
            "quantile empirique lineaire de numpy, perte comptee positive"
        ),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"var_geometric_basket_{args.profile}"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(report, indent=1, default=float), encoding="utf-8"
    )

    csv_path = out_dir / f"{stem}.csv"
    keys = sorted({k for row in hs_rows + mc_rows for k in row})
    lines = ["scenario_set," + ",".join(keys)]
    for tag, rows in (("historical", hs_rows), ("monte_carlo", mc_rows)):
        for row in rows:
            lines.append(tag + "," + ",".join(str(row.get(k, "")) for k in keys))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("")
    log("VaR 99% (perte, en unites de prix x n_contracts)")
    header = (
        f"  {'bras':26s} {'VaR':>12s} {'erreur':>12s} {'err rel':>9s} "
        f"{'|e| prix':>10s} {'|e| PnL':>10s} {'ratio':>7s} {'cout s':>9s}"
    )
    log(header)
    for tag, rows in (("HS", hs_rows), ("MC", mc_rows)):
        for row in rows:
            log(f"  {tag} {row['arm']:23s} {row['var_99']:12.4f} "
                f"{row['var_99_error']:12.4f} "
                f"{row['var_99_relative_error'] * 100:8.2f}% "
                f"{row['price_error_mean_abs']:10.6f} "
                f"{row['pnl_error_mean_abs']:10.6f} "
                f"{row['cancellation_ratio']:7.3f} "
                f"{row['total_seconds']:9.2f}")
    log("")
    log(f"ecrit {json_path}")
    log(f"ecrit {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
