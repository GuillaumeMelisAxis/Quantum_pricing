r"""VaR 1 jour d'un livre de 6 options sur panier geometrique: QTT contre Monte Carlo.

Trois revalorisations du meme livre, sur les *memes* scenarios, dans les memes
coordonnees de marche ``(S_1, ..., S_5, K, r, T)``:

  * ``closed_form`` - Black-Scholes panier geometrique. Exacte a la precision
    machine, donc elle fournit la VaR de reference. Tout ecart mesure ici est
    une erreur de revalorisation et jamais une erreur d'echantillonnage des
    scenarios: c'est pour cela que les trois bras partagent un unique jeu de
    10 000 trajectoires.
  * ``monte_carlo`` - Monte Carlo imbrique. Chaque scenario est revalorise par
    simulation terminale des cinq sous-jacents correles, avec nombres
    aleatoires communs (CRN) d'un scenario a l'autre.
  * ``qtt`` - le surrogate QTT du depot, sur la **grille de risque hybride
    13-D** ``(S_1, sigma_1, ..., S_5, sigma_5, K, r, T)``, famille
    ``bounded_standardized_risk``, interpolation cubique sur tous les axes. Il
    n'est calibre que sur des **puts**; les calls sont obtenus par parite
    call/put, dont le terme ``e^{-rT} (F - K)`` est exact en forme fermee.
    L'erreur sur un call est donc exactement l'erreur sur le put de meme
    strike, pas une seconde approximation empilee.

Le choix de la grille 13-D n'est pas une generalite gratuite. Sa coordonnee de
strike est standardisee par ``sigma_B sqrt(T)``, donc une meme coordonnee
designe un strike different a chaque maturite: c'est ce qui lui permet de
resoudre la structure en moneyness *de facon coherente sur toute l'echelle de
maturites*, ce qu'une grille de prix a maturite grossiere ne fait pas. Un livre
etale de 1 mois a 2 ans est precisement le cas ou cela compte. La volatilite
est portee comme coordonnee bien qu'elle soit figee dans cette experience: le
surrogate reste alors utilisable pour une VaR qui choque aussi la surface, au
prix d'un cout d'initialisation plus eleve.

Le temps total d'un bras inclut son cout d'initialisation. Pour le QTT cela
veut dire la construction de la grille *et* l'initialisation des noeuds -
echantillonnage ANOVA d'ordre 2 puis TT-cross - payee une seule fois avant la
premiere requete. Pour le Monte Carlo imbrique cela veut dire le tirage des
normales communes. La forme fermee ne paie aucune initialisation. Le script
reporte les trois totaux et le nombre de revalorisations au-dela duquel le QTT
devient le moins cher, parce qu'un surrogate qui gagne en cout par requete mais
perd en cout total sur un run de VaR n'a rien gagne du tout.

Une VaR est un quantile d'une *difference* de prix, PnL(s) = V(s) - V(0). Si
l'erreur de revalorisation e(s) varie lentement avec les facteurs de risque
elle se retranche entre s et 0 et disparait largement du PnL; si elle est un
bruit independant d'un scenario a l'autre elle ne se retranche pas et elle
repousse le quantile vers l'exterieur. Le script mesure les deux regimes plutot
que de les supposer: erreur de prix, erreur de PnL, et leur rapport.

    .venv/Scripts/python.exe scripts/validation/benchmark_var_qtt_basket_book.py --profile smoke
    .venv/Scripts/python.exe scripts/validation/benchmark_var_qtt_basket_book.py --profile full
"""

from __future__ import annotations

import os

# Le nombre de threads BLAS est lu a l'import de numpy, donc il doit etre fixe
# avant. Un cout par requete qui depend silencieusement du nombre de coeurs
# disponibles n'est pas un nombre reproductible, et ce script compare des couts.
for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from stngpr.config import VolatilityExtendedConfig
from stngpr.portfolio import Portfolio
from stngpr.pricers import (
    geometric_basket_call,
    geometric_basket_parity_offset,
    geometric_basket_put,
)
from stngpr.risk import (
    compare_loss_distributions,
    domain_coverage,
    var_es_uncertainty,
)
from stngpr.scenarios import GBMScenarioGenerator
from stngpr.vol_extension import VOL_RISK_GRID_MODES, VolExtendedPriceSurrogate

# ==========================================================================
# 1. Donnees - cinq actions synthetiques
# ==========================================================================
#
# Le marche est synthetique et entierement specifie ici: c'est la "section
# data" de l'experience. Les niveaux sont choisis autour de 100 et les
# volatilites implicites s'etalent de 18% a 30%, de sorte qu'aucun actif ne
# domine le panier et que la volatilite du panier ne soit pas celle d'un actif
# unique deguise. La correlation est equicorrelee a 0.30, la valeur par defaut
# de ``PaperConfig``: assez basse pour que la diversification joue, assez haute
# pour que les cinq facteurs ne soient pas independants.
#
# Aucun dividende: le panier geometrique a alors un forward analytique simple,
# et l'offset de parite reste exactement ``e^{-rT}(F - K)``.

ASSETS = (
    # (ticker, spot, volatilite implicite constante)
    ("SYN-A", 95.0, 0.18),
    ("SYN-B", 100.0, 0.22),
    ("SYN-C", 105.0, 0.25),
    ("SYN-D", 90.0, 0.30),
    ("SYN-E", 110.0, 0.20),
)
CORRELATION_RHO = 0.30
RISK_FREE_RATE = 0.03
DIVIDEND_YIELDS = (0.0,) * 5

# ==========================================================================
# 2. Le livre - 6 options longues sur le panier geometrique equipondere
# ==========================================================================
#
# Le strike est fige a la date 0 a partir de ``G_0``; il ne bouge plus ensuite,
# donc la moneyness derive avec les scenarios, ce qui est le comportement d'un
# vrai livre.
#
# Le melange call/put n'est pas decoratif. Un livre uniquement long de puts est
# monotone decroissant en chaque spot et sa perte extreme vit dans un seul coin
# de la distribution; en ajoutant les calls, le livre devient une position
# longue de convexite dont les deux queues comptent, et la VaR devient
# sensible au *milieu* de la distribution des spots. C'est le cas ou une erreur
# de revalorisation lisse a le plus de chances de ne pas se retrancher.
#
# Deux dispositions sont disponibles.
#
# ``otm_term_ladder`` (defaut) n'a que des options hors de la monnaie - puts
# sous 100%, calls au-dessus - et six maturites distinctes de 1 mois a 2 ans.
# L'appariement moneyness/maturite n'est pas libre: la moneyness est mesuree en
# ecarts-types du panier, ``log(m) / (sigma_B sqrt(T))``, et un put 80% a un
# mois se trouve a cinq ecarts-types, c'est-a-dire qu'il vaut zero et ne pese
# rien dans la VaR. Les moneyness les plus eloignees sont donc portees par les
# maturites les plus longues, ce qui laisse chaque position entre 0.6 et 1.2
# ecart-type de la monnaie: six positions reellement vivantes plutot que trois
# positions et trois zeros.
#
# ``matched_strikes`` est la disposition de la premiere campagne - call et put
# au meme strike, maturite unique de 1 an - conservee pour que le rapport
# correspondant reste reproductible. C'est aussi la seule ou le livre contient
# des paires call/put appariees.

BOOK_LAYOUTS = {
    # (type, moneyness K/G_0, maturite en annees)
    "otm_term_ladder": (
        ("put", 0.95, 1.0 / 12.0),
        ("call", 1.05, 0.25),
        ("put", 0.90, 0.50),
        ("call", 1.10, 1.00),
        ("put", 0.80, 1.50),
        ("call", 1.20, 2.00),
    ),
    "matched_strikes": (
        ("call", 0.80, 1.0),
        ("put", 0.80, 1.0),
        ("call", 1.00, 1.0),
        ("put", 1.00, 1.0),
        ("call", 1.20, 1.0),
        ("put", 1.20, 1.0),
    ),
}
DEFAULT_BOOK_LAYOUT = "otm_term_ladder"
BOOK_QUANTITY = 1.0

HORIZON_DAYS = 1.0
TRADING_DAYS = 252.0
LEVELS = (0.95, 0.975, 0.99)
# Le niveau de l'Expected Shortfall reglementaire (FRTB, Bale III): l'ES a
# 97.5% remplace la VaR a 99% comme mesure de capital. Il est calibre pour etre
# du meme ordre qu'une VaR 99% sous loi normale, mais il lit toute la queue au
# lieu d'un seul quantile - ici la moyenne des 250 pires pertes sur 10 000.
ES_LEVEL = 0.975

PROFILES = {
    "smoke": {
        "scenarios": 2_000,
        "budget": 20_000,
        "anova": 1_000,
        "inner_paths": 1_024,
        "mc_sweep": (256, 1_024, 4_096),
        "qtt_sweep": (10_000, 20_000),
    },
    "full": {
        "scenarios": 10_000,
        "budget": 150_000,
        "anova": 2_000,
        "inner_paths": 4_096,
        "mc_sweep": (1_024, 4_096, 16_384),
        "qtt_sweep": (25_000, 50_000, 100_000, 150_000, 200_000),
    },
}


@dataclass(frozen=True)
class Market:
    """Etat de marche synthetique, en clair et en un seul endroit."""

    tickers: tuple[str, ...]
    spots: np.ndarray
    volatilities: np.ndarray
    correlation: np.ndarray
    dividends: np.ndarray
    rate: float

    @classmethod
    def synthetic(cls) -> Market:
        n = len(ASSETS)
        correlation = np.full((n, n), CORRELATION_RHO, dtype=float)
        np.fill_diagonal(correlation, 1.0)
        return cls(
            tickers=tuple(row[0] for row in ASSETS),
            spots=np.asarray([row[1] for row in ASSETS], dtype=float),
            volatilities=np.asarray([row[2] for row in ASSETS], dtype=float),
            correlation=correlation,
            dividends=np.asarray(DIVIDEND_YIELDS, dtype=float),
            rate=float(RISK_FREE_RATE),
        )

    @property
    def n_assets(self) -> int:
        return int(self.spots.size)

    @property
    def geometric_basket(self) -> float:
        return float(np.exp(np.mean(np.log(self.spots))))

    @property
    def basket_volatility(self) -> float:
        weights = np.full(self.n_assets, 1.0 / self.n_assets)
        scaled = self.volatilities * weights
        return float(np.sqrt(scaled @ self.correlation @ scaled))

    def describe(self) -> dict:
        return {
            "assets": [
                {
                    "ticker": ticker,
                    "spot": float(spot),
                    "implied_volatility": float(vol),
                    "dividend_yield": float(q),
                }
                for ticker, spot, vol, q in zip(
                    self.tickers, self.spots, self.volatilities, self.dividends
                )
            ],
            "correlation_model": f"equicorrelation rho={CORRELATION_RHO}",
            "correlation": self.correlation.tolist(),
            "risk_free_rate": self.rate,
            "geometric_basket_spot": self.geometric_basket,
            "geometric_basket_volatility": self.basket_volatility,
        }




def _maturity_tag(maturity: float) -> str:
    """Etiquette lisible d'une maturite: 1m, 3m, 6m, 1y, 18m, 2y."""
    months = maturity * 12.0
    if abs(months - round(months)) < 1e-9 and months < 12.0:
        return f"{round(months)}m"
    if abs(maturity - round(maturity)) < 1e-9:
        return f"{round(maturity)}y"
    return f"{months:.0f}m"


@dataclass(frozen=True)
class Book:
    """Six positions longues, decrites par strike, maturite, type et quantite."""

    portfolio: Portfolio
    option_types: tuple[str, ...]
    labels: tuple[str, ...]
    moneyness: tuple[float, ...]
    layout: str

    @classmethod
    def build(cls, basket_spot: float, layout: str = DEFAULT_BOOK_LAYOUT) -> Book:
        if layout not in BOOK_LAYOUTS:
            raise ValueError(f"unknown book layout: {layout}")
        positions = BOOK_LAYOUTS[layout]
        strikes = [basket_spot * ratio for _, ratio, _ in positions]
        maturities = [maturity for _, _, maturity in positions]
        types = tuple(kind for kind, _, _ in positions)
        ratios = tuple(float(ratio) for _, ratio, _ in positions)
        labels = tuple(
            f"{kind}_{round(100 * ratio)}_{_maturity_tag(maturity)}"
            for kind, ratio, maturity in positions
        )
        portfolio = Portfolio.from_arrays(
            strikes=strikes,
            maturities=maturities,
            quantities=BOOK_QUANTITY,
            labels=labels,
        )
        return cls(portfolio, types, labels, ratios, layout)

    def __len__(self) -> int:
        return len(self.portfolio)

    @property
    def is_call(self) -> np.ndarray:
        return np.asarray([kind == "call" for kind in self.option_types], dtype=bool)

    @property
    def maturities(self) -> np.ndarray:
        return self.portfolio.maturities

    def standardized_moneyness(self, basket_volatility: float) -> np.ndarray:
        """``log(m) / (sigma_B sqrt(T))``: a combien d'ecarts-types du strike.

        C'est la mesure qui dit si une position hors de la monnaie est encore
        vivante. La moneyness seule ne le dit pas: 80% est loin a un mois et
        proche a deux ans.
        """
        return np.log(np.asarray(self.moneyness)) / (
            float(basket_volatility) * np.sqrt(self.maturities)
        )

    def describe(self, basket_volatility: float | None = None) -> list[dict]:
        rows = []
        standardized = (
            None
            if basket_volatility is None
            else self.standardized_moneyness(basket_volatility)
        )
        for index, (trade, kind, ratio) in enumerate(
            zip(self.portfolio.trades, self.option_types, self.moneyness)
        ):
            row = {
                **trade.describe(),
                "option_type": kind,
                "direction": "long",
                "moneyness": float(ratio),
                "in_the_money": bool(
                    (kind == "call" and ratio < 1.0)
                    or (kind == "put" and ratio > 1.0)
                ),
            }
            if standardized is not None:
                row["standardized_moneyness"] = float(standardized[index])
            rows.append(row)
        return rows


# ==========================================================================
# 3. Les trois revalorisations
# ==========================================================================


def price_book_closed_form(market: Market, parameters: np.ndarray, is_call: np.ndarray):
    """Forme fermee: put et call panier geometrique, chacun par sa propre formule.

    ``parameters`` est le tenseur ``(n_etats, n_trades, n_assets + 3)`` produit
    par ``Portfolio.market_parameters``. La reference ne passe pas par la
    parite, pour que la parite reste une propriete verifiable du bras QTT et
    non une definition partagee par les deux bras.
    """
    n = market.n_assets
    flat = parameters.reshape(-1, parameters.shape[-1])
    calls = np.broadcast_to(is_call, parameters.shape[:2]).reshape(-1)
    values = np.empty(flat.shape[0], dtype=float)
    for mask, pricer in ((~calls, geometric_basket_put), (calls, geometric_basket_call)):
        if not np.any(mask):
            continue
        rows = flat[mask]
        values[mask] = pricer(
            rows[:, :n], rows[:, n], rows[:, n + 1], rows[:, n + 2],
            market.volatilities, market.correlation, market.dividends,
        )
    return values.reshape(parameters.shape[:2])


class NestedMonteCarloBook:
    """Monte Carlo imbrique a nombres aleatoires communs pour le livre entier.

    Un unique jeu de normales correlees est tire a la construction et reutilise
    pour chaque scenario et chaque position. Ce contrat CRN n'est pas un detail
    d'implementation: sans lui l'erreur de revalorisation est un bruit
    independant d'un scenario a l'autre, elle ne se retranche pas dans le PnL et
    la VaR obtenue mesure surtout la variance du pricer.

    Deux moteurs sont exposes.  ``collapsed=False`` simule les cinq
    sous-jacents et recompose le panier, ce que paie un moteur de panier
    generique qui ignore la forme du payoff.  ``collapsed=True`` exploite le
    fait que le facteur multiplicatif du panier geometrique est commun a tous
    les scenarios et le precalcule: c'est exact, mais c'est une optimisation qui
    n'existe que pour ce payoff, donc elle est reportee comme borne inferieure
    de cout et non comme le cout du Monte Carlo.
    """

    def __init__(
        self,
        market: Market,
        n_paths: int,
        seed: int,
        antithetic: bool = True,
        batch_elements: int = 8_000_000,
    ):
        self.market = market
        self.n_paths = int(n_paths)
        self.antithetic = bool(antithetic)
        if self.n_paths < 2:
            raise ValueError("n_paths must be at least two")
        # The generic engine materializes (states, paths, assets); the batch is
        # sized by element count rather than by state count so that raising the
        # path budget trades speed for time and never for memory.
        self.batch_states = max(
            1, int(batch_elements) // (self.n_paths * market.n_assets)
        )

        rng = np.random.default_rng(int(seed))
        if self.antithetic:
            half = (self.n_paths + 1) // 2
            base = rng.standard_normal((half, market.n_assets))
            normals = np.concatenate((base, -base), axis=0)[: self.n_paths]
        else:
            normals = rng.standard_normal((self.n_paths, market.n_assets))
        self._normals = normals @ np.linalg.cholesky(market.correlation).T

    def _growth_factors(self, rate: float, maturity: float) -> np.ndarray:
        """``(n_paths, n_assets)`` facteurs multiplicatifs ``S_T / S_0``.

        Sous GBM a parametres constants ils ne dependent pas du spot, donc ils
        sont communs a tous les scenarios de meme ``(r, T)``. Les recalculer par
        scenario serait du travail redondant, pas un Monte Carlo plus honnete.
        """
        sigma = self.market.volatilities
        drift = (rate - self.market.dividends - 0.5 * sigma**2) * maturity
        return np.exp(drift + sigma * np.sqrt(maturity) * self._normals)

    def __call__(
        self,
        parameters: np.ndarray,
        is_call: np.ndarray,
        collapsed: bool = False,
    ) -> np.ndarray:
        n = self.market.n_assets
        n_states, n_trades, _ = parameters.shape
        is_call = np.asarray(is_call, dtype=bool)
        values = np.empty((n_states, n_trades), dtype=float)

        # Le panier terminal ne depend que de l'etat et de (r, T), pas du
        # strike: on simule une fois par groupe (r, T) et on valorise toutes les
        # positions du groupe sur les memes trajectoires.
        keys = parameters[0, :, n + 1 :]
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        weights = np.full(n, 1.0 / n)

        for group, (rate, maturity) in enumerate(unique):
            columns = np.flatnonzero(inverse == group)
            strikes = parameters[0, columns, n]
            calls = is_call[columns]
            discount = float(np.exp(-rate * maturity))
            factors = self._growth_factors(float(rate), float(maturity))
            basket_factor = np.exp(np.log(factors) @ weights) if collapsed else None

            for start in range(0, n_states, self.batch_states):
                spots = parameters[start : start + self.batch_states, columns[0], :n]
                if collapsed:
                    basket_spot = np.exp(np.log(spots) @ weights)
                    basket = basket_spot[:, None] * basket_factor[None, :]
                else:
                    terminal = spots[:, None, :] * factors[None, :, :]
                    basket = np.exp(np.log(terminal) @ weights)
                for column, strike, call in zip(columns, strikes, calls):
                    payoff = (
                        np.maximum(basket - strike, 0.0)
                        if call
                        else np.maximum(strike - basket, 0.0)
                    )
                    values[start : start + len(spots), column] = discount * payoff.mean(
                        axis=1
                    )
        return values

    def standard_error(self, parameters: np.ndarray, is_call: np.ndarray) -> np.ndarray:
        """Erreur type Monte Carlo par position, au point de marche fourni."""
        n = self.market.n_assets
        weights = np.full(n, 1.0 / n)
        is_call = np.asarray(is_call, dtype=bool)
        errors = np.empty(parameters.shape[1], dtype=float)
        for column in range(parameters.shape[1]):
            row = parameters[0, column]
            rate, maturity = float(row[n + 1]), float(row[n + 2])
            factors = self._growth_factors(rate, maturity)
            basket = np.exp(np.log(row[:n] * factors) @ weights)
            strike = float(row[n])
            payoff = (
                np.maximum(basket - strike, 0.0)
                if is_call[column]
                else np.maximum(strike - basket, 0.0)
            )
            payoff = payoff * np.exp(-rate * maturity)
            errors[column] = payoff.std(ddof=1) / np.sqrt(payoff.size)
        return errors


def to_extended_market(market: Market, parameters: np.ndarray, config) -> np.ndarray:
    """Lift ``(S_1..S_5, K, r, T)`` rows to the 13-D market vector.

    The volatilities are constants of this experiment, not risk factors, so the
    lift is an insertion of fixed columns at the positions the layout assigns
    them. Doing it here rather than inside the pricer keeps the scenario tensor
    in the package-wide 8-D market convention, which every other arm and every
    diagnostic already speaks.
    """
    flat = np.atleast_2d(np.asarray(parameters, dtype=float))
    n = market.n_assets
    out = np.empty((flat.shape[0], config.n_dimensions), dtype=float)
    out[:, list(config.spot_columns)] = flat[:, :n]
    out[:, list(config.volatility_columns)] = market.volatilities[None, :]
    out[:, config.strike_column] = flat[:, n]
    out[:, config.rate_column] = flat[:, n + 1]
    out[:, config.maturity_column] = flat[:, n + 2]
    return out


class QTTBookPricer:
    """Surrogate QTT 13-D calibre sur les puts, etendu aux calls par parite.

    Le surrogate ne voit jamais un call. Le call est reconstruit par
    ``C = P_hat + e^{-rT}(F - K)``, ou l'offset est la forme fermee exacte du
    forward du panier geometrique. Cet offset ne coute aucun travail de pricing
    et n'introduit aucune erreur, donc l'erreur sur un call est exactement
    l'erreur du surrogate sur le put de meme strike et meme maturite.

    L'entree reste le tenseur de marche 8-D du portefeuille; le passage en
    13-D est fait ici, et il est compte dans le temps de revalorisation parce
    que c'est un cout que cette grille impose a chaque requete.
    """

    def __init__(self, market: Market, surrogate: VolExtendedPriceSurrogate):
        self.market = market
        self.surrogate = surrogate
        self.config = surrogate.config

    def __call__(self, parameters: np.ndarray, is_call: np.ndarray) -> np.ndarray:
        n = self.market.n_assets
        flat = parameters.reshape(-1, parameters.shape[-1])
        values = np.asarray(
            self.surrogate.price(to_extended_market(self.market, flat, self.config)),
            dtype=float,
        ).reshape(-1)
        calls = np.broadcast_to(is_call, parameters.shape[:2]).reshape(-1)
        if np.any(calls):
            rows = flat[calls]
            values[calls] = values[calls] + geometric_basket_parity_offset(
                rows[:, :n], rows[:, n], rows[:, n + 1], rows[:, n + 2],
                self.market.volatilities, self.market.correlation, self.market.dividends,
            )
        return values.reshape(parameters.shape[:2])


# ==========================================================================
# 4. Agregation du risque
# ==========================================================================


def timed(function, repeats: int = 1):
    """Run ``function`` ``repeats`` times and return its result and timing spread.

    A single wall-clock reading on a laptop is not a cost measurement: the same
    TT-cross fit was observed at 3.6 s and at 10.1 s on this machine depending
    on what else the CPU was doing. The median over repeats is reported as the
    cost, and the min/max are kept alongside it so that a reader can see how
    much of a quoted ratio is real and how much is scheduling noise.
    """
    repeats = max(1, int(repeats))
    times = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        times.append(perf_counter() - start)
    return result, {
        "median": float(np.median(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "repeats": repeats,
        "samples": [float(value) for value in times],
    }


@dataclass
class Arm:
    """Un bras de revalorisation: ses valeurs, ses temps, ses diagnostics."""

    name: str
    base_trade_values: np.ndarray
    scenario_trade_values: np.ndarray
    quantities: np.ndarray
    initialisation_time: float
    revaluation_time: float
    pricer_evaluations: int
    metadata: dict = field(default_factory=dict)
    initialisation_timing: dict = field(default_factory=dict)
    revaluation_timing: dict = field(default_factory=dict)

    @property
    def base_value(self) -> float:
        return float(self.base_trade_values @ self.quantities)

    @property
    def scenario_values(self) -> np.ndarray:
        return self.scenario_trade_values @ self.quantities

    @property
    def pnl(self) -> np.ndarray:
        return self.scenario_values - self.base_value

    @property
    def losses(self) -> np.ndarray:
        return -self.pnl

    @property
    def total_time(self) -> float:
        return float(self.initialisation_time + self.revaluation_time)

    @property
    def seconds_per_revaluation(self) -> float:
        return float(self.revaluation_time / max(self.pricer_evaluations, 1))

    def risk_summary(self, levels=LEVELS) -> dict:
        pnl = self.pnl
        return {
            "base_value": self.base_value,
            "pnl": {
                "mean": float(np.mean(pnl)),
                "std": float(np.std(pnl, ddof=1)),
                "min": float(np.min(pnl)),
                "max": float(np.max(pnl)),
            },
            "measures": {
                f"{level:g}": var_es_uncertainty(self.losses, level)
                for level in levels
            },
        }

    def cost_summary(self) -> dict:
        return {
            "initialisation_time": float(self.initialisation_time),
            "revaluation_time": float(self.revaluation_time),
            "total_time": self.total_time,
            "pricer_evaluations": int(self.pricer_evaluations),
            "seconds_per_revaluation": self.seconds_per_revaluation,
            "initialisation_timing": dict(self.initialisation_timing),
            "revaluation_timing": dict(self.revaluation_timing),
        }


def cancellation_report(reference: Arm, approximate: Arm) -> dict:
    """Erreur au niveau du prix contre erreur au niveau du PnL.

    ``ratio`` est la fraction de l'erreur de prix qui survit au passage au PnL.
    Proche de 0, l'erreur de revalorisation est un biais lisse qui se retranche
    entre le scenario et le point de base et la VaR est bien meilleure que le
    prix; proche de 1 ou au-dela, elle ne se retranche pas et la VaR herite de
    toute l'erreur de prix.
    """
    price_error = approximate.scenario_values - reference.scenario_values
    base_error = approximate.base_value - reference.base_value
    pnl_error = price_error - base_error
    mean_price_error = float(np.mean(np.abs(price_error)))
    mean_pnl_error = float(np.mean(np.abs(pnl_error)))
    return {
        "base_price_error": float(base_error),
        "scenario_price_mean_absolute_error": mean_price_error,
        "scenario_price_mean_bias": float(np.mean(price_error)),
        "scenario_price_max_absolute_error": float(np.max(np.abs(price_error))),
        "pnl_mean_absolute_error": mean_pnl_error,
        "pnl_max_absolute_error": float(np.max(np.abs(pnl_error))),
        "cancellation_ratio": (
            mean_pnl_error / mean_price_error if mean_price_error > 0.0 else None
        ),
    }


def break_even(reference: Arm, candidate: Arm) -> dict:
    """Revalorisations au-dela desquelles le candidat amortit son initialisation."""
    saving = reference.seconds_per_revaluation - candidate.seconds_per_revaluation
    extra_setup = candidate.initialisation_time - reference.initialisation_time
    return {
        "versus": reference.name,
        "speedup_per_revaluation": (
            reference.seconds_per_revaluation / candidate.seconds_per_revaluation
            if candidate.seconds_per_revaluation > 0.0
            else None
        ),
        "extra_initialisation_time": float(extra_setup),
        "break_even_revaluations": (
            float(extra_setup / saving) if saving > 0.0 else None
        ),
        "total_time_ratio": (
            candidate.total_time / reference.total_time
            if reference.total_time > 0.0
            else None
        ),
    }


def _rms_over_seeds(per_seed, metric: str) -> dict:
    """Moyenne quadratique d'une erreur relative sur les graines, par niveau.

    L'erreur est une magnitude: une moyenne signee sur les graines se
    compenserait et sous-estimerait l'erreur d'un run quelconque.
    """
    return {
        f"{level:g}": float(
            np.sqrt(np.mean([entry[metric][f"{level:g}"] ** 2 for entry in per_seed]))
        )
        for level in LEVELS
    }


def _range_over_seeds(per_seed, metric: str) -> dict:
    return {
        f"{level:g}": [
            float(min(entry[metric][f"{level:g}"] for entry in per_seed)),
            float(max(entry[metric][f"{level:g}"] for entry in per_seed)),
        ]
        for level in LEVELS
    }


def _seed_errors(accuracy: dict) -> dict:
    """Erreurs relatives de VaR et d'ES d'un run, a chaque niveau."""
    return {
        "var_relative_error": {
            f"{level:g}": accuracy["measures"][f"{level:g}"]["var_relative_error"]
            for level in LEVELS
        },
        "es_relative_error": {
            f"{level:g}": accuracy["measures"][f"{level:g}"][
                "expected_shortfall_relative_error"
            ]
            for level in LEVELS
        },
    }


def monte_carlo_path_sweep(
    market: Market,
    book: Book,
    base_parameters: np.ndarray,
    scenario_parameters: np.ndarray,
    reference: Arm,
    path_counts,
    seeds,
) -> list[dict]:
    """VaR error and total cost of the nested Monte Carlo against its path budget.

    Comparing one Monte Carlo configuration against one surrogate answers the
    wrong question: the two are not at the same accuracy, so whichever is
    cheaper is cheaper at the accuracy it happens to have. This sweep gives the
    Monte Carlo accuracy/cost curve, which is what a QTT total time has to be
    read against - the fair statement is the path budget at which Monte Carlo
    *matches* the surrogate, and what that budget costs.

    Each path budget is run under several independent seeds. Under common
    random numbers the VaR error at a fixed budget is a single draw of a random
    quantity, not a converged number: a first pass of this sweep returned 0.54%
    at 16 384 paths and 1.17% at 65 536, which reads as divergence and is only
    seed noise. Averaging over seeds is what makes the ``1/sqrt(n)`` law
    visible and the extrapolation below legitimate.
    """
    is_call = book.is_call
    quantities = book.portfolio.quantities
    n_evaluations = int(
        base_parameters.shape[0] * base_parameters.shape[1]
        + scenario_parameters.shape[0] * scenario_parameters.shape[1]
    )
    seeds = tuple(int(value) for value in seeds)
    rows = []
    for n_paths in path_counts:
        per_seed, totals, price_errors, pnl_errors = [], [], [], []
        inits, revals = [], []
        for seed in seeds:
            start = perf_counter()
            engine = NestedMonteCarloBook(market, int(n_paths), seed=seed)
            init_time = perf_counter() - start
            start = perf_counter()
            base = engine(base_parameters, is_call)
            scenarios = engine(scenario_parameters, is_call)
            reval_time = perf_counter() - start
            arm = Arm(
                name=f"monte_carlo_{n_paths}_{seed}",
                base_trade_values=base[0],
                scenario_trade_values=scenarios,
                quantities=quantities,
                initialisation_time=init_time,
                revaluation_time=reval_time,
                pricer_evaluations=n_evaluations,
            )
            accuracy = compare_loss_distributions(
                reference.losses, arm.losses, levels=LEVELS
            )
            per_seed.append({
                "seed": seed,
                **_seed_errors(accuracy),
                "var_absolute_error": {
                    f"{level:g}": accuracy["measures"][f"{level:g}"][
                        "var_absolute_error"
                    ]
                    for level in LEVELS
                },
            })
            totals.append(arm.total_time)
            inits.append(init_time)
            revals.append(reval_time)
            pnl_errors.append(accuracy["loss_mae"])
            price_errors.append(
                float(np.mean(np.abs(arm.scenario_values - reference.scenario_values)))
            )

        mean_relative = _rms_over_seeds(per_seed, "var_relative_error")
        es_relative = _rms_over_seeds(per_seed, "es_relative_error")
        rows.append({
            "inner_paths": int(n_paths),
            "seeds": list(seeds),
            # Init et cout quotidien sont separes parce que le surrogate QTT
            # s'initialise une fois et se reutilise chaque jour: confondre les
            # deux dans un "total" repond a la question du premier jour, pas a
            # celle d'un run de VaR quotidien.
            "initialisation_time": float(np.median(inits)),
            "revaluation_time": float(np.median(revals)),
            "total_time": float(np.median(totals)),
            "total_time_min": float(np.min(totals)),
            "total_time_max": float(np.max(totals)),
            "var_relative_error": mean_relative,
            "var_relative_error_range": _range_over_seeds(
                per_seed, "var_relative_error"
            ),
            "es_relative_error": es_relative,
            "es_relative_error_range": _range_over_seeds(
                per_seed, "es_relative_error"
            ),
            "var_relative_error_by_seed": per_seed,
            "pnl_mean_absolute_error": float(np.mean(pnl_errors)),
            "price_mean_absolute_error": float(np.mean(price_errors)),
        })
        print(f"[mc sweep] paths={n_paths:>7} total {rows[-1]['total_time']:>7.2f}s "
              f"VaR95 err {mean_relative['0.95']:.2%} "
              f"VaR99 err {mean_relative['0.99']:.2%} "
              f"ES97.5 err {es_relative['0.975']:.2%} "
              f"(rms over {len(seeds)} seeds)")
    return rows


def qtt_budget_sweep(
    market: Market,
    book: Book,
    base_parameters: np.ndarray,
    scenario_parameters: np.ndarray,
    reference: Arm,
    build_surrogate,
    budgets,
    anova_samples: int,
    truncation,
    seeds,
) -> list[dict]:
    """QTT accuracy and total cost against the TT-cross evaluation budget.

    Measured inside the same process as the Monte Carlo sweep, because the wall
    clock on this machine drifted by a factor of 2.6 between two runs doing
    byte-identical work. Two curves that are to be read against each other have
    to be measured under the same machine state, or the comparison is a
    comparison of scheduling.

    Each budget is fitted under several independent TT seeds, for exactly the
    reason the Monte Carlo sweep uses several seeds. ``teneva.cross`` is called
    without a tolerance, so it stops wherever the requested budget runs out,
    at an arbitrary point of a sweep; the accuracy at a fixed budget is
    therefore a draw, not a converged number. A first single-seed pass on this
    13-D risk grid returned 4.67%, 12.95%, 5.17%, 0.06% and 0.44% at 25k, 50k,
    100k, 150k and 200k - two orders of magnitude of swing with a smoothly
    growing rank. Reporting any one of those as "the" accuracy would be
    reporting a coin flip.
    """
    is_call = book.is_call
    quantities = book.portfolio.quantities
    n_evaluations = int(
        base_parameters.shape[0] * base_parameters.shape[1]
        + scenario_parameters.shape[0] * scenario_parameters.shape[1]
    )

    seeds = tuple(int(value) for value in seeds)
    rows = []
    for budget in budgets:
        per_seed, totals, inits, revals = [], [], [], []
        ranks, price_errors, pnl_errors = [], [], []
        for seed in seeds:
            # La grille est reconstruite avec le surrogate: a 13-D sa
            # construction fait partie du cout d'initialisation, elle n'est
            # pas negligeable comme l'etait celle de la grille de prix 8-D.
            surrogate = build_surrogate(seed)

            def fit_at(surrogate=surrogate, budget=budget):
                return surrogate.fit(
                    int(budget),
                    anova_samples=anova_samples,
                    truncation=truncation,
                    log=False,
                )

            diagnostics, fit_timing = timed(fit_at, 1)
            pricer = QTTBookPricer(market, surrogate)

            def revalue(pricer=pricer):
                return (
                    pricer(base_parameters, is_call),
                    pricer(scenario_parameters, is_call),
                )

            (base, scenarios), reval_timing = timed(revalue, 1)
            arm = Arm(
                name=f"qtt_{budget}_{seed}",
                base_trade_values=base[0],
                scenario_trade_values=scenarios,
                quantities=quantities,
                initialisation_time=fit_timing["median"],
                revaluation_time=reval_timing["median"],
                pricer_evaluations=n_evaluations,
            )
            accuracy = compare_loss_distributions(
                reference.losses, arm.losses, levels=LEVELS
            )
            per_seed.append({
                "seed": seed,
                "effective_rank": float(diagnostics.effective_rank),
                "parameter_count": int(diagnostics.parameter_count),
                **_seed_errors(accuracy),
            })
            totals.append(arm.total_time)
            inits.append(fit_timing["median"])
            revals.append(reval_timing["median"])
            ranks.append(float(diagnostics.effective_rank))
            pnl_errors.append(accuracy["loss_mae"])
            price_errors.append(
                float(np.mean(np.abs(arm.scenario_values - reference.scenario_values)))
            )

        mean_relative = _rms_over_seeds(per_seed, "var_relative_error")
        spread = _range_over_seeds(per_seed, "var_relative_error")
        es_relative = _rms_over_seeds(per_seed, "es_relative_error")
        rows.append({
            "budget": int(budget),
            "seeds": list(seeds),
            "effective_rank": float(np.mean(ranks)),
            "initialisation_time": float(np.median(inits)),
            "revaluation_time": float(np.median(revals)),
            "total_time": float(np.median(totals)),
            "var_relative_error": mean_relative,
            "var_relative_error_range": spread,
            "es_relative_error": es_relative,
            "es_relative_error_range": _range_over_seeds(
                per_seed, "es_relative_error"
            ),
            "var_relative_error_by_seed": per_seed,
            "pnl_mean_absolute_error": float(np.mean(pnl_errors)),
            "price_mean_absolute_error": float(np.mean(price_errors)),
        })
        print(f"[qtt sweep] budget={budget:>7} total {rows[-1]['total_time']:>7.2f}s "
              f"(init {rows[-1]['initialisation_time']:.2f}s) "
              f"erank {rows[-1]['effective_rank']:.2f} "
              f"VaR95 err {mean_relative['0.95']:.2%} "
              f"[{spread['0.95'][0]:.2%}-{spread['0.95'][1]:.2%}] "
              f"VaR99 err {mean_relative['0.99']:.2%} "
              f"ES97.5 err {es_relative['0.975']:.2%} "
              f"(rms over {len(seeds)} seeds)")
    return rows


AMORTISATION_HORIZONS = (1, 5, 21, 63, 252)


METRIC_LABELS = {
    "var_relative_error": "VaR",
    "es_relative_error": "ES",
}


def amortisation_table(
    qtt_rows,
    mc_rows,
    level: float = 0.95,
    metric: str = "var_relative_error",
) -> dict:
    """Cout total apres ``N`` runs de VaR quotidiens, par configuration.

    Le surrogate QTT s'initialise une fois - grille, ANOVA, TT-cross - et se
    reutilise tel quel les jours suivants: seuls les spots bougent, et la
    grille les couvre deja. Le Monte Carlo imbrique ne paie rien d'avance et
    tout a chaque run. Un total unique melange donc deux regimes: il repond a
    la question du premier jour et a aucune autre.

        cout(N) = initialisation + N x cout_quotidien

    Le point de croisement entre deux configurations est le nombre de runs au
    dela duquel la plus chere a initialiser devient la moins chere au total.
    Il n'a de sens qu'entre configurations de precision comparable, ce que la
    colonne d'erreur laisse verifier.
    """
    key = f"{level:g}"

    def describe(rows, label_key, prefix):
        out = []
        for row in rows:
            out.append({
                "configuration": f"{prefix} {row[label_key]}",
                label_key: row[label_key],
                "var_relative_error": {
                    f"{lv:g}": row["var_relative_error"][f"{lv:g}"]
                    for lv in LEVELS
                },
                "es_relative_error": {
                    f"{lv:g}": row["es_relative_error"][f"{lv:g}"]
                    for lv in LEVELS
                },
                "initialisation_time": float(row["initialisation_time"]),
                "daily_var_time": float(row["revaluation_time"]),
                "cost_after_runs": {
                    str(n): float(
                        row["initialisation_time"] + n * row["revaluation_time"]
                    )
                    for n in AMORTISATION_HORIZONS
                },
            })
        return out

    qtt = describe(qtt_rows, "budget", "QTT")
    monte_carlo = describe(mc_rows, "inner_paths", "MC")

    # Croisements: pour chaque paire, le nombre de runs ou les deux totaux
    # s'egalisent. Negatif ou infini quand une configuration domine partout.
    crossovers = []
    for a in qtt:
        for b in monte_carlo:
            saving = b["daily_var_time"] - a["daily_var_time"]
            extra = a["initialisation_time"] - b["initialisation_time"]
            if saving > 0.0:
                runs = extra / saving
                verdict = (
                    "QTT moins cher des le premier run"
                    if runs <= 1.0
                    else f"QTT moins cher a partir du run {int(np.ceil(runs))}"
                )
            else:
                runs = None
                verdict = "QTT jamais moins cher: cout quotidien superieur"
            qtt_error = a[metric][key]
            mc_error = b[metric][key]
            ratio = mc_error / qtt_error if qtt_error > 0.0 else None
            # Un QTT moins cher mais moins precis n'a rien gagne: le verdict le
            # dit, pour que la derniere colonne ne puisse pas se lire seule.
            if ratio is not None and 1.0 / ratio >= 1.1 and runs is not None:
                verdict += f" (mais {1.0 / ratio:.1f}x moins precis)"
            crossovers.append({
                "qtt": a["configuration"],
                "monte_carlo": b["configuration"],
                "qtt_relative_error": qtt_error,
                "monte_carlo_relative_error": mc_error,
                "accuracy_ratio": ratio,
                "break_even_runs": runs,
                "verdict": verdict,
            })

    return {
        "level": float(level),
        "metric": metric,
        "horizons": list(AMORTISATION_HORIZONS),
        "qtt": qtt,
        "monte_carlo": monte_carlo,
        "crossovers": crossovers,
        "note": (
            "cost(N) = initialisation + N * daily_var_time; the QTT "
            "initialisation is paid once and reused on every later run"
        ),
    }


def iso_accuracy_crossover(
    sweep_rows,
    target_error: float,
    level: float,
    metric: str = "var_relative_error",
) -> dict:
    """Path budget and cost at which Monte Carlo reaches a target VaR error.

    The VaR error of a Monte Carlo estimator falls like the square root of the
    path count, so the crossover is read on a log-log line through the sweep
    rather than by interpolating between two arbitrary neighbours. It is an
    extrapolation whenever the target sits below every measured point, which
    the output states rather than hides.
    """
    key = f"{level:g}"
    paths = np.asarray([row["inner_paths"] for row in sweep_rows], dtype=float)
    errors = np.asarray([row[metric][key] for row in sweep_rows], dtype=float)
    times = np.asarray([row["total_time"] for row in sweep_rows], dtype=float)
    usable = np.isfinite(errors) & (errors > 0.0)
    if usable.sum() < 2 or not np.isfinite(target_error) or target_error <= 0.0:
        return {"level": level, "target_relative_error": target_error, "paths": None}

    slope, intercept = np.polyfit(
        np.log(paths[usable]), np.log(errors[usable]), deg=1
    )
    required_paths = float(np.exp((np.log(target_error) - intercept) / slope))
    # Cost is linear in the path count once the book and the scenarios are
    # fixed, so the time is scaled from the most expensive measured point.
    heaviest = int(np.argmax(paths))
    required_time = float(
        times[heaviest] * required_paths / paths[heaviest]
    )
    return {
        "level": level,
        "metric": metric,
        "target_relative_error": float(target_error),
        "fitted_error_exponent": float(slope),
        "paths": required_paths,
        "estimated_total_time": required_time,
        "extrapolated": bool(
            required_paths > paths.max() or required_paths < paths.min()
        ),
    }


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.2f} %"


def _int(value) -> str:
    """Separateur de milliers a la francaise, sans toucher a la ponctuation."""
    return f"{int(value):,}".replace(",", " ")


def _crossover_rows(amort, add, low=0.5, high=2.0):
    """Croisements entre configurations de precision voisine uniquement.

    Hors de ``[low, high]`` le rapport de precision est trop grand pour qu'un
    croisement de cout dise quoi que ce soit: c'est le piege de lecture d'une
    premiere version de ce tableau, ou une ligne « QTT moins cher des le 1er
    run » pouvait cacher un QTT deux fois moins precis.
    """
    kept = [
        row for row in amort["crossovers"]
        if row["accuracy_ratio"] is not None
        and low <= row["accuracy_ratio"] <= high
    ]
    if not kept:
        add(f"_Aucune paire de precision comparable (rapport entre {low:g} et "
            f"{high:g})._")
        return
    add("| QTT | Monte Carlo | Err. QTT | Err. MC | Rapport MC/QTT | Croisement |")
    add("|---|---|---:|---:|---:|---|")
    for row in kept:
        runs = row["break_even_runs"]
        if runs is None:
            when = "jamais : cout quotidien QTT plus eleve"
        elif runs <= 1.0:
            when = "des le 1er run"
        else:
            when = f"a partir du run {int(np.ceil(runs))}"
        # Sous 1.1x l'ecart de precision est dans le bruit inter-graines:
        # « 1.0x moins precis » ne dirait rien.
        if 1.0 / row["accuracy_ratio"] >= 1.1:
            when += f" (QTT {1.0 / row['accuracy_ratio']:.1f}x moins precis)"
        add(f"| {row['qtt']} | {row['monte_carlo']} | "
            f"{_pct(row['qtt_relative_error'])} | "
            f"{_pct(row['monte_carlo_relative_error'])} | "
            f"{row['accuracy_ratio']:.1f}x | {when} |")


def benchmark_markdown(results: dict) -> str:
    """Le benchmark en tableaux markdown, prets a etre inseres dans un rapport.

    Ecrit a cote du JSON. Le JSON porte tout, y compris ce que personne ne
    lira; ce fichier-ci porte ce qui se cite, dans l'ordre ou cela se cite, et
    avec les reserves attachees aux chiffres plutot que releguees ailleurs.
    """
    amort = results.get("amortisation")
    amort_es = results.get("amortisation_es")
    if not amort:
        return ""
    book = results["book"]
    setup = results["risk_setup"]
    grid = results["qtt_grid"]
    arms = results["arms"]
    comparison = results["comparison"]
    value = book["base_value_closed_form"]
    es_key = f"{ES_LEVEL:g}"

    lines = []
    add = lines.append

    add("# Benchmark VaR / ES 1 jour - QTT 13-D contre Monte Carlo imbrique")
    add("")
    add(f"Livre de {len(book['positions'])} options sur panier geometrique, "
        f"valeur {value:.4f}. "
        f"{_int(setup['scenarios'])} scenarios GBM correles, horizon "
        f"{setup['horizon_days']:g} jour. "
        f"Grille QTT {grid['dimensions']}-D `{grid['mode']}`, layout "
        f"`{grid['layout']}`, {grid['qtt_cores']} coeurs.")
    add("")
    add("Toutes les erreurs des sections 2 a 6 sont des moyennes quadratiques "
        "sur 3 graines independantes : a budget fixe, ni le TT-cross ni le "
        "Monte Carlo ne produisent un nombre reproductible.")
    add("")

    # ---- 1. mesures de risque des bras principaux
    add("## 1. Mesures de risque - bras principaux")
    add("")
    qtt_budget = arms["qtt"]["metadata"]["budget"]
    mc_paths = arms["monte_carlo"]["metadata"]["inner_paths"]
    add(f"QTT au budget {_int(qtt_budget)}, Monte Carlo a {_int(mc_paths)} "
        "chemins internes, une graine chacun.")
    add("")
    add("| Bras | VaR 95 % | VaR 99 % | **ES 97,5 %** | Err. VaR 95 % | "
        "Err. VaR 99 % | **Err. ES 97,5 %** |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    labels = {
        "closed_form": "Forme fermee (reference)",
        "monte_carlo": "Monte Carlo imbrique",
        "qtt": "QTT 13-D",
    }
    for name in ("closed_form", "monte_carlo", "qtt"):
        measures = arms[name]["risk"]["measures"]
        if name == "closed_form":
            errors = ("-", "-", "-")
        else:
            cmp = comparison[name]["loss_distribution"]["measures"]
            signed = []
            for key, field, ref_field in (
                ("0.95", "surrogate_var", "reference_var"),
                ("0.99", "surrogate_var", "reference_var"),
                (es_key, "surrogate_expected_shortfall",
                 "reference_expected_shortfall"),
            ):
                ref = cmp[key][ref_field]
                signed.append(f"{(cmp[key][field] - ref) / ref * 100:+.2f} %")
            errors = tuple(signed)
        add(f"| {labels[name]} | {measures['0.95']['var']:.6f} | "
            f"{measures['0.99']['var']:.6f} | "
            f"**{measures[es_key]['expected_shortfall']:.6f}** | "
            f"{errors[0]} | {errors[1]} | {errors[2]} |")
    add("")
    ref_es = arms["closed_form"]["risk"]["measures"][es_key]
    lo, hi = ref_es["expected_shortfall_ci"]
    add(f"Intervalle de confiance a 95 % de l'ES 97,5 % de reference, du seul "
        f"fait de l'echantillonnage des {_int(setup['scenarios'])} scenarios : "
        f"`[{lo:.6f}, {hi:.6f}]` (moyenne de queue sur "
        f"{ref_es['tail_scenarios']} scenarios, erreur type asymptotique).")
    add("")
    verdict = []
    for name in ("monte_carlo", "qtt"):
        es = arms[name]["risk"]["measures"][es_key]["expected_shortfall"]
        inside = lo <= es <= hi
        verdict.append(f"{labels[name]} : {'dedans' if inside else 'dehors'}")
    add(" ; ".join(verdict) + ".")
    add("")

    # ---- 2. configurations QTT
    add("## 2. Configurations QTT")
    add("")
    add("| Budget TT-cross | Err. VaR 95 % | Err. VaR 99 % | **Err. ES 97,5 %** | "
        "Initialisation (une fois) | VaR / ES quotidienne |")
    add("|---|---:|---:|---:|---:|---:|")
    for row in amort["qtt"]:
        add(f"| {_int(row['budget'])} | "
            f"{_pct(row['var_relative_error']['0.95'])} | "
            f"{_pct(row['var_relative_error']['0.99'])} | "
            f"**{_pct(row['es_relative_error'][es_key])}** | "
            f"{row['initialisation_time']:.1f} s | "
            f"{row['daily_var_time']:.1f} s |")
    add("")

    # ---- 3. configurations Monte Carlo
    add("## 3. Configurations Monte Carlo")
    add("")
    add("| Chemins internes | Err. VaR 95 % | Err. VaR 99 % | **Err. ES 97,5 %** | "
        "Initialisation (une fois) | VaR / ES quotidienne |")
    add("|---|---:|---:|---:|---:|---:|")
    for row in amort["monte_carlo"]:
        add(f"| {_int(row['inner_paths'])} | "
            f"{_pct(row['var_relative_error']['0.95'])} | "
            f"{_pct(row['var_relative_error']['0.99'])} | "
            f"**{_pct(row['es_relative_error'][es_key])}** | "
            f"{row['initialisation_time']:.2f} s | "
            f"{row['daily_var_time']:.1f} s |")
    add("")
    add("La VaR et l'ES sortent du meme jeu de pertes : le cout quotidien est "
        "celui de la revalorisation, l'ES n'y ajoute qu'un tri de "
        f"{_int(setup['scenarios'])} nombres.")
    add("")

    # ---- 4. amortissement
    add("## 4. Amortissement : cout total apres N runs")
    add("")
    add("`cout(N) = initialisation + N x cout quotidien`. Le surrogate QTT "
        "s'initialise une fois et se reutilise tel quel les jours suivants : "
        "seuls les spots bougent, et la grille les couvre deja.")
    add("")
    horizons = amort["horizons"]
    header = " | ".join(f"N = {n}" for n in horizons)
    add(f"| Configuration | Err. VaR 95 % | Err. ES 97,5 % | {header} |")
    add("|---|---:|---:|" + "---:|" * len(horizons))
    for row in amort["qtt"] + amort["monte_carlo"]:
        costs = " | ".join(
            f"{row['cost_after_runs'][str(n)]:.0f} s" for n in horizons
        )
        add(f"| {row['configuration']} | "
            f"{_pct(row['var_relative_error']['0.95'])} | "
            f"{_pct(row['es_relative_error'][es_key])} | {costs} |")
    add("")

    # ---- 5 et 6. croisements
    add("## 5. Croisements a precision comparable - ES 97,5 %")
    add("")
    add("Seules les paires dont le rapport de precision MC/QTT est compris "
        "entre 0,5 et 2 sont listees : au-dela, un croisement de cout ne dit "
        "rien. Rapport > 1 : le QTT est le plus precis des deux.")
    add("")
    if amort_es:
        _crossover_rows(amort_es, add)
    add("")
    es_iso = results.get("iso_accuracy_es_versus_qtt")
    if es_iso and es_iso.get("paths"):
        add(f"A precision egale sur l'ES 97,5 % "
            f"({_pct(es_iso['target_relative_error'])}, erreur QTT moyennee "
            f"sur graines au budget {_int(qtt_budget)}), le Monte Carlo "
            f"demanderait {_int(es_iso['paths'])} chemins"
            + (" (extrapole)" if es_iso["extrapolated"] else " (mesure)")
            + f", soit ~{es_iso['estimated_total_time']:.0f} s par run.")
        add("")

    add("## 6. Croisements a precision comparable - VaR 95 %")
    add("")
    _crossover_rows(amort, add)
    add("")

    # ---- reserves
    add("## Reserves")
    add("")
    add("1. **Les temps absolus portent une dispersion machine d'environ "
        "+-2.5x** pour un travail identique. Les rapports intra-run sont "
        "exploitables, les secondes prises isolement beaucoup moins.")
    add("2. **L'erreur de risque du QTT n'est pas monotone en budget** alors "
        "que son erreur de prix l'est : le budget ne peut pas etre regle sur "
        "l'erreur de prix.")
    add(f"3. **L'ES 97,5 % est une moyenne sur les {ref_es['tail_scenarios']} "
        "pires scenarios**, pas un quantile : il est moins sensible qu'une "
        "VaR a la position d'une seule perte, et son intervalle de confiance "
        "est asymptotique (erreur type de la moyenne de queue).")
    add("4. **L'initialisation QTT n'est amortissable que tant que les spots "
        "restent dans la boite de la grille.** Mesure ici : "
        f"{setup['grid_domain_coverage']['fraction_outside_any_axis']:.2%} de "
        "scenarios hors boite sur un horizon d'un jour.")
    add("5. **Le cout de mise au point du budget** - balayer plusieurs budgets "
        "sur plusieurs graines contre une reference exacte - n'est compte "
        "dans aucun tableau.")
    add("")
    return "\n".join(lines)


# ==========================================================================
# 5. Orchestration
# ==========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--mode",
        choices=VOL_RISK_GRID_MODES,
        default="bounded_standardized_risk",
        help=(
            "13-D risk grid family. 'standardized_risk' cannot be built at "
            "13-D (see CLAUDE.md gotcha 7); the bounded variant is the general "
            "answer"
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("interleaved", "blocked"),
        default="interleaved",
        help=(
            "QTT mode ordering. Measured on this grid at budget 150k: "
            "interleaved 0.06%% VaR error against blocked 0.60%%, at the same "
            "effective rank. CLAUDE.md gotcha 3 prefers blocked for a "
            "geometric basket, but that was established on the pricing "
            "grids; on the standardized-risk grid the strike coordinate is "
            "itself a function of sigma, so the sigma-strike coupling no "
            "longer sits where blocked puts it"
        ),
    )
    parser.add_argument(
        "--book",
        choices=tuple(BOOK_LAYOUTS),
        default=DEFAULT_BOOK_LAYOUT,
        help="Book layout; 'matched_strikes' reproduces the first campaign",
    )
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--inner-paths", type=int, default=None)
    parser.add_argument(
        "--spot-nodes",
        type=int,
        default=32,
        help=(
            "Nodes per spot axis, power of two. 32 over [5, 150] is a cell of "
            "about 4.7%% of a spot near 100, while a one-day move is about "
            "1.3%%: base point and scenario then share a cell and the P&L is a "
            "difference taken below the grid scale"
        ),
    )
    parser.add_argument("--vol-nodes", type=int, default=16)
    parser.add_argument(
        "--strike-nodes",
        type=int,
        default=64,
        help="Nodes on the standardized strike axis, power of two",
    )
    parser.add_argument(
        "--maturity-nodes",
        type=int,
        default=32,
        help=(
            "Nodes on the maturity axis, power of two. The standardized-risk "
            "grids need far more than the config default of 8: their strike "
            "coordinate is a function of (r, T, sigma), so a fixed coordinate "
            "names a different strike at every maturity and the surface moves "
            "along T much faster than on a moneyness grid (CLAUDE.md gotcha 6)"
        ),
    )
    parser.add_argument("--rate-nodes", type=int, default=8)
    parser.add_argument(
        "--qtt-budget-sweep",
        type=int,
        nargs="*",
        default=None,
        help=(
            "TT-cross budgets for the QTT accuracy/cost curve, measured in this "
            "same process as the Monte Carlo curve; empty disables the sweep"
        ),
    )
    parser.add_argument(
        "--mc-path-sweep",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Inner path budgets for the Monte Carlo accuracy/cost curve; "
            "empty disables the sweep"
        ),
    )
    parser.add_argument("--horizon-days", type=float, default=HORIZON_DAYS)
    parser.add_argument("--scenario-seed", type=int, default=20260921)
    parser.add_argument("--mc-seed", type=int, default=4242)
    parser.add_argument(
        "--mc-sweep-seeds",
        type=int,
        nargs="+",
        default=(4242, 911, 20260921, 77, 5150),
        help="Independent seeds per path budget in the Monte Carlo sweep",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
        help=(
            "Repetitions of every timed section; the median is reported as the "
            "cost and the min/max spread is kept in the JSON"
        ),
    )
    parser.add_argument("--tt-seed", type=int, default=20260327)
    parser.add_argument(
        "--tt-sweep-seeds",
        type=int,
        nargs="+",
        default=(20260327, 7, 991),
        help=(
            "Independent TT-cross seeds per budget. The accuracy at a fixed "
            "budget is a draw, not a converged number, so a single seed is a "
            "coin flip"
        ),
    )
    parser.add_argument(
        "--truncation",
        type=float,
        default=1e-8,
        help="Post-fit TT price-norm truncation; 0 disables it",
    )
    parser.add_argument(
        "--real-world-drift",
        type=float,
        default=None,
        help="Annualized expected log-return; default is the risk-neutral drift",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    n_scenarios = args.scenarios or profile["scenarios"]
    budget = args.budget or profile["budget"]
    inner_paths = args.inner_paths or profile["inner_paths"]
    tag = args.tag or f"{args.book}_{args.profile}_{args.mode}"
    horizon = args.horizon_days / TRADING_DAYS

    market = Market.synthetic()
    book = Book.build(market.geometric_basket, layout=args.book)
    is_call = book.is_call
    quantities = book.portfolio.quantities

    config = VolatilityExtendedConfig(
        n_assets=market.n_assets,
        layout=args.layout,
        dividends=market.dividends,
        correlation=market.correlation,
        spot_nodes=args.spot_nodes,
        volatility_nodes=args.vol_nodes,
        strike_nodes=args.strike_nodes,
        rate_nodes=args.rate_nodes,
        maturity_nodes=args.maturity_nodes,
        seed=args.tt_seed,
    )
    physical_shape = config.physical_shape
    if not (
        config.volatility_bounds[0]
        <= market.volatilities.min()
        <= market.volatilities.max()
        <= config.volatility_bounds[1]
    ):
        raise SystemExit(
            "the market volatilities fall outside the grid's volatility box"
        )

    print(f"[setup] book={args.book} profile={args.profile} mode={args.mode} "
          f"layout={args.layout} scenarios={n_scenarios} budget={budget} "
          f"inner_paths={inner_paths} horizon={horizon:.6f}y")
    print(f"[setup] 13-D grid {physical_shape} = {config.n_dimensions} axes, "
          f"{sum(config.qtt_bits)} QTT cores")
    standardized = book.standardized_moneyness(market.basket_volatility)
    for row, z in zip(book.describe(), standardized):
        print(f"  {row['label']:<16} K={row['strike']:8.4f} T={row['maturity']:.4f} "
              f"m={row['moneyness']:.2f} ({z:+.2f} sd, "
              f"{'ITM' if row['in_the_money'] else 'OTM'})")

    # ---- scenarios, partages par les trois bras --------------------------
    generator = GBMScenarioGenerator(
        volatilities=market.volatilities,
        correlation=market.correlation,
        dividends=market.dividends,
        drift=(
            None
            if args.real_world_drift is None
            else np.full(market.n_assets, args.real_world_drift)
        ),
    )
    start = perf_counter()
    scenario_spots = generator.simulate(
        market.spots,
        horizon,
        n_scenarios,
        rate=market.rate,
        rng=np.random.default_rng(args.scenario_seed),
        antithetic=True,
    )
    scenario_time = perf_counter() - start

    base_parameters = book.portfolio.market_parameters(
        market.spots[None, :], market.rate, maturity_shift=0.0
    )
    scenario_parameters = book.portfolio.market_parameters(
        scenario_spots, market.rate, maturity_shift=horizon
    )
    n_evaluations = int(
        base_parameters.shape[0] * base_parameters.shape[1]
        + scenario_parameters.shape[0] * scenario_parameters.shape[1]
    )

    arms: dict[str, Arm] = {}

    # ---- bras 1: forme fermee (reference) --------------------------------
    def value_closed_form():
        return (
            price_book_closed_form(market, base_parameters, is_call),
            price_book_closed_form(market, scenario_parameters, is_call),
        )

    (base_values, scenario_values), closed_form_timing = timed(
        value_closed_form, args.timing_repeats
    )
    arms["closed_form"] = Arm(
        name="closed_form",
        base_trade_values=base_values[0],
        scenario_trade_values=scenario_values,
        quantities=quantities,
        initialisation_time=0.0,
        revaluation_time=closed_form_timing["median"],
        pricer_evaluations=n_evaluations,
        metadata={"description": "Black-Scholes geometric basket, exact"},
        revaluation_timing=closed_form_timing,
    )
    print(f"[closed_form] {closed_form_timing['median']:.3f}s, "
          f"base value {arms['closed_form'].base_value:.6f}")

    # ---- bras 2: Monte Carlo imbrique ------------------------------------
    mc, mc_init_timing = timed(
        lambda: NestedMonteCarloBook(market, inner_paths, seed=args.mc_seed),
        args.timing_repeats,
    )
    mc_init_time = mc_init_timing["median"]

    (mc_base, mc_scenarios), mc_timing = timed(
        lambda: (mc(base_parameters, is_call), mc(scenario_parameters, is_call)),
        args.timing_repeats,
    )
    mc_time = mc_timing["median"]
    arms["monte_carlo"] = Arm(
        name="monte_carlo",
        base_trade_values=mc_base[0],
        scenario_trade_values=mc_scenarios,
        quantities=quantities,
        initialisation_time=mc_init_time,
        revaluation_time=mc_time,
        pricer_evaluations=n_evaluations,
        metadata={
            "inner_paths": int(inner_paths),
            "antithetic": True,
            "common_random_numbers": True,
            "engine": "generic five-asset terminal simulation",
            "base_standard_error": mc.standard_error(base_parameters, is_call).tolist(),
        },
        initialisation_timing=mc_init_timing,
        revaluation_timing=mc_timing,
    )
    print(f"[monte_carlo] init {mc_init_time:.3f}s + reval {mc_time:.3f}s")

    # Borne inferieure de cout: le moteur qui sait que le panier geometrique
    # se reduit a un facteur multiplicatif commun a tous les scenarios.
    (collapsed_base, collapsed_scenarios), collapsed_timing = timed(
        lambda: (
            mc(base_parameters, is_call, collapsed=True),
            mc(scenario_parameters, is_call, collapsed=True),
        ),
        args.timing_repeats,
    )
    collapsed_time = collapsed_timing["median"]
    collapsed_gap = float(
        np.max(np.abs(collapsed_scenarios @ quantities - mc_scenarios @ quantities))
    )
    arms["monte_carlo"].metadata["payoff_aware_engine"] = {
        "revaluation_time": collapsed_time,
        "speedup": mc_time / collapsed_time if collapsed_time > 0.0 else None,
        "max_portfolio_value_gap": collapsed_gap,
        "note": "exact same estimator, geometric collapse precomputed",
    }
    print(f"[monte_carlo] payoff-aware lower bound {collapsed_time:.3f}s "
          f"(gap {collapsed_gap:.2e})")
    del collapsed_base, collapsed_scenarios

    # ---- bras 3: QTT calibre sur les puts --------------------------------
    def build_surrogate(seed=None):
        # La construction de la grille 13-D fait partie de l'initialisation:
        # elle resout les bornes standardisees sur toute la boite de
        # volatilite, ce qui n'est pas gratuit, contrairement au produit
        # tensoriel d'une grille de prix.
        return VolExtendedPriceSurrogate(
            config,
            args.mode,
            basket_kind="geometric",
            cubic_axes=("all",),
            seed=args.tt_seed if seed is None else int(seed),
        )

    surrogate, grid_timing = timed(build_surrogate, args.timing_repeats)
    grid_time = grid_timing["median"]
    grid = surrogate.grid
    grid_description = surrogate.grid_description

    def fit_nodes():
        # Le seed est fixe, donc chaque repetition refait exactement le meme
        # ajustement: seule la mesure du temps change, jamais le surrogate.
        return surrogate.fit(
            budget,
            anova_samples=profile["anova"],
            truncation=None if args.truncation == 0.0 else args.truncation,
            log=False,
        )

    diagnostics, fit_timing = timed(fit_nodes, args.timing_repeats)
    fit_time = fit_timing["median"]

    # Les deux echelles que la grille doit resoudre pour une VaR 1 jour: le
    # deplacement du panier sur l'horizon, et l'horizon lui-meme sur l'axe des
    # maturites. Une maille plus large qu'elles signifie que le PnL est une
    # difference prise sous la resolution de la grille.
    one_day_move = float(
        market.geometric_basket * market.basket_volatility * np.sqrt(horizon)
    )
    maturity_axis = grid.axes[config.maturity_column]
    # Le livre couvre maintenant plusieurs maturites, donc la maille pertinente
    # est celle de chaque position; on garde la pire, qui est celle qui borne
    # la qualite de l'interpolation en T.
    maturity_cells = {}
    for maturity in sorted(set(book.maturities.tolist())):
        upper = int(np.clip(
            np.searchsorted(maturity_axis, maturity), 1, len(maturity_axis) - 1
        ))
        maturity_cells[f"{maturity:.6g}"] = float(
            maturity_axis[upper] - maturity_axis[upper - 1]
        )
    maturity_cell = max(maturity_cells.values())

    cubic_columns = surrogate.cubic_columns
    qtt = QTTBookPricer(market, surrogate)
    (qtt_base, qtt_scenarios), qtt_timing = timed(
        lambda: (qtt(base_parameters, is_call), qtt(scenario_parameters, is_call)),
        args.timing_repeats,
    )
    qtt_time = qtt_timing["median"]

    arms["qtt"] = Arm(
        name="qtt",
        base_trade_values=qtt_base[0],
        scenario_trade_values=qtt_scenarios,
        quantities=quantities,
        initialisation_time=grid_time + fit_time,
        revaluation_time=qtt_time,
        pricer_evaluations=n_evaluations,
        metadata={
            "calibrated_on": "geometric basket puts only",
            "calls_from": "closed-form put-call parity offset e^{-rT}(F - K)",
            "dimensions": int(config.n_dimensions),
            "mode": args.mode,
            "layout": args.layout,
            "fit_target": surrogate.target,
            "qtt_cores": int(sum(config.qtt_bits)),
            "grid": grid_description,
            "grid_build_time": grid_time,
            "fit_wall_time_total": fit_time,
            "cross_wall_time": diagnostics.wall_time,
            "anova_samples": int(profile["anova"]),
            "budget": int(budget),
            "function_evaluations": int(diagnostics.function_evaluations),
            "sweeps": int(diagnostics.sweeps),
            "stop": diagnostics.stop,
            "effective_rank": diagnostics.effective_rank,
            "maximum_rank": diagnostics.maximum_rank,
            "parameter_count": diagnostics.parameter_count,
            "pre_truncation_parameter_count": diagnostics.pre_truncation_parameter_count,
            "truncation": diagnostics.truncation,
            "cubic_columns": list(cubic_columns),
            "physical_shape": list(physical_shape),
            "one_day_basket_move": one_day_move,
            "spot_cell_width": float(np.diff(grid.axes[0]).mean()),
            "spot_cell_over_one_day_move": float(
                np.diff(grid.axes[0]).mean() / one_day_move
            ),
            "maturity_cell_width_by_book_maturity": maturity_cells,
            "worst_maturity_cell_width": maturity_cell,
            "worst_maturity_cell_over_horizon": float(maturity_cell / horizon),
        },
        initialisation_timing={
            "grid": grid_timing,
            "fit": fit_timing,
            "median": grid_time + fit_time,
        },
        revaluation_timing=qtt_timing,
    )
    print(f"[qtt] spot cell {np.diff(grid.axes[0]).mean():.3f} vs one-day move "
          f"{one_day_move:.3f}; worst maturity cell {maturity_cell:.4f}y vs "
          f"horizon {horizon:.4f}y")
    print(f"[qtt] init {grid_time + fit_time:.3f}s "
          f"(grid {grid_time:.3f}s + fit {fit_time:.3f}s) + reval {qtt_time:.3f}s, "
          f"erank {diagnostics.effective_rank:.2f}")

    # ---- courbe cout/precision du Monte Carlo ----------------------------
    sweep_paths = (
        profile["mc_sweep"] if args.mc_path_sweep is None else args.mc_path_sweep
    )
    sweep_rows = (
        monte_carlo_path_sweep(
            market, book, base_parameters, scenario_parameters,
            arms["closed_form"], sweep_paths, seeds=args.mc_sweep_seeds,
        )
        if sweep_paths
        else []
    )

    # ---- courbe cout/precision du QTT ------------------------------------
    qtt_sweep_budgets = (
        profile["qtt_sweep"] if args.qtt_budget_sweep is None else args.qtt_budget_sweep
    )
    qtt_sweep_rows = (
        qtt_budget_sweep(
            market, book, base_parameters, scenario_parameters,
            arms["closed_form"], build_surrogate, qtt_sweep_budgets,
            anova_samples=profile["anova"],
            truncation=None if args.truncation == 0.0 else args.truncation,
            seeds=args.tt_sweep_seeds,
        )
        if qtt_sweep_budgets
        else []
    )

    # ---- diagnostics de domaine ------------------------------------------
    model_points = surrogate.to_model(
        to_extended_market(
            market,
            scenario_parameters.reshape(-1, scenario_parameters.shape[-1]),
            config,
        )
    )
    coverage = domain_coverage(model_points, grid.bounds)

    # ---- comparaisons -----------------------------------------------------
    reference = arms["closed_form"]
    comparison = {}
    for name in ("monte_carlo", "qtt"):
        arm = arms[name]
        comparison[name] = {
            "loss_distribution": compare_loss_distributions(
                reference.losses, arm.losses, levels=LEVELS
            ),
            "error_cancellation": cancellation_report(reference, arm),
            "cost_versus_closed_form": break_even(reference, arm),
            "per_trade_price_error": {
                label: float(value)
                for label, value in zip(
                    book.labels,
                    np.mean(
                        np.abs(arm.scenario_trade_values - reference.scenario_trade_values),
                        axis=0,
                    ),
                )
            },
        }
    comparison["qtt"]["cost_versus_monte_carlo"] = break_even(
        arms["monte_carlo"], arms["qtt"]
    )

    # La parite est une identite, pas une approximation: l'erreur du call doit
    # etre exactement celle du put de meme strike. On le verifie plutot que de
    # l'affirmer.
    # Le livre n'a plus de paires call/put appariees, donc l'audit ne peut plus
    # comparer deux colonnes existantes: il reprice, aux parametres exacts de
    # chaque call, le put fantome de meme strike et meme maturite. L'erreur du
    # call doit alors egaler celle de ce put, puisque l'offset de parite est
    # commun aux deux et exact.
    call_columns = np.flatnonzero(is_call)
    qtt_error = arms["qtt"].scenario_trade_values - reference.scenario_trade_values
    if call_columns.size:
        shadow = scenario_parameters[:, call_columns, :]
        shadow_puts = np.zeros(shadow.shape[1], dtype=bool)
        shadow_qtt = qtt(shadow, shadow_puts)
        shadow_reference = price_book_closed_form(market, shadow, shadow_puts)
        parity_residual = float(
            np.max(np.abs(qtt_error[:, call_columns] - (shadow_qtt - shadow_reference)))
        )
    else:
        parity_residual = 0.0
    comparison["qtt"]["parity_consistency"] = {
        "description": (
            "max |call error - error of the same-strike same-maturity put|, "
            "the shadow put repriced through the same surrogate"
        ),
        "max_residual": parity_residual,
        "audited_calls": [book.labels[j] for j in call_columns],
    }

    results = {
        "profile": args.profile,
        "platform": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "machine": platform.platform(),
            "blas_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "data": market.describe(),
        "book": {
            "layout": book.layout,
            "positions": book.describe(market.basket_volatility),
            "strike_reference": "geometric basket spot at t=0",
            "moneyness": list(book.moneyness),
            "maturity_years": book.maturities.tolist(),
            "all_out_of_the_money": bool(
                not any(row["in_the_money"] for row in book.describe())
            ),
            "base_value_closed_form": reference.base_value,
            "base_trade_values_closed_form": {
                label: float(value)
                for label, value in zip(book.labels, reference.base_trade_values)
            },
        },
        "qtt_grid": {
            "dimensions": int(config.n_dimensions),
            "mode": args.mode,
            "layout": args.layout,
            "physical_shape": list(physical_shape),
            "qtt_cores": int(sum(config.qtt_bits)),
            "volatility_bounds": list(config.volatility_bounds),
            "cubic_columns": list(cubic_columns),
        },
        "risk_setup": {
            "horizon_days": args.horizon_days,
            "horizon_years": horizon,
            "scenarios": int(n_scenarios),
            "antithetic": True,
            "levels": list(LEVELS),
            "scenario_model": generator.describe(market.rate, horizon),
            "scenario_generation_time": scenario_time,
            "maturities_aged_by_horizon": True,
            "grid_domain_coverage": coverage,
        },
        "arms": {
            name: {
                "risk": arm.risk_summary(),
                "cost": arm.cost_summary(),
                "metadata": arm.metadata,
            }
            for name, arm in arms.items()
        },
        "comparison": comparison,
        "monte_carlo_path_sweep": sweep_rows,
        "qtt_budget_sweep": qtt_sweep_rows,
    }

    if sweep_rows and qtt_sweep_rows:
        results["amortisation"] = amortisation_table(qtt_sweep_rows, sweep_rows)
        results["amortisation_es"] = amortisation_table(
            qtt_sweep_rows, sweep_rows, level=ES_LEVEL, metric="es_relative_error"
        )

    if sweep_rows:
        # La cible est l'erreur du QTT *moyennee sur les graines* au budget
        # retenu, jamais celle de la graine unique du bras principal: sinon le
        # croisement est cale sur un tirage chanceux et l'avantage annonce est
        # celui d'un coup de chance.
        qtt_errors = comparison["qtt"]["loss_distribution"]["measures"]
        at_budget = next(
            (row for row in qtt_sweep_rows if row["budget"] == int(budget)), None
        )
        crossover = {}
        for level in LEVELS:
            key = f"{level:g}"
            if at_budget is None:
                target = qtt_errors[key]["var_relative_error"]
                source = "single-seed headline arm"
            else:
                target = at_budget["var_relative_error"][key]
                source = f"rms over {len(at_budget['seeds'])} TT seeds"
            crossover[key] = iso_accuracy_crossover(sweep_rows, target, level)
            crossover[key]["target_source"] = source
        for entry in crossover.values():
            if entry.get("estimated_total_time") is not None:
                entry["qtt_total_time"] = arms["qtt"].total_time
                entry["qtt_total_time_advantage"] = (
                    entry["estimated_total_time"] / arms["qtt"].total_time
                )
        results["iso_accuracy_versus_qtt"] = crossover

        # Meme lecture pour l'ES reglementaire: cible = erreur d'ES du QTT
        # moyennee sur graines au budget retenu.
        es_key = f"{ES_LEVEL:g}"
        if at_budget is None:
            es_target = qtt_errors[es_key]["expected_shortfall_relative_error"]
            es_source = "single-seed headline arm"
        else:
            es_target = at_budget["es_relative_error"][es_key]
            es_source = f"rms over {len(at_budget['seeds'])} TT seeds"
        es_crossover = iso_accuracy_crossover(
            sweep_rows, es_target, ES_LEVEL, metric="es_relative_error"
        )
        es_crossover["target_source"] = es_source
        if es_crossover.get("estimated_total_time") is not None:
            es_crossover["qtt_total_time"] = arms["qtt"].total_time
            es_crossover["qtt_total_time_advantage"] = (
                es_crossover["estimated_total_time"] / arms["qtt"].total_time
            )
        results["iso_accuracy_es_versus_qtt"] = es_crossover

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"var_qtt_basket_book_{tag}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    markdown = benchmark_markdown(results)
    if markdown:
        markdown_path = output_dir / f"var_qtt_benchmark_{tag}.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        print(f"wrote {markdown_path}")

    csv_path = output_dir / f"var_qtt_basket_book_{tag}.csv"
    header = (
        "arm,level,var,expected_shortfall,var_absolute_error,"
        "var_relative_error,es_absolute_error,es_relative_error,"
        "initialisation_time,revaluation_time,total_time"
    )
    rows = [header]
    for name, arm in arms.items():
        summary = arm.risk_summary()
        for level in LEVELS:
            key = f"{level:g}"
            measure = summary["measures"][key]
            if name == "closed_form":
                abs_error = rel_error = es_abs = es_rel = 0.0
            else:
                errors = comparison[name]["loss_distribution"]["measures"][key]
                abs_error = errors["var_absolute_error"]
                rel_error = errors["var_relative_error"]
                es_abs = errors["expected_shortfall_absolute_error"]
                es_rel = errors["expected_shortfall_relative_error"]
            rows.append(
                f"{name},{key},{measure['var']:.10g},"
                f"{measure['expected_shortfall']:.10g},{abs_error:.10g},"
                f"{rel_error:.10g},{es_abs:.10g},{es_rel:.10g},"
                f"{arm.initialisation_time:.6f},"
                f"{arm.revaluation_time:.6f},{arm.total_time:.6f}"
            )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ---- resume lisible ---------------------------------------------------
    print()
    print(f"base portfolio value (closed form): {reference.base_value:.6f}")
    print(f"{'arm':<13} {'VaR 95%':>11} {'VaR 99%':>11} {'ES 97.5%':>11} "
          f"{'init once':>10} {'daily VaR':>10} {'day 1':>9}")
    for name, arm in arms.items():
        summary = arm.risk_summary()
        print(f"{name:<13} "
              f"{summary['measures']['0.95']['var']:>11.6f} "
              f"{summary['measures']['0.99']['var']:>11.6f} "
              f"{summary['measures']['0.975']['expected_shortfall']:>11.6f} "
              f"{arm.initialisation_time:>9.3f} "
              f"{arm.revaluation_time:>9.3f} {arm.total_time:>9.3f}")
    print()
    for name in ("monte_carlo", "qtt"):
        measures = comparison[name]["loss_distribution"]["measures"]
        cancel = comparison[name]["error_cancellation"]
        print(f"{name}: VaR95 err {measures['0.95']['var_absolute_error']:.3e} "
              f"({measures['0.95']['var_relative_error']:.2%}), "
              f"VaR99 err {measures['0.99']['var_absolute_error']:.3e} "
              f"({measures['0.99']['var_relative_error']:.2%}), "
              f"ES97.5 err "
              f"{measures['0.975']['expected_shortfall_absolute_error']:.3e} "
              f"({measures['0.975']['expected_shortfall_relative_error']:.2%}), "
              f"price MAE {cancel['scenario_price_mean_absolute_error']:.3e} -> "
              f"PnL MAE {cancel['pnl_mean_absolute_error']:.3e} "
              f"(ratio {cancel['cancellation_ratio']:.3f})")
    print()
    if sweep_rows:
        for level in LEVELS:
            entry = results["iso_accuracy_versus_qtt"][f"{level:g}"]
            if entry.get("estimated_total_time") is None:
                continue
            print(f"iso-accuracy at {level:g}: Monte Carlo needs "
                  f"{entry['paths']:.3g} paths "
                  f"({'extrapolated' if entry['extrapolated'] else 'measured'}) "
                  f"= {entry['estimated_total_time']:.1f}s total, "
                  f"against {entry['qtt_total_time']:.1f}s for QTT "
                  f"({entry['qtt_total_time_advantage']:.1f}x)")
        print()
    es_iso = results.get("iso_accuracy_es_versus_qtt")
    if es_iso and es_iso.get("estimated_total_time") is not None:
        print(f"iso-accuracy on ES 97.5: Monte Carlo needs "
              f"{es_iso['paths']:.3g} paths "
              f"({'extrapolated' if es_iso['extrapolated'] else 'measured'}) "
              f"= {es_iso['estimated_total_time']:.1f}s total, against "
              f"{es_iso['qtt_total_time']:.1f}s for QTT "
              f"({es_iso['qtt_total_time_advantage']:.1f}x)")
        print()
    print(f"qtt parity residual (call err - put err): {parity_residual:.3e}")
    print(f"scenarios outside the grid box: "
          f"{coverage['fraction_outside_any_axis']:.2%}")
    print(f"wrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
