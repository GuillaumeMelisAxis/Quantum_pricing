r"""VaR 1 jour d'un put sur panier geometrique a 5 sous-jacents, via RiskEngine.

Difference avec ``benchmark_var_geometric_basket.py``, dont ce script reprend la
machinerie de revalorisation: les chocs de spot ne sont plus simules, ce sont les
rendements quotidiens reellement observes sur un an, lus par
``riskengine.data.market_data_fetcher.MarketDataFetcher``. La VaR est calculee
par ``riskengine.metrics.var_calculator.VaRCalculator`` (VaR 95, VaR 99, ES 99)
sur des scenarios produits par
``riskengine.scenarios.historical_scenario_generator.HistoricalScenarioGenerator``.

Ce qui reste synthetique, et pourquoi: les chocs de volatilite. La base de
calibrations SVI ne couvre que quatre sous-jacents, et leur intersection ne
fournit que 83 jours calibres sur quatorze mois, dont 62 seulement sont des
ecarts d'un jour. Une VaR 99% assise sur 62 observations repose sur la pire
d'entre elles: ce n'est pas un quantile, c'est un maximum. Les chocs de vol sont
donc generes, mais **conditionnellement aux rendements observes**, de sorte que
l'effet de levier relie la vol simulee au spot reel plutot qu'a un spot simule.

Trois revalorisations sur le meme jeu de scenarios - forme fermee (exacte, donc
reference), grille hybride QTT, Monte Carlo - et le cout d'initialisation compte.
La VaR Monte Carlo au sens generateur de scenarios n'est pas calculee ici:
Monte Carlo n'intervient que comme methode de revalorisation.

    .venv/Scripts/python.exe scripts/validation/benchmark_var_riskengine.py
"""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse
import json
import platform
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stngpr.config import VolatilityExtendedConfig
from stngpr.vol_extension import VolExtendedPriceSurrogate

from riskengine.data.market_data_fetcher import MarketDataFetcher
from riskengine.metrics.var_calculator import VaRCalculator
from riskengine.scenarios.historical_scenario_generator import (
    HistoricalScenarioGenerator,
)

from benchmark_var_geometric_basket import (
    bootstrap_var_interval,
    cancellation_report,
    price_analytic,
    price_monte_carlo,
    timed_call,
)

CONFIDENCE_LEVELS = (0.95, 0.99)


# ==========================================================================
# 1. Portefeuille
# ==========================================================================

# Cinq sous-jacents, deux secteurs. Le decoupage sert uniquement a donner sa
# structure de blocs a la matrice de correlation.
TICKERS = ("AAPL", "MSFT", "NVDA", "JPM", "XOM")
SECTORS = (0, 0, 0, 1, 1)          # 0 = technologie, 1 = hors technologie
SECTOR_NAMES = {0: "technologie", 1: "hors technologie"}
INTRA_SECTOR = {0: 0.70, 1: 0.55}  # correlation intra-secteur
INTER_SECTOR = 0.35                # correlation inter-secteurs

# Volatilites arbitraires, du meme ordre que les vols implicites a un an de ces
# noms. Ce sont des parametres du portefeuille, pas des estimations.
VOLATILITIES = (0.28, 0.32, 0.45, 0.24, 0.28)


def sector_block_correlation(
    sectors=SECTORS, intra=INTRA_SECTOR, inter=INTER_SECTOR
) -> np.ndarray:
    """Correlation en blocs sectoriels, definie positive (verifiee par Cholesky).

    Le seul fait stylise encode est celui qui compte pour un panier: deux titres
    du meme secteur bougent ensemble plus que deux titres de secteurs
    differents. Une equicorrelation ne le voit pas, et c'est exactement ce qui
    fixe la volatilite du panier, donc le prix de l'option.
    """
    n = len(sectors)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            matrix[i, j] = (
                intra[sectors[i]] if sectors[i] == sectors[j] else inter
            )
    np.linalg.cholesky(matrix)  # leve si la matrice n'est pas definie positive
    return matrix


@dataclass(frozen=True)
class BasketPortfolio:
    """``n_contracts`` puts sur panier geometrique equipondere de cinq titres.

    Les sous-jacents sont **rebases a 100** a la date de valorisation. Seuls les
    rendements relatifs observes entrent dans les scenarios, donc le niveau
    nominal des titres ne fait pas partie de la definition du portefeuille; le
    rebasement le rend explicite et garde toutes les coordonnees a l'interieur
    de la boite de la grille, qui couvre S dans [5, 150].
    """

    tickers: tuple = TICKERS
    n_contracts: float = 1000.0
    spot: float = 100.0
    volatilities: tuple = VOLATILITIES
    strike: float = 100.0          # a la monnaie: G(S(0)) = 100
    rate: float = 0.03
    maturity: float = 1.0
    correlation_matrix: tuple = field(default=None, repr=False)

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def correlation(self) -> np.ndarray:
        if self.correlation_matrix is None:
            return sector_block_correlation()
        return np.asarray(self.correlation_matrix, dtype=float)

    @property
    def dividends(self) -> np.ndarray:
        return np.zeros(self.n_assets)

    def base_market_point(self) -> np.ndarray:
        """(S_1, sigma_1, ..., S_5, sigma_5, K, r, T)."""
        n = self.n_assets
        row = np.empty(2 * n + 3)
        for i in range(n):
            row[2 * i] = self.spot
            row[2 * i + 1] = self.volatilities[i]
        row[2 * n] = self.strike
        row[2 * n + 1] = self.rate
        row[2 * n + 2] = self.maturity
        return row


# ==========================================================================
# 2. Donnees
# ==========================================================================


@dataclass(frozen=True)
class VolShockParameters:
    """Parametres du choc de volatilite synthetique.

    Identiques a ceux du generateur entierement simule, pour que les deux
    experiences restent comparables.
    """

    vol_mean_reversion: float = 3.0    # kappa, par an, sur log(sigma)
    vol_stationary_sd: float = 0.35    # ecart-type stationnaire de log(sigma)
    vol_correlation: float = 0.50      # correlation cible des innovations de vol
    leverage: float = -0.60            # corr(rendement standardise, innovation)
    trading_days: float = 252.0
    seed: int = 20260918


def market_return_scenarios(tickers, history_period: str = "1y"):
    """Rendements quotidiens observes, via RiskEngine, tels quels.

    ``MarketDataFetcher.get_historical_returns`` renvoie des rendements simples
    (``pct_change``). Ils sont passes sans retraitement a
    ``HistoricalScenarioGenerator``, dont c'est le contrat: une journee passee,
    un scenario, et l'index de dates est conserve pour pouvoir nommer ensuite
    les journees responsables des pires pertes.
    """
    fetcher = MarketDataFetcher(list(tickers), history_period=history_period)
    prices = fetcher.get_historical_prices()[list(tickers)]
    returns = fetcher.get_historical_returns()[list(tickers)].dropna()
    generator = HistoricalScenarioGenerator(returns, keep_dates=True)
    scenarios = generator.generate()
    info = {
        "source": "riskengine.data.market_data_fetcher.MarketDataFetcher (yfinance)",
        "scenario_generator": (
            "riskengine.scenarios.historical_scenario_generator."
            "HistoricalScenarioGenerator"
        ),
        "tickers": list(tickers),
        "history_period": history_period,
        "n_price_days": int(prices.shape[0]),
        "n_scenarios": int(scenarios.shape[0]),
        "first_date": str(scenarios.index[0].date()),
        "last_date": str(scenarios.index[-1].date()),
        "return_convention": "rendement simple quotidien (pct_change)",
        "realized_annualized_vol": {
            ticker: float(scenarios[ticker].std() * np.sqrt(252))
            for ticker in tickers
        },
        "empirical_correlation": scenarios.corr().round(4).to_dict(),
    }
    return scenarios, info


def synthetic_vol_shocks(returns: pd.DataFrame, params: VolShockParameters):
    r"""Chocs de volatilite simules, conditionnels aux rendements observes.

    Le choc d'un jour sur la volatilite de l'actif i est

        dlog sigma_i = eta xi_i
        xi_i = lam z_i + sqrt(1 - lam^2) [ sqrt(rho') g + sqrt(1 - rho') h_i ]

    ou ``z_i`` est le rendement **observe** du jour, standardise par son
    ecart-type d'echantillon, ``g`` un facteur de volatilite commun et ``h_i``
    un bruit idiosyncratique. C'est le point important: le terme de levier est
    porte par le vrai rendement de marche, donc une seance de forte baisse
    produit mecaniquement une hausse de volatilite dans le meme scenario. Un
    choc de vol tire independamment casserait cette dependance, qui est
    precisement ce que paie un vendeur d'option.

    L'amplitude eta = s_inf sqrt(1 - phi^2) est celle d'un AR(1) sur log sigma
    de vitesse kappa et d'ecart-type stationnaire s_inf; a un jour elle vaut
    environ 5,4% en variation logarithmique. La volatilite courante du
    portefeuille etant le point de depart, le terme de rappel vers la moyenne
    est nul au premier pas et n'apparait pas.

    ``rho'`` est resolu pour que la correlation des innovations de vol atteigne
    ``vol_correlation`` compte tenu de la correlation *empirique* des rendements
    observes; la fonction leve si la cible est inatteignable, plutot que de
    produire une structure de dependance incoherente.
    """
    dt = 1.0 / params.trading_days
    phi = float(np.exp(-params.vol_mean_reversion * dt))
    eta = params.vol_stationary_sd * np.sqrt(1.0 - phi**2)
    lam = params.leverage

    values = returns.to_numpy(dtype=float)
    days, n = values.shape
    z = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)

    empirical = float(
        np.mean(np.corrcoef(z, rowvar=False)[~np.eye(n, dtype=bool)])
    )
    rho_prime = (params.vol_correlation - lam**2 * empirical) / (1.0 - lam**2)
    if not 0.0 <= rho_prime <= 1.0:
        raise ValueError(
            f"vol_correlation={params.vol_correlation} inatteignable: "
            f"rho'={rho_prime:.3f} hors de [0, 1] avec une correlation "
            f"empirique des rendements de {empirical:.3f}"
        )

    rng = np.random.default_rng(params.seed)
    g = rng.standard_normal((days, 1))
    h = rng.standard_normal((days, n))
    xi = lam * z + np.sqrt(1.0 - lam**2) * (
        np.sqrt(rho_prime) * g + np.sqrt(1.0 - rho_prime) * h
    )
    log_vol_change = eta * xi

    frame = pd.DataFrame(log_vol_change, index=returns.index, columns=returns.columns)
    realized_leverage = float(np.mean([
        np.corrcoef(values[:, i], log_vol_change[:, i])[0, 1] for i in range(n)
    ]))
    info = {
        "model": "dlog sigma = eta * xi, xi conditionnel aux rendements observes",
        "parameters": asdict(params),
        "phi": phi,
        "eta": float(eta),
        "rho_prime": float(rho_prime),
        "empirical_return_correlation": empirical,
        "daily_log_vol_change_sd": float(log_vol_change.std()),
        "realized_leverage": realized_leverage,
        "realized_vol_shock_correlation": float(
            np.mean(np.corrcoef(log_vol_change, rowvar=False)[~np.eye(n, dtype=bool)])
        ),
        "why_synthetic": (
            "l'intersection des calibrations SVI disponibles ne couvre que "
            "4 sous-jacents et 62 ecarts d'un jour; une VaR 99% sur 62 "
            "observations est un maximum, pas un quantile"
        ),
    }
    return frame, info


def build_scenario_points(portfolio: BasketPortfolio, returns: pd.DataFrame,
                          log_vol_change: pd.DataFrame):
    """Applique chocs de spot et de vol au point de marche -> points de marche.

    Le choc de spot est le rendement simple observe, le choc de vol est
    multiplicatif. ``K``, ``r`` et ``T`` sont geles: a l'horizon d'un jour la
    maturite residuelle n'est pas roulee, donc le P&L ne contient pas de theta.
    C'est la convention de ``MarketOptionDataFetcher.option_pnl_scenarios`` dans
    RiskEngine, reprise ici pour rester comparable.
    """
    n = portfolio.n_assets
    m = returns.shape[0]
    points = np.empty((m, 2 * n + 3))
    spots = portfolio.spot * (1.0 + returns.to_numpy(dtype=float))
    vols = np.asarray(portfolio.volatilities, dtype=float) * np.exp(
        log_vol_change.to_numpy(dtype=float)
    )
    for i in range(n):
        points[:, 2 * i] = spots[:, i]
        points[:, 2 * i + 1] = vols[:, i]
    points[:, 2 * n] = portfolio.strike
    points[:, 2 * n + 1] = portfolio.rate
    points[:, 2 * n + 2] = portfolio.maturity
    box = {
        "spot_min": float(spots.min()), "spot_max": float(spots.max()),
        "vol_min": float(vols.min()), "vol_max": float(vols.max()),
        "grid_box_spot": [5.0, 150.0], "grid_box_vol": [0.05, 0.80],
        "spot_outside_grid_box": int(
            np.sum((spots < 5.0) | (spots > 150.0))
        ),
        "vol_outside_grid_box": int(np.sum((vols < 0.05) | (vols > 0.80))),
    }
    return points, box


# ==========================================================================
# 3. Revalorisation et VaR
# ==========================================================================


def var_row(name, prices, base_price, exact_prices, exact_base, portfolio,
            index, seconds, setup_seconds, extra=None):
    """Un bras: VaR 95 / 99 et ES 99 par le VaRCalculator de RiskEngine."""
    pnl = pd.Series(
        portfolio.n_contracts * (prices - base_price), index=index, name="pnl"
    )
    exact_pnl = pd.Series(
        portfolio.n_contracts * (exact_prices - exact_base), index=index
    )
    calculator = VaRCalculator(pnl, horizon=1)
    reference = VaRCalculator(exact_pnl, horizon=1)

    row = {
        "arm": name,
        "base_price": float(base_price),
        "revaluation_seconds": float(seconds),
        "setup_seconds": float(setup_seconds),
        "total_seconds": float(seconds + setup_seconds),
        "seconds_per_revaluation": float(seconds / prices.size),
        "n_scenarios": int(prices.size),
        "var_95": calculator.var_95(),
        "var_99": calculator.var_99(),
        "es_99": calculator.es_99(),
        "var_95_error": calculator.var_95() - reference.var_95(),
        "var_99_error": calculator.var_99() - reference.var_99(),
        "es_99_error": calculator.es_99() - reference.es_99(),
        "var_99_relative_error": (
            calculator.var_99() / reference.var_99() - 1.0
            if reference.var_99() != 0 else float("nan")
        ),
        "var_95_relative_error": (
            calculator.var_95() / reference.var_95() - 1.0
            if reference.var_95() != 0 else float("nan")
        ),
    }
    row.update(cancellation_report(prices, exact_prices, base_price, exact_base))
    if extra:
        row.update(extra)
    return row


def run_arms(points, index, portfolio, surrogates, mc_paths, crn_paths,
             mc_seed, log):
    """Les trois revalorisations, sur le meme jeu de scenarios."""
    base_point = portfolio.base_market_point().reshape(1, -1)

    exact, exact_seconds = timed_call(lambda: price_analytic(portfolio, points))
    exact_base = float(price_analytic(portfolio, base_point)[0])
    exact_pnl = pd.Series(
        portfolio.n_contracts * (exact - exact_base), index=index
    )

    extra = {"family": "closed_form"}
    for level in CONFIDENCE_LEVELS:
        tag = f"{int(level * 100)}"
        extra[f"var_{tag}_bootstrap_ci"] = bootstrap_var_interval(
            exact_pnl.to_numpy(), level
        )
    extra["worst_scenarios"] = VaRCalculator(exact_pnl).worst_scenarios(5)

    rows = [var_row("closed_form", exact, exact_base, exact, exact_base,
                    portfolio, index, exact_seconds, 0.0, extra)]
    log(f"  forme fermee            : {exact_seconds:8.3f} s")

    for budget, entry in surrogates.items():
        model = entry["model"]
        prices, seconds = timed_call(lambda m=model: m.price(points))
        base = float(model.price(base_point)[0])
        rows.append(var_row(
            f"hybrid_grid_{budget}", prices, base, exact, exact_base,
            portfolio, index, seconds, entry["training_seconds"],
            {"family": "hybrid_grid", "budget": int(budget),
             "effective_rank": entry["effective_rank"],
             "parameter_count": entry["parameter_count"]},
        ))
        log(f"  grille hybride {budget:>7,}  : {seconds:8.3f} s")

    for paths in mc_paths:
        prices, seconds = timed_call(
            lambda p=paths: price_monte_carlo(portfolio, points, p, mc_seed, False)[0]
        )
        base = float(
            price_monte_carlo(portfolio, base_point, paths, mc_seed + 1, False)[0][0]
        )
        rows.append(var_row(
            f"monte_carlo_{paths}", prices, base, exact, exact_base,
            portfolio, index, seconds, 0.0,
            {"family": "monte_carlo", "paths": int(paths),
             "common_random_numbers": False},
        ))
        log(f"  MC {paths:>7,} independant: {seconds:8.3f} s")

    for paths in crn_paths:
        prices, seconds = timed_call(
            lambda p=paths: price_monte_carlo(portfolio, points, p, mc_seed, True)[0]
        )
        base = float(
            price_monte_carlo(portfolio, base_point, paths, mc_seed, True)[0][0]
        )
        rows.append(var_row(
            f"monte_carlo_crn_{paths}", prices, base, exact, exact_base,
            portfolio, index, seconds, 0.0,
            {"family": "monte_carlo_crn", "paths": int(paths),
             "common_random_numbers": True},
        ))
        log(f"  MC {paths:>7,} CRN        : {seconds:8.3f} s")

    return rows


def break_even(rows):
    """Revalorisations au-dela desquelles la grille coute moins cher que MC."""
    grids = [r for r in rows if r["family"] == "hybrid_grid"]
    mcs = [r for r in rows if r["family"].startswith("monte_carlo")]
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


# ==========================================================================
# 4. Orchestration
# ==========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-period", default="1y")
    parser.add_argument("--mode", default="price_adaptive")
    parser.add_argument("--maturity-nodes", type=int, default=32)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[20_000, 60_000, 150_000])
    parser.add_argument("--mc-paths", type=int, nargs="+",
                        default=[4_096, 16_384, 65_536, 262_144])
    parser.add_argument("--crn-paths", type=int, nargs="+",
                        default=[16_384, 262_144])
    parser.add_argument("--anova-samples", type=int, default=1000)
    parser.add_argument("--vol-seed", type=int, default=20260918)
    parser.add_argument("--mc-seed", type=int, default=4242)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    def log(message):
        if not args.quiet:
            print(message, flush=True)

    portfolio = BasketPortfolio()
    correlation = portfolio.correlation
    config = VolatilityExtendedConfig(
        maturity_nodes=args.maturity_nodes, correlation=correlation
    )

    log("1. rendements de marche (RiskEngine / yfinance)")
    returns, market_info = market_return_scenarios(
        portfolio.tickers, args.history_period
    )
    log(f"   {market_info['n_scenarios']} scenarios, "
        f"{market_info['first_date']} -> {market_info['last_date']}")
    for ticker, vol in market_info["realized_annualized_vol"].items():
        log(f"   {ticker:6s} vol annualisee realisee {vol:.3f}")

    log("2. chocs de volatilite synthetiques, conditionnels aux rendements")
    vol_params = VolShockParameters(seed=args.vol_seed)
    log_vol_change, vol_info = synthetic_vol_shocks(returns, vol_params)
    log(f"   eta {vol_info['eta']:.4f}, rho' {vol_info['rho_prime']:.3f}")
    log(f"   levier realise {vol_info['realized_leverage']:.3f}, "
        f"correlation des chocs de vol "
        f"{vol_info['realized_vol_shock_correlation']:.3f}")

    log("3. scenarios de marche")
    points, box = build_scenario_points(portfolio, returns, log_vol_change)
    log(f"   spot [{box['spot_min']:.2f}, {box['spot_max']:.2f}], "
        f"vol [{box['vol_min']:.3f}, {box['vol_max']:.3f}]")
    log(f"   hors boite de la grille: {box['spot_outside_grid_box']} spots, "
        f"{box['vol_outside_grid_box']} vols")

    log("4. apprentissage des grilles hybrides (cout d'initialisation)")
    surrogates = {}
    for budget in args.budgets:
        model = VolExtendedPriceSurrogate(config, args.mode, seed=config.seed)
        start = perf_counter()
        diagnostics = model.fit(budget, anova_samples=args.anova_samples, log=False)
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

    log("5. revalorisation et VaR")
    rows = run_arms(points, returns.index, portfolio, surrogates,
                    args.mc_paths, args.crn_paths, args.mc_seed, log)

    report = {
        "experiment": (
            "VaR 1 jour, put sur panier geometrique 5 sous-jacents, "
            "scenarios historiques de marche"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "portfolio": {
            "tickers": list(portfolio.tickers),
            "sectors": {
                ticker: SECTOR_NAMES[sector]
                for ticker, sector in zip(TICKERS, SECTORS)
            },
            "n_contracts": portfolio.n_contracts,
            "rebased_spot": portfolio.spot,
            "volatilities": dict(zip(portfolio.tickers, portfolio.volatilities)),
            "strike": portfolio.strike,
            "rate": portfolio.rate,
            "maturity": portfolio.maturity,
            "correlation": correlation.tolist(),
            "correlation_construction": {
                "intra_sector": {SECTOR_NAMES[k]: v for k, v in INTRA_SECTOR.items()},
                "inter_sector": INTER_SECTOR,
                "eigenvalues": np.linalg.eigvalsh(correlation).tolist(),
            },
        },
        "surrogate_configuration": {
            "mode": args.mode,
            "maturity_nodes": int(args.maturity_nodes),
            "physical_shape": list(config.physical_shape),
            "qtt_cores": len(config.qtt_shape),
            "seed": int(config.seed),
        },
        "data_generation": {
            "spot_shocks": market_info,
            "vol_shocks": vol_info,
            "scenario_box": box,
            "shock_convention": (
                "spot: rendement simple observe; vol: multiplicatif "
                "exp(dlog sigma); K, r, T geles"
            ),
            "maturity_convention": "T gelee a T_0, le P&L exclut le theta",
        },
        "var_calculator": (
            "riskengine.metrics.var_calculator.VaRCalculator, horizon 1 jour, "
            "percentile lineaire"
        ),
        "results": rows,
        "break_even": break_even(rows),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "var_riskengine_basket.json"
    json_path.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")

    keys = sorted({k for row in rows for k in row if not isinstance(row[k], (dict, list))})
    csv_path = out_dir / "var_riskengine_basket.csv"
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(k, "")) for k in keys))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("")
    log(f"  {'bras':26s} {'VaR 95':>10s} {'VaR 99':>10s} {'ES 99':>10s} "
        f"{'err VaR99':>10s} {'err rel':>9s} {'ratio':>7s} {'cout s':>9s}")
    for row in rows:
        log(f"  {row['arm']:26s} {row['var_95']:10.2f} {row['var_99']:10.2f} "
            f"{row['es_99']:10.2f} {row['var_99_error']:10.2f} "
            f"{row['var_99_relative_error'] * 100:8.2f}% "
            f"{row['cancellation_ratio']:7.3f} {row['total_seconds']:9.2f}")
    ci = rows[0]["var_99_bootstrap_ci"]
    log("")
    log(f"  IC bootstrap 95% de la VaR 99% de reference: "
        f"[{ci['lower']:.2f}, {ci['upper']:.2f}]")
    log("")
    log(f"ecrit {json_path}")
    log(f"ecrit {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
