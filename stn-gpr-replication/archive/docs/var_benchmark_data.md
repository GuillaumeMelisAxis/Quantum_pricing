# Section *data* — benchmark VaR sur put panier géométrique

Matière première pour la section données du rapport. Tout est produit par
[`scripts/validation/benchmark_var_riskengine.py`](../scripts/validation/benchmark_var_riskengine.py)
et sérialisé dans `results/var_riskengine_basket.json`, bloc `data_generation`.

Le dispositif est **hybride, et l'hybridation est assumée** : les chocs de spot
sont des rendements de marché réellement observés, les chocs de volatilité sont
simulés. La section 3.3 explique pourquoi, chiffres à l'appui — c'est le point
qu'un rapporteur attaquera en premier, il doit être traité de front.

---

## 1. Ce qui vient du marché et ce qui n'en vient pas

| Composant | Source | Nature |
|---|---|---|
| Rendements quotidiens des 5 sous-jacents | `riskengine.data.market_data_fetcher.MarketDataFetcher` (yfinance) | **observé** |
| Génération des scénarios | `riskengine.scenarios.historical_scenario_generator.HistoricalScenarioGenerator` | observé, tel quel |
| VaR 95 / VaR 99 / ES 99 | `riskengine.metrics.var_calculator.VaRCalculator` | — |
| Chocs de volatilité implicite | modèle AR(1) conditionnel aux rendements observés | **simulé** |
| Volatilités initiales, corrélation | paramètres choisis | **arbitraires, assumés** |
| Prix de référence | forme fermée panier géométrique | exact à la précision machine |

Le prix de référence étant exact, **tout écart mesuré dans ce benchmark est une
erreur de revalorisation et rien d'autre.** C'est ce qui rend la comparaison
concluante : les trois moteurs de revalorisation (forme fermée, grille hybride
QTT, Monte Carlo) partagent toujours le même jeu de scénarios.

## 2. Le portefeuille

1 000 contrats d'un put européen sur panier géométrique équipondéré de cinq
sous-jacents, maturité 1 an.

| Sous-jacent | Secteur | σ_i(0) |
|---|---|---|
| AAPL | technologie | 0,28 |
| MSFT | technologie | 0,32 |
| NVDA | technologie | 0,45 |
| JPM | hors technologie | 0,24 |
| XOM | hors technologie | 0,28 |

Strike K = 100, taux r = 0,03, maturité T = 1 an, dividendes nuls.

**Volatilités arbitraires**, de l'ordre des vols implicites à un an de ces noms.
Ce sont des paramètres du portefeuille, pas des estimations : aucune nappe n'est
calibrée ici. À comparer aux vols annualisées *réalisées* sur la fenêtre, qui
sont une quantité différente et ne servent pas au pricing : AAPL 0,251,
MSFT 0,326, NVDA 0,381, JPM 0,222, XOM 0,260.

**Rebasement à 100.** Les sous-jacents sont rebasés à 100 à la date de
valorisation. Seuls les rendements relatifs observés entrent dans les scénarios,
donc le niveau nominal des titres ne fait pas partie de la définition du
portefeuille — AAPL cote 337 et MSFT 498, ce qui sortirait de la boîte de la
grille (S ∈ [5, 150]) sans rien apporter. Le rebasement rend ce choix explicite
plutôt qu'implicite.

## 3. Les données

### 3.1 Chocs de spot — observés

`MarketDataFetcher(history_period="1y")` puis `get_historical_returns()`, qui
renvoie des rendements **simples** (`pct_change`). Ils sont passés sans
retraitement à `HistoricalScenarioGenerator(keep_dates=True)`, dont c'est le
contrat : une journée passée, un scénario, index de dates conservé. C'est cet
index qui permet ensuite à `VaRCalculator.worst_scenarios()` de **nommer les
journées responsables des pires pertes** — ce que la VaR historique sait faire et
que les méthodes paramétriques ne savent pas.

| | |
|---|---|
| Fenêtre | 19/09/2025 → 17/09/2026 |
| Jours de cotation | 251 |
| Scénarios | **250** |

Aucune filtration (FHS, dévolatilisation) n'est appliquée : simulation
historique brute.

### 3.2 Matrice de corrélation — arbitraire, en blocs sectoriels

```
        AAPL  MSFT  NVDA   JPM   XOM
AAPL    1.00  0.70  0.70  0.35  0.35
MSFT    0.70  1.00  0.70  0.35  0.35
NVDA    0.70  0.70  1.00  0.35  0.35
JPM     0.35  0.35  0.35  1.00  0.55
XOM     0.35  0.35  0.35  0.55  1.00
```

0,70 intra-technologie, 0,55 intra-hors-technologie, 0,35 inter-secteurs.
Valeurs propres {0,30 ; 0,30 ; 0,45 ; 1,02 ; 2,93} — définie positive, vérifiée
par Cholesky à la construction. Le seul fait stylisé encodé est celui qui compte
pour un panier : deux titres du même secteur bougent ensemble plus que deux
titres de secteurs différents. Une équicorrélation ne le voit pas, et c'est
exactement ce qui fixe la volatilité du panier, donc le prix de l'option.

> **À déclarer dans le rapport.** Cette corrélation est un *paramètre de modèle*,
> pas une estimation. La corrélation empirique des rendements observés sur la
> fenêtre est nettement plus faible (0,07 à 0,25 intra-tech, légèrement négative
> avec XOM). Les deux coexistent sans incohérence — l'option est valorisée sous
> une corrélation de modèle, les scénarios portent la dépendance réalisée du
> marché, comme dans tout système de risque où le modèle de pricing et les
> scénarios ont des sources distinctes. Mais le lecteur doit savoir laquelle est
> laquelle.

### 3.3 Chocs de volatilité — simulés, et pourquoi

**Pourquoi pas les nappes SVI de `MarketOptionDataFetcher`.** La base de
calibrations a été interrogée, voici ce qu'elle contient :

| Ticker | Jours calibrés | Fenêtre |
|---|---|---|
| AAPL | 126 | 10/06/2025 → 17/09/2026 |
| MSFT | 120 | 17/07/2025 → 17/09/2026 |
| AMZN | 115 | 17/07/2025 → 17/09/2026 |
| GOOGL | 115 | 17/07/2025 → 17/09/2026 |
| TSLA | **1** | — |

Deux obstacles dirimants :

1. **Quatre sous-jacents seulement** portent une nappe exploitable, pas cinq.
2. Une fois les quatre intersectés, il reste **83 jours calibrés** sur quatorze
   mois, avec un trou de 139 jours, et **62 seulement** sont des écarts d'un
   jour.

Une VaR 99 % assise sur 62 observations repose sur la pire d'entre elles : ce
n'est pas un quantile, c'est un maximum. Les chocs de vol sont donc simulés.

**Le modèle, et sa propriété importante.** Le choc d'un jour sur la volatilité
de l'actif i est

```
Δlog σ_i = η · ξ_i
ξ_i      = λ · z_i + √(1−λ²) · [ √ρ' · g + √(1−ρ') · h_i ]
```

où **`z_i` est le rendement réellement observé du jour**, standardisé par son
écart-type d'échantillon, `g` un facteur de volatilité commun et `h_i` un bruit
idiosyncratique. C'est le point qui justifie la construction : le terme de levier
est porté par le vrai rendement de marché, donc une séance de forte baisse
produit mécaniquement une hausse de volatilité **dans le même scénario**. Un choc
de vol tiré indépendamment casserait cette dépendance, qui est précisément ce que
paie un vendeur d'option.

L'amplitude `η = s_∞√(1−φ²)` est celle d'un AR(1) sur log σ de vitesse κ et
d'écart-type stationnaire s_∞, avec `φ = exp(−κ/252)`. La volatilité courante
étant le point de départ, le terme de rappel vers la moyenne est nul au premier
pas et n'apparaît pas.

| Paramètre | Valeur | Réalisé |
|---|---|---|
| κ (retour à la moyenne, /an) | 3,0 | φ = 0,9882 |
| s_∞ (écart-type stationnaire de log σ) | 0,35 | η = **0,0537** |
| λ (levier) | −0,60 | −0,567 |
| Corrélation cible des chocs de vol | 0,50 | 0,472 |
| ρ' (résolu) | — | 0,760 |

`ρ'` est résolu pour atteindre la corrélation cible **compte tenu de la
corrélation empirique des rendements observés** ; la fonction lève une exception
si la cible est inatteignable, plutôt que de produire une structure de dépendance
incohérente. Le levier réalisé (−0,567) est légèrement atténué par rapport à sa
cible (−0,60) parce que le diagnostic mesure `corr(rendement, Δlog σ)` alors que
la cible porte sur `corr(z, ξ)`.

### 3.4 Application des chocs et garde-fou

```
S_i(s) = 100 × (1 + r_i,s)                ← rendement simple observé
σ_i(s) = σ_i(0) × exp(Δlog σ_i,s)         ← choc multiplicatif
```

`K`, `r` et `T` sont gelés. **Maturité gelée** : à l'horizon d'un jour la
maturité résiduelle n'est pas roulée, donc le P&L ne contient pas de thêta.
C'est la convention de `MarketOptionDataFetcher.option_pnl_scenarios` dans
RiskEngine, reprise ici pour rester comparable.

Enveloppe effectivement atteinte par les 250 scénarios, contre la boîte de la
grille QTT :

| Coordonnée | Étendue des scénarios | Boîte de la grille | Hors boîte |
|---|---|---|---|
| Spot | [90,01 ; 115,51] | [5 ; 150] | **0** |
| Volatilité | [0,211 ; 0,510] | [0,05 ; 0,80] | **0** |

Zéro coordonnée hors boîte : la grille interpole partout, elle n'extrapole
jamais. Sans cette vérification, l'écart mesuré pourrait être une erreur
d'extrapolation déguisée en erreur d'interpolation.

## 4. Calcul de la VaR

`VaRCalculator(pnl, horizon=1)` de RiskEngine, appliqué au P&L de chaque bras :

```
VaR_α = −percentile(PnL, (1−α)·100, method="linear")
ES_α  = −moyenne( PnL | PnL ≤ percentile(PnL, (1−α)·100) )
```

Niveaux rapportés : **VaR 95 %, VaR 99 %, ES 99 %**. Perte comptée positive.
Horizon 1 jour, donc pas de mise à l'échelle en √h.

La VaR Monte Carlo au sens *générateur de scénarios* n'est pas calculée : dans
cette première étape, Monte Carlo n'intervient que comme méthode de
**revalorisation**, en face de la forme fermée et de la grille hybride.

## 5. Reproductibilité

| Élément | Graine / paramètre |
|---|---|
| Chocs de volatilité | `--vol-seed 20260918` |
| Revalorisation Monte Carlo | `--mc-seed 4242` |
| TT-cross | `config.seed = 20260327` |
| Fenêtre de marché | `--history-period 1y` |

Threads BLAS épinglés à 1 avant l'import de numpy : un coût par requête qui
dépend silencieusement du nombre de cœurs n'est pas un nombre reproductible.
Python, numpy, pandas et plateforme sont enregistrés dans le bloc `environment`
du JSON.

> **Limite de reproductibilité à signaler.** Les rendements viennent de yfinance
> avec `history_period="1y"`, donc **la fenêtre glisse avec la date d'exécution**.
> Rejouer le script six mois plus tard ne redonnera pas les mêmes 250 scénarios.
> Pour figer le jeu de données au moment de la soumission, sauvegarder le
> DataFrame de rendements à côté du JSON et le relire plutôt que de le
> re-télécharger.

Dépendance : `market_option_data_fetcher` importe `psycopg2`, qui n'est **pas**
dans les dépendances déclarées de RiskEngine (`METADATA` liste numpy, pandas,
yfinance, matplotlib, scipy, django). Installé ici via `psycopg2-binary` pour
pouvoir interroger la couverture SVI de la section 3.3.

Commande :

```
.venv/Scripts/python.exe scripts/validation/benchmark_var_riskengine.py
```
