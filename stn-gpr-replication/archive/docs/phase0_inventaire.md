# Phase 0 — Inventaire du workspace

Racine de projet : `stn-gpr-replication/` (autonome : `pyproject.toml`, `.venv`,
`src/stngpr/`, `tests/`). Interpréteur : `.venv/Scripts/python.exe`.

État au moment de l'inventaire :

| Élément | Valeur |
|---|---|
| Commit | `9c9f1fded4d284d8a981f56209646c8e3e5bd2f5` (branche `adding_vol`) |
| Arbre de travail | propre (`git diff --stat` vide) |
| Tests | `104 passed, 5 subtests passed` en 19 s |
| Python | 3.12.9 |
| numpy / scipy / teneva | 2.5.2 / 1.18.0 / **0.14.11** |
| Plateforme | Windows-11-10.0.26200, Intel64 Family 6 Model 186 (Raptor Lake-P), 20 CPU logiques |

Aucun calcul de production n'a été lancé. Aucun fichier source n'a été modifié.

---

## 1. Points d'entrée pour construire une grille

### 1.1 Grille 8D (volatilité gelée)

`stngpr.risk_grids.build_greek_coordinate_grid(config, mode, monitor_maturities=None, monitor_rate=0.03)`
→ `(QTTGrid, transform, description)`. Fichier : [risk_grids.py:292](src/stngpr/risk_grids.py#L292).

- `config` : `stngpr.config.PaperConfig` (dataclass **gelée**, donc `dataclasses.replace`
  pour changer `physical_shape`).
- `mode ∈ TENSOR_COMPATIBLE_GREEK_GRID_MODES = ("m_uniform", "price_adaptive",
  "gamma_monitor", "standardized_risk", "bounded_standardized_risk")`.
- Les axes sont construits ainsi, sans paramètre libre :
  - spots : `np.linspace(*spot_bounds, physical_shape[j])`
  - coordonnée strike : dépend du mode ; `bounded_standardized_risk` → `np.linspace(-1, 1, physical_shape[n_assets])`
  - taux : `np.linspace(*rate_bounds, physical_shape[-2])`
  - maturité : `short_maturity_axis(*maturity_bounds, physical_shape[-1], power=2.0)`
- `PaperConfig` — valeurs par défaut : `n_assets=5`, `volatilities=0.20×5`,
  `dividends=0`, `correlation=equicorrelation(5, 0.30)`, `spot_bounds=(5,150)`,
  `strike_bounds=(1,200)`, `rate_bounds=(0.005,0.08)`, `maturity_bounds=(1/365,3)`,
  `physical_shape=(32,32,32,32,32,64,8,8)`, `seed=20260327`.
  `physical_shape` est validée : longueur `n_assets+3`, chaque entrée puissance de 2.

### 1.2 Grille 13D (volatilité coordonnée)

`stngpr.vol_extension.build_vol_grid(config, mode, **kwargs)` dispatche
([vol_extension.py:648](src/stngpr/vol_extension.py#L648)) vers :

- `build_vol_extended_risk_grid(config, mode, basket_kind="geometric",
  volatility_scale="log", monitor_maturities=None, monitor_rate=0.03,
  monitor_volatilities=None)` pour les familles « risque »
  ([vol_extension.py:495](src/stngpr/vol_extension.py#L495)) ;
- `build_vol_extended_grid(config, mode, basket_kind="geometric",
  volatility_scale="log")` pour les familles « pricing » `GRID_MODES = ("paper",
  "paper_adaptive_maturity", "moneyness_uniform",
  "moneyness_adaptive_uniform_maturity", "moneyness_adaptive")`.

`config` : `stngpr.config.VolatilityExtendedConfig` (gelée). Par défaut :
`n_assets=5`, `layout="interleaved"`, `dividends=0`,
`correlation=equicorrelation(5,0.30)`, `spot_bounds=(5,150)`,
**`volatility_bounds=(0.05,0.80)`**, `strike_bounds=(1,200)`,
`rate_bounds=(0.005,0.08)`, `maturity_bounds=(1/365,3)`, `spot_nodes=32`,
**`volatility_nodes=16`**, `strike_nodes=64`, `rate_nodes=8`, `maturity_nodes=8`,
`seed=20260327`.

`volatility_scale="linear"` existe et donne des **nœuds uniformes en σ** (et non en
`log σ`) — voir `market_axes` ([vol_extension.py:106](src/stngpr/vol_extension.py#L106)) :
`np.linspace(*volatility_bounds, volatility_nodes)`. C'est ce que demande la Phase 1.

### 1.3 Correspondance avec les deux grilles physiques nommées

`scripts/validation/compare_pricing_risk_grids.py:66` fixe explicitement les deux
grilles de production, et c'est la seule source de vérité sur leurs noms :

```python
PRODUCTION_GRIDS = {
    "pricing_grid":     {"coordinate_mode": "price_adaptive",
                         "physical_shape": (32,32,32,32,32,64,8,8)},
    "risk_hybrid_grid": {"coordinate_mode": "bounded_standardized_risk",
                         "physical_shape": (64,64,64,64,64,512,8,64)},
}
```

Vérifié : `build_greek_coordinate_grid(replace(PaperConfig(),
physical_shape=(64,64,64,64,64,512,8,64)), "bounded_standardized_risk")` donne
`bits = (6,6,6,6,6,9,3,6)`, soit **48 cœurs QTT**. Conforme aux paramètres gelés.

---

## 2. Point d'entrée TT-cross

`stngpr.tt_surrogate.TTPriceSurrogate.fit` ([tt_surrogate.py:124](src/stngpr/tt_surrogate.py#L124)) :

```python
fit(max_evals, anova_samples=2_000, max_sweeps=20,
    rank_increment=2, truncation=1e-8, log=True) -> FitDiagnostics
```

Mapping vers `teneva.cross` :

| Spécification | Paramètre | Valeur par défaut (= « inchangé ») |
|---|---|---|
| Budget | `m=int(max_evals)` | — (argument obligatoire) |
| Initialisation ANOVA | `teneva.anova(i_anova, y_anova, r=2, order=2, seed=self.seed)` sur `anova_samples` tirages | `anova_samples=2000`, `r=2`, `order=2` |
| Sweeps max | `nswp=int(max_sweeps)` | `20` |
| Incrément de rang | `dr_min=1`, `dr_max=int(rank_increment)` | `rank_increment=2` |
| Tolérance | `e` et `e_vld` **jamais passés** | absente → `stop="m"` systématique |
| Seed | `np.random.default_rng(seed)` pour les tirages ANOVA + `seed=` de `teneva.anova` | `TTPriceSurrogate(grid, pricer, seed=0)` |
| Troncature SVD post-fit | `truncation` → `teneva.truncate` | **`1e-8`** ; `truncation=None` la désactive |
| Autres (`teneva.cross` par défaut) | `tau=1.1`, `tau0=1.05`, `k0=100`, `m_cache_scale=5`, `cache={}` activé | |

Chemin de haut niveau : `VolExtendedPriceSurrogate(config, mode, basket_kind,
volatility_scale, target="auto", cubic_axes=("all",), market_pricer=None,
seed=None, **grid_kwargs)` puis `.fit(budget, anova_samples=..., **kwargs)`
(les kwargs remontent inchangés à `TTPriceSurrogate.fit`), puis `.price(market_points)`.

Deux points à retenir pour le protocole :

- **Il faut passer explicitement `truncation=None`** ; le défaut tronque à `1e-8`.
  `scripts/validation/compare_pricing_risk_grids.py:126` le fait déjà et utilise
  déjà les seeds `20260401…20260405` — c'est le précédent le plus proche du protocole demandé.
- L'absence de tolérance signifie que le cross consomme **tout** le budget
  (gotcha 9 du `CLAUDE.md`), ce qui est cohérent avec « aucun réglage a posteriori ».

---

## 3. Nombre réel d'appels oracle — le compteur existe

`FitDiagnostics.function_evaluations = int(info["m"]) + int(anova_samples)`
([tt_surrogate.py:162](src/stngpr/tt_surrogate.py#L162)).

`info["m"]` provient de `teneva.cross`. Dans `_func_eval`
(`.venv/Lib/site-packages/teneva/cross.py:229`), avec `cache` non nul (c'est le cas :
`cache={}` est passé) :

```python
I_new = np.array([i for i in I if tuple(i) not in cache])
...
info['m']       += len(I_new)
info['m_cache'] += len(I) - len(I_new)
```

Donc `info["m"]` compte **uniquement les indices réellement évalués**, pas les
relectures de cache, et pas le budget nominal. C'est bien un nombre d'appels réalisés.

Trois nuances à documenter dans le manifeste :

1. Les `anova_samples` tirages sont évalués **hors cache** et **avant** le cross ;
   ils ne comptent pas dans le budget `m` de `teneva`. Avec le défaut de 2 000,
   un budget nominal de 20 000 donne ≈ 22 000 appels réalisés. L'abscisse de la
   Figure 1 doit donc bien être `function_evaluations`, pas le budget.
2. Les indices ANOVA ne sont pas injectés dans le cache du cross : un nœud tiré dans
   les deux phases est payé deux fois. C'est le compte honnête d'appels.
3. `teneva` s'arrête *avant* de dépasser `m` (`if info['m'] + len(I_new) > m_max`),
   donc `info["m"] ≤ budget` strictement, et `stop="m"`. `stop="conv"` reste
   possible si `m_cache > 5 · m`.

---

## 4. Rangs de lien cœur par cœur — exposés, mais pas par `FitDiagnostics`

`FitDiagnostics` ne porte que des agrégats (`effective_rank`, `maximum_rank`,
`parameter_count`, et leurs variantes `pre_truncation_*`).

Le profil complet est dans `CoreDiagnostics`
([tt_surrogate.py:27](src/stngpr/tt_surrogate.py#L27)) :

```python
bond_ranks  = [int(core.shape[-1]) for core in cores[:-1]]   # r_1 … r_{d-1}
core_shapes = [[r_left, 2, r_right] for core in cores]
```

Obtention après un fit : `model.surrogate.apply_truncation(None)` renvoie un
`CoreDiagnostics` sur les cœurs bruts (et remet `self.cores` sur les cœurs non
tronqués, ce qui est exactement ce que veut le protocole). Précédent :
`compare_pricing_risk_grids.py:136` sérialise `asdict(model.apply_truncation(None))`.

`bond_ranks` omet `r_0 = r_d = 1` ; il faut les rajouter pour obtenir `r_0 … r_d`.

- **Rang effectif** (définition de la base) = `teneva.erank`
  (`.venv/Lib/site-packages/teneva/props.py:10`) : le rang constant `r` qui donne le
  même nombre de paramètres que la décomposition réelle.
- **Paramètres TT** = `sum(core.size)` = `Σ_k n_k · r_{k-1} · r_k` sur les cœurs
  binaires (`n_k = 2`). Coïncide exactement avec la définition du protocole.
- **Rangs internes au bloc σ / lien frontière** : immédiats à extraire de
  `bond_ranks` une fois connue la position des cœurs σ dans la chaîne — voir le
  manque bloquant B2.

---

## 5. Oracle analytique du panier géométrique

`stngpr.pricers.geometric_basket_put(spots, strikes, rates, maturities,
volatilities, correlation, dividends=None)` ([pricers.py:8](src/stngpr/pricers.py#L8)).

**Il accepte un vecteur de volatilités hétérogène.** Code :

```python
sigma = np.atleast_2d(np.asarray(volatilities, dtype=float))
if sigma.shape[1] != n_assets or sigma.shape[0] not in (1, m):
    raise ValueError("volatilities must be one vector or one vector per row")
```

soit un vecteur partagé `(1, n)`, soit **un vecteur par ligne `(m, n)`**. La forme
quadratique `σ_G² = wᵗ(σσᵗ∘C)w` et l'ajustement de drift `Σ w_i σ_i²` sont calculés
par ligne, sans jamais construire de tableau `(m,n,n)`. Déterministe, fermé
(`scipy.special.ndtr`).

Deux enveloppes :

- 8D : `EuropeanGeometricBasketPricer(config)` — lit `config.volatilities` (vecteur gelé).
- 13D : `EuropeanGeometricBasketVolPricer(config)` — lit les colonnes σ de la ligne.

---

## 6. Interpolant de production

Chemin de production : `TTPriceSurrogate.predict_factorized(points,
cubic_columns=(), batch_size=2048)` ([tt_surrogate.py:218](src/stngpr/tt_surrogate.py#L218)),
utilisé par `VolExtendedPriceSurrogate.price`.

- **Ordre par axe** : décidé par `cubic_columns`. Un axe hors de la liste est
  **linéaire** (2 nœuds) ; un axe dans la liste est **cubique** (Lagrange à 4 nœuds,
  stencil `clip(lower-1, 0, n-4) + arange(4)`) — construction dans
  `QTTGrid.axis_stencils` ([grids.py:130](src/stngpr/grids.py#L130)).
- **Défaut de production** : `VolExtendedPriceSurrogate(cubic_axes=("all",))`
  → cubique **sur tous les axes**.
- **Ajout d'axes** : `resolve_cubic_columns(config, names)`
  ([vol_extension.py:761](src/stngpr/vol_extension.py#L761)), noms acceptés
  `CUBIC_AXIS_NAMES = ("none", "all", "spots", "vols", "m", "r", "T")`. Les groupes
  `spots` / `vols` se résolvent via `config.spot_columns` / `config.volatility_columns`,
  donc l'interpolant suit automatiquement le `layout`.
- Coût : `O(Σ_j k_j q_j)` au lieu de `O(Π_j k_j)` — un axe cubique de plus coûte une
  paire de chaînes sur ce seul axe.

**Plancher d'interpolation** : `VolExtendedPriceSurrogate.price_interpolation_floor(
market_points, cubic_columns=(), batch_size=8, max_stencil=1<<20)`
([vol_extension.py:906](src/stngpr/vol_extension.py#L906)) — même interpolant, même
grille, valeurs exactes de l'oracle aux nœuds du stencil, aucune compression TT.
Pour le 8D il existe `coordinates.oracle_hybrid_cubic_predict` et
`coordinates.CachedGridInterpolator`. Voir le manque bloquant B4 : cet appel
**énumère le stencil produit** et ne peut pas être cubique sur 13 axes.

**NMAE** : `stngpr.validation.error_metrics(y_true, y_pred)["normalized_mae"]`
([validation.py:44](src/stngpr/validation.py#L44)) = `mean|pred − true| / mean|true|`,
`None` si `mean|true| ≤ 1e-14`. Définition utilisée telle quelle, non redéfinie.

**Horloge** : `time.perf_counter` partout (`tt_surrogate.py`, `risk.py`, scripts) —
monotone. Conforme.

---

## 7. Coïncidence des nœuds (Phase 2) — outillage déjà présent

Tout existe :

- `grid_geometry.physical_log_moneyness_nodes(grid, transform, n_assets, rate,
  maturity, reference_spot=100.0)` — axe coordonnée 8D → `m = log(K/G)`
  ([grid_geometry.py:6](src/stngpr/grid_geometry.py#L6)).
- `vol_extension.strike_axis_log_moneyness_nodes(config, grid, transform, rate,
  maturity, volatility=None, reference_spot=100.0)` — même chose en 13D, avec un
  argument `volatility` scalaire (à passer `σ_ref`)
  ([vol_extension.py:663](src/stngpr/vol_extension.py#L663)).
- `grid_geometry.normalized_node_displacement(reference_nodes, candidate_nodes,
  domain=None)` — renvoie `normalized_l1 / normalized_l2 / normalized_linf` et leurs
  variantes absolues ([grid_geometry.py:39](src/stngpr/grid_geometry.py#L39)).
  `normalized_l2` = `D_2`, `normalized_linf` = `D_∞`.

Lecture du code, sans exécution : le critère devrait passer. `StandardizedRiskTransform._ridge`
(8D, [risk_grids.py:179](src/stngpr/risk_grids.py#L179)) et
`VolStandardizedRiskTransform._ridge_and_scale` (13D,
[vol_extension.py:268](src/stngpr/vol_extension.py#L268)) implémentent la **même**
formule (`carry = r − w·q − 0.5·w·σ² + 0.5·σ_G²`, `ridge = (carry + 0.5·σ_G²)·T`,
`scale = σ_G·√T`), et `_basket_log_moneyness_bounds` est la même fonction appelée sur
les deux configs, qui portent les mêmes `spot_bounds` et `strike_bounds`. Les axes
`r` et `T` sont construits par les mêmes expressions. À nombres de nœuds appariés,
l'écart attendu est de l'ordre de l'arrondi flottant. **Ce n'est pas une mesure :
c'est une lecture. La Phase 2 reste à exécuter.**

---

## 8. L'espace 13D avec axes de volatilité est-il implémenté ?

**Partiellement.** Ce qui est là :

- 13 dimensions = `2·n_assets + 3` avec **une σ par actif**, sur les deux familles de
  grilles (pricing et risque), en échelle σ linéaire ou logarithmique.
- Les transformations σ-conscientes : `VolStandardizedRiskTransform` et
  `BoundedVolStandardizedRiskTransform` recalculent `σ_G(σ)` et le ridge par ligne.
  Elles prennent les indices de colonnes en argument explicite, donc elles sont
  **agnostiques au layout** — c'est ce qui rend les manques ci-dessous réparables.
- `VolExtendedPriceSurrogate` fait la chaîne complète : grille, transformation,
  cible `P/G`, TT-cross, prédiction cubique factorisée, plancher d'interpolation.
- La grille de contrôle de la Phase 6b (`32^5 × 64 × 8 × 8` + axes σ, mode
  `price_adaptive`) se construit directement.

Ce qui n'est pas là : voir la section suivante.

---

# Liste de manques bloquants

## B1 — `d_σ` n'est pas un paramètre (bloque la Phase 4)

`VolatilityExtendedConfig` impose `n_dimensions = 2·n_assets + 3` avec exactement
une coordonnée σ par actif : `volatility_columns` est dérivé de `n_assets`, pas d'un
compte indépendant. Recherche exhaustive (`d_sigma`, `n_vol`, `vol_axes`,
`n_volatility`, `promoted`, `frozen_vol`, `sigma_ref`) : **aucune occurrence** dans
`src`, `scripts`, `tests`.

Conséquence : `d_σ ∈ {1, 2, 3}` n'est pas construisible. `d_σ = 0` existe mais sous
une **autre classe de config** (`PaperConfig`) et un **autre constructeur de grille**
(`build_greek_coordinate_grid`), pas comme cas limite du chemin 13D.

Il faut un mécanisme unique : `d_σ` axes σ promus (les `d_σ` premiers actifs, ordre
documenté), les `n − d_σ` autres volatilités gelées à `σ_ref` et injectées par
l'oracle, `d_σ = 0` et `d_σ = 5` sortant du **même** constructeur.

## B2 — Aucun layout ne place le bloc σ en fin de chaîne (bloque la Phase 4 et la Figure 2)

Vérifié par construction :

| `layout` | `physical_shape` (n_σ = 16, grille risk-hybrid) | colonnes σ |
|---|---|---|
| `"blocked"` | `(64,64,64,64,64, 16,16,16,16,16, 512,8,64)` | 5–9 |
| `"interleaved"` | `(64,16,64,16,64,16,64,16,64,16, 512,8,64)` | 1,3,5,7,9 |

Dans les deux cas le bloc σ est **avant** `(K, r, T)`. Le protocole exige le bloc σ
**en fin de chaîne**, de sorte que la chaîne `d_σ = 0` soit exactement la chaîne
`d_σ = 5` privée des cœurs σ. Avec `blocked`, les 18 cœurs de `(K, r, T)` sont
déplacés de 20 positions (5 × log2 16) : la vérification explicite demandée
échouerait, et l'alignement « sur les cœurs partagés » de la Figure 2 n'a pas de sens.

Il faut un layout `(S_1…S_5, K, r, T, σ_1…σ_{d_σ})`. `strike_column`, `rate_column`,
`maturity_column` sont actuellement câblés sur `2·n_assets{,+1,+2}` — c'est le point
à changer. La Phase 6a a besoin en plus d'un layout σ-en-tête ; `interleaved` couvre
la troisième permutation.

## B3 — La cible de fit diffère entre 8D et 13D (bloque la comparabilité de la Phase 4)

- Chemin 8D (`compare_pricing_risk_grids.py:524`) : `TransformedPricer(market_pricer,
  transform)` → le cross apprend le **prix brut `P`**.
- Chemin 13D (`VolExtendedPriceSurrogate`, `target="auto"`) : tout mode hors
  `PAPER_STRIKE_MODES` passe par `BasketScaledModelPricer` → le cross apprend **`P/G(S)`**.

Le protocole n'autorise que `d_σ` et le budget à varier. Ici la cible varierait
aussi, et c'est précisément le levier qui fait passer la MAE 13D de ≈ 7 à ≈ 0.1
(gotcha 1 du `CLAUDE.md`) : la comparaison mélangerait l'effet de dimension et
l'effet de cible.

`BasketScaledModelPricer(model_pricer, spot_columns, basket_kind)` est générique et
s'applique tel quel au chemin 8D (`spot_columns = range(n_assets)`). **Décision
requise** : unifier sur `P/G` pour toutes les configurations (recommandé — c'est
obligatoire à 13D), ou sur `P` (impraticable à 13D).

## B4 — Le plancher d'interpolation de production n'est pas calculable en l'état (bloque la Phase 1)

`price_interpolation_floor` appelle `oracle_hybrid_cubic_predict`, qui énumère le
**stencil produit** via `QTTGrid.hybrid_cubic_stencil`. L'interpolant de production
est cubique sur tous les axes, donc 4 nœuds par axe :

| Configuration | Coins par point de test | Statut |
|---|---|---|
| 8D, cubique tous axes | `4^8` = 65 536 | faisable |
| 13D, cubique tous axes | `4^13` = 67 108 864 | **refusé** par `max_stencil = 2^20` |

C'est le gotcha 8 du `CLAUDE.md`, et `benchmark_vol_risk_grids.py` le contourne en
rapportant une échelle de planchers partiellement cubiques — ce que le protocole
interdit explicitement (« plancher d'interpolation … en appliquant l'interpolant de
production »).

Il existe une réduction **exacte**, pas une approximation : pour un panier
géométrique avec cible `P/G` sur une grille en moneyness ou en coordonnée
standardisée, `P/G` est homogène de degré 0 en `(G, K)` et `m` ne dépend pas des
spots ; la valeur à un nœud du stencil est donc **identique pour les `4^5`
combinaisons de nœuds spot**, et les poids de Lagrange somment exactement à 1 par
axe. Le stencil effectif se réduit aux axes non-spot :

| Configuration | Coins effectifs après réduction |
|---|---|
| 8D (`m, r, T`) | `4^3` = 64 |
| 13D (`σ_1…σ_5, m, r, T`) | `4^8` = 65 536 |

À 2 000 points de test cela fait ≈ 1.31 × 10⁸ évaluations vectorisées de la formule
fermée par candidat `n_σ`, avec batching obligatoire côté mémoire. Faisable, mais
**cela demande du code nouveau** (chemin de plancher réduit, plus un test d'égalité
contre le chemin produit sur une grille jouet où `4^d` est énumérable).
**Décision requise** avant la Phase 1.

## B5 — Placement de `σ_ref` dans la maille : la spécification est en tension avec elle-même (bloque la Phase 1)

Sur `σ ∈ [0.05, 0.60]`, nœuds uniformes, `σ_ref = 0.20` :

| `n_σ` | pas `h` | nœuds encadrants | position fractionnaire de `σ_ref` dans sa maille |
|---|---|---|---|
| 4 | 0.183333 | 0.050000 / 0.233333 | 0.8182 |
| 8 | 0.078571 | 0.128571 / 0.207143 | 0.9091 |
| 16 | 0.036667 | **0.196667** / 0.233333 | **0.0909** |
| 32 | 0.017742 | 0.191935 / 0.209677 | 0.4545 |

Aucun `n_σ` ne place un nœud **exactement** sur 0.20, donc la clause « si la
discrétisation naturelle place un nœud sur 0.20, décale les bornes » ne se déclenche
pas. Mais `n_σ = 16` met un nœud à 3.3 × 10⁻³ de `σ_ref`, soit 9 % de la maille :
la MAE 13D sur la tranche commune y serait quasi exempte d'erreur d'interpolation
σ-directionnelle, exactement le biais que la règle veut éviter. Et `n_σ = 8` est
symétriquement proche de l'autre nœud (9 % de la maille, de l'autre côté).

Ces fractions sont `9/11, 10/11, 1/11, 5/11` — conséquence de
`(0.20 − 0.05)/(0.60 − 0.05) = 3/11`. **Aucun décalage marginal des bornes ne les
rapproche simultanément de 0.5** : les fractions bougent comme `3δ, 7δ, 15δ, 31δ`.
La seule valeur qui met `σ_ref` au milieu d'une maille pour les quatre candidats à
la fois est `(σ_ref − a)/(b − a) = 1/2`, c'est-à-dire un domaine **symétrique autour
de 0.20** — soit `[0.05, 0.35]`, ce qui n'est pas un décalage marginal et contredit
le domaine `[0.05, 0.60]` donné dans le protocole.

Trois issues possibles, aucune que je choisirai seul :

1. garder `[0.05, 0.60]` et accepter les fractions ci-dessus, en documentant que
   `n_σ ∈ {8, 16}` est favorable au 13D (la mesure « hors nœud » à `σ = 0.235` reste
   le garde-fou) ;
2. passer à `[0.05, 0.35]` (`σ_ref` exactement au milieu de la maille centrale pour
   les quatre candidats), et documenter que le domaine σ est plus étroit que prévu ;
3. décaler les bornes par candidat `n_σ` — mais le domaine devient alors une variable
   supplémentaire entre configurations, ce que le protocole interdit.

## B6 — Points mineurs, non bloquants, à figer dans le manifeste

- `volatility_bounds` par défaut est `(0.05, 0.80)`, non `(0.05, 0.60)` : à passer
  explicitement à la construction.
- `volatility_scale` par défaut est `"log"` : la Phase 1 demande des nœuds uniformes
  en σ, donc `volatility_scale="linear"` explicite partout.
- `truncation=None` explicite à chaque `fit` (le défaut est `1e-8`).
- `maturity_nodes=64` et `strike_nodes=512` doivent être passés explicitement
  (défauts 8 et 64). Le gotcha 6 du `CLAUDE.md` dit que 8 nœuds de maturité font
  stagner `bounded_standardized_risk` à sa propre limite d'interpolation ; 64 est
  bien le bon choix pour la grille risk-hybrid.
- `n_σ ∈ {4,8,16,32}` satisfait la contrainte « puissance de 2 » de
  `physical_shape`, et 4 nœuds sont le minimum pour l'interpolation cubique
  (`axis_stencils` lève `ValueError` en dessous).
- Nombre de cœurs QTT : `48 + d_σ · log2(n_σ)`. Pour `d_σ = 5`, `n_σ = 32` → 73.
  Entrées logiques : `2^(nombre de cœurs)`.
- Aucun utilitaire n'existe pour le **temps moyen par requête apparié sur 150 points
  identiques entre configurations** ; `benchmark_vol_risk_grids.py` a des helpers de
  chronométrage réutilisables. À définir : « mêmes vecteurs de paramètres » = mêmes
  valeurs sur les 8 axes partagés, les entrées σ non promues étant remplies à `σ_ref`.
- `results/*.json` est gitignoré ; les livrables CSV/PDF/SVG demandés devront aller
  dans un répertoire versionné ou explicitement listé.

---

# Résumé : ce que j'attends avant la Phase 1

1. **B3** — cible de fit unifiée : `P/G` pour toutes les configurations, y compris
   `d_σ = 0` ? (recommandé)
2. **B4** — autorisation d'implémenter le plancher d'interpolation réduit (exact, par
   indépendance aux spots de `P/G`), avec test d'équivalence contre le chemin produit
   sur grille jouet ?
3. **B5** — laquelle des trois issues pour le domaine σ et le placement de `σ_ref` ?
4. **B1 + B2** — confirmation que je peux ajouter au code source le compte `d_σ`
   indépendant et les layouts σ-en-queue / σ-en-tête. Ces deux ajouts touchent
   `config.py` et `vol_extension.py` ; sans eux, ni la Phase 4 ni la Phase 6a ne sont
   exécutables telles que spécifiées.

Rien n'est codé ni lancé avant ta réponse.
