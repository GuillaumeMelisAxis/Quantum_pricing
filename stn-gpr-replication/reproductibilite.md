# Informations de reproductibilité

Réponse point par point aux six demandes du relecteur. Chaque valeur a été relue
dans le code du dépôt (commit de travail courant, `teneva 0.14.11`,
`numpy 2.5.2`, `scipy 1.18.0`, Python 3.12.9) ; les rares valeurs reprises du
texte du papier sans vérification possible dans le code sont signalées
**[papier]**, et ce qui manque encore est signalé **[manquant]**.

## 0. Expériences couvertes

| Code | Expérience | Script | Profil |
|------|------------|--------|--------|
| **E-8D** | Put européen sur panier géométrique, grilles G00 / G11 (tableau européen du papier) | `scripts/reproduce_european.py` | `intermediate` |
| **A-8D** | Put américain sur panier arithmétique, robustesse G10 / G11 (tableau américain) | `scripts/validation/validate_american_grid_robustness.py` | `intermediate` |
| **A-scr** | Criblage américain 2×2 à budget 2 000 | `scripts/reproduce_american.py` | **[papier]** : 10 points, LSMC 500 chemins |
| **G-8D** | Grecs (Delta, Gamma, Hessien 5×5) : grille de prix contre grille de risque hybride | `scripts/validation/compare_pricing_risk_grids.py` | `paper` |
| **Q-13D** | Benchmark requête par requête QTT contre MC (figure *accuracy vs time*, E3) | `scripts/validation/benchmark_vol_risk_grids.py` | `intermediate`, run « quiet » |
| **R-13D** | VaR / ES 1 jour d'un livre d'options | `scripts/validation/benchmark_var_qtt_basket_book.py` | `full` |

Correspondance des grilles du papier avec les modes du code :
G00 = `paper`, G01 = `paper_adaptive_maturity`,
G10 = `moneyness_adaptive_uniform_maturity`, G11 = `moneyness_adaptive`.

Paramètres de marché communs aux expériences 8-D : 5 actifs, volatilités toutes
égales à 20 %, corrélation constante 0,30 hors diagonale, dividendes nuls,
panier équipondéré. En 13-D la volatilité de chaque actif devient une
coordonnée de la grille (même corrélation, dividendes nuls). Graine de
configuration : `20260327`. Générateur : `numpy.random.default_rng` (PCG64).

---

## 1. Table des axes

Toutes les grilles sont des produits tensoriels ; chaque nombre de nœuds est une
puissance de 2 et chaque axe est quantisé en `log2(n)` cœurs binaires
(encodage big-endian). Un point de requête hors de la boîte est **ramené sur la
borne** (extrapolation constante), sur chaque axe indépendamment.

| Coordonnée | Bornes | Carte (placement des nœuds) | Nœuds | Interpolation | Expériences |
|------------|--------|-----------------------------|------:|---------------|-------------|
| Spot `S_i` (×5) | [5 ; 150] | uniforme | 32 | linéaire | E-8D, A-8D, A-scr |
| Spot `S_i` (×5) | [5 ; 150] | uniforme | 32 | linéaire, ou cubique pour les Grecs | G-8D, grille de prix |
| Spot `S_i` (×5) | [5 ; 150] | uniforme | 64 | cubique | G-8D, grille de risque |
| Spot `S_i` (×5) | [5 ; 150] | uniforme | 32 | cubique | Q-13D, R-13D |
| Volatilité `σ_i` (×5) | [0,05 ; 0,80], soit `log σ` ∈ [−2,996 ; −0,223] | uniforme en `log σ` | 16 | cubique | Q-13D, R-13D |
| Strike absolu `K` | [1 ; 200] | uniforme | 64 | linéaire | G00, G01 |
| Log-moneyness `m = log(K/G)` | [log(1/150) ; log(200/5)] = [−5,011 ; 3,689] | sinh centré en 0, concentration c = 3 (voir ci-dessous) | 64 | linéaire | G10, G11 (E-8D, A-8D, A-scr) |
| Log-moneyness `m` | idem | sinh, c = 3 | 64 | cubique | G-8D grille de prix ; Q-13D (`price_adaptive`) |
| Coordonnée de risque bornée `ẑ` | [−1 ; 1], extrémités envoyées exactement sur [−5,011 ; 3,689] en `m`, pour tout `(r, T, σ)` | uniforme en `ẑ` | 512 | cubique | G-8D grille de risque |
| Coordonnée de risque bornée `ẑ` | idem | uniforme en `ẑ` | 64 | cubique | R-13D |
| Taux `r` | [0,005 ; 0,08] | uniforme | 8 | linéaire (8-D) ; cubique (13-D) | toutes |
| Maturité `T` | [1/365 ; 3] ans | uniforme | 8 | linéaire | G00, G10 |
| Maturité `T` | [1/365 ; 3] ans | quadratique | 8 | linéaire | G01, G11 ; G-8D grille de prix |
| Maturité `T` | [1/365 ; 3] ans | quadratique | 64 | linéaire | G-8D grille de risque |
| Maturité `T` | [1/365 ; 3] ans | quadratique | 32 | cubique | Q-13D, R-13D |

**Définitions des cartes**

- **sinh** (`sinh_centered_axis`) : avec `u_k` uniforme sur [−1 ; 1],
  `m_k = −|m_min|·sinh(c|u_k|)/sinh(c)` pour `u_k < 0` et
  `m_k = m_max·sinh(c·u_k)/sinh(c)` sinon, avec c = 3. Avec 64 nœuds, le pas va
  de 0,035 au voisinage de la monnaie à 0,458 aux extrémités.
- **quadratique** (`short_maturity_axis`) : `T_k = T_min + (T_max − T_min)·u_k²`,
  avec `u_k` uniforme sur [0 ; 1]. Les 8 nœuds sont 0,0027 ; 0,0639 ; 0,2474 ;
  0,5533 ; 0,9814 ; 1,532 ; 2,205 ; 3,0.
- **Coordonnée de risque bornée** (`BoundedStandardizedRiskTransform`,
  `BoundedVolStandardizedRiskTransform` en 13-D) :
  `z = asinh((m − m_ridge(r,T)) / (σ_B √T))`, avec la crête
  `m_ridge = (carry + σ_B²/2)·T` et `carry = r − w·q − ½ w·σ² + ½ σ_B²`.
  `z` est ensuite normalisé : `ẑ = 2(z − z_lo)/(z_hi − z_lo) − 1`, où
  `[z_lo, z_hi]` est l'image de [m_min ; m_max] à `(r, T, σ)` donnés. En 8-D,
  `σ_B` est la volatilité figée du panier ; en 13-D elle est recalculée sur
  chaque ligne à partir des `σ_i`.
- **Forme physique** : G00–G11 `(32⁵, 64, 8, 8)`, 37 cœurs QTT ; grille de
  risque G-8D `(64⁵, 512, 8, 64)`, 48 cœurs ; Q-13D et R-13D
  `(32, 16, 32, 16, 32, 16, 32, 16, 32, 16, 64, 8, 32)`, 59 cœurs, axes
  entrelacés `(S_1, σ_1, …, S_5, σ_5, coordonnée de strike, r, T)`.

**Cible ajustée.** En 8-D, le TT ajuste le prix brut. En 13-D, il ajuste
`P / G(S)` (prix divisé par le spot du panier), et la prédiction est
remultipliée par le `G(S)` exact du point de requête.

---

## 2. Solveur TT-cross

**Paramètres communs à toutes les expériences**

| Élément | Valeur |
|---------|--------|
| Bibliothèque | `teneva` 0.14.11, fonction `teneva.cross` : TT-cross à rang adaptatif par `maxvol_rect`, balayages alternés gauche-droite puis droite-gauche |
| Critère d'arrêt | **Uniquement le budget d'évaluations `m`.** Aucune tolérance n'est passée (`e = None`, `e_vld = None`). Arrêt de sécurité à `nswp = 20` balayages, jamais atteint. Le diagnostic indique `stop = "m"` dans tous les runs. |
| Croissance du rang | `dr_min = 1`, `dr_max = 2` : au plus 2 lignes ajoutées par liaison et par balayage. Paramètres maxvol : `tau = 1,1`, `tau0 = 1,05`, `k0 = 100`. |
| Rang maximal | **Aucun plafond explicite.** Le rang est borné par le budget et le nombre de balayages. Rangs maximaux observés avant troncature : 32 (Q-13D, 150k) et 35 (R-13D, 300k). Pour E-8D et A-8D, la valeur est dans le JSON (`maximum_rank`), qui n'est pas versionné. **[manquant]** |
| Initialisation | `teneva.anova` à l'ordre 2, convertie en TT de rang 2, à partir de `N_ANOVA` multi-indices tirés uniformément sur la grille, avec la graine TT. Ces évaluations de l'oracle sont comptées dans les « oracle evaluations » publiées. |
| Troncature après ajustement | `teneva.truncate` : arrondi SVD à précision relative `e = 1e-8`, sauf mention contraire ci-dessous. |
| Cache | Dictionnaire d'évaluations (`cache`) : un multi-indice déjà évalué n'est pas recalculé. |

**Paramètres par expérience**

| Expérience | Budgets TT-cross | `N_ANOVA` | Graines TT | Troncature |
|------------|------------------|----------:|------------|------------|
| E-8D | 5 000 ; 10 000 ; 20 000 ; 50 000 | 2 000 | 20260327 | 1e-8 |
| A-8D | 7 500 ; 9 000 ; 12 000 | 1 000 | 20260327 – 20260331 | 1e-8 |
| A-scr | 2 000 | **[papier]** | **[papier]** | 1e-8 |
| G-8D | 200 000 | 2 000 | 20260401 – 20260405 | **aucune** (cœurs bruts) |
| Q-13D | 20 000 ; 60 000 ; 150 000 | 1 000 | 20260327 | 1e-8 |
| R-13D | 50 000 ; 100 000 ; 150 000 ; 200 000 ; 300 000 ; 400 000 | 2 000 | 20260327 ; 7 ; 991 | 1e-8 |

Le nombre total d'évaluations de l'oracle dépasse légèrement « budget +
`N_ANOVA` », parce que `teneva` évalue par lots entiers. Par exemple, au budget
5 000 de E-8D, on compte 6 992 évaluations.

---

## 3. Schéma d'interpolation cubique

**Il ne s'agit ni d'une spline naturelle ni d'une spline not-a-knot.** C'est une
**interpolation de Lagrange locale à quatre points, sur nœuds non uniformes**,
appliquée axe par axe (produit tensoriel) — `QTTGrid.axis_stencils`.

- **Stencil.** Pour un point `x` situé dans la maille `[x_k ; x_{k+1}]`, on
  utilise les nœuds `x_{k−1}, x_k, x_{k+1}, x_{k+2}`. Près d'un bord, ce
  stencil est décalé vers l'intérieur (indice de départ ramené à
  `[0 ; n − 4]`), ce qui le rend unilatéral.
- **Poids.** Polynômes de Lagrange exacts sur les quatre nœuds réels :
  `w_a(x) = ∏_{b≠a} (x − x_b)/(x_a − x_b)`.
- **Régularité.** L'interpolant est **C⁰ mais pas C¹**. À chaque nœud, le
  stencil passe de `[k−2 ; k+1]` à `[k−1 ; k+2]` : les deux polynômes valent
  bien `f(x_k)` en ce point, mais leurs dérivées diffèrent. Entre deux nœuds,
  c'est un polynôme de degré 3, donc sa dérivée seconde est affine par
  morceaux et **discontinue aux nœuds**.
- **Contraction.** Le produit tensoriel est contracté cœur par cœur
  (`predict_factorized`) en `O(Σ k_j q_j)` au lieu des `4^d` coins du stencil.
  Le résultat est identique, à la précision machine près, au stencil explicite
  `predict_hybrid_cubic`.

**Quels axes sont cubiques, par expérience**

| Expérience | Axes cubiques | Axes linéaires |
|------------|---------------|----------------|
| E-8D, A-8D, A-scr | aucun | **tous** (multilinéaire, `TTPriceSurrogate.predict`) |
| G-8D (règle `unified_risk_cubic`) | les 5 spots et la coordonnée de strike / de risque | `r`, `T` |
| Q-13D, R-13D | les 13 axes | aucun |

Les résultats de prix européens et américains du papier reposent donc sur
l'interpolation **multilinéaire**. Le plancher d'interpolation cité dans le
papier (MAE 0,08310) est d'ailleurs le plancher multilinéaire.

**Lien avec les Grecs (G-8D)**

Les Grecs sont des différences finies centrées, à strike fixe, de cette unique
surface interpolée (`finite_difference_fixed_hybrid_component_arrays`) :

- pas relatif `h_i = 0,002 · S_i` (minimum 1e-6) ;
- Delta : `(P(S+h) − P(S−h)) / 2h` ;
- Gamma diagonal : `(P(S+h) − 2P(S) + P(S−h)) / h²` ;
- Gamma croisé : `(P₊₊ − P₊₋ − P₋₊ + P₋₋) / (4 h_i h_j)` ;
- 51 prix par point (1 + 10 + 40). La transformation de coordonnées est
  recalculée après chaque bosse, donc la log-moneyness bouge à `K` fixé.

Conséquence directe sur Γ, que le relecteur a raison de soulever :

- Sur la grille de risque à 64 nœuds de spot, la maille spot vaut
  145/63 ≈ 2,30, alors que `2h ≈ 0,1 à 0,6`. La plupart des stencils de
  différences finies restent dans une seule maille, et Γ y est la dérivée
  seconde d'un polynôme de degré 3.
- Quand le stencil **chevauche un nœud**, Γ combine deux polynômes différents
  et capte la discontinuité de la dérivée première, d'amplitude `O(h_grille³)`,
  divisée par `h²`.
- Une spline C² (naturelle ou not-a-knot) supprimerait ce saut, mais au prix
  d'un support global. Elle n'a pas été testée.

---

## 4. Spécification du LSMC (A-8D, A-scr)

Classe `AmericanArithmeticBasketLSMC`, mode `policy_mode = "refit"` :
estimateur standard, une régression par état.

| Élément | Valeur |
|---------|--------|
| Algorithme | Longstaff–Schwartz, récursion arrière sur les flux de trésorerie |
| Pas de temps | `n_steps` pas égaux, `Δt = T / n_steps` ; incréments log-normaux **exacts** du GBM (pas d'erreur de discrétisation du schéma, seulement celle de la grille d'exercice) |
| Dates d'exercice | pas 1 à `n_steps − 1`, plus la maturité ; à `t = 0`, valeur = `max(exercice immédiat, moyenne actualisée)` |
| Variable d'état | le panier arithmétique `B = moyenne des S_i` (une seule variable, pas les cinq spots) |
| Base de régression | `{1, B/K, (B/K)²}` : polynôme quadratique en moneyness du panier |
| Chemins utilisés | **chemins dans la monnaie uniquement** (`payoff > 0`) ; pas de régression si moins de 3 chemins ITM à cette date |
| Solveur | `numpy.linalg.lstsq` |
| Régression contre évaluation | **même jeu de chemins** (estimateur « in-sample ») |
| Aléa | normales `(n_steps, n_paths, 5)` tirées une fois par instance, corrélées par Cholesky ; **nombres aléatoires communs** à tous les états d'un même appel ; pas de variables antithétiques |
| Étiquettes d'entraînement (A-8D) | 2 000 chemins, 15 pas, graine 20260327 |
| Référence indépendante (A-8D) | 25 000 chemins, 60 pas, graines 901 à 905, prix moyen des 5 graines |
| Points de test (A-8D) | 200 états stratifiés : 5 classes de moneyness `K/B` (0,5–0,8 ; 0,8–0,95 ; 0,95–1,05 ; 1,05–1,2 ; 1,2–2,0) × 4 classes de maturité (≤ 0,25 ; 0,25–1 ; 1–2 ; 2–3 ans), 40 par classe de moneyness |

**Erreur type de la référence 25k**, calculée à partir des 5 graines stockées
dans `results/validation/american_grid_robustness.cache.npz` :

| Quantité | Valeur |
|----------|-------:|
| Erreur type de la moyenne sur 5 graines, en moyenne sur les 200 points | **0,00338** |
| … médiane | 0,00026 |
| … maximum | 0,0307 |
| Écart-type d'une graine à 25k chemins, moyenne | 0,00756 |
| Prix de référence moyen | 9,127 |

Comparée aux erreurs du surrogate dans le tableau américain :

| MAE du surrogate | Erreur type moyenne / MAE |
|------------------|--------------------------:|
| 0,0555 (G11, 9 000, meilleur cas) | **6,1 %** |
| 0,0621 (G11, 7 500) | 5,4 % |
| 0,1177 (G10, 7 500) | 2,9 % |

**Le bruit d'échantillonnage de la référence est donc bien petit, mais
l'affirmation ne couvre pas le biais.** L'erreur type ne mesure que la
dispersion d'une graine à l'autre. Le biais de régression et de discrétisation
de la même référence, mesuré contre un arbre binomial en 1-D, vaut en moyenne
0,01365 **[papier]**, soit **25 %** de la meilleure MAE du surrogate. Il est du
même ordre que l'écart entre l'oracle d'entraînement et la référence
(MAE 0,01344, également **[papier]**). La distribution de l'erreur type est par
ailleurs très asymétrique : la médiane est 13 fois plus petite que la moyenne,
quelques états concentrent l'incertitude.

---

## 5. Réglages du benchmark GPR

Classe `ExactLaplacianGPR` (`src/stngpr/baselines.py`), appelée par
`reproduce_european.py` et `reproduce_american.py`.

| Élément | Valeur |
|---------|--------|
| Implémentation | `sklearn.gaussian_process.GaussianProcessRegressor`, GPR **exacte** (Cholesky en `O(n³)`) |
| Noyau | Laplacien en norme L1 : `k(x,y) = exp(−‖x − y‖₁ / L)`, **une seule longueur de corrélation isotrope `L`**. C'est le noyau du papier STN-GPR ; à une dimension, il coïncide avec Matérn ν = 1/2. |
| Entrées | 8 coordonnées du **même mode de grille** que le TT comparé (strike absolu pour G00, log-moneyness pour G11), mises à l'échelle sur [0 ; 1]⁸ par `MinMaxScaler`, ajusté sur l'ensemble d'entraînement |
| Sorties | `normalize_y = True` |
| Bruit | `alpha = 1e-10` : interpolation sans bruit plus un terme de régularisation numérique. **C'est aussi le cas pour les étiquettes LSMC bruitées de l'américain**, et c'est un choix discutable. |
| Hyperparamètres | `L` estimé par maximum de vraisemblance marginale, L-BFGS-B (`fmin_l_bfgs_b`), départ `L = 1`, bornes [1e-3 ; 1e5], **0 redémarrage** (`n_restarts_optimizer = 0`) |
| Ensemble d'entraînement | points i.i.d. uniformes dans la boîte de marché ; étiquettes en forme fermée (européen) ou par l'oracle LSMC d'entraînement (américain) |
| Tailles d'entraînement | E-8D : 500 ; 1 000 ; 2 000 ; 5 000. Américain (`reproduce_american.py`, profil `intermediate`) : 250 ; 500 ; 1 000 |
| Test | le même ensemble de 2 000 points uniformes que le TT (E-8D) |
| Graine | 20260327 |

**[manquant]**

- Les longueurs `L` ajustées pour les runs du papier sont enregistrées dans le
  champ `kernel` de chaque JSON de résultats. Ces fichiers ne sont pas versionnés
  et ceux des runs E-8D / A-8D ne sont plus présents dans `results/`. À titre
  indicatif, d'autres runs donnent `L` entre 6 et 23 sur [0 ; 1]⁸.
- Le tableau européen du papier ne contient pas de colonne GPR. La comparaison
  annoncée dans l'introduction (« evaluate it against GPR under comparable
  training budgets ») n'a donc pas encore de résultats chiffrés dans le texte.

---

## 6. Pricer Monte Carlo et threading

### 6.1 MC du benchmark requête par requête (Q-13D)

Fonction `basket_put_monte_carlo`, appelée par `benchmark_vol_risk_grids.py`.

| Élément | Valeur |
|---------|--------|
| Schéma | **un seul pas exact** jusqu'à l'échéance : simulation de la valeur terminale des 5 actifs corrélés (Cholesky), payoff `max(K − G_T, 0)` actualisé. Pas de discrétisation temporelle (payoff européen). |
| Volatilités | celles de chaque point de requête (coordonnées 13-D) |
| Chemins | 4 096 ; 16 384 ; 65 536 ; 262 144 (de 2¹² à 2¹⁸) |
| Réduction de variance | **aucune** : ni antithétiques, ni variable de contrôle, ni quasi-Monte Carlo |
| Aléa | un générateur par appel, graine `seed_base + 1 009 × réplication`, 3 réplications ; tirages indépendants d'une requête à l'autre (pas de nombres aléatoires communs) |
| Simulation des 5 actifs | volontairement **générique** : la réduction exacte à une dimension du panier géométrique n'est pas utilisée |
| Chronométrage MC | 8 requêtes, médiane de 5 répétitions après 1 tour d'échauffement |
| Chronométrage QTT | lot de 1 000 points de test, médiane de 5 répétitions après échauffement ; interpolation cubique factorisée sur les 13 axes, lots de 2 048 |
| Points de test | 1 000 points uniformes dans la boîte, `σ` tiré uniformément en `log σ`, prix exacts en forme fermée |

Pour mémoire, le Monte Carlo imbriqué de R-13D est différent : un pas exact,
4 096 chemins (1 024 à 65 536 dans le balayage), **antithétiques et nombres
aléatoires communs** à tous les scénarios, un jeu terminal par maturité
distincte du livre.

### 6.2 Threading et parallélisme

**Machine** : Intel Core i7-13700H, **14 cœurs physiques** (6 Performance avec
hyperthreading et 8 Efficiency), **20 processeurs logiques**, Windows 11, mode
d'alimentation « Utilisation normale ». BLAS : `scipy-openblas` 0.3.34
(`DYNAMIC_ARCH`, `MAX_THREADS = 24`, `NO_AFFINITY`). **Aucune expérience
n'utilise de multiprocessing** (ni `multiprocessing`, ni `joblib`, ni
`concurrent.futures`).

| Expérience | Threads BLAS | Bilan |
|------------|--------------|-------|
| Q-13D (MC et QTT) | 1 : `OMP`, `MKL`, `OPENBLAS`, `NUMEXPR` et `VECLIB_MAXIMUM_THREADS` fixés à 1 avant l'import de numpy | **parallélisme apparié** (mono-thread des deux côtés) |
| R-13D (MC, QTT, forme fermée) | 1, même mécanisme | apparié |
| E-8D (TT et GPR) | **non fixés** : OpenBLAS par défaut, jusqu'à 20 threads pour les appels BLAS/LAPACK (Cholesky de la GPR, QR et SVD de `teneva`, produits matriciels) | apparié **au sein du script**, mais les temps publiés ne sont pas mono-thread |
| A-8D, A-scr, G-8D | non fixés | idem |

Le second facteur, plus important que les threads BLAS, est le **type de cœur**.
Aucune affinité n'est fixée, et l'ordonnanceur Windows place le processus
Python tantôt sur un cœur P, tantôt sur un cœur E. Mesures épinglées sur cette
machine, au même calcul près :

| Charge | Cœur P | Cœur E | Rapport |
|--------|-------:|-------:|--------:|
| Charge fixe QR + produits matriciels | 0,055 s | 0,127 – 0,210 s | 2,3 – 3,8× |
| Ajustement QTT, Q-13D, budget 20k | 4,4 s | 12,9 s | 2,9× |
| Ajustement QTT, Q-13D, budget 150k | 15,9 s | 36,0 s | 2,3× |
| Requête QTT, Q-13D, budget 150k | 3,25e-4 s | 5,96e-4 s | 1,8× |

Les temps de construction publiés pour Q-13D (16 / 22 / 41 s) ont été mesurés
sur des cœurs E. Dans la figure *accuracy vs time*, les points QTT à 20k et 60k
ont un coût par requête de cœur E, et le point à 150k un coût de cœur P. Le
palier apparent de 3,5e-4 s par requête entre 60k et 150k est donc un artefact :
sur un même cœur, ce coût double entre les deux budgets.

**Pour des comparaisons de temps crédibles**, il faudrait :

1. fixer les threads BLAS à 1 dans **tous** les scripts, ou au même nombre de
   threads pour chaque méthode comparée ;
2. épingler le processus sur les cœurs P (processeurs logiques 0 à 11), et
   enregistrer ce choix dans les sorties ;
3. relancer les mesures de temps de Q-13D, R-13D, E-8D et A-8D. Les précisions
   ne changent pas : elles ne dépendent que du budget, de la graine et de la
   grille, et les rangs obtenus sont identiques sur les deux types de cœur.

---

## 7. Ce qui reste à compléter

| Élément | Où le trouver ou comment le produire |
|---------|--------------------------------------|
| Rangs maximaux de E-8D et A-8D | relancer les scripts, champ `maximum_rank` du JSON |
| Longueurs `L` ajustées de la GPR | relancer `reproduce_european.py --profile intermediate`, champ `kernel` |
| Résultats GPR chiffrés dans le papier | idem ; aucune colonne GPR dans les tableaux actuels |
| Réglages exacts du criblage A-scr | non retrouvés dans un profil du code ; repris du papier (10 points, 500 chemins) |
| Temps mono-thread sur cœurs P | correction proposée en 6.2, puis nouveaux runs |
