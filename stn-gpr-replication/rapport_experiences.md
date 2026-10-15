# Rapport d'experiences — cout tensoriel en fonction du nombre de facteurs de volatilite

Genere par `scripts/experiments/build_report.py` a partir des seuls artefacts de `experiments/`. Aucun chiffre n'est saisi a la main. Une mesure absente s'ecrit `NOT RUN`.

## 1. Manifeste de run

### Environnement

| item | value |
|---|---|
| commit | `f39cfe54238ddb4833a91a875ef4ee1db14e2c46` |
| branch | `dimensionality_experiments` |
| working tree | dirty (6 files) |
| python | 3.12.9 |
| numpy | 2.5.2 |
| scipy | 1.18.0 |
| teneva | 0.14.11 |
| platform | Windows-11-10.0.26200-SP0 |
| processor | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel |
| clock | `time.perf_counter` (monotone) |

### Paramètres gelés

| parameter | value |
|---|---|
| grille principale | risk-hybrid `bounded_standardized_risk`, `64^5 x 512 x 8 x 64` (48 coeurs QTT) |
| grille de controle (6b) | `price_adaptive`, `32^5 x 64 x 8 x 8` |
| oracle | formule fermee du panier geometrique (deterministe) |
| correlations | equicorrelation(5, 0.30) |
| dividendes | zeros(5) |
| cible de fit | P / G(S) in every configuration |
| interpolant | mixed-order unified interpolant, cubic on every axis |
| troncature SVD post-fit | aucune (refusee par l'API) |
| init ANOVA | `teneva.anova(r=2, order=2)`, 2000 tirages |
| sweeps max | 20 |
| increment de rang | `dr_min=1`, `dr_max=2` |
| tolerance TT-cross | aucune (`e` / `e_vld` non passes) |
| sigma_ref | 0.2 |
| domaine sigma | `[0.05, 0.6]` |
| sigma hors noeud | 0.235 |
| axes sigma | raw sigma, uniform nodes |
| actifs promus | the first d_sigma assets, fixed order |
| d_sigma | `[0, 1, 2, 3, 5]` |
| budgets TT-cross | `[20000, 60000, 150000, 300000]` |
| seeds | `[20260401, 20260402, 20260403, 20260404, 20260405]` |
| seed jeu de test (axes partages) | 20260327 |
| seed jeu de test (boite sigma) | 20260328 |
| points de test | 2000 |
| points de latence apparies | 150 |
| seuil UNRESOLVED | NMAE > 3 x plancher |
| n_sigma retenu | 16 |

### Tests bloquants

**Phase 1** — critere : plus petit `n_sigma` dont le plancher 13D ne depasse pas le plancher 8D de plus de 10 %. Retenu : **n_sigma = 16**.

**Phase 2** — Over 81 (r, T) regimes and every d_sigma in [0, 1, 2, 3, 5], the sigma-augmented grid evaluated at sigma = 0.2 places its log-moneyness nodes at max D_2 = 0.000e+00 and max D_inf = 0.000e+00 from the frozen-volatility production grid: PASS against the 1e-12 machine-precision gate.

## 2. Tableaux

### Phase 1 — choix de `n_sigma`

| n_sigma | L_sigma | coeurs QTT | plancher 13D | plancher 8D | ratio | critere | retenu | sigma_ref dans la maille |
|---|---|---|---|---|---|---|---|---|
| 4 | 2 | 58 | 1.3911e-04 | 2.3484e-06 | 59.2393 | FAIL | non | 0.8182 |
| 8 | 3 | 63 | 4.6464e-06 | 2.3484e-06 | 1.9786 | FAIL | non | 0.9091 |
| 16 | 4 | 68 | 2.3995e-06 | 2.3484e-06 | 1.0218 | PASS | **oui** | 0.0909 |
| 32 | 5 | 73 | 2.3336e-06 | 2.3484e-06 | 0.9937 | PASS | non | 0.4545 |

Plancher calcule par le chemin a stencil spot reduit : 64 coins effectifs contre 65536 pour le stencil produit, residu de collapse 1.68e-15 (norme NMAE, seuil 1e-12).

**Justification.** Les 2 candidats plus grossiers echouent nettement — ratio 59.24, 1.98 pour n_sigma = 4, 8 — donc les axes de volatilite relevent bel et bien le plancher d'interpolation de la grille en dessous de 16 noeuds. A n_sigma = 16 le plancher 13D vaut 2.399e-06 contre 2.348e-06 pour la grille 8D sur la tranche commune, soit un ratio de 1.0218, dans la tolerance de 10 %. Le critere retenant le plus petit candidat conforme, n_sigma = 16 est fige pour toutes les phases suivantes, y compris son placement defavorable de sigma_ref dans la maille (0.0909 de la maille, soit a 3.33e-03 du noeud inferieur), que le balayage de maille de la phase 4 quantifie.

### Phase 2 — coincidence des noeuds

9 taux x 9 maturites, sigma = 0.2, seuil `1e-12`.

| d_sigma | max D_2 | max D_inf | verdict |
|---|---|---|---|
| 0 | 0.000e+00 | 0.000e+00 | PASS |
| 1 | 0.000e+00 | 0.000e+00 | PASS |
| 2 | 0.000e+00 | 0.000e+00 | PASS |
| 3 | 0.000e+00 | 0.000e+00 | PASS |
| 5 | 0.000e+00 | 0.000e+00 | PASS |

Over 81 (r, T) regimes and every d_sigma in [0, 1, 2, 3, 5], the sigma-augmented grid evaluated at sigma = 0.2 places its log-moneyness nodes at max D_2 = 0.000e+00 and max D_inf = 0.000e+00 from the frozen-volatility production grid: PASS against the 1e-12 machine-precision gate.

### Phase 3 — dry run

Chiffres jetes : ce run ne sert qu'a verifier que la chaine s'execute de bout en bout et que tous les compteurs sont collectes.

| d_sigma | coeurs | appels realises | stop | rang effectif | parametres TT | fit (s) |
|---|---|---|---|---|---|---|
| 0 | 48 | 2492 | m | 5.33 | 2634 | 5.3 |
| 5 | 68 | 2489 | m | 4.68 | 2916 | 19.8 |

Tous les compteurs requis collectes : **oui**.

### Phase 4 — balayage en `d_sigma`

125 fits au total. Moyennes ± ecarts-types sur les 5 seeds. Rang et NMAE sur la meme ligne ; un fit `UNRESOLVED` reste dans le tableau et est exclu de toute conclusion sur la compressibilite.

| d_sigma | budget | appels realises | coeurs | log2 entrees | rang effectif | rang max | max r bloc sigma | r frontiere | parametres TT | NMAE tranche | plancher | x plancher | NMAE pave sigma | NMAE hors noeud | statut | fit (s) | s/requete |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 20000 | 21854 | 48 | 48 | 13.60 ± 0.11 | 17.4 | NOT APPLICABLE | NOT APPLICABLE | 17062 | 4.817e-05 ± 1.1e-05 | 2.348e-06 | 20.51 | NOT APPLICABLE | NOT APPLICABLE | 5/5 UNRESOLVED | 4 | 6.64e-05 |
| 0 | 60000 | 61670 | 48 | 48 | 19.98 ± 0.07 | 25.4 | NOT APPLICABLE | NOT APPLICABLE | 36818 | 3.435e-06 ± 5.6e-07 | 2.348e-06 | 1.46 | NOT APPLICABLE | NOT APPLICABLE | resolved | 8 | 1.02e-04 |
| 0 | 150000 | 151335 | 48 | 48 | 27.19 ± 0.05 | 33.4 | NOT APPLICABLE | NOT APPLICABLE | 68128 | 2.366e-06 ± 1.1e-08 | 2.348e-06 | 1.01 | NOT APPLICABLE | NOT APPLICABLE | resolved | 19 | 3.21e-04 |
| 0 | 300000 | 301378 | 48 | 48 | 34.24 ± 0.09 | 41.4 | NOT APPLICABLE | NOT APPLICABLE | 108023 | 2.349e-06 ± 1.8e-10 | 2.348e-06 | 1.00 | NOT APPLICABLE | NOT APPLICABLE | resolved | 45 | 6.86e-04 |
| 1 | 20000 | 21886 | 52 | 52 | 13.02 ± 0.06 | 17.8 | 8.0 | 10.4 | 17014 | 4.909e-04 ± 1.4e-04 | 2.358e-06 | 208.18 | 4.349e-04 | 5.037e-04 | 5/5 UNRESOLVED | 5 | 5.97e-05 |
| 1 | 60000 | 61817 | 52 | 52 | 19.58 ± 0.08 | 24.2 | 8.0 | 16.0 | 38410 | 1.122e-05 ± 2.2e-06 | 2.358e-06 | 4.76 | 1.089e-05 | 1.138e-05 | 5/5 UNRESOLVED | 8 | 9.16e-05 |
| 1 | 150000 | 151878 | 52 | 52 | 26.73 ± 0.10 | 32.2 | 8.0 | 16.0 | 71553 | 2.519e-06 ± 8.1e-08 | 2.358e-06 | 1.07 | 2.555e-06 | 2.520e-06 | resolved | 19 | 3.20e-04 |
| 1 | 300000 | 301295 | 52 | 52 | 33.50 ± 0.16 | 39.8 | 8.0 | 16.0 | 112373 | 2.368e-06 ± 2.1e-09 | 2.358e-06 | 1.00 | 2.429e-06 | 2.357e-06 | resolved | 38 | 6.91e-04 |
| 2 | 20000 | 21907 | 56 | 56 | 12.80 ± 0.19 | 17.0 | 13.2 | 10.6 | 17758 | 1.240e-03 ± 5.1e-04 | 2.413e-06 | 514.07 | 1.326e-03 | 1.359e-03 | 5/5 UNRESOLVED | 7 | 6.27e-05 |
| 2 | 60000 | 61728 | 56 | 56 | 19.05 ± 0.18 | 23.6 | 21.2 | 17.6 | 39271 | 5.204e-05 ± 5.0e-05 | 2.413e-06 | 21.57 | 3.969e-05 | 5.367e-05 | 5/5 UNRESOLVED | 11 | 1.33e-04 |
| 2 | 150000 | 151564 | 56 | 56 | 26.03 ± 0.14 | 31.2 | 29.2 | 25.4 | 73276 | 4.209e-06 ± 8.0e-07 | 2.413e-06 | 1.74 | 4.242e-06 | 4.187e-06 | resolved | 27 | 3.33e-04 |
| 2 | 300000 | 301359 | 56 | 56 | 32.75 ± 0.08 | 38.8 | 37.2 | 33.4 | 115996 | 2.589e-06 ± 4.5e-08 | 2.413e-06 | 1.07 | 2.599e-06 | 2.541e-06 | resolved | 39 | 7.07e-04 |
| 3 | 20000 | 21889 | 60 | 60 | 12.37 ± 0.21 | 16.2 | 14.0 | 10.4 | 17798 | 1.635e-03 ± 3.7e-04 | 2.481e-06 | 658.76 | 1.740e-03 | 1.741e-03 | 5/5 UNRESOLVED | 9 | 6.38e-05 |
| 3 | 60000 | 61853 | 60 | 60 | 18.62 ± 0.18 | 22.8 | 22.0 | 16.6 | 40288 | 9.715e-05 ± 4.1e-05 | 2.481e-06 | 39.15 | 8.216e-05 | 1.062e-04 | 5/5 UNRESOLVED | 12 | 1.07e-04 |
| 3 | 150000 | 151663 | 60 | 60 | 25.49 ± 0.11 | 30.6 | 29.6 | 24.2 | 75476 | 1.111e-05 ± 2.9e-06 | 2.481e-06 | 4.48 | 1.237e-05 | 1.259e-05 | 5/5 UNRESOLVED | 20 | 1.73e-04 |
| 3 | 300000 | 301519 | 60 | 60 | 32.14 ± 0.10 | 37.4 | 35.6 | 30.2 | 119928 | 3.374e-06 ± 3.6e-07 | 2.481e-06 | 1.36 | 3.412e-06 | 3.518e-06 | resolved | 190 | 7.02e-04 |
| 5 | 20000 | 21937 | 68 | 68 | 11.75 ± 0.22 | 15.0 | 14.2 | 12.4 | 18278 | 3.887e-03 ± 1.3e-03 | 2.640e-06 | 1472.69 | 5.793e-03 | 5.358e-03 | 5/5 UNRESOLVED | 34 | 1.60e-04 |
| 5 | 60000 | 61859 | 68 | 68 | 17.53 ± 0.19 | 21.2 | 20.0 | 17.6 | 40656 | 6.192e-04 ± 1.4e-04 | 2.640e-06 | 234.59 | 4.221e-04 | 6.078e-04 | 5/5 UNRESOLVED | 40 | 2.99e-04 |
| 5 | 150000 | 151664 | 68 | 68 | 24.29 ± 0.19 | 29.2 | 26.2 | 23.4 | 78006 | 1.078e-04 ± 4.5e-05 | 2.640e-06 | 40.83 | 1.192e-04 | 1.088e-04 | 5/5 UNRESOLVED | 56 | 3.65e-04 |
| 5 | 300000 | 301393 | 68 | 68 | 30.91 ± 0.18 | 36.4 | 34.2 | 31.4 | 126230 | 1.096e-05 ± 3.3e-06 | 2.640e-06 | 4.15 | 1.117e-05 | 1.211e-05 | 4/5 UNRESOLVED | 100 | 1.11e-03 |

#### Palier post-hoc (hors pre-enregistrement)

Budget(s) `[600000]`, ajoute(s) **apres** que l'echelle pre-enregistree ait ete executee et examinee, et applique(s) uniformement a tous les `d_sigma` — pas a une configuration isolee. Les planchers d'interpolation sont ceux de la phase 4, reutilises et non recalcules, donc les deux paliers se comparent au meme plancher exact. Ces lignes ne doivent pas etre presentees comme pre-enregistrees.

| d_sigma | budget | appels realises | coeurs | log2 entrees | rang effectif | rang max | max r bloc sigma | r frontiere | parametres TT | NMAE tranche | plancher | x plancher | NMAE pave sigma | NMAE hors noeud | statut | fit (s) | s/requete |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 600000 | 601175 | 48 | 48 | 43.31 ± 0.09 | 51.0 | NOT APPLICABLE | NOT APPLICABLE | 172744 | 2.348e-06 ± 6.1e-13 | 2.348e-06 | 1.00 | NOT APPLICABLE | NOT APPLICABLE | resolved | 80 | 8.37e-04 |
| 1 | 600000 | 601225 | 52 | 52 | 42.02 ± 0.13 | 49.8 | 8.0 | 16.0 | 176748 | 2.358e-06 ± 5.4e-10 | 2.358e-06 | 1.00 | 2.420e-06 | 2.346e-06 | resolved | 95 | 8.71e-04 |
| 2 | 600000 | 600919 | 56 | 56 | 41.19 ± 0.13 | 49.0 | 45.2 | 41.4 | 183420 | 2.433e-06 ± 5.9e-09 | 2.413e-06 | 1.01 | 2.460e-06 | 2.379e-06 | resolved | 87 | 8.56e-04 |
| 3 | 600000 | 601196 | 60 | 60 | 40.54 ± 0.05 | 47.0 | 46.0 | 40.6 | 190837 | 2.535e-06 ± 2.1e-08 | 2.481e-06 | 1.02 | 2.552e-06 | 2.464e-06 | resolved | 76 | 9.01e-04 |
| 5 | 600000 | 601159 | 68 | 68 | 38.80 ± 0.22 | 45.2 | 42.2 | 39.4 | 198848 | 4.361e-06 ± 5.2e-07 | 2.640e-06 | 1.65 | 4.638e-06 | 4.630e-06 | resolved | 172 | 1.44e-03 |

**Verification de l'ordre des modes** (chaine `d_sigma=0` = chaine `d_sigma=k` privee des coeurs sigma) :

- `shared_core_order_preserved` : **verifie sur les 125 fits**
- `sigma_cores_contiguous` : **verifie sur les 125 fits**
- `sigma_block_at_chain_tail` : **verifie sur les 125 fits**

Temps total : 2.60 h.

### Runs diagnostiques (parametre gele modifie)

**d_sigma=5 a n_sigma=32 (n_sigma etait gele a 16)** (`phase4c_nsigma32_*`)

n_sigma = 32, 2000 points de test (axes partages strictement identiques a ceux du balayage principal).

| d_sigma | budget | appels realises | coeurs | log2 entrees | rang effectif | rang max | max r bloc sigma | r frontiere | parametres TT | NMAE tranche | plancher | x plancher | NMAE pave sigma | NMAE hors noeud | statut | fit (s) | s/requete |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 600000 | 601462 | 73 | 73 | 37.91 ± 0.11 | 43.6 | 42.4 | 39.0 | 204200 | 4.311e-06 ± 4.7e-07 | 2.354e-06 | 1.83 | 4.885e-06 | 4.635e-06 | resolved | 207 | 2.21e-03 |

### Phase 6 — controles

**6a — sensibilite a l'ordre des modes.**

| ordre | rang effectif | parametres TT | max r bloc sigma | NMAE tranche | seeds | UNRESOLVED |
|---|---|---|---|---|---|---|
| head | 24.31 ± 0.12 | 78131 | 27.3 | 2.287e-04 ± 1.7e-04 | 3 | 3 |
| interleaved | 23.79 ± 0.04 | 74803 | 28.7 | 6.140e-05 ± 1.0e-05 | 3 | 3 |
| tail | 24.40 ± 0.15 | 78717 | 26.3 | 1.309e-04 ± 4.3e-05 | 3 | 3 |

**6b — controle sur la grille de pricing.**

| d_sigma | rang effectif | parametres TT | max r bloc sigma | NMAE tranche | seeds | UNRESOLVED |
|---|---|---|---|---|---|---|
| 0 | 30.51 ± 0.23 | 65290 | NOT APPLICABLE | 1.685e-05 ± 6.8e-11 | 5 | 0 |
| 5 | 26.17 ± 0.17 | 75420 | 29.8 | 1.988e-04 ± 8.2e-05 | 5 | 5 |

## 3. Figures

Format vectoriel, sans titre integre.

**Echelle de budgets pre-enregistree, telle que produite par la phase 4**

| figure | PDF | SVG |
|---|---|---|
| Figure 1 — cout statistique | `experiments\figures\figure1_statistical_cost.pdf` | `experiments\figures\figure1_statistical_cost.svg` |
| Figure 2 — profil de rang | `experiments\figures\figure2_rank_profile.pdf` | `experiments\figures\figure2_rank_profile.svg` |
| Figure 3 — pente dimensionnelle | `experiments\figures\figure3_dimensional_slope.pdf` | `experiments\figures\figure3_dimensional_slope.svg` |
| Figure S1 — NMAE sur une maille sigma | `experiments\figures\figureS1_sigma_cell.pdf` | `experiments\figures\figureS1_sigma_cell.svg` |

**Memes figures, palier post-hoc inclus**

| figure | PDF | SVG |
|---|---|---|
| Figure 1 — cout statistique | `experiments\figures_extended\figure1_statistical_cost.pdf` | `experiments\figures_extended\figure1_statistical_cost.svg` |
| Figure 2 — profil de rang | `experiments\figures_extended\figure2_rank_profile.pdf` | `experiments\figures_extended\figure2_rank_profile.svg` |
| Figure 3 — pente dimensionnelle | `experiments\figures_extended\figure3_dimensional_slope.pdf` | `experiments\figures_extended\figure3_dimensional_slope.svg` |
| Figure S1 — NMAE sur une maille sigma | `experiments\figures_extended\figureS1_sigma_cell.pdf` | `experiments\figures_extended\figureS1_sigma_cell.svg` |

## 4. Observations

Section descriptive. Aucune interpretation, aucune redaction destinee au manuscrit.

- Au plus grand budget (600000), le rang de lien maximal interne au bloc sigma vaut 8.0 a d_sigma=1, 45.2 a d_sigma=2, 46.0 a d_sigma=3, 42.2 a d_sigma=5.
- Le rang effectif global vaut 43.31 a d_sigma=0, 42.02 a d_sigma=1, 41.19 a d_sigma=2, 40.54 a d_sigma=3, 38.80 a d_sigma=5.
- Le nombre de parametres TT passe de 172744 (d_sigma=0) a 198848 (d_sigma=5), facteur 1.15, tandis que le nombre d'entrees logiques passe de 2^48 a 2^68.
- La NMAE sur la tranche commune au plus grand budget vaut 2.348e-06 a d_sigma=0, 2.358e-06 a d_sigma=1, 2.433e-06 a d_sigma=2, 2.535e-06 a d_sigma=3, 4.361e-06 a d_sigma=5.
- Configurations comportant au moins un fit `UNRESOLVED` : d_sigma=0 budget=20000 (5/5); d_sigma=1 budget=20000 (5/5); d_sigma=1 budget=60000 (5/5); d_sigma=2 budget=20000 (5/5); d_sigma=2 budget=60000 (5/5); d_sigma=3 budget=20000 (5/5); d_sigma=3 budget=60000 (5/5); d_sigma=3 budget=150000 (5/5); d_sigma=5 budget=20000 (5/5); d_sigma=5 budget=60000 (5/5); d_sigma=5 budget=150000 (5/5); d_sigma=5 budget=300000 (4/5).
- **balayage pre-enregistre (phase4_extended), budget [300000]**
- Balayage de la maille sigma (20 valeurs de 0.196667 a 0.233333, bornes = noeuds de grille) : la NMAE a sigma = 0.200526 vaut 1.0992e-05, soit le rang 3 sur 20 par valeur croissante ; l'intervalle sur la maille est [1.0764e-05, 1.2096e-05], mediane 1.1746e-05. Le minimum de la maille est atteint a sigma = 0.196667. Amplitude relative sur la maille : 1.124x.
- **balayage rejoue sur fits resolus, budget [600000]**
- Balayage de la maille sigma (20 valeurs de 0.196667 a 0.233333, bornes = noeuds de grille) : la NMAE a sigma = 0.200526 vaut 4.3765e-06, soit le rang 3 sur 20 par valeur croissante ; l'intervalle sur la maille est [4.3359e-06, 4.7786e-06], mediane 4.6476e-06. Le minimum de la maille est atteint a sigma = 0.198596. Amplitude relative sur la maille : 1.102x.
- Plancher d'interpolation sur la meme maille : 2.4969e-06 a sigma_ref, intervalle [2.1558e-06, 3.1229e-06], minimum a sigma = 0.196667.

## 5. Anomalies et echecs

- 59 fits sur 125 depassent le plancher d'interpolation de leur grille d'un facteur superieur a 3 et sont marques `UNRESOLVED`. Ils restent dans les tableaux et sont exclus de toute conclusion sur la compressibilite.
- d_sigma=2, budget=60000 : la seed 20260403 s'ecarte de la mediane des seeds de 348 % en NMAE (1.398e-04 contre 3.122e-05). Rapportee, non exclue.
- d_sigma=3, budget=60000 : la seed 20260403 s'ecarte de la mediane des seeds de 125 % en NMAE (1.630e-04 contre 7.260e-05). Rapportee, non exclue.
- PLAFOND DE RANG — AUCUN RANG DE CE RAPPORT N'EST CONVERGE. Verification : scripts/experiments/verify_rank_ceiling.py, artefacts verification_plafond_rang.{csv,json}. (1) TT-cross n'impose aucun cap absolu : teneva.maxvol_rect renvoie r+dr avec dr <= dr_max, borne seulement par le nombre de lignes de la matrice haute (2*r_gauche pour un coeur binaire) ; aucun parametre de plafond n'existe. (2) La borne de matricisation QTT min(2^k, 2^(d-k)) ne mord qu'aux deux extremites de la chaine (bonds 0-5 et les 5-6 derniers, qui la saturent exactement) ; a l'interieur elle vaut jusqu'a 2^24 et ne mord jamais. (3) Ce qui mord est le calendrier d'increment : maxvol_rect ajoute au plus dr_max=2 lignes par appel, chaque lien est visite deux fois par sweep, et teneva.anova(r=2) demarre a rang 2, d'ou r_k <= 2 + 2*dr_max*nswp. Le rang maximal observe vaut 0.96 a 1.07 fois cette borne sur les 25 configurations : il y est colle. Et nswp est fixe par le budget, aucune tolerance n'etant passee a teneva.cross (stop='m' partout).
- CONSEQUENCE : LA COMPARAISON DE RANG ENTRE d_sigma EST CONFONDUE. Plus de coeurs coutent plus d'evaluations par sweep, donc nswp baisse quand d_sigma monte : a 600000 appels, nswp vaut 12.0, 12.0, 11.6, 11.0, 11.0 pour d_sigma = 0,1,2,3,5, et le plafond correspondant 50.0, 50.0, 48.4, 46.0, 46.0. La decroissance du rang effectif (43.31 -> 38.80) suit donc le plafond et non une propriete du probleme. Normalise par le plafond, le rang effectif est plat : 0.866, 0.840, 0.851, 0.881, 0.843. De meme le rang maximal interne au bloc sigma vaut 0.93 a 1.00 fois le plafond des d_sigma=2, donc lui aussi y est colle.
- RETRACTATION. L'affirmation 'le rang du bloc sigma sature en d_sigma' n'est PAS etablie par ces donnees et ne doit pas figurer dans le manuscrit. Ce qui est mesure est que rien dans la chaine — axes partages comme bloc sigma — n'atteint son rang structurel a aucun budget teste : tout est colle au plafond du calendrier d'increment. La prediction d'un rang de bloc sigma borne par la regularite des deux agregats scalaires reste NON TESTEE. La tester demanderait de passer une tolerance e ou e_vld a teneva.cross, ou d'augmenter rank_increment, tous deux parametres geles par le protocole.
- CE QUI SURVIT A CETTE RETRACTATION : les enonces fondes sur la precision, qui ne dependent pas du rang. A 600000 appels, d_sigma = 0,1,2,3 atteignent leur plancher d'interpolation (x1.00, x1.00, x1.01, x1.02) et d_sigma=5 s'en approche (x1.65) ; le nombre de parametres TT croit d'un facteur 1.15 de d_sigma=0 a 5 quand les entrees logiques passent de 2^48 a 2^68. Ce dernier point reste toutefois un enonce A BUDGET FIXE : les parametres valent environ 2 * (nombre de coeurs) * r^2 avec r colle au plafond, donc leur croissance mesure la taille du TT obtenu pour un budget d'oracle donne, pas la compressibilite structurelle de la representation.
- PALIER POST-HOC. Le budget 600000 a ete ajoute APRES execution et examen de l'echelle pre-enregistree {20000, 60000, 150000, 300000}, sur demande. Applique uniformement aux cinq d_sigma ; planchers reutilises, donc les deux paliers se comparent au plancher identique. Tableau distinct.
- 6a REJOUEE A 600000, HORS PROTOCOLE (phase6a_ordre_modes_600k*). A 150000, budget specifie, les neuf fits sont UNRESOLVED dans les trois ordres et la comparaison mesure une vitesse de convergence. A 600000 la lecture s'inverse : 'interleaved' paraissait ~2x meilleur que 'tail' en NMAE (6.14e-05 contre 1.31e-04) ; une fois resolus, 'tail' est legerement meilleur (4.684e-06 contre 5.143e-06) tandis que 'interleaved' est 5 % moins cher en parametres (189881 contre 199891). L'ordre 'tail' de l'experience principale n'est pas favorise ; il reste le seul ordre additif, seul a permettre l'alignement coeur par coeur de la figure 2.
- 6a a 600000 : l'ordre 'tail' n'a pas ete refitte, il est repris de phase4b_resultats.csv restreint aux trois seeds du controle (configuration identique, determinisme verifie). SEED DIVERGENTE : dans l'ordre 'head', la seed 20260403 donne 4.003e-05, huit fois ses homologues (5.159e-06 et 5.474e-06), et reste UNRESOLVED ; elle explique a elle seule l'ecart-type de 2.0e-05 et la moyenne de 1.689e-05 de cette ligne, qui n'est donc pas interpretable telle quelle. Rapportee, non exclue.
- 6a a 600000 : planchers repris du run a 150000 (independants du budget). Le run precedent ne stockait pas le sous-echantillon de controle, donc aucune verification d'invariance par permutation n'etait possible pour le premier ordre reutilise ('head') ; la reference a ete etablie sur lui et 'interleaved' verifie contre elle a 0.00e+00 sur 32 points. L'invariance avait ete mesuree a 5.58e-16 lors du run a 150000.
- RUN DIAGNOSTIQUE HORS PROTOCOLE : d_sigma=5 a n_sigma=32 modifie un parametre gele et produit une autre grille (73 coeurs, plancher 2.354e-06 contre 2.640e-06). Doubler la resolution sigma abaisse le plancher de 10.8 % mais la NMAE de 1.1 % seulement, et le rapport au plancher se degrade (1.65 -> 1.83) : l'erreur residuelle a d_sigma=5 n'est pas limitee par la resolution de l'axe sigma.
- BALAYAGE DE MAILLE REJOUE a 600000 sur fits resolus : la dispersion entre seeds passe de 82.6 % a 25.6 % a sigma fixe, l'amplitude sur la maille restant de 10.2 % contre 44.9 % pour le plancher. Les deux balayages sont rapportes ; le plancher, sans TT, est partage. Le balayage echantillonne 20 valeurs de noeud a noeud et sigma_ref = 0.20 n'est pas l'un des 20 (voisins 0.198596 et 0.200526) : ECART AU PROTOCOLE non corrige a posteriori.
- BUG CORRIGE PENDANT LA PHASE 6 : le premier passage de 6b n'executait que d_sigma=0, un 'break' terminant la boucle sur d_sigma au lieu de sauter la seule chaine 8D redondante. Corrige en 'continue', logique reverifiee, 6b entierement relancee. 6a n'utilisait qu'une valeur de d_sigma et n'etait pas affectee.
- La colonne 'max r bloc sigma' ne mesure pas la meme chose a d_sigma=1 qu'au-dela : a d_sigma=1 le bloc ne contient qu'un axe physique et ses liens internes sont ceux de sa decomposition QTT binaire ; a partir de d_sigma=2 ils incluent les liens entre axes sigma. La discontinuite du panneau gauche de la figure 3 est definitionnelle, pas physique.
- TEMPS D'HORLOGE CONTAMINES : phase4c_nsigma32, phase4d_cell600k, la phase 6 et sa reprise a 600000 ont tourne partiellement en concurrence ; leurs colonnes fit_seconds ne sont pas comparables au balayage principal. Les grandeurs numeriques ne le sont pas, le determinisme a seed fixee ayant ete verifie. Voir aussi le fit isole a 777.8 s (d_sigma=3, budget 300000, seed 20260404) contre ~42 s pour ses homologues.
- Le manifeste rapporte l'arbre de travail comme 'dirty' : le code de l'experience n'est pas commite, donc le hash de commit ne suffit pas a reproduire.
