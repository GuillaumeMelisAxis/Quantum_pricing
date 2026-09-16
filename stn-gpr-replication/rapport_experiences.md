# Rapport d'experiences — cout tensoriel en fonction du nombre de facteurs de volatilite

Genere par `scripts/experiments/build_report.py` a partir des seuls artefacts de `experiments/`. Aucun chiffre n'est saisi a la main. Une mesure absente s'ecrit `NOT RUN`.

## 1. Manifeste de run

### Environnement

| item | value |
|---|---|
| commit | `9c9f1fded4d284d8a981f56209646c8e3e5bd2f5` |
| branch | `dimensionality_experiments` |
| working tree | dirty (8 files) |
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

100 fits. Moyennes ± ecarts-types sur les 5 seeds. Rang et NMAE sur la meme ligne ; un fit `UNRESOLVED` reste dans le tableau et est exclu de toute conclusion sur la compressibilite.

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

**Verification de l'ordre des modes** (chaine `d_sigma=0` = chaine `d_sigma=k` privee des coeurs sigma) :

- `shared_core_order_preserved` : **verifie sur les 100 fits**
- `sigma_cores_contiguous` : **verifie sur les 100 fits**
- `sigma_block_at_chain_tail` : **verifie sur les 100 fits**

Temps total : 1.83 h.

### Phase 6 — controles

`NOT RUN`

## 3. Figures

Format vectoriel, sans titre integre.

| figure | PDF | SVG |
|---|---|---|
| Figure 1 — cout statistique | `experiments\figures\figure1_statistical_cost.pdf` | `experiments\figures\figure1_statistical_cost.svg` |
| Figure 2 — profil de rang | `experiments\figures\figure2_rank_profile.pdf` | `experiments\figures\figure2_rank_profile.svg` |
| Figure 3 — pente dimensionnelle | `experiments\figures\figure3_dimensional_slope.pdf` | `experiments\figures\figure3_dimensional_slope.svg` |
| Figure S1 — NMAE sur une maille sigma | `experiments\figures\figureS1_sigma_cell.pdf` | `experiments\figures\figureS1_sigma_cell.svg` |

## 4. Observations

Section descriptive. Aucune interpretation, aucune redaction destinee au manuscrit.

- Au plus grand budget (300000), le rang de lien maximal interne au bloc sigma vaut 8.0 a d_sigma=1, 37.2 a d_sigma=2, 35.6 a d_sigma=3, 34.2 a d_sigma=5.
- Le rang effectif global vaut 34.24 a d_sigma=0, 33.50 a d_sigma=1, 32.75 a d_sigma=2, 32.14 a d_sigma=3, 30.91 a d_sigma=5.
- Le nombre de parametres TT passe de 108023 (d_sigma=0) a 126230 (d_sigma=5), facteur 1.17, tandis que le nombre d'entrees logiques passe de 2^48 a 2^68.
- La NMAE sur la tranche commune au plus grand budget vaut 2.349e-06 a d_sigma=0, 2.368e-06 a d_sigma=1, 2.589e-06 a d_sigma=2, 3.374e-06 a d_sigma=3, 1.096e-05 a d_sigma=5.
- Configurations comportant au moins un fit `UNRESOLVED` : d_sigma=0 budget=20000 (5/5); d_sigma=1 budget=20000 (5/5); d_sigma=1 budget=60000 (5/5); d_sigma=2 budget=20000 (5/5); d_sigma=2 budget=60000 (5/5); d_sigma=3 budget=20000 (5/5); d_sigma=3 budget=60000 (5/5); d_sigma=3 budget=150000 (5/5); d_sigma=5 budget=20000 (5/5); d_sigma=5 budget=60000 (5/5); d_sigma=5 budget=150000 (5/5); d_sigma=5 budget=300000 (4/5).
- Balayage de la maille sigma (20 valeurs de 0.196667 a 0.233333, bornes = noeuds de grille) : la NMAE a sigma = 0.200526 vaut 1.0992e-05, soit le rang 3 sur 20 par valeur croissante ; l'intervalle sur la maille est [1.0764e-05, 1.2096e-05], mediane 1.1746e-05. Le minimum de la maille est atteint a sigma = 0.196667.
- Plancher d'interpolation sur la meme maille : 2.4969e-06 a sigma_ref, intervalle [2.1558e-06, 3.1229e-06], minimum a sigma = 0.196667.

## 5. Anomalies et echecs

- 59 fits sur 100 depassent le plancher d'interpolation de leur grille d'un facteur superieur a 3 et sont marques `UNRESOLVED`. Ils restent dans les tableaux et sont exclus de toute conclusion sur la compressibilite.
- d_sigma=2, budget=60000 : la seed 20260403 s'ecarte de la mediane des seeds de 348 % en NMAE (1.398e-04 contre 3.122e-05). Rapportee, non exclue.
- d_sigma=3, budget=60000 : la seed 20260403 s'ecarte de la mediane des seeds de 125 % en NMAE (1.630e-04 contre 7.260e-05). Rapportee, non exclue.
- Le balayage de maille sigma echantillonne 20 valeurs uniformes de noeud a noeud ; sigma_ref = 0.20 ne fait pas partie des 20 echantillons (les plus proches sont 0.198596 et 0.200526). ECART AU PROTOCOLE, non corrige a posteriori : la mesure a exactement sigma_ref existe par ailleurs (colonne 'NMAE tranche' de la phase 4) et se compare directement a l'intervalle de la maille.
- La colonne 'max r bloc sigma' ne mesure pas la meme chose a d_sigma=1 qu'au-dela. A d_sigma=1 le bloc sigma ne contient qu'un seul axe physique, donc ses liens internes sont les liens de la decomposition QTT binaire de cet axe ; a partir de d_sigma=2 ils incluent les liens entre axes sigma, qui portent le couplage inter-actifs. La discontinuite visible sur le panneau gauche de la figure 3 entre d_sigma=1 et d_sigma=2 est de nature definitionnelle, pas physique.
- Un fit isole a mis 777.8 s la ou ses cinq homologues en ont mis ~42 s (d_sigma=3, budget 300000, seed 20260404), soit un facteur 18 en temps d'horloge. Ses grandeurs numeriques (rang effectif 31.99, NMAE 3.887e-06, parametres 118864) sont alignees sur celles de ses homologues : anomalie de temps machine, pas de calcul. Les temps de fit de cette configuration sont a lire avec cette reserve.
- Aucun budget ne resout les deux chaines de la figure 2 simultanement : a 300000 appels, d_sigma=0 resout 5 seeds sur 5 et d_sigma=5 une sur cinq. La figure porte cette fraction en clair.
- Le livrable est nomme 'phase4_agrege.csv' et non 'phase4_agrégé.csv' : nom de fichier ASCII, par prudence sur un systeme de fichiers Windows.
- Le manifeste rapporte l'arbre de travail comme 'dirty' : le code de l'experience (sigma_axes.py, sigma_surrogate.py, scripts/experiments/, tests/test_sigma_axes.py) n'est pas commite. Un tiers ne peut pas reproduire a partir du seul hash de commit tant que ce n'est pas fait.
