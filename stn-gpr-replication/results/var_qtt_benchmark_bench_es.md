# Benchmark VaR / ES 1 jour - QTT 13-D contre Monte Carlo imbrique

Livre de 6 options sur panier geometrique, valeur 9.2519. 10 000 scenarios GBM correles, horizon 1 jour. Grille QTT 13-D `bounded_standardized_risk`, layout `interleaved`, 59 coeurs.

Toutes les erreurs des sections 2 a 6 sont des moyennes quadratiques sur 3 graines independantes : a budget fixe, ni le TT-cross ni le Monte Carlo ne produisent un nombre reproductible.

## 1. Mesures de risque - bras principaux

QTT au budget 300 000, Monte Carlo a 4 096 chemins internes, une graine chacun.

| Bras | VaR 95 % | VaR 99 % | **ES 97,5 %** | Err. VaR 95 % | Err. VaR 99 % | **Err. ES 97,5 %** |
|---|---:|---:|---:|---:|---:|---:|
| Forme fermee (reference) | 0.743980 | 0.898179 | **0.888192** | - | - | - |
| Monte Carlo imbrique | 0.703391 | 0.849436 | **0.839600** | -5.46 % | -5.43 % | -5.47 % |
| QTT 13-D | 0.743389 | 0.897812 | **0.887773** | -0.08 % | -0.04 % | -0.05 % |

Intervalle de confiance a 95 % de l'ES 97,5 % de reference, du seul fait de l'echantillonnage des 10 000 scenarios : `[0.882777, 0.893606]` (moyenne de queue sur 250 scenarios, erreur type asymptotique).

Monte Carlo imbrique : dehors ; QTT 13-D : dedans.

## 2. Configurations QTT

| Budget TT-cross | Err. VaR 95 % | Err. VaR 99 % | **Err. ES 97,5 %** | Initialisation (une fois) | VaR / ES quotidienne |
|---|---:|---:|---:|---:|---:|
| 50 000 | 11.12 % | 8.09 % | **8.14 %** | 5.4 s | 8.4 s |
| 100 000 | 4.29 % | 5.40 % | **5.32 %** | 9.5 s | 14.4 s |
| 150 000 | 1.81 % | 1.26 % | **1.26 %** | 14.0 s | 16.3 s |
| 200 000 | 0.33 % | 0.39 % | **0.39 %** | 20.1 s | 18.6 s |
| 300 000 | 0.11 % | 0.10 % | **0.09 %** | 33.2 s | 22.8 s |
| 400 000 | 0.44 % | 0.43 % | **0.43 %** | 47.1 s | 22.6 s |

## 3. Configurations Monte Carlo

| Chemins internes | Err. VaR 95 % | Err. VaR 99 % | **Err. ES 97,5 %** | Initialisation (une fois) | VaR / ES quotidienne |
|---|---:|---:|---:|---:|---:|
| 1 024 | 6.60 % | 6.71 % | **6.71 %** | 0.00 s | 3.3 s |
| 4 096 | 4.11 % | 3.99 % | **4.00 %** | 0.00 s | 11.8 s |
| 16 384 | 0.82 % | 0.91 % | **0.90 %** | 0.00 s | 50.1 s |
| 65 536 | 0.18 % | 0.15 % | **0.15 %** | 0.00 s | 218.3 s |

La VaR et l'ES sortent du meme jeu de pertes : le cout quotidien est celui de la revalorisation, l'ES n'y ajoute qu'un tri de 10 000 nombres.

## 4. Amortissement : cout total apres N runs

`cout(N) = initialisation + N x cout quotidien`. Le surrogate QTT s'initialise une fois et se reutilise tel quel les jours suivants : seuls les spots bougent, et la grille les couvre deja.

| Configuration | Err. VaR 95 % | Err. ES 97,5 % | N = 1 | N = 5 | N = 21 | N = 63 | N = 252 |
|---|---:|---:|---:|---:|---:|---:|---:|
| QTT 50000 | 11.12 % | 8.14 % | 14 s | 47 s | 181 s | 532 s | 2113 s |
| QTT 100000 | 4.29 % | 5.32 % | 24 s | 81 s | 311 s | 914 s | 3626 s |
| QTT 150000 | 1.81 % | 1.26 % | 30 s | 95 s | 356 s | 1040 s | 4118 s |
| QTT 200000 | 0.33 % | 0.39 % | 39 s | 113 s | 410 s | 1189 s | 4698 s |
| QTT 300000 | 0.11 % | 0.09 % | 56 s | 147 s | 512 s | 1468 s | 5773 s |
| QTT 400000 | 0.44 % | 0.43 % | 70 s | 160 s | 522 s | 1471 s | 5743 s |
| MC 1024 | 6.60 % | 6.71 % | 3 s | 17 s | 70 s | 209 s | 836 s |
| MC 4096 | 4.11 % | 4.00 % | 12 s | 59 s | 249 s | 747 s | 2986 s |
| MC 16384 | 0.82 % | 0.90 % | 50 s | 250 s | 1052 s | 3156 s | 12623 s |
| MC 65536 | 0.18 % | 0.15 % | 218 s | 1091 s | 4583 s | 13750 s | 55000 s |

## 5. Croisements a precision comparable - ES 97,5 %

Seules les paires dont le rapport de precision MC/QTT est compris entre 0,5 et 2 sont listees : au-dela, un croisement de cout ne dit rien. Rapport > 1 : le QTT est le plus precis des deux.

| QTT | Monte Carlo | Err. QTT | Err. MC | Rapport MC/QTT | Croisement |
|---|---|---:|---:|---:|---|
| QTT 50000 | MC 1024 | 8.14 % | 6.71 % | 0.8x | jamais : cout quotidien QTT plus eleve (QTT 1.2x moins precis) |
| QTT 100000 | MC 1024 | 5.32 % | 6.71 % | 1.3x | jamais : cout quotidien QTT plus eleve |
| QTT 100000 | MC 4096 | 5.32 % | 4.00 % | 0.8x | jamais : cout quotidien QTT plus eleve (QTT 1.3x moins precis) |
| QTT 150000 | MC 16384 | 1.26 % | 0.90 % | 0.7x | des le 1er run (QTT 1.4x moins precis) |
| QTT 300000 | MC 65536 | 0.09 % | 0.15 % | 1.6x | des le 1er run |

A precision egale sur l'ES 97,5 % (0.09 %, erreur QTT moyennee sur graines au budget 300 000), le Monte Carlo demanderait 148 390 chemins (extrapole), soit ~494 s par run.

## 6. Croisements a precision comparable - VaR 95 %

| QTT | Monte Carlo | Err. QTT | Err. MC | Rapport MC/QTT | Croisement |
|---|---|---:|---:|---:|---|
| QTT 50000 | MC 1024 | 11.12 % | 6.60 % | 0.6x | jamais : cout quotidien QTT plus eleve (QTT 1.7x moins precis) |
| QTT 100000 | MC 1024 | 4.29 % | 6.60 % | 1.5x | jamais : cout quotidien QTT plus eleve |
| QTT 100000 | MC 4096 | 4.29 % | 4.11 % | 1.0x | jamais : cout quotidien QTT plus eleve |
| QTT 200000 | MC 65536 | 0.33 % | 0.18 % | 0.5x | des le 1er run (QTT 1.8x moins precis) |
| QTT 300000 | MC 65536 | 0.11 % | 0.18 % | 1.7x | des le 1er run |
| QTT 400000 | MC 16384 | 0.44 % | 0.82 % | 1.9x | a partir du run 2 |

## Reserves

1. **Les temps absolus portent une dispersion machine d'environ +-2.5x** pour un travail identique. Les rapports intra-run sont exploitables, les secondes prises isolement beaucoup moins.
2. **L'erreur de risque du QTT n'est pas monotone en budget** alors que son erreur de prix l'est : le budget ne peut pas etre regle sur l'erreur de prix.
3. **L'ES 97,5 % est une moyenne sur les 250 pires scenarios**, pas un quantile : il est moins sensible qu'une VaR a la position d'une seule perte, et son intervalle de confiance est asymptotique (erreur type de la moyenne de queue).
4. **L'initialisation QTT n'est amortissable que tant que les spots restent dans la boite de la grille.** Mesure ici : 0.00% de scenarios hors boite sur un horizon d'un jour.
5. **Le cout de mise au point du budget** - balayer plusieurs budgets sur plusieurs graines contre une reference exacte - n'est compte dans aucun tableau.
