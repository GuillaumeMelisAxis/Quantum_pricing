# Benchmark VaR 1 jour - QTT 13-D contre Monte Carlo imbrique

Livre de 6 options sur panier geometrique, valeur 9.2519. 10000 scenarios GBM correles, horizon 1 jour. Grille QTT 13-D `bounded_standardized_risk`, layout `interleaved`, 59 coeurs.

Reference exacte (Black-Scholes panier geometrique) : VaR 95 % = 0.743980, VaR 99 % = 0.898179.

Toutes les erreurs sont des moyennes quadratiques sur 3 graines independantes : a budget fixe, ni le TT-cross ni le Monte Carlo ne produisent un nombre reproductible.

## 1. Configurations QTT

| Budget TT-cross | Err. VaR 95 % | Err. VaR 99 % | Initialisation (une fois) | VaR quotidienne |
|---|---:|---:|---:|---:|
| 50 000 | 11.12 % | 8.09 % | 5.7 s | 8.0 s |
| 100 000 | 4.29 % | 5.40 % | 8.3 s | 12.3 s |
| 150 000 | 1.81 % | 1.26 % | 12.4 s | 14.0 s |
| 200 000 | 0.33 % | 0.39 % | 20.2 s | 16.6 s |
| 300 000 | 0.11 % | 0.10 % | 32.7 s | 21.3 s |
| 400 000 | 0.44 % | 0.43 % | 118.0 s | 48.2 s |

## 2. Configurations Monte Carlo

| Chemins internes | Err. VaR 95 % | Err. VaR 99 % | Initialisation (une fois) | VaR quotidienne |
|---|---:|---:|---:|---:|
| 1 024 | 6.60 % | 6.71 % | 0.00 s | 3.2 s |
| 4 096 | 4.11 % | 3.99 % | 0.00 s | 12.0 s |
| 16 384 | 0.82 % | 0.91 % | 0.00 s | 47.4 s |
| 65 536 | 0.18 % | 0.15 % | 0.00 s | 210.2 s |

## 3. Amortissement : cout total apres N runs de VaR

`cout(N) = initialisation + N x VaR quotidienne`. Le surrogate QTT s'initialise une fois et se reutilise tel quel les jours suivants : seuls les spots bougent, et la grille les couvre deja.

| Configuration | Err. VaR 95 % | N = 1 | N = 5 | N = 21 | N = 63 | N = 252 |
|---|---:|---:|---:|---:|---:|---:|
| QTT 50000 | 11.12 % | 14 s | 46 s | 174 s | 510 s | 2024 s |
| QTT 100000 | 4.29 % | 21 s | 70 s | 266 s | 781 s | 3100 s |
| QTT 150000 | 1.81 % | 26 s | 83 s | 307 s | 897 s | 3553 s |
| QTT 200000 | 0.33 % | 37 s | 103 s | 370 s | 1069 s | 4215 s |
| QTT 300000 | 0.11 % | 54 s | 139 s | 481 s | 1376 s | 5407 s |
| QTT 400000 | 0.44 % | 166 s | 359 s | 1131 s | 3158 s | 12277 s |
| MC 1024 | 6.60 % | 3 s | 16 s | 67 s | 202 s | 809 s |
| MC 4096 | 4.11 % | 12 s | 60 s | 253 s | 758 s | 3033 s |
| MC 16384 | 0.82 % | 47 s | 237 s | 994 s | 2983 s | 11934 s |
| MC 65536 | 0.18 % | 210 s | 1051 s | 4415 s | 13244 s | 52975 s |

## 4. Croisements a precision comparable

Un croisement n'a de sens qu'entre configurations de precision voisine ; la colonne `rapport` le rend verifiable (1.0 = precisions egales, 10 = le Monte Carlo est dix fois moins precis).

| QTT | Monte Carlo | Err. QTT | Err. MC | Rapport | Croisement |
|---|---|---:|---:|---:|---|
| QTT 100000 | MC 4096 | 4.29 % | 4.11 % | 1.0x | jamais |
| QTT 100000 | MC 1024 | 4.29 % | 6.60 % | 1.5x | jamais |
| QTT 50000 | MC 1024 | 11.12 % | 6.60 % | 0.6x | jamais |
| QTT 300000 | MC 65536 | 0.11 % | 0.18 % | 1.7x | des le 1er run |
| QTT 200000 | MC 65536 | 0.33 % | 0.18 % | 0.5x | des le 1er run |
| QTT 400000 | MC 16384 | 0.44 % | 0.82 % | 1.9x | jamais |
| QTT 150000 | MC 16384 | 1.81 % | 0.82 % | 0.5x | des le 1er run |
| QTT 150000 | MC 4096 | 1.81 % | 4.11 % | 2.3x | jamais |
| QTT 400000 | MC 65536 | 0.44 % | 0.18 % | 0.4x | des le 1er run |
| QTT 200000 | MC 16384 | 0.33 % | 0.82 % | 2.4x | des le 1er run |
| QTT 50000 | MC 4096 | 11.12 % | 4.11 % | 0.4x | run 2 |
| QTT 150000 | MC 1024 | 1.81 % | 6.60 % | 3.6x | jamais |

## Reserves

1. **Les temps absolus portent une dispersion machine d'environ +-2.5x** pour un travail identique. Les rapports intra-run sont exploitables, les secondes prises isolement beaucoup moins.
2. **L'erreur de VaR du QTT n'est pas monotone en budget** alors que son erreur de prix l'est : le budget ne peut pas etre regle sur l'erreur de prix.
3. **L'initialisation QTT n'est amortissable que tant que les spots restent dans la boite de la grille.** Mesure ici : 0.00% de scenarios hors boite sur un horizon d'un jour. A verifier a nouveau si le point de marche derive sur plusieurs mois.
4. **Le cout de mise au point du budget** - balayer plusieurs budgets sur plusieurs graines contre une reference exacte - n'est compte dans aucun tableau.
