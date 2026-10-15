# Benchmark VaR 1 jour - QTT 13-D contre Monte Carlo imbrique

Livre de 6 options sur panier geometrique, valeur 9.2519. 2000 scenarios GBM correles, horizon 1 jour. Grille QTT 13-D `bounded_standardized_risk`, layout `interleaved`, 59 coeurs.

Reference exacte (Black-Scholes panier geometrique) : VaR 95 % = 0.747329, VaR 99 % = 0.890015.

Toutes les erreurs sont des moyennes quadratiques sur 3 graines independantes : a budget fixe, ni le TT-cross ni le Monte Carlo ne produisent un nombre reproductible.

## 1. Configurations QTT

| Budget TT-cross | Err. VaR 95 % | Err. VaR 99 % | Initialisation (une fois) | VaR quotidienne |
|---|---:|---:|---:|---:|
| 10 000 | 46.91 % | 47.75 % | 3.2 s | 0.4 s |
| 20 000 | 15.20 % | 22.05 % | 3.5 s | 0.8 s |

## 2. Configurations Monte Carlo

| Chemins internes | Err. VaR 95 % | Err. VaR 99 % | Initialisation (une fois) | VaR quotidienne |
|---|---:|---:|---:|---:|
| 256 | 8.70 % | 8.52 % | 0.00 s | 0.2 s |
| 1 024 | 5.68 % | 6.11 % | 0.00 s | 0.6 s |
| 4 096 | 3.21 % | 3.16 % | 0.00 s | 2.4 s |

## 3. Amortissement : cout total apres N runs de VaR

`cout(N) = initialisation + N x VaR quotidienne`. Le surrogate QTT s'initialise une fois et se reutilise tel quel les jours suivants : seuls les spots bougent, et la grille les couvre deja.

| Configuration | Err. VaR 95 % | N = 1 | N = 5 | N = 21 | N = 63 | N = 252 |
|---|---:|---:|---:|---:|---:|---:|
| QTT 10000 | 46.91 % | 4 s | 5 s | 12 s | 30 s | 112 s |
| QTT 20000 | 15.20 % | 4 s | 7 s | 20 s | 53 s | 202 s |
| MC 256 | 8.70 % | 0 s | 1 s | 3 s | 10 s | 41 s |
| MC 1024 | 5.68 % | 1 s | 3 s | 13 s | 40 s | 161 s |
| MC 4096 | 3.21 % | 2 s | 12 s | 49 s | 148 s | 593 s |

## 4. Croisements a precision comparable

Un croisement n'a de sens qu'entre configurations de precision voisine ; la colonne `rapport` le rend verifiable (1.0 = precisions egales, 10 = le Monte Carlo est dix fois moins precis).

| QTT | Monte Carlo | Err. QTT | Err. MC | Rapport | Croisement |
|---|---|---:|---:|---:|---|
| QTT 20000 | MC 256 | 15.20 % | 8.70 % | 0.6x | jamais |
| QTT 20000 | MC 1024 | 15.20 % | 5.68 % | 0.4x | jamais |
| QTT 20000 | MC 4096 | 15.20 % | 3.21 % | 0.2x | run 3 |
| QTT 10000 | MC 256 | 46.91 % | 8.70 % | 0.2x | jamais |
| QTT 10000 | MC 1024 | 46.91 % | 5.68 % | 0.1x | run 16 |
| QTT 10000 | MC 4096 | 46.91 % | 3.21 % | 0.1x | run 2 |

## Reserves

1. **Les temps absolus portent une dispersion machine d'environ +-2.5x** pour un travail identique. Les rapports intra-run sont exploitables, les secondes prises isolement beaucoup moins.
2. **L'erreur de VaR du QTT n'est pas monotone en budget** alors que son erreur de prix l'est : le budget ne peut pas etre regle sur l'erreur de prix.
3. **L'initialisation QTT n'est amortissable que tant que les spots restent dans la boite de la grille.** Mesure ici : 0.00% de scenarios hors boite sur un horizon d'un jour. A verifier a nouveau si le point de marche derive sur plusieurs mois.
4. **Le cout de mise au point du budget** - balayer plusieurs budgets sur plusieurs graines contre une reference exacte - n'est compte dans aucun tableau.
