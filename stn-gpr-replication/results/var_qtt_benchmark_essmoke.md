# Benchmark VaR / ES 1 jour - QTT 13-D contre Monte Carlo imbrique

Livre de 6 options sur panier geometrique, valeur 9.2519. 2000 scenarios GBM correles, horizon 1 jour. Grille QTT 13-D `bounded_standardized_risk`, layout `interleaved`, 59 coeurs.

Toutes les erreurs des sections 2 a 6 sont des moyennes quadratiques sur 3 graines independantes : a budget fixe, ni le TT-cross ni le Monte Carlo ne produisent un nombre reproductible.

## 1. Mesures de risque - bras principaux

QTT au budget 20 000, Monte Carlo a 1 024 chemins internes, une graine chacun.

| Bras | VaR 95 % | VaR 99 % | **ES 97,5 %** | Err. VaR 95 % | Err. VaR 99 % | **Err. ES 97,5 %** |
|---|---:|---:|---:|---:|---:|---:|
| Forme fermee (reference) | 0.747329 | 0.890015 | **0.871814** | - | - | - |
| Monte Carlo imbrique | 0.669517 | 0.796867 | **0.780412** | -10.41 % | -10.47 % | -10.48 % |
| QTT 13-D | 0.791578 | 0.894666 | **0.880939** | +5.92 % | +0.52 % | +1.05 % |

Intervalle de confiance a 95 % de l'ES 97,5 % de reference, du seul fait de l'echantillonnage des 2000 scenarios : `[0.859452, 0.884176]` (moyenne de queue sur 50 scenarios, erreur type asymptotique).

Monte Carlo imbrique : dehors ; QTT 13-D : dedans.

## 2. Configurations QTT

| Budget TT-cross | Err. VaR 95 % | Err. VaR 99 % | **Err. ES 97,5 %** | Initialisation (une fois) | VaR / ES quotidienne |
|---|---:|---:|---:|---:|---:|
| 10 000 | 46.91 % | 47.75 % | **47.22 %** | 3.7 s | 0.5 s |
| 20 000 | 15.20 % | 22.05 % | **21.31 %** | 4.1 s | 1.0 s |

## 3. Configurations Monte Carlo

| Chemins internes | Err. VaR 95 % | Err. VaR 99 % | **Err. ES 97,5 %** | Initialisation (une fois) | VaR / ES quotidienne |
|---|---:|---:|---:|---:|---:|
| 256 | 8.70 % | 8.52 % | **8.54 %** | 0.00 s | 0.2 s |
| 1 024 | 5.68 % | 6.11 % | **6.06 %** | 0.00 s | 0.8 s |
| 4 096 | 3.21 % | 3.16 % | **3.17 %** | 0.00 s | 2.7 s |

La VaR et l'ES sortent du meme jeu de pertes : le cout quotidien est celui de la revalorisation, l'ES n'y ajoute qu'un tri de 2 000 nombres.

## 4. Amortissement : cout total apres N runs

`cout(N) = initialisation + N x cout quotidien`. Le surrogate QTT s'initialise une fois et se reutilise tel quel les jours suivants : seuls les spots bougent, et la grille les couvre deja.

| Configuration | Err. VaR 95 % | Err. ES 97,5 % | N = 1 | N = 5 | N = 21 | N = 63 | N = 252 |
|---|---:|---:|---:|---:|---:|---:|---:|
| QTT 10000 | 46.91 % | 47.22 % | 4 s | 6 s | 14 s | 35 s | 128 s |
| QTT 20000 | 15.20 % | 21.31 % | 5 s | 9 s | 25 s | 66 s | 252 s |
| MC 256 | 8.70 % | 8.54 % | 0 s | 1 s | 5 s | 14 s | 54 s |
| MC 1024 | 5.68 % | 6.06 % | 1 s | 4 s | 17 s | 50 s | 201 s |
| MC 4096 | 3.21 % | 3.17 % | 3 s | 13 s | 57 s | 170 s | 678 s |

## 5. Croisements a precision comparable - ES 97,5 %

Seules les paires dont le rapport de precision MC/QTT est compris entre 0,5 et 2 sont listees : au-dela, un croisement de cout ne dit rien. Rapport > 1 : le QTT est le plus precis des deux.

_Aucune paire de precision comparable (rapport entre 0.5 et 2)._

A precision egale sur l'ES 97,5 % (21.31 %, erreur QTT moyennee sur graines au budget 20 000), le Monte Carlo demanderait 22 chemins (mesure), soit ~0 s par run.

## 6. Croisements a precision comparable - VaR 95 %

| QTT | Monte Carlo | Err. QTT | Err. MC | Rapport MC/QTT | Croisement |
|---|---|---:|---:|---:|---|
| QTT 20000 | MC 256 | 15.20 % | 8.70 % | 0.6x | jamais : VaR quotidienne QTT plus chere (QTT 1.7x moins precis) |

## Reserves

1. **Les temps absolus portent une dispersion machine d'environ +-2.5x** pour un travail identique. Les rapports intra-run sont exploitables, les secondes prises isolement beaucoup moins.
2. **L'erreur de risque du QTT n'est pas monotone en budget** alors que son erreur de prix l'est : le budget ne peut pas etre regle sur l'erreur de prix.
3. **L'ES 97,5 % est une moyenne sur les 50 pires scenarios**, pas un quantile : il est moins sensible qu'une VaR a la position d'une seule perte, et son intervalle de confiance est asymptotique (erreur type de la moyenne de queue).
4. **L'initialisation QTT n'est amortissable que tant que les spots restent dans la boite de la grille.** Mesure ici : 0.00% de scenarios hors boite sur un horizon d'un jour.
5. **Le cout de mise au point du budget** - balayer plusieurs budgets sur plusieurs graines contre une reference exacte - n'est compte dans aucun tableau.
