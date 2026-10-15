# VaR 1 jour d'un livre d'options sur panier géométrique : QTT contre Monte Carlo

> **Superseded — conserve pour le registre, ne pas citer les chiffres de cout.**
>
> Remplace par [`rapport_var_qtt_panier_otm13d.md`](rapport_var_qtt_panier_otm13d.md).
> Deux raisons, dans cet ordre d'importance :
>
> 1. **Une seule graine TT par budget.** Toute la section 4.6 et le point 3 du
>    §7 reposent sur un ajustement unique. Mesure faite depuis sur 3 graines :
>    l'erreur du QTT a budget fixe varie d'un ordre de grandeur d'une graine a
>    l'autre, donc les avantages « 11.6x a 31x » annonces ici ne sont pas
>    etablis. Les chiffres de *precision* par graine restent valides.
> 2. **Grille de prix 8-D.** Cette campagne utilisait
>    `moneyness_adaptive` en 8-D, qui ne convient pas a un livre etage en
>    maturites. Le script est passe a la grille de risque hybride 13-D et ne
>    reproduit plus ce chemin.
>
> Ce qui tient toujours : la donnee synthetique, le protocole de scenarios, le
> test d'appartenance a l'intervalle de confiance (§4.2), la compensation
> d'erreur dans le P&L (§4.3), et l'exactitude de la parite call/put (§4.4).

Expérience de revalorisation complète (*full revaluation*) d'un petit livre
d'options sur panier, sous 10 000 scénarios GBM multi-actifs corrélés. Trois
méthodes de revalorisation sont comparées sur **les mêmes scénarios** : la forme
fermée Black-Scholes panier géométrique (référence exacte), un Monte Carlo
imbriqué, et le surrogate QTT du dépôt.

Script : [`scripts/validation/benchmark_var_qtt_basket_book.py`](scripts/validation/benchmark_var_qtt_basket_book.py)
Figure : [`scripts/figures/plot_var_qtt_versus_monte_carlo.py`](scripts/figures/plot_var_qtt_versus_monte_carlo.py)
Sorties : `results/var_qtt_basket_book_headline.{json,csv}`, `experiments/figures/var_qtt_versus_monte_carlo.pdf`

---

## 1. Données

Marché entièrement synthétique, spécifié dans le script
(`ASSETS`, `CORRELATION_RHO`, `RISK_FREE_RATE`).

### 1.1 Sous-jacents

Cinq actions synthétiques, volatilités implicites **constantes** (la volatilité
n'est pas un facteur de risque ici : elle est figée et ne bouge pas entre la
date 0 et l'horizon).

| Ticker | Spot `S_0` | Vol. implicite | Dividende |
|--------|-----------:|---------------:|----------:|
| SYN-A  |  95.00     | 18 %           | 0 %       |
| SYN-B  | 100.00     | 22 %           | 0 %       |
| SYN-C  | 105.00     | 25 %           | 0 %       |
| SYN-D  |  90.00     | 30 %           | 0 %       |
| SYN-E  | 110.00     | 20 %           | 0 %       |

Les niveaux sont dispersés autour de 100 et les volatilités s'étalent de 18 % à
30 %, de sorte qu'aucun actif ne domine le panier et que la volatilité du panier
ne soit pas celle d'un actif unique déguisé.

### 1.2 Corrélation et taux

Matrice de corrélation équicorrélée, `ρ = 0.30` hors diagonale (valeur par
défaut de `PaperConfig`) :

```
 1.00  0.30  0.30  0.30  0.30
 0.30  1.00  0.30  0.30  0.30
 0.30  0.30  1.00  0.30  0.30
 0.30  0.30  0.30  1.00  0.30
 0.30  0.30  0.30  0.30  1.00
```

Taux sans risque `r = 3 %`, constant. Pas de dividendes : le panier géométrique
a alors un forward analytique simple et l'offset de parité reste exactement
`e^{-rT}(F - K)`.

### 1.3 Panier

Panier géométrique équipondéré, `G = (∏ S_i)^{1/5}`.

- `G_0 = 99.7492`
- `σ_B = 15.34 %` (volatilité lognormale du panier, effet de diversification :
  bien en dessous des 18–30 % individuels)

---

## 2. Le livre

Six positions, **toutes longues**, quantité 1 contrat chacune, maturité 1 an.
Trois moneyness — 80 %, 100 %, 120 % de `G_0` — et pour chacune un call et un
put.

| Position | Type | Strike | Moneyness | Quantité |
|----------|------|-------:|----------:|---------:|
| `call_80`  | call |  79.80 |  80 % | +1 |
| `put_80`   | put  |  79.80 |  80 % | +1 |
| `call_100` | call |  99.75 | 100 % | +1 |
| `put_100`  | put  |  99.75 | 100 % | +1 |
| `call_120` | call | 119.70 | 120 % | +1 |
| `put_120`  | put  | 119.70 | 120 % | +1 |

**Valeur de base (forme fermée) : 53.6597.**

Le strike est figé à la date 0 à partir de `G_0` ; il ne bouge plus ensuite,
donc la moneyness dérive avec les scénarios.

Le mélange call/put n'est pas décoratif. Un livre uniquement long de puts est
monotone décroissant en chaque spot et sa perte extrême vit dans un seul coin de
la distribution ; en ajoutant les calls, le livre devient une position longue de
convexité dont les deux queues comptent. C'est le cas où une erreur de
revalorisation lisse a le plus de chances de ne pas se retrancher dans le P&L.

---

## 3. Protocole

### 3.1 Scénarios

- GBM multi-actifs corrélé, **un pas exact** (la loi de `S_T` sous GBM est connue
  en forme fermée : pas de discrétisation de trajectoire, donc pas de biais de
  discrétisation).
- 10 000 scénarios, variables antithétiques.
- Horizon 1 jour ouvré = `1/252 = 0.003968` an.
- Drift risque-neutre (`r − q − σ²/2`). Sur un jour, l'écart avec un drift
  réel-monde est négligeable devant le terme de diffusion.
- Maturités vieillies de l'horizon : la valeur de base est prise en `T = 1.00`,
  les scénarios en `T = 0.99603`. Le thêta d'un jour est donc dans le P&L.
- **Les trois bras partagent le même jeu de scénarios**, dans le même ordre.
  Tout écart mesuré est donc une erreur de revalorisation, jamais une erreur
  d'échantillonnage des scénarios.

`P&L(s) = V(s) − V(0)` ; `perte = −P&L` ; `VaR_α = quantile_α(perte)`.

### 3.2 Les trois bras

| Bras | Description |
|------|-------------|
| `closed_form` | Black-Scholes panier géométrique. Exacte à la précision machine. **Référence.** Le call est calculé par sa propre formule `d₁/d₂`, pas par parité, pour que la parité reste une propriété vérifiable du bras QTT et non une définition partagée. |
| `monte_carlo` | Monte Carlo imbriqué : simulation terminale des cinq sous-jacents corrélés à chaque scénario, 4 096 chemins, antithétiques, **nombres aléatoires communs (CRN)** d'un scénario à l'autre. Les calls sont pricés directement, pas par parité. |
| `qtt` | Surrogate QTT, grille `moneyness_adaptive` 8-D `(S₁…S₅, log(K/G), r, T)`, forme physique `(32,32,32,32,32,64,8,8)`, interpolation cubique sur les 8 axes, TT-cross à 50 000 évaluations (+ 2 000 ANOVA), troncature `1e-8`. **Calibré uniquement sur des puts.** |

### 3.3 Parité call/put

Le QTT ne voit jamais un call. Le call est reconstruit par

```
C = P̂ + e^{−rT} (F − K),   F = G_0 · exp(carry · T)
```

où `carry = r − w·q − ½(σ²·w) + ½σ_B²`. L'offset est model-exact et ne coûte
aucun travail de pricing.

Trois fonctions ont été ajoutées à [`src/stngpr/pricers.py`](src/stngpr/pricers.py) :
`geometric_basket_moments`, `geometric_basket_parity_offset`,
`geometric_basket_call`, plus la classe `EuropeanGeometricBasketCallPricer` et
cinq tests dans `tests/test_core.py` (parité, réduction 1-D à Black-Scholes,
forward contre simulation, call MC contre forme fermée).

### 3.4 Comptage du temps

Le temps total d'un bras **inclut son initialisation** :

- QTT : construction de la grille + échantillonnage ANOVA d'ordre 2 + TT-cross.
- Monte Carlo : tirage des normales corrélées communes.
- Forme fermée : aucune initialisation.

Chaque section chronométrée est répétée 3 fois, la médiane est reportée, le
min/max reste dans le JSON. Threads BLAS fixés à 1 avant l'import de numpy.

---

## 4. Résultats

### 4.1 VaR et ES à 1 jour

10 000 scénarios, livre de valeur 53.6597.

| Bras | VaR 95 % | VaR 99 % | ES 95 % | ES 99 % |  95 % |  99 % |
|------|---------:|---------:|--------:|--------:|--------------:|--------------:|
| `closed_form` (réf.) | **0.448272** | **0.547887** | 0.506446 | 0.573877 | — | — |
| `monte_carlo` (4 096 ch.) | 0.430296 | 0.529880 | 0.488374 | 0.555770 | **−4.01 %** | **−3.29 %** |
| `qtt` (budget 50k) | 0.450330 | 0.550081 | 0.508645 | 0.576003 | **+0.46 %** | **+0.40 %** |

### 4.2 Le test qui tranche

Avec 10 000 scénarios, la VaR de référence est elle-même bruitée. L'intervalle
de confiance exact d'ordre (statistique d'ordre binomiale, 95 %) est :

- VaR 95 % : `[0.441507, 0.454015]`
- VaR 99 % : `[0.540337, 0.553500]`

**La VaR du QTT tombe à l'intérieur de cet intervalle aux deux niveaux. Celle du
Monte Carlo imbriqué en sort aux deux niveaux.**

Autrement dit : à 10 000 scénarios, le QTT est statistiquement indiscernable de
la revalorisation exacte, tandis que le Monte Carlo imbriqué à 4 096 chemins
introduit un biais que l'échantillonnage des scénarios n'explique pas.

### 4.3 Compensation de l'erreur dans le P&L

Une VaR est un quantile d'une *différence* de prix. Si l'erreur de
revalorisation varie lentement avec les facteurs de risque, elle se retranche
entre le scénario et le point de base.

| Bras | MAE prix (portefeuille) | MAE P&L | Ratio de compensation |
|------|------------------------:|--------:|----------------------:|
| `monte_carlo` | 0.5100 (0.95 % de la valeur) | 0.01393 | **0.027** |
| `qtt` | 0.000620 (0.0012 %) | 0.000781 | **1.26** |

Les deux régimes annoncés se lisent directement :

- Le Monte Carlo a une erreur de prix **énorme** (biais de −0.51, soit −0.95 %
  du livre) mais quasi constante d'un scénario à l'autre grâce aux CRN : 97 %
  s'annule dans le P&L. Sans CRN, cette colonne serait du bruit indépendant et
  la VaR serait bien pire.
- Le QTT a une erreur de prix **800 fois plus petite**, mais qui ne se
  compense pas (ratio 1.26) : ce qui reste est du bruit d'interpolation à une
  échelle plus fine que le déplacement d'un jour. C'est pour cela que
  l'avantage du QTT sur la VaR (8×) est bien moindre que son avantage sur le
  prix (800×).

Classement des scénarios (les deux méthodes identifient parfaitement la queue) :

| Bras | Corrélation de rang de Spearman | Recouvrement de la queue à 1 % |
|------|--------------------------------:|-------------------------------:|
| `monte_carlo` | 1.0000000 | 100 % |
| `qtt` | 0.9999999982 | 100 % |

Les 100 pires scénarios sont les mêmes pour les trois méthodes. L'erreur est
entièrement dans le **niveau**, pas dans l'identification du risque.

### 4.4 Parité call/put — vérification

Résidu max `|erreur call − erreur put|` à strikes appariés : **5.24 × 10⁻¹⁴**.

Erreur de prix moyenne par position (QTT) :

| Strike | call | put |
|--------|-----:|----:|
| 80 %  | 6.724340 × 10⁻⁴ | 6.724340 × 10⁻⁴ |
| 100 % | 1.093231 × 10⁻³ | 1.093231 × 10⁻³ |
| 120 % | 4.958020 × 10⁻⁴ | 4.958020 × 10⁻⁴ |

Identiques à 12 chiffres significatifs. **La parité n'empile pas une seconde
approximation** : l'erreur sur un call est exactement l'erreur du surrogate sur
le put de même strike. Calibrer le QTT sur les seuls puts est donc gratuit — il
n'y a rien à gagner à l'entraîner aussi sur les calls.

(À comparer avec le bras Monte Carlo, où call et put sont pricés
indépendamment : MAE `call_100` = 0.210 contre `put_100` = 0.154.)

### 4.5 Coût

Temps de ce run (médiane sur 3 répétitions), 60 006 revalorisations de position :

| Bras | Initialisation | Revalorisation | **Total** |
|------|---------------:|---------------:|----------:|
| `closed_form` | 0 s | 0.007 s | **0.007 s** |
| `monte_carlo` (4 096 ch.) | 0.0002 s | 7.39 s | **7.39 s** |
| `qtt` (budget 50k) | 9.14 s | 2.80 s | **11.94 s** |

**Tel quel, le QTT est plus lent que le Monte Carlo en temps total** (11.94 s
contre 7.39 s) — mais huit fois plus précis. Comparer deux méthodes qui ne sont
pas à la même précision ne dit rien : celle qui est moins chère l'est à la
précision qu'elle a.

### 4.6 À précision égale

Courbe coût/précision des deux méthodes, mesurée **dans le même processus**
(voir figure). L'erreur du Monte Carlo à budget fixé est un tirage aléatoire :
chaque budget est donc répété sur 5 graines indépendantes et l'erreur reportée
est la moyenne quadratique.

| Chemins MC | Temps total |  95 % |  99 % |
|-----------:|------------:|--------------:|--------------:|
|  1 024 |  1.97 s |  9.21 % | 10.67 % |
|  4 096 |  7.43 s |  2.67 % |  3.05 % |
| 16 384 | 28.70 s |  1.19 % |  1.45 % |
| 65 536 | 94.46 s |  0.68 % |  0.87 % |

| Budget TT | Init. | Temps total | rang eff. | params |  95 % |  99 % |
|----------:|------:|------------:|----------:|-------:|--------------:|--------------:|
|  10 000 | 1.66 s | 2.73 s | 5.20 | 1 910 | 2.486 % | 2.299 % |
|  20 000 | 2.02 s | 3.22 s | 6.13 | 2 656 | 6.437 % | 7.780 % |
|  35 000 | 2.57 s | 3.61 s | 6.89 | 3 354 | **0.454 %** | **0.388 %** |
|  50 000 | 3.35 s | 4.50 s | 7.30 | 3 762 | 0.459 % | 0.401 % |
| 100 000 | 8.73 s | 9.85 s | 7.50 | 3 972 | 0.463 % | 0.404 % |

> **Attention en lisant les deux tableaux ci-dessus contre le §4.5.** La même
> configuration QTT (budget 50 000) y apparaît à 3.35 s d'initialisation, alors
> que le §4.5 la donne à 9.14 s — *dans le même processus, même graine, même
> travail*. Les deux mesures sont réelles ; l'écart de 2.7× est la dispersion
> machine documentée au §5.1. Le bras QTT est mesuré juste après le bras Monte
> Carlo, qui alloue un pic de ~1.6 Go, ce qui est l'explication la plus
> plausible mais n'a pas été isolée. Aucune conclusion de ce rapport ne repose
> sur laquelle des deux est « la bonne ».

Extrapolation le long de la loi en `n^{-1/2}` ajustée sur la courbe MC
(exposants mesurés −0.62 et −0.60) :

| Niveau | Chemins MC requis | Temps total MC estimé | Avantage QTT |
|--------|------------------:|----------------------:|-------------:|
| 95 % | 96 016 (23× le budget mesuré) | ≈ 138 s | **11.6 ×** à **31 ×** |
| 99 % | 179 904 (44× le budget mesuré) | ≈ 259 s | **21.7 ×** à **58 ×** |

La fourchette est l'effet direct de la dispersion ci-dessus : borne basse avec
le temps total QTT du §4.5 (11.94 s), borne haute avec celui de la courbe
(4.50 s). Le chiffre à citer est donc l'ordre de grandeur — **un facteur 10 à
50** — pas une décimale.

Formulation indépendante de la machine, et de loin la plus solide : pour
atteindre la précision du QTT, le Monte Carlo imbriqué doit exécuter **23 à 44
fois l'arithmétique** de sa configuration à 4 096 chemins. Ce rapport-là ne
dépend d'aucune horloge.

### 4.7 Point de fonctionnement

Le QTT **plafonne à 0.46 % / 0.40 % dès 35 000 évaluations**. Passer à 400 000
(mesuré séparément) ne change rien au-delà du troisième chiffre tout en
multipliant l'initialisation par 20. C'est le symptôme connu du dépôt
(`CLAUDE.md`, gotcha 9) : `teneva.cross` est appelé sans tolérance et tourne
jusqu'au budget demandé, donc le rang continue de croître longtemps après que
l'erreur a cessé de baisser.

Ce qui fixe le plancher de 0.46 % est **l'axe log-moneyness**, pas les spots ni
la maturité (tous à budget 50 000) :

| Variante |  95 % |  99 % |
|----------|--------------:|--------------:|
| Grille papier `(32⁵, 64, 8, 8)` | 0.46 % | 0.40 % |
| Spots 32 → 64 | 0.47 % | 0.49 % |
| Maturité 8 → 32 | 0.52 % | 0.54 % |
| **Strike 64 → 128** | **0.16 %** | **0.15 %** |
| Les trois ensemble | 0.27 % | 0.30 % |

Doubler le seul axe de strike **divise l'erreur par ~2.8**. Raffiner les spots
ou la maturité ne fait rien, voire dégrade légèrement : à budget constant, une
grille plus grande est un ajustement plus difficile.

C'était contre-intuitif a priori. La maille spot (4.68) vaut 4.9 fois le
déplacement du panier sur un jour (0.96), et la maille de maturité autour de
`T = 1` vaut 139 fois l'horizon — on pouvait donc s'attendre à ce que ces deux
axes dominent. Ils ne dominent pas : le prix est lisse et quasi linéaire le long
de ces deux directions à ces échelles, et l'interpolation cubique le capture.
C'est la courbure en moneyness qui n'est pas capturée.

**Recommandation** : budget 35 000–50 000, axe de strike à 128 nœuds. Cela donne
0.16 % / 0.15 % d'erreur de VaR.

---

## 5. Menaces à la validité

1. **Les temps absolus sont peu fiables sur cette machine.** Le bras Monte
   Carlo, travail strictement identique, a été mesuré à 2.74 s dans un run et à
   7.39 s dans un autre ; à l'intérieur d'un même run ses trois répétitions
   donnent `[3.80, 7.50, 7.39]` s. Les répétitions du QTT sont serrées
   (`[9.36, 9.14, 9.13]` s pour l'ajustement), donc la dispersion n'est pas
   uniforme. **Tous les ratios de temps cités sont intra-run**, jamais entre
   runs, et portent quand même une dispersion de l'ordre de ±2×. Les conclusions
   de *précision* sont exactes et reproductibles, elles ne sont pas touchées.
2. **Le QTT n'est pas monotone en budget.** 20 000 évaluations donnent 6.44 %,
   soit *pire* que 10 000 (2.49 %). Le rang adapté par TT-cross à un budget
   intermédiaire est arbitraire, pas convergent. Un budget ne doit pas être
   choisi sur un point isolé.
3. **Sur ce produit, la forme fermée gagne largement** : 7 ms contre 7–12 s.
   Le panier géométrique a une solution analytique, donc la comparaison
   QTT/Monte Carlo ne vaut que comme *proxy* des produits où cette colonne
   n'existe pas (panier arithmétique, américain). Le résultat à transposer est
   le rapport QTT/Monte Carlo, pas les temps absolus.
4. **Le Monte Carlo imbriqué est mesuré avec un moteur générique** qui simule
   les cinq sous-jacents. Un moteur conscient du payoff peut précalculer le
   facteur multiplicatif du panier géométrique, commun à tous les scénarios :
   mesuré ici à **2.9× plus rapide**, pour un estimateur strictement identique
   (écart max `4.97 × 10⁻¹⁴`). L'avantage QTT à précision égale tomberait donc
   d'environ 11.6× à ~4× face à un tel moteur. Cette optimisation n'existe que
   pour le panier géométrique.
5. **Volatilité figée.** Elle n'est pas un facteur de risque ici. Une VaR qui
   choque aussi la surface de volatilité demanderait le chemin 13-D
   (`vol_extension.py`), nettement plus coûteux à initialiser.
6. **Horizon 1 jour uniquement.** À 10 jours le déplacement des spots est ~3×
   plus grand, la compensation d'erreur dans le P&L change de régime, et le
   classement des méthodes peut bouger.
7. **Aucun scénario ne sort de la boîte de la grille** (0.00 %). C'est propre
   ici, mais c'est une conséquence de l'horizon court : à horizon plus long ce
   diagnostic doit être relu avant de faire confiance à un chiffre QTT.
8. **Une seule graine de scénarios.** Les VaR reportées héritent de
   l'échantillonnage de ces 10 000 trajectoires ; les intervalles du §4.2 le
   quantifient, mais l'expérience n'a pas été répétée sur plusieurs jeux de
   scénarios.

---

## 6. Reproduire

```bash
# Run principal (~10 min) : les trois bras + les deux courbes coût/précision
.venv/Scripts/python.exe scripts/validation/benchmark_var_qtt_basket_book.py \
    --profile full --budget 50000 --tag headline

# Variante grille raffinée (axe de strike à 128 nœuds)
.venv/Scripts/python.exe scripts/validation/benchmark_var_qtt_basket_book.py \
    --profile full --budget 50000 --strike-nodes 128 --mc-path-sweep --tag refined

# Vérification rapide (~1 min)
.venv/Scripts/python.exe scripts/validation/benchmark_var_qtt_basket_book.py --profile smoke

# Figure
.venv/Scripts/python.exe scripts/figures/plot_var_qtt_versus_monte_carlo.py
```

Environnement : Python 3.12.9, numpy 2.5.2, teneva 0.14.11, Windows 11,
`OMP_NUM_THREADS=1`. Graines : scénarios 20260921, Monte Carlo 4242, TT 20260327.

---

## 7. Ce qu'il faut retenir

1. **Le QTT reproduit la VaR exacte à 0.46 % / 0.40 %** à 1 jour, ce qui le
   place *à l'intérieur* de l'intervalle de confiance d'échantillonnage de la
   référence à 10 000 scénarios. Le Monte Carlo imbriqué à 4 096 chemins en
   sort, avec −4.01 % / −3.29 %.
2. **La parité call/put est gratuite et exacte.** Calibrer le surrogate sur les
   seuls puts ne coûte rien : le résidu `|err. call − err. put|` vaut
   5.2 × 10⁻¹⁴. Cela divise par deux la surface à apprendre.
3. **À précision égale, le QTT est moins cher en temps total, initialisation
   comprise, d'un facteur 10 à 50** selon la dispersion machine — ou, sans
   horloge : le Monte Carlo doit faire 23× (95 %) à 44× (99 %) l'arithmétique
   de sa configuration mesurée pour y arriver. Mais tel quel, à budget non
   ajusté, le QTT est 1.6× *plus* cher que le Monte Carlo tout en étant 8× plus
   précis.
4. **Le budget TT plafonne à 35 000 évaluations.** Au-delà, l'initialisation
   grossit sans que la VaR bouge.
5. **L'axe de strike est le goulot, pas les spots ni la maturité.** Le passer à
   128 nœuds amène la VaR à 0.16 % / 0.15 %.
6. Les deux méthodes approchées **identifient parfaitement les scénarios de
   queue** (recouvrement 100 %, Spearman ≈ 1). L'erreur est un décalage de
   niveau, pas une erreur de classement du risque.
