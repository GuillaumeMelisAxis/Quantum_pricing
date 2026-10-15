# VaR et ES 1 jour d'un livre OTM à maturités étagées : QTT 13-D contre Monte Carlo

Revalorisation complète d'un livre de six options **hors de la monnaie**, de
maturité 1 mois à 2 ans, sous 10 000 scénarios GBM multi-actifs corrélés. Trois
méthodes de revalorisation sur **les mêmes scénarios** : forme fermée
Black-Scholes panier géométrique (référence exacte), Monte Carlo imbriqué, et
le surrogate QTT sur la **grille de risque hybride 13-D**. Mesures : VaR 95 %,
VaR 99 % et **Expected Shortfall 97,5 %** (mesure réglementaire FRTB).

Script : [`scripts/validation/benchmark_var_qtt_basket_book.py`](scripts/validation/benchmark_var_qtt_basket_book.py)
Figures : [`scripts/figures/plot_var_qtt_versus_monte_carlo.py`](scripts/figures/plot_var_qtt_versus_monte_carlo.py) (`--metric var` / `--metric es`)
Sorties :
- `results/var_qtt_benchmark_bench_es.md` — **les tableaux du benchmark, prêts à insérer**
- `results/var_qtt_basket_book_bench_es.{json,csv}` — run de référence pour le coût (§6) et l'ES (§5.5)
- `results/var_qtt_basket_book_otm13d_final.{json,csv}` — run des §4–5.4
- `experiments/figures/var_qtt_versus_monte_carlo.pdf`, `experiments/figures/es_qtt_versus_monte_carlo.pdf`

Les deux runs partagent graines, livre et scénarios : leurs **précisions sont
identiques au dernier chiffre** (elles ne dépendent que du triplet budget,
graine, grille). Seuls leurs temps diffèrent, dans la dispersion machine du §7.

> Remplace [`rapport_var_qtt_panier.md`](rapport_var_qtt_panier.md), qui portait
> sur un livre apparié à maturité unique et sur une grille de **prix 8-D**. Les
> conclusions de coût de ce rapport-là ne se transposent pas ici : elles ont été
> établies sur une graine TT unique, et la grille 8-D n'est pas utilisable pour
> un livre étagé en maturités. Le script ne reproduit plus ce chemin.

---

## 1. Données

Marché synthétique, inchangé par rapport à la campagne précédente, spécifié
dans le script (`ASSETS`, `CORRELATION_RHO`, `RISK_FREE_RATE`).

### 1.1 Sous-jacents

Cinq actions synthétiques, volatilités implicites **constantes** — la
volatilité n'est pas un facteur de risque ici, elle ne bouge pas entre la date 0
et l'horizon.

| Ticker | Spot `S_0` | Vol. implicite | Dividende |
|--------|-----------:|---------------:|----------:|
| SYN-A  |  95.00     | 18 %           | 0 %       |
| SYN-B  | 100.00     | 22 %           | 0 %       |
| SYN-C  | 105.00     | 25 %           | 0 %       |
| SYN-D  |  90.00     | 30 %           | 0 %       |
| SYN-E  | 110.00     | 20 %           | 0 %       |

### 1.2 Corrélation et taux

Équicorrélation `ρ = 0.30` hors diagonale, taux sans risque `r = 3 %` constant,
pas de dividendes.

### 1.3 Panier

Panier géométrique équipondéré `G = (∏ S_i)^{1/5}` : `G_0 = 99.7492`,
`σ_B = 15.34 %`.

---

## 2. Le livre — six options OTM, échelle de maturités

Toutes longues, quantité 1 contrat, strikes figés à la date 0 à partir de `G_0`.
Moneyness de 80 % à 120 %, six maturités distinctes de 1 mois à 2 ans.

| Position | Type | Strike | Moneyness | Maturité | Écarts-types OTM | Valeur (forme fermée) |
|----------|------|-------:|----------:|---------:|-----------------:|----------------------:|
| `put_95_1m`   | put  |  94.76 |  95 % | 1 mois  | −1.16 | 0.2481 |
| `call_105_3m` | call | 104.74 | 105 % | 3 mois  | +0.64 | 1.3373 |
| `put_90_6m`   | put  |  89.77 |  90 % | 6 mois  | −0.97 | 0.7851 |
| `call_110_1y` | call | 109.72 | 110 % | 1 an    | +0.62 | 2.9674 |
| `put_80_18m`  | put  |  79.80 |  80 % | 18 mois | −1.19 | 0.7301 |
| `call_120_2y` | call | 119.70 | 120 % | 2 ans   | +0.84 | 3.1838 |

**Valeur de base : 9.2519.**

### Pourquoi cet appariement moneyness/maturité

L'appariement n'est pas libre. Ce qui dit si une position OTM est encore vivante
n'est pas la moneyness mais la moneyness *standardisée*,
`log(m) / (σ_B √T)`. Un put à 80 % à un mois est à **−5.0 écarts-types** : il
vaut zéro, il ne pèse rien dans la VaR, et l'inclure reviendrait à faire
l'expérience sur trois positions et trois zéros.

Les moneyness les plus éloignées sont donc portées par les maturités les plus
longues. Les six positions tiennent ainsi entre **0.62 et 1.19 écart-type** de
la monnaie : six positions réellement vivantes, et un livre qui couvre
véritablement le plan (strike × maturité).

Le mélange call/put reste délibéré : un livre uniquement long de puts est
monotone décroissant en chaque spot et sa perte extrême vit dans un seul coin de
la distribution. Avec les calls, le livre est une position longue de convexité
dont les deux queues comptent.

---

## 3. Protocole

### 3.1 Scénarios

GBM multi-actifs corrélé, **un pas exact** (loi de `S_T` connue en forme fermée,
donc pas de biais de discrétisation), 10 000 scénarios antithétiques, horizon
1 jour ouvré = `1/252 = 0.003968` an, drift risque-neutre, maturités vieillies de
l'horizon. **Les trois bras partagent le même jeu de scénarios**, dans le même
ordre : tout écart mesuré est une erreur de revalorisation, jamais une erreur
d'échantillonnage des scénarios.

`P&L(s) = V(s) − V(0)` ; `perte = −P&L` ; `VaR_α = quantile_α(perte)`.

### 3.2 La grille QTT — risque hybride 13-D

| | |
|---|---|
| Coordonnées | `(S_1, σ_1, …, S_5, σ_5, K, r, T)` — 13 axes |
| Famille | `bounded_standardized_risk` |
| Layout | `interleaved` |
| Forme physique | `(32, 16, 32, 16, 32, 16, 32, 16, 32, 16, 64, 8, 32)` |
| Cœurs QTT | **59** |
| Cible d'ajustement | `P / G` |
| Interpolation | cubique sur les 13 axes |
| Troncature post-fit | `1e-8` |

Trois choix méritent justification.

**`bounded_standardized_risk`.** Le mode `standardized_risk` non borné ne se
construit pas en 13-D : un unique intervalle `z` global doit couvrir tout
`(r, T, σ)`, et avec σ de 0.05 à 0.80 l'enveloppe atteint `|z| ≈ 8.7`, ce qui
projette le coin opposé de la boîte sur `|log(K/G)| ≈ 2.6 × 10³` — un strike qui
n'est pas un flottant. La variante bornée est la réponse générale
(`CLAUDE.md` gotcha 7).

**`maturity_nodes = 32` et non 8.** La coordonnée de strike de cette grille est
fonction de `(r, T, σ)`, donc une même coordonnée désigne un strike différent à
chaque maturité et la surface bouge le long de `T` bien plus vite que sur une
grille de moneyness. Avec 32 nœuds, la pire maille du livre tombe à 0.159 an
(à `T = 2`) contre 0.673 an avec 8 nœuds.

**`interleaved` et non `blocked` — mesuré, pas supposé.** Le gotcha 3 du dépôt
recommande `blocked` pour un panier géométrique ajusté en `P/G`, parce que le
prix y est exactement indépendant des spots. Sur cette grille, c'est faux :

| Layout | Err. VaR 95 % | Err. VaR 99 % | Rang effectif |
|--------|--------------:|--------------:|--------------:|
| `blocked` | 0.60 % | 0.32 % | 23.2 |
| **`interleaved`** | **0.06 %** | **0.16 %** | 22.0 |

Facteur 10 à rang effectif équivalent (budget 150k, graine unique). Le gotcha a
été établi sur les grilles de *prix* ; ici la coordonnée de strike est elle-même
fonction de σ, donc le couplage σ↔strike ne se trouve plus là où `blocked` le
place.

**La volatilité est portée comme coordonnée bien qu'elle soit figée** dans cette
expérience. C'est ce qui rend le surrogate réutilisable pour une VaR qui
choquerait aussi la surface — au prix d'une initialisation et d'une requête
nettement plus chères (§6).

### 3.3 Les trois bras

| Bras | Description |
|------|-------------|
| `closed_form` | Black-Scholes panier géométrique, exacte à la précision machine. **Référence.** Le call est calculé par sa propre formule `d₁/d₂`, pas par parité. |
| `monte_carlo` | Monte Carlo imbriqué, 4 096 chemins, antithétiques, **nombres aléatoires communs (CRN)**. Calls pricés directement. |
| `qtt` | Surrogate 13-D ci-dessus, **calibré uniquement sur des puts**, calls par parité. |

### 3.4 Décompte du Monte Carlo imbriqué

| | |
|---|---:|
| Revalorisations de position | 60 006 |
| Simulations d'état (une par seau de maturité) | 60 006 |
| Valeurs terminales d'actif calculées | 1.229 × 10⁹ |
| Évaluations de payoff | 2.458 × 10⁸ |
| **Normales réellement tirées** | **20 480, une seule fois** |

La dernière ligne est le contrat CRN : les 4 096 × 5 normales sont tirées à la
construction et réutilisées pour tous les scénarios et toutes les positions.
C'est ce qui fait que l'erreur de prix se retranche dans le P&L (§5.3).

Le livre ayant **six maturités distinctes**, le moteur simule un jeu terminal
par seau `(r, T)` : six fois plus de travail de simulation que le livre à
maturité unique de la campagne précédente, pour le même nombre de chemins.

### 3.5 Parité call/put

Le QTT ne voit jamais un call :
`C = P̂ + e^{−rT}(F − K)` avec `F = G_0 · exp(carry · T)`. L'offset est
model-exact et ne coûte aucun travail de pricing.

Le livre n'ayant plus de paires call/put appariées, l'audit reprice, **aux
paramètres exacts de chaque call, le put fantôme de même strike et même
maturité** à travers le même surrogate, et vérifie que l'erreur du call égale
celle de ce put.

### 3.6 Comptage du temps

Le temps total inclut l'initialisation : construction de la grille 13-D +
ANOVA d'ordre 2 + TT-cross pour le QTT, tirage des normales pour le Monte Carlo,
rien pour la forme fermée. Le passage 8-D → 13-D des points de marché est compté
dans la **revalorisation**, puisque c'est un coût que cette grille impose à
chaque requête. Threads BLAS fixés à 1 avant l'import de numpy.

---

## 4. Le problème de méthode central : la VaR n'hérite pas de la convergence de l'ajustement

**Aucune des deux méthodes approchées ne produit un nombre reproductible à
budget fixé.** Un premier passage à graine unique donnait, pour le QTT, 4.67 %,
12.95 %, 5.17 %, **0.06 %** et 0.44 % à 25k, 50k, 100k, 150k et 200k — deux
ordres de grandeur d'oscillation avec un rang effectif qui croît régulièrement.
Publier le 0.06 % aurait été publier un tirage.

Chaque budget est donc mesuré sous **trois graines indépendantes** — graines
TT-cross pour le QTT, graines Monte Carlo pour le MC — et l'erreur reportée est
la moyenne quadratique, avec son étendue.

Mais la dispersion entre graines n'est que la moitié de l'histoire, et pas la
plus importante. **L'ajustement TT, lui, converge parfaitement** ; c'est le
passage de l'erreur de prix à l'erreur de VaR qui ne suit pas. Le §4.3
l'établit et l'explique.

### 4.1 Monte Carlo : le tirage existe, mais il est gouverné

| Chemins | graine 4242 | graine 911 | graine 20260921 | RMS | étendue / RMS |
|--------:|------------:|-----------:|----------------:|----:|--------------:|
|  1 024 | 10.40 % | 4.52 % | 1.43 % | 6.60 % | 1.36 |
|  4 096 |  5.46 % | 1.30 % | 4.39 % | 4.11 % | 1.01 |
| 16 384 |  1.19 % | 0.59 % | 0.49 % | 0.82 % | 0.85 |

À 4 096 chemins, deux graines diffèrent d'un facteur 4.2. Mais le RMS descend de
façon monotone (exposant ajusté −0.75) **et l'étendue relative se resserre**.
Payer des chemins achète de la précision, de façon fiable.

### 4.2 QTT : le tirage n'est pas gouverné

| Budget | Rang eff. | **MAE prix** | Err. VaR 95 % (RMS) | Étendue sur graines | Err. VaR 99 % |
|-------:|----------:|-------------:|--------------------:|---------------------|--------------:|
|  25k | 12.5 | 3.69 × 10⁻¹ |  3.45 % | 1.15 – 4.67 % | 4.34 % |
|  50k | 15.7 | 1.54 × 10⁻¹ | **11.12 %** | 7.95 – 12.95 % | 8.09 % |
| 100k | 19.8 | 4.49 × 10⁻² |  4.29 % | 1.66 – 5.17 % | 5.40 % |
| 150k | 22.1 | 1.66 × 10⁻² |  1.81 % | 0.06 – 2.38 % | 1.26 % |
| 200k | 23.2 | 6.31 × 10⁻³ |  0.33 % | 0.12 – 0.44 % | 0.39 % |
| **300k** | **26.0** | 3.50 × 10⁻³ | **0.11 %** | **0.08 – 0.13 %** | **0.10 %** |
| 400k | 26.5 | **2.04 × 10⁻³** |  0.44 % | 0.15 – 0.71 % | 0.43 % |

Le creux à 50k est **réel, pas un accident de graine** : les trois graines y sont
entre 7.95 % et 12.95 %, toutes pires qu'à 25k. Et 400k redégrade après 300k.

**300k est le point de fonctionnement** : c'est le seul budget où les trois
graines se rejoignent étroitement (0.08–0.13 %), signe d'un régime réellement
atteint plutôt que d'un tirage.

### 4.3 Pourquoi un budget supérieur peut donner une VaR moins bonne

La colonne **MAE prix** ci-dessus est **strictement décroissante**, sur tous les
budgets, facteur 180 de bout en bout. L'ajustement TT ne se dégrade jamais : il
apprend la surface de prix exactement comme on l'espère. Le budget 400k a même
la **meilleure erreur de prix de tout le balayage** — et une VaR deux fois moins
bonne qu'à 300k.

Ce qui n'est pas monotone, c'est le **taux de survie** de l'erreur, la fraction
qui passe du prix au P&L :

| Budget | 25k | 50k | 100k | 150k | 200k | 300k | 400k |
|--------|----:|----:|-----:|-----:|-----:|-----:|-----:|
| MAE P&L / MAE prix | 0.261 | 0.409 | 0.331 | 0.329 | **0.162** | 0.221 | **0.589** |

Un facteur 3.6 d'amplitude, sans tendance.

**Le mécanisme.** Une VaR est un quantile de `P&L(s) = V(s) − V(0)`. Sur un
horizon d'un jour, `s` et le point de base sont distants d'environ 1 % en spot et
de `1/252` an en maturité — un déplacement minuscule. Seule la part de l'erreur
de pricing qui *varie* sur ce déplacement survit ; la part lisse se retranche.

Or TT-cross est un **interpolant, pas une projection** : il est exact aux pivots
sélectionnés et ondule entre eux, et `maxvol_rect` re-sélectionne ces pivots à
chaque balayage (`dr_max = 2` lignes ajoutées par passage). Augmenter le budget
ne raffine donc pas la même approximation — **cela change quels pivots sont
choisis**, donc où l'erreur s'annule et à quelle pente elle remonte entre deux
zéros. L'amplitude baisse proprement, mais la **texture spatiale** de l'erreur
est rebattue à chaque budget, et c'est elle qui décide du taux de survie. Que le
nuage de scénarios tombe à cheval sur un pivot ou à l'intérieur d'une zone lisse
est une loterie retirée à chaque budget. La VaR, qui ne lit qu'une statistique
d'ordre (la 500ᵉ pire perte sur 10 000), y est encore plus exposée.

C'est aussi ce qui explique la dispersion entre graines à budget fixé : une autre
graine ANOVA donne d'autres pivots, donc une autre texture, donc un autre taux
de survie.

**Le Monte Carlo n'a pas ce problème, pour la raison inverse.** Sous CRN son
erreur est dominée par un biais commun à tout le nuage de scénarios, que la
différence `V(s) − V(0)` retranche presque entièrement : taux de survie 0.028,
0.062, 0.073 puis **0.012** à 1k, 4k, 16k et 64k chemins. Son erreur de *prix*
n'est même pas monotone — moyennée sur 3 graines, elle remonte de 0.55 % à
0.69 % du livre entre 16k et 64k chemins, parce qu'elle mesure la taille d'un
biais tiré au hasard — mais son erreur de *risque* descend proprement (MAE P&L
3.7 × 10⁻³ → 7.7 × 10⁻⁴, erreur de VaR 0.82 % → 0.18 %).

C'est l'image miroir exacte du QTT : **chez le QTT le prix converge et le risque
oscille ; chez le Monte Carlo le prix oscille et le risque converge.** Dans les
deux cas, la qualité du pricing mesurée en prix ne prédit pas la qualité de la
mesure de risque.

> Une version précédente de ce paragraphe affirmait que les courbes prix et VaR
> du Monte Carlo « descendent en parallèle ». C'était vrai jusqu'à 16k chemins,
> faux dès que le point à 64k a été mesuré.

**Conséquences pratiques**, et c'est la partie qui compte :

1. **On ne peut pas régler le budget sur l'erreur de prix et espérer que la VaR
   suive.** Le critère d'arrêt naturel du TT (norme globale) n'est pas celui qui
   nous intéresse. Ni le budget, ni la tolérance `e_vld` de `teneva`, ne
   contrôlent la régularité de l'erreur à l'échelle du déplacement d'un jour.
2. **Le budget ne peut être ni choisi sur un point isolé, ni extrapolé.** Il doit
   être balayé, sur plusieurs graines, contre la métrique de risque elle-même —
   un coût de mise au point à part entière, non compté dans les tableaux du §6,
   et qui suppose une référence exacte pour arbitrer.

> Cette explication corrige une première version de ce rapport, qui attribuait
> l'oscillation au fait que `teneva.cross` est appelé sans tolérance et s'arrête
> **en plein milieu d'un balayage**, laissant la chaîne dans un état mixte
> (cœurs `0…i−1` réajustés au balayage courant, cœurs `i+1…d−1` encore ceux du
> précédent). Ce mécanisme existe bien — il est vérifiable dans
> `teneva/cross.py`, et `stop: "m"` le déclenche au niveau du cœur — mais il
> n'est pas la cause : s'il abîmait l'ajustement, la MAE de prix ne serait pas
> monotone. Elle l'est.

---

## 5. Résultats

### 5.1 VaR et ES à 1 jour

Livre de valeur 9.2519, 10 000 scénarios, QTT au budget 300k.

| Bras | VaR 95 % | VaR 99 % | ES 95 % | ES 99 % | Err. VaR 95 % | Err. VaR 99 % |
|------|---------:|---------:|--------:|--------:|--------------:|--------------:|
| `closed_form` (réf.) | **0.743980** | **0.898179** | 0.833831 | 0.933517 | — | — |
| `monte_carlo` (4 096 ch.) | 0.703391 | 0.849436 | 0.788211 | 0.882201 | −5.46 % | −5.43 % |
| `qtt` (300k) | 0.743389 | 0.897812 | 0.833202 | 0.933557 | −0.08 % | −0.04 % |

Les deux chiffres d'erreur de cette table sont ceux d'**une graine**. À lire
avec le §4 : la graine MC affichée (4242) est la **pire** des trois (RMS 4.11 %),
la graine QTT affichée (20260327) est représentative de sa plage étroite
(RMS 0.11 %, étendue 0.08–0.13 %).

La VaR vaut ici **8.0 % de la valeur du livre**, contre 0.84 % pour le livre
apparié à maturité unique de la campagne précédente : un livre uniquement OTM
est bien moins cher et bien plus risqué à valeur égale.

### 5.2 Le test qui tranche

Avec 10 000 scénarios la VaR de référence est elle-même bruitée. Intervalle de
confiance exact d'ordre (statistique d'ordre binomiale, 95 %) :

- VaR 95 % : `[0.733164, 0.753143]`
- VaR 99 % : `[0.887035, 0.906363]`

**La VaR du QTT tombe à l'intérieur aux deux niveaux (0.743389 et 0.897812).
Celle du Monte Carlo en sort aux deux (0.703391 et 0.849436).**

À 10 000 scénarios, le QTT au point de fonctionnement est statistiquement
indiscernable de la revalorisation exacte ; le Monte Carlo à 4 096 chemins
introduit un biais que l'échantillonnage des scénarios n'explique pas.

### 5.3 Compensation de l'erreur dans le P&L

| Bras | MAE prix (portefeuille) | MAE P&L | Ratio de compensation |
|------|------------------------:|--------:|----------------------:|
| `monte_carlo` | 0.4975 (5.4 % de la valeur) | 0.02188 | **0.044** |
| `qtt` | 0.001728 (0.019 %) | 0.001079 | **0.624** |

Le Monte Carlo a une erreur de prix énorme — un biais de −0.498 sur un livre à
9.25 — mais quasi constante d'un scénario à l'autre grâce aux CRN : 96 %
s'annule dans le P&L. Le QTT a une erreur de prix **288 fois plus petite**, qui
se compense beaucoup moins (0.624). C'est pourquoi son avantage sur la VaR est
moindre que son avantage sur le prix.

Classement des scénarios :

| Bras | Spearman | Recouvrement de la queue à 1 % |
|------|---------:|-------------------------------:|
| `monte_carlo` | 1.000000000 | 100 % |
| `qtt` | 0.999999999 | 100 % |

Les 100 pires scénarios sont les mêmes pour les trois méthodes. **L'erreur est
entièrement dans le niveau, pas dans l'identification du risque.**

### 5.4 Parité call/put

Résidu max `|erreur call − erreur du put fantôme de même strike et maturité|` :
**5.02 × 10⁻¹⁴**.

La parité n'empile pas une seconde approximation : l'erreur sur un call est
exactement l'erreur du surrogate sur le put correspondant. Calibrer le QTT sur
les seuls puts est gratuit, et cela divise par deux la surface à apprendre.

Erreur de prix moyenne par position (QTT, budget 300k) :

| Position | MAE | Position | MAE |
|----------|----:|----------|----:|
| `put_95_1m`   | 6.16 × 10⁻⁴ | `call_110_1y` | 6.01 × 10⁻⁴ |
| `call_105_3m` | 6.57 × 10⁻⁴ | `put_80_18m`  | 6.74 × 10⁻⁴ |
| `put_90_6m`   | 1.33 × 10⁻³ | `call_120_2y` | 1.10 × 10⁻³ |

Homogène sur toute l'échelle de maturités : la grille de risque tient sa
promesse de résoudre la structure en moneyness de façon cohérente de 1 mois à
2 ans. (À comparer au Monte Carlo, dont la MAE va de 1.24 × 10⁻² à
1.76 × 10⁻¹ selon la position.)

### 5.5 Expected Shortfall 97,5 %

L'ES à 97,5 % est la mesure de capital réglementaire FRTB (Bâle III), qui
remplace la VaR à 99 %. Sur 10 000 scénarios c'est la **moyenne des 250 pires
pertes**. Même livre, mêmes scénarios, mêmes graines ; l'ES ne coûte rien de
plus que la VaR, il sort du même vecteur de pertes.

**Bras principaux** (QTT au budget 300k, MC à 4 096 chemins, une graine chacun) :

| Bras | ES 97,5 % | Erreur | Dans l'IC de la référence ? |
|------|----------:|-------:|:---------------------------:|
| Forme fermée (réf.) | **0.888192** | — | — |
| Monte Carlo imbriqué | 0.839600 | −5.47 % | **non** |
| QTT 13-D | 0.887773 | **−0.05 %** | **oui** |

Intervalle de confiance à 95 % de l'ES de référence, du seul fait de
l'échantillonnage des 10 000 scénarios : `[0.882777, 0.893606]` — erreur type
asymptotique de la moyenne de queue sur 250 scénarios. Même verdict que pour la
VaR : le QTT est statistiquement indiscernable de la référence exacte, le Monte
Carlo à 4 096 chemins ne l'est pas.

**Moyenne quadratique sur 3 graines** :

| Budget QTT | Err. ES 97,5 % | Étendue | | Chemins MC | Err. ES 97,5 % |
|-----------:|---------------:|--------:|-|-----------:|---------------:|
|  50k | 8.14 % | 7.07 – 9.85 % | | 1 024 | 6.71 % |
| 100k | 5.32 % | 1.40 – 6.75 % | | 4 096 | 4.00 % |
| 150k | 1.26 % | 0.16 – 1.94 % | | 16 384 | 0.90 % |
| 200k | 0.39 % | 0.02 – 0.48 % | | 65 536 | **0.15 %** |
| **300k** | **0.09 %** | **0.05 – 0.12 %** | | | |
| 400k | 0.43 % | 0.20 – 0.66 % | | | |

Trois observations.

**L'erreur d'ES 97,5 % reproduit presque exactement l'erreur de VaR 99 %**, sur
toutes les configurations des deux méthodes : 8.14 / 8.09 %, 5.32 / 5.40 %,
1.26 / 1.26 %, 0.39 / 0.39 %, 0.09 / 0.10 %, 0.43 / 0.43 % pour le QTT ;
6.71 / 6.71 %, 4.00 / 3.99 %, 0.90 / 0.91 %, 0.15 / 0.15 % pour le MC. C'est la
calibration voulue par le régulateur (ES 97,5 % ≈ VaR 99 % sous loi normale),
et ici elle tient aussi pour les *erreurs*.

**Moyenner sur 250 scénarios ne lisse pas l'erreur.** On pourrait attendre de
l'ES, moyenne de queue, qu'il soit plus robuste que la VaR, statistique d'ordre
isolée. Il ne l'est pas : la dispersion entre graines est à peine plus serrée,
et la non-monotonie du QTT est intacte (0.09 % à 300k, 0.43 % à 400k). La raison
est la même qu'au §5.3 : les deux méthodes classent les scénarios de queue
parfaitement (Spearman ≈ 1, recouvrement 100 %), donc leur erreur n'est pas un
bruit qui se moyenne sur la queue, c'est un **décalage de niveau cohérent** sur
toute la queue — et une moyenne de décalages identiques est ce décalage.

**À précision égale, même conclusion que pour la VaR.** Le Monte Carlo
atteindrait l'erreur QTT moyenne de 0.09 % vers 1.48 × 10⁵ chemins
(extrapolation de 2.3× au-delà du dernier point mesuré), soit ~494 s par run
contre 52 s pour le QTT au premier jour, initialisation comprise (§6.2).

---

## 6. Coût

Run `bench_es` (médiane sur 3 répétitions), 60 006 revalorisations par run de
VaR / ES. Tous les rapports de cette section sont **intra-run**.

| Bras | Initialisation (une fois) | VaR / ES quotidienne | **Premier jour** |
|------|--------------------------:|---------------------:|-----------------:|
| `closed_form` | 0 s | 0.006 s | **0.006 s** |
| `monte_carlo` (4 096 ch.) | ~0 s | 13.04 s | **13.04 s** |
| `qtt` (300k) | 29.22 s | 22.77 s | **51.99 s** |

### 6.1 Le QTT n'est plus « cher à construire, gratuit à interroger »

En 8-D la revalorisation QTT coûtait 2.8 s. Ici elle coûte **22.8 s**, soit
~78 % du coût d'initialisation et davantage que le Monte Carlo entier à 4 096
chemins. 59 cœurs à contracter en cubique sur 13 axes, sur 60 006 points, se
paient. Le surrogate stocke 78 042 paramètres contre 3 762 en 8-D.

C'est le prix de la généralité : la volatilité est une coordonnée, donc la
grille sait pricer des chocs de vol qu'on ne lui demande pas ici.

### 6.2 À précision égale

Cible : l'erreur QTT **moyennée sur les graines** au budget 300k, jamais celle
de la graine unique du bras principal. Le nombre de chemins Monte Carlo requis
est lu sur la loi ajustée aux **quatre** points mesurés (1k à 64k).

| Mesure | Cible QTT | Chemins MC requis | Au-delà du dernier point mesuré | Temps MC / run | Avantage QTT, jour 1 | Avantage QTT, en régime |
|--------|----------:|------------------:|--------------------------------:|---------------:|---------------------:|------------------------:|
| VaR 95 % | 0.107 % | 1.49 × 10⁵ | 2.3× | ≈ 497 s | **9.6×** | **21.8×** |
| VaR 99 % | 0.096 % | 1.44 × 10⁵ | 2.2× | ≈ 480 s | **9.2×** | **21.1×** |
| **ES 97,5 %** | **0.092 %** | **1.48 × 10⁵** | 2.3× | **≈ 494 s** | **9.5×** | **21.7×** |

« Jour 1 » compare au coût QTT initialisation comprise (52.0 s) ; « en régime »
au seul coût quotidien (22.8 s), puisque le surrogate est réutilisé tel quel les
jours suivants.

Formulation indépendante de toute horloge : pour atteindre la précision du QTT,
le Monte Carlo imbriqué doit exécuter **35 à 36 fois l'arithmétique** de sa
configuration à 4 096 chemins.

> **Correction.** Une version précédente annonçait ~20× (95 %) à ~29× (99 %) au
> premier jour. Ces chiffres reposaient sur une extrapolation d'un facteur 19
> au-delà du dernier point mesuré (16 384 chemins), avec un exposant ajusté de
> −0.75 sur trois points. La mesure du point à 65 536 chemins a raidi l'exposant
> à −0.89 / −0.94 : le Monte Carlo converge plus vite qu'estimé, il faut 2 à 3
> fois moins de chemins, et l'avantage du QTT est divisé d'autant. L'extrapolation
> résiduelle n'est plus que de ~2.3×.

### 6.3 Où va le coût du Monte Carlo

Le moteur générique simule les cinq sous-jacents. Un moteur conscient du payoff
peut précalculer le facteur multiplicatif du panier géométrique, commun à tous
les scénarios : mesuré à **7.4× plus rapide** dans ce run (1.76 s contre
13.04 s), pour un estimateur strictement identique (écart max 7.3 × 10⁻¹⁴). Le
gain est bien plus grand que sur le livre à maturité unique (2.9×) parce qu'il
économise aussi le travail des six seaux de maturité.

Face à un tel moteur, l'avantage QTT à précision égale tomberait de ~9.5× à
**~1.3× au premier jour** et de ~22× à **~3× en régime**. Cette optimisation
n'existe que pour le panier géométrique ; c'est précisément pourquoi elle borne
ce que cette expérience peut dire des produits où elle n'existe pas.

### 6.4 Amortissement

Le surrogate QTT s'initialise une fois et se réutilise chaque jour : seuls les
spots bougent, et la grille les couvre déjà. Le Monte Carlo imbriqué ne paie rien
d'avance et tout à chaque run. Coût cumulé après N runs quotidiens,
`initialisation + N × coût quotidien`, pour les deux configurations les mieux
appariées en précision (balayage du run `bench_es`) :

| Configuration | Err. ES 97,5 % | Init (une fois) | Quotidien | N = 1 | N = 21 (1 mois) | N = 252 (1 an) |
|---------------|---------------:|----------------:|----------:|------:|----------------:|---------------:|
| QTT 300k | **0.09 %** | 33.2 s | 22.8 s | 56 s | 512 s | 5 773 s |
| MC 65 536 | 0.15 % | 0 s | 218.3 s | 218 s | 4 583 s | 55 000 s |

Le QTT est le plus précis des deux **et** le moins cher dès le premier run. Le
croisement tombe à **0.17 run** : l'initialisation (33 s) est plus petite que
l'économie d'un seul run (196 s). L'amortissement n'est donc pas ce qui fait
basculer la décision — il ne fait qu'élargir l'écart, jusqu'à **9.5× sur un
an**. La part de l'initialisation dans le coût QTT passe de 59 % au premier jour
à 6.5 % après un mois et 0.6 % après un an.

Le tableau complet, tous budgets × tous niveaux de chemins, avec les croisements
filtrés aux paires de précision comparable, est dans
`results/var_qtt_benchmark_bench_es.md`.

---

## 7. Menaces à la validité

1. **Les temps absolus sont peu fiables sur cette machine.** Le bras Monte
   Carlo, travail strictement identique, a été mesuré à 11.5 s, 11.6 s et
   26.6 s selon le run ; à l'intérieur du run final, le balayage QTT a
   chronométré le budget 150k (48.8 s) **plus lentement que le 200k** (32.7 s),
   et le 300k à 68.7 s alors que le bras principal au même budget donnait
   27.3 s. Dispersion de l'ordre de ±2.5×. Tous les ratios cités sont
   intra-run, et la figure trace l'erreur contre des **comptes d'évaluations**,
   pas contre des secondes, précisément pour cette raison. **Les conclusions de
   précision sont exactes et reproductibles** — elles ne dépendent que du
   triplet (budget, graine, grille) — et ne sont pas touchées.
2. **Le budget QTT n'est pas choisissable a priori, et aucun substitut bon
   marché n'est disponible** (§4.3). L'erreur de prix, qui serait le critère
   d'arrêt naturel et se mesure sans référence de risque, est monotone et ne
   signale donc *pas* les mauvais budgets. Il faut balayer contre la métrique de
   risque elle-même, sur plusieurs graines — ce qui suppose une référence exacte
   pour arbitrer, dont on ne dispose pas sur les produits qui motivent la
   méthode. Ce coût de mise au point n'est pas compté au §6. C'est l'obstacle
   opérationnel le plus sérieux relevé par cette campagne.
3. **Le point de fonctionnement 300k est un minimum local sur un balayage fini.**
   400k est pire. Rien ne garantit qu'un balayage plus fin ne révélerait pas
   d'autres creux, et le mécanisme du §4.3 donne plutôt lieu de le craindre :
   rien ne lisse le taux de survie d'un budget au suivant.
4. **Sur ce produit, la forme fermée gagne massivement** : 6 ms contre 12–47 s.
   Le panier géométrique a une solution analytique, donc la comparaison
   QTT/Monte Carlo ne vaut que comme *proxy* des produits où cette colonne
   n'existe pas (panier arithmétique, américain).
5. **Volatilité figée.** Elle est portée comme coordonnée mais n'est pas
   choquée. Une VaR qui choque aussi la surface utiliserait la même grille sans
   la reconstruire — c'est l'argument principal en faveur du 13-D — mais cette
   expérience ne le démontre pas, elle en paie seulement le coût.
6. **Horizon 1 jour uniquement.** À 10 jours le déplacement des spots est ~3×
   plus grand et la compensation d'erreur change de régime.
7. **Aucun scénario ne sort de la boîte de la grille** (0.00 %) — conséquence de
   l'horizon court ; à horizon plus long ce diagnostic doit être relu.
8. **Une seule graine de scénarios.** Les VaR héritent de l'échantillonnage de
   ces 10 000 trajectoires ; les intervalles du §5.2 le quantifient, mais
   l'expérience n'a pas été répétée sur plusieurs jeux de scénarios.
9. **Trois graines par point, c'est peu** pour une moyenne quadratique. Les
   étendues reportées sont indicatives, pas des intervalles de confiance.

---

## 8. Reproduire

```bash
# Benchmark complet (~30 min) : trois bras, VaR 95/99 et ES 97,5,
# 6 budgets QTT x 4 niveaux de chemins MC, 3 graines chacun,
# et les tableaux markdown dans results/var_qtt_benchmark_bench_es.md
.venv/Scripts/python.exe scripts/validation/benchmark_var_qtt_basket_book.py \
    --profile full --budget 300000 \
    --qtt-budget-sweep 50000 100000 150000 200000 300000 400000 \
    --mc-path-sweep 1024 4096 16384 65536 \
    --mc-sweep-seeds 4242 911 20260921 --tag bench_es

# Vérification rapide (~2 min)
.venv/Scripts/python.exe scripts/validation/benchmark_var_qtt_basket_book.py --profile smoke

# Contrôle de layout
.venv/Scripts/python.exe scripts/validation/benchmark_var_qtt_basket_book.py \
    --profile full --layout blocked --timing-repeats 1 \
    --mc-path-sweep --qtt-budget-sweep --tag probe_blocked

# Figures (VaR 95 % puis ES 97,5 %)
.venv/Scripts/python.exe scripts/figures/plot_var_qtt_versus_monte_carlo.py --metric var
.venv/Scripts/python.exe scripts/figures/plot_var_qtt_versus_monte_carlo.py --metric es
```

Environnement : Python 3.12.9, numpy 2.5.2, teneva 0.14.11, Windows 11,
`OMP_NUM_THREADS=1`. Graines : scénarios 20260921, Monte Carlo 4242,
TT 20260327 (balayages : MC 4242/911/20260921, TT 20260327/7/991).

---

## 9. Ce qu'il faut retenir

1. **Au point de fonctionnement (budget 300k), le QTT 13-D reproduit la VaR
   exacte à 0.11 % près** (moyenne sur 3 graines, étendue 0.08–0.13 %), ce qui
   le place *à l'intérieur* de l'intervalle de confiance d'échantillonnage de la
   référence. Le Monte Carlo à 4 096 chemins en sort, avec −5.46 % / −5.43 %.
2. **L'ajustement TT converge, la VaR n'en hérite pas.** L'erreur de *prix*
   décroît strictement sur tout le balayage, facteur 180 — le budget 400k a la
   meilleure de toutes. L'erreur de *VaR* fait 3.45 → 11.12 → 4.29 → 1.81 →
   0.33 → **0.11** → 0.44 %. Ce qui oscille est le taux de survie de l'erreur
   dans la différence `V(s) − V(0)` : TT-cross est un interpolant, exact à ses
   pivots, et augmenter le budget rebat quels pivots sont choisis, donc la
   texture spatiale de l'erreur. **Conséquence : on ne peut pas régler le budget
   sur l'erreur de prix et attendre que la VaR suive** (§4.3). Le Monte Carlo
   est l'image miroir : sous CRN son erreur de prix est un biais commun qui
   peut même remonter avec les chemins, mais elle se retranche dans le P&L et
   son erreur de risque converge proprement. Dans les deux cas, l'erreur de prix
   ne prédit pas l'erreur de risque.
3. **La parité call/put est gratuite et exacte** : résidu 5.0 × 10⁻¹⁴, audité en
   repriçant le put fantôme de même strike et même maturité.
4. **`interleaved` bat `blocked` d'un facteur 10** sur cette grille, contre la
   recommandation du `CLAUDE.md` établie sur les grilles de prix.
5. **Le QTT 13-D n'est plus gratuit à interroger** : 22.8 s de revalorisation
   contre 2.8 s en 8-D, soit plus que le Monte Carlo à 4 096 chemins. À
   précision égale il reste **~9.5× moins cher au premier jour,
   initialisation comprise, et ~22× en régime** quotidien (le Monte Carlo doit
   exécuter 35–36× l'arithmétique de sa configuration à 4 096 chemins) — mais
   seulement ~1.3× au premier jour et ~3× en régime face à un moteur Monte Carlo
   conscient du payoff. (Une version précédente annonçait 20–29× sur la foi
   d'une extrapolation d'un facteur 19 ; mesurer le point à 65 536 chemins l'a
   ramenée à 2.3× et a divisé l'avantage par deux.)
6. **Les six maturités distinctes coûtent cher au Monte Carlo** — un seau de
   simulation chacune — et rien au QTT, qui price n'importe quel `(K, T)` au même
   prix. C'est l'argument structurel en faveur du surrogate sur un livre étagé.
7. Les deux méthodes approchées **identifient parfaitement les scénarios de
   queue** (recouvrement 100 %, Spearman ≈ 1). L'erreur est un décalage de
   niveau, pas une erreur de classement du risque.
8. **ES 97,5 %** : le QTT au budget 300k donne 0.887773 contre 0.888192 exact
   (**−0.05 %**, 0.09 % en moyenne sur 3 graines), à l'intérieur de
   l'intervalle de confiance de la référence ; le Monte Carlo à 4 096 chemins
   donne 0.839600 (−5.47 %), à l'extérieur. L'erreur d'ES reproduit presque
   exactement celle de la VaR 99 % sur toutes les configurations, et **moyenner
   sur les 250 pires scénarios ne lisse rien** : l'erreur est un décalage de
   niveau cohérent sur toute la queue, pas un bruit (§5.5).
9. **L'amortissement élargit l'écart mais ne fait pas basculer la décision** :
   à précision appariée (QTT 300k contre MC 65 536), le QTT est moins cher dès
   le premier run (croisement à 0.17 run), et 9.5× moins cher sur un an (§6.4).
