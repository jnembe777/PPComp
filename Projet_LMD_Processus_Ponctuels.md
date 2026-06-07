# Projet : Codage LMD Versatile pour Classes de Processus Ponctuels

> **Auteur de référence :** J. Nembé | **Principe central :** Longueur Minimale de Description (MDL/LMD) | **Version :** 6.0 — Cartouche ABCDEFGH

---

## Table des Matières

1. [Objectif du Projet](#objectif)
2. [Cadre Mathématique Général](#cadre-math)
3. [Cartouche de Méthode ABCDEFGH](#cartouche)
4. [Dimension B — Modes de Codage Couleur (correction fondamentale)](#dim-b)
5. [Versatilité des Représentations R1–R4b (Dim. C)](#versatilite)
6. [Estimation de l'Intensité — Familles (Dim. E)](#estimation)
7. [Critère LMD](#critere-lmd)
8. [Classes de Processus Ponctuels (Dim. A)](#classes-pp)
9. [Codage Uniforme — Formules L1–L4 Corrigées](#codage-uniforme)
10. [Représentation Spatiale Binaire Bitwise](#bitwise)
11. [Extension CPPM — Décomposition Monochromatique](#cppm)
12. [6 Fonctionnalités Avancées](#fonctionnalites)
13. [Architecture 9 Agents](#architecture)
14. [Algorithme Global — Pipeline ABCDEFGH](#algorithme)
15. [Arbre de Décision Adaptatif](#arbre-decision)
16. [Métriques de Performance](#metriques)
17. [Feuille de Route](#roadmap)
18. [Références](#references)

---

## 1. Objectif du Projet {#objectif}

Développer un **framework de compression vidéo adaptatif** basé sur le principe de la **Longueur Minimale de Description (MDL/LMD)** appliqué à diverses classes de processus ponctuels. Le système sélectionne automatiquement, pour chaque bloc vidéo, la configuration optimale décrite par un **cartouche de 8 dimensions (ABCDEFGH)** encodé sur 17 bits.

Les capacités du système :

1. **Classer** le type de processus ponctuel (dim. A) parmi 5 classes.
2. **Choisir le mode de codage couleur** optimal (dim. B) parmi séquentiel, uniforme, Huffman ou universel.
3. **Sélectionner la représentation** temporelle la plus compacte (dim. C) parmi 5 options (R1–R4b).
4. **Comparer** le codage statistique MDL et le codage uniforme (dim. D).
5. **Estimer** la fonction d'intensité dans la famille optimale (dim. E).
6. **Gérer** la richesse chromatique hiérarchique (dim. F) et la résolution spatiale quadtree (dim. G).

---

## 2. Cadre Mathématique Général {#cadre-math}

Soit $(\Omega, \mathcal{F}, P)$ un espace de probabilité. On observe un **processus ponctuel marqué** $N$ sur $[0, T]$.

Un réalisé est noté $\omega = \{(\tau_k, m_k)\}_{k=1}^N$, où :
- $\tau_k$ : temps de saut (dans $[0, r]$ après discrétisation)
- $m_k \in \mathcal{E}$ : marque (couleur, état, etc.)

### 2.1. Modèle d'Intensité Multiplicative (Aalen)

$$\lambda(t) = \alpha(t) \cdot Y(t)$$

| Terme | Description |
|-------|-------------|
| $\alpha(t) \in \mathcal{F}$ | Fonction d'intensité inconnue à estimer |
| $Y(t) \in \{0,1\}$ | Processus prévisible (pixel actif) |

**Vraisemblance :**

$$L_{\theta}(N) = \exp\!\left(\int_0^T \log(\alpha_{\theta}(s))\,dN(s) - \int_0^T \alpha_{\theta}(s)Y(s)\,ds\right)$$

### 2.2. Estimateur MDL & Borne de Hellinger

$$\hat{P}_n = \arg\min_{Q \in \mathcal{P}}\Big(-\log Q(X^n) + C_n(Q)\Big)$$

**Borne de Hellinger** (versatile_main.tex, Théorème 1) :

$$H^2(P, \hat{P}_n) \leq \min_{Q \in \Gamma_n}\left[H^2(P, Q) + \frac{1}{n}C_n(Q)\right] \quad \text{p.s.}$$

---

## 3. Cartouche de Méthode ABCDEFGH {#cartouche}

Chaque bloc vidéo est encodé avec un **header de 17 bits** décrivant intégralement la méthode de compression choisie. Ce cartouche est le vecteur de décision du pipeline.

| Dim. | Bits | Valeurs | Description |
|------|------|---------|-------------|
| **A** | 3 | 5 | Type de processus ponctuel |
| **B** | 2 | 4 | **Mode de codage couleur** ← *correction fondamentale* |
| **C** | 3 | 5 | Représentation temporelle R1–R4b |
| **D** | 2 | 3 | Mode de compression (Uniforme / Universel / MDL) |
| **E** | 2 | 4 | Famille de fonctions pour l'estimation de $\hat{\alpha}(t)$ |
| **F** | 2 | 3 | Niveau chromatique hiérarchique (8/16/24 bits) |
| **G** | 2 | 4 | Résolution spatiale quadtree (16px → 2px) |
| **H** | 1 | 2 | Mode temporel (continu / discret) |
| **Total** | **17** | | **Header par bloc** |

```python
@dataclass
class Cartouche:
    A: int  # 3b : 0=Marqué 1=Mono 2=VecMarg 3=VecJoint 4=Markov
    B: int  # 2b : 0=Ba_seq 1=Bb_unif 2=Bc_huf 3=Bd_elias
    C: int  # 3b : 0=R1 1=R2 2=R3 3=R4a 4=R4b
    D: int  # 2b : 0=Uniforme 1=Universel 2=MDL
    E: int  # 2b : 0=Hist 1=Splines 2=Ondelettes 3=Trig
    F: int  # 2b : 1=8b 2=16b 3=24b
    G: int  # 2b : 0=16px 1=8px 2=4px 3=2px
    H: int  # 1b : 0=continu 1=discret

    def encode(self) -> int:      # → 17 bits
        return ((self.A&7)<<14 | (self.B&3)<<12 | (self.C&7)<<9
               | (self.D&3)<<7  | (self.E&3)<<5  | (self.F&3)<<3
               | (self.G&3)<<1  | (self.H&1))
```

---

## 4. Dimension B — Modes de Codage Couleur {#dim-b}

> **Correction fondamentale.** Les formules L1–L4 telles que classiquement formulées (`L4 = log N + log C(r,N) + N·log m`) ne sont valables que lorsque le codage de la couleur est **uniforme (mode Bb)**. Le terme `N·log₂m` est le coût couleur uniforme. Le codage couleur constitue une dimension indépendante (dim. B) avec quatre options.

La décomposition générale est :

$$L_i(B) = L^{\text{temporel}}_i + C_{\text{color}}(B)$$

où $L^{\text{temporel}}_i$ est **indépendante de B** et $C_{\text{color}}(B)$ varie selon le mode.

### 4.1. Quatre modes (dim. B)

**Ba — Séquentiel**

La couleur est indiquée **une seule fois en tête de bloc**. Chaque saut hérite de la couleur courante. Seules les transitions explicites $(\tau_k, \text{nouvelle\_couleur})$ sont encodées.

$$C_{\text{color}}(\text{Ba}) = \log_2 m + N_{\text{trans}} \cdot \log_2 m$$

Optimal si le processus couleur varie lentement ($N_{\text{trans}} \ll N$).

**Bb — Uniforme**

Chaque saut porte sa marque codée uniformément sur $m$ couleurs.

$$C_{\text{color}}(\text{Bb}) = N \cdot \log_2 m \quad \leftarrow \textbf{terme des formules L1–L4 actuelles}$$

Optimal si la distribution $\hat{P}(m_k)$ est uniforme ou si $m \leq 4$.

**Bc — Huffman (entropique)**

Code à longueur variable sur la distribution empirique $\hat{P}(m_k)$.

$$C_{\text{color}}(\text{Bc}) = N \cdot H(\hat{P}) + D_{\text{huf}}$$

où $H(\hat{P}) = -\sum_c p_c \log_2 p_c \leq \log_2 m$ et $D_{\text{huf}} \approx m \cdot (\lfloor\log_2 m\rfloor + 1)$ bits (overhead dictionnaire).

**Seuil de rentabilité Huffman :**

$$N > N^* = \frac{D_{\text{huf}}}{\log_2 m - H(\hat{P})} \quad \Longrightarrow \quad C_{\text{color}}(\text{Bc}) < C_{\text{color}}(\text{Bb})$$

Exemple : $m=16$, $H=2.8$ bits, $D_{\text{huf}}=80$ bits → $N^* = 80/(4-2.8) = 67$ sauts.

**Bd — Universel Elias (m inconnu)**

Si $m$ n'est pas connu au décodeur a priori. Code $\delta$ d'Elias sur chaque marque $m_k+1$.

$$C_{\text{color}}(\text{Bd}) \approx N \cdot \left(\log_2 m + 2\log_2\log_2 m\right)$$

### 4.2. Condition Ba < Bc

Le mode séquentiel est préférable à Huffman si les transitions sont suffisamment rares :

$$N_{\text{trans}} < \frac{N \cdot H(\hat{P}) + D_{\text{huf}} - \log_2 m}{\log_2 m}$$

### 4.3. Cas monochromatique (A = Ab)

Quelle que soit la valeur de B, si le processus est monochromatique (dim. A = Ab), le terme couleur est **nul** : la couleur est portée par l'indice du processus, aucune marque n'est codée explicitement.

$$C_{\text{color}} = 0 \quad \Rightarrow \quad L_4^{\text{mono}} = \log_2 N + \log_2\binom{r}{N}$$

### 4.4. Couleurs hiérarchiques avec Huffman (F × B = Bc)

Pour $L$ niveaux actifs de la hiérarchie chromatique (dim. F), avec un dictionnaire Huffman par niveau :

$$C_{\text{color\_hier}}(\text{Bc}, L) = N \cdot \sum_{i=1}^L H(\hat{P}_i) + \sum_{i=1}^L D_{\text{huf}_i} \leq N \cdot 8L \text{ bits}$$

Si les niveaux sont indépendants : $H(m_1,\ldots,m_L) = \sum H(m_i)$.
Si les niveaux sont corrélés : codage conditionnel $H(m_2|m_1) < H(m_2)$ → gain supplémentaire.

---

## 5. Versatilité des Représentations R1–R4b (Dim. C) {#versatilite}

Avant tout codage, le système sélectionne la représentation temporelle la plus compacte. La **partie temporelle** est indépendante du mode couleur (dim. B).

| Code | Représentation | Structure | $L^{\text{temporel}}$ | Optimal quand |
|------|---------------|-----------|----------------------|---------------|
| **R1** | Timestamps | $[\tau_1,\ldots,\tau_N]$ trié | $\log_2 N + N\log_2 r$ | $N/r < 0.10$ |
| **R2** | Count | $[n_1,\ldots,n_r]$ histogramme | $r \cdot \log_2(N/r+1)$ | $N/r > 0.80$ |
| **R3** | Intervalles | $[\Delta_1,\ldots,\Delta_N]$ | $N\log_2 r$ | $R_{\text{temp}} > 1.5$ |
| **R4a** | Boolean | $b \in \{0,1\}^r$ | $r$ | $N/r \in [0.50, 0.80]$ |
| **R4b ★** | Combinatoire | $(N, \text{Index})$ | $\log_2 N + \log_2\binom{r}{N}$ | **Borne inférieure générale** |

> **Critère de choix :** $C^* = \arg\min_{C} \left[L^{\text{temporel}}_C + C_{\text{color}}(B)\right]$

### 5.1. Représentation R4b — Adresse Combinatoire

À partir du vecteur booléen $b \in \{0,1\}^r$ avec $|b| = N$ bits à 1, l'index est le rang de $b$ dans l'énumération lexicographique des $\binom{r}{N}$ vecteurs de longueur $r$ ayant exactement $N$ uns :

$$\text{Index} = \sum_{\substack{k=0 \\ b_k=1}}^{r-1} \binom{r-k-1}{N - \text{count}(b,k) - 1}$$

Calcul en $O(N)$. Décodage inverse : `decode_comb_index(N, Index, r)` en $O(N)$.

### 5.2. Primitives d'Accès (API Universelle)

| Primitive | Signature | Représentation cible |
|-----------|-----------|---------------------|
| `get_jump_times()` | `() → ndarray` | R1, R4b |
| `get_color_at(t)` | `(t:int) → int` | R2, R3 |
| `get_count_in_bin(b)` | `(b:int) → int` | R2 |
| `has_jump_at(t)` | `(t:int) → bool` | R4a en O(1) |
| `get_N()` | `() → int` | R4b |
| `get_comb_index()` | `() → int` | R4b |
| `get_color_distribution()` | `() → dict` | Huffman (Bc) |

### 5.3. Optimisation globale (C × B)

L'optimisation explore les $5 \times 4 = 20$ combinaisons (représentation × mode couleur) :

$$\arg\min_{(C,B)} \left[ L^{\text{temporel}}_C + C_{\text{color}}(B) \right]$$

---

## 6. Estimation de l'Intensité — Familles (Dim. E) {#estimation}

Pour le codage MDL statistique (dim. D = Dc), on estime $\hat{\alpha}(t)$ dans une famille $\mathcal{F}$ :

| Famille | Formulation | Points forts |
|---------|-------------|--------------|
| **Histogrammes** | $\alpha(t) = \sum_i c_i \mathbb{I}_{[t_i,t_{i+1}]}(t)$ | Simplicité, interprétabilité |
| **Splines** | Polynômes par morceaux $C^k$ | Lissage continu, adaptatif ★ |
| **Ondelettes** | Décomposition multi-résolution | Singularités locales |
| **Trig.** | $\alpha(t) = \sum(a_k\cos kt + b_k\sin kt)$ | Données périodiques |

**Pénalité de complexité :** $C_n(\mathcal{F}) = (k+4) \cdot \log_2 N / 2$ où $k$ est le nombre de paramètres.

---

## 7. Critère LMD {#critere-lmd}

### 7.1. Estimateur MDL

$$\hat{P}_n = \arg\min_{Q \in \mathcal{P}}\Big(-\log Q(X^n) + C_n(Q)\Big)$$

### 7.2. Codage MDL en 3 étapes (dim. D = Dc)

| Étape | Opération | Formule |
|-------|-----------|---------|
| **1 — Sélection** | $\arg\min_{\mathcal{F}}(-\log L_{\mathcal{F}} + C_n(\mathcal{F}))$ | $(k+4)\log_2 N/2$ |
| **2 — Estimation** | $\hat{\alpha}(t)$ dans la famille sélectionnée | Modèle Aalen $\lambda=\hat{\alpha}\cdot Y$ |
| **3 — Codage** | Codage arithmétique via $\hat{\Lambda}(t) = \int_0^t\hat{\alpha}(s)\,ds$ | $\tau_k \to \hat{\Lambda}(\tau_k)/\hat{\Lambda}(T) \sim \text{Uniforme}[0,1]$ |

### 7.3. Borne de Hellinger Étendue

La borne reste valide avec le terme couleur et les extensions spatiales :

$$C_n^{\text{total}}(Q) = C_n^{\text{PP}}(Q) + C_{\text{color}}(B) + L \cdot 8 + \log_2(4) \cdot \frac{4^d-1}{3}$$

---

## 8. Classes de Processus Ponctuels (Dim. A) {#classes-pp}

| Type | Dim. A | Structure de données | Terme couleur | Test de classification |
|------|--------|---------------------|---------------|----------------------|
| **Réel Marqué** | Aa | `[(τ_k, m_k)]` | $C_{\text{color}}(B)$ | Défaut |
| **Monochromatique** | Ab | `{c → [τ_k^(c)]}` | **0** (couleur = indice) | $\sum_c \log\binom{r}{N_c} < L_t + C_{\text{color}}(B)$ |
| **Vectoriel Marginal** | Ac | `[ProcessData_c]` | $C_{\text{color}}(B)$ par composante | POPCOUNT(XOR)/n > 0.15 |
| **Vectoriel Joint** | Ad | Matrice $\Lambda(t)$ m×m | $C_{\text{color}}$ loi jointe | POPCOUNT(XOR)/n < 0.15 |
| **Markovien** | Ae | `N_hj[h,j] + Y_h[t]` | $C_{\text{color}}$ des transitions | Test $\chi^2$ matrice transitions |

### 8.1. Règle Marqué vs Monochromatique

La comparaison dépend du mode B choisi :

$$\text{Choisir Ab si} \quad \sum_{c=1}^m \log_2\binom{r}{N_c} < L^{\text{temporel}}(\text{R4b}) + C_{\text{color}}(B_{\text{optimal}})$$

---

## 9. Codage Uniforme — Formules L1–L4 Corrigées {#codage-uniforme}

> **Principe :** $L_i(B) = L^{\text{temporel}}_i + C_{\text{color}}(B)$. Le mode **Bb uniforme** redonne exactement les formules classiques.

| Code | $L^{\text{temporel}}$ | $+ C(\text{Ba})$ | $+ C(\text{Bb})$ ← **formule actuelle** | $+ C(\text{Bc})$ Huffman | $+ C(\text{Bd})$ Elias |
|------|----------------------|------------------|-----------------------------------------|--------------------------|------------------------|
| **L1** | $r$ | $\log m + N_t\log m$ | $r\log_2 m$ | $r \cdot H(\hat{P}) + D_{\text{huf}}$ | $r \cdot L^*(m)$ |
| **L2** | $\log N + N\log r$ | $\log m + N_t\log m$ | $N\log_2(rm)$ | $N \cdot H(\hat{P}) + D_{\text{huf}}$ | $N \cdot L^*(m)$ |
| **L3** | $r$ | $\log m + N_t\log m$ | $N\log_2 m$ → $r + N\log m$ | $N \cdot H(\hat{P}) + D_{\text{huf}}$ | $N \cdot L^*(m)$ |
| **L4 ★** | $\log N + \log\binom{r}{N}$ | $\log m + N_t\log m$ | $N\log_2 m$ → **L4 classique** | $N \cdot H(\hat{P}) + D_{\text{huf}}$ | $N \cdot L^*(m)$ |
| **L4_mono** | $\log N + \log\binom{r}{N}$ | **C = 0** (Ab) | **C = 0** | **C = 0** | **C = 0** |

**Règle de décision :** Le codage MDL est retenu si $L_{\text{MDL}} < \min_B\min_i L_i(B)$.

---

## 10. Représentation Spatiale Binaire Bitwise {#bitwise}

Pour chaque couleur $c$ : $M_c(t,p) = \mathbb{I}\{C(p,t) = c\}$ — 1 bit par pixel par couleur.

$$\Delta_c(t) = M_c(t) \oplus M_c(t+1) \quad \Rightarrow \quad N_c = \text{POPCOUNT}(\Delta_c)$$

| Opération | Méthode classique | Méthode bitwise | Gain (AVX2) |
|-----------|-------------------|----------------|-------------|
| Comparaison frames | `pixel != pixel+1` | `XOR(reg_t, reg_t+1)` | ×64 |
| Comptage $N$ | Boucle incrémentale | `POPCOUNT(XOR)` | ×10 |
| Localisation $\tau_k$ | Parcours liste | `TZCNT` itératif | variable |
| Homogénéité $H_s$ | Variance flottante | `POPCOUNT(AND(M,voisins))` | ×32 |
| Corrélation couleurs | Pearson float | `POPCOUNT(XOR(M_{c1},M_{c2}))/n` | ×20 |
| Mémoire | 1 octet/pixel | 1 bit/pixel/couleur | **×8** |

**Structure C++ :** `struct alignas(32) BinaryMask { uint64_t data[4]; }` (256 pixels = 4×64 bits).

### 7 Features Bitwise

| Feature | Formule bitwise | Rôle dans le cartouche |
|---------|-----------------|------------------------|
| $\lambda_{\text{avg}}$ | `POPCOUNT(XOR_total) / (r·n_pix)` | Seuil D : MDL vs Uniforme |
| $H_s$ | `POPCOUNT(AND(M_c, voisins)) / n` | Classification A (spatial) |
| $\rho_{\text{corr}}$ | `1 - POPCOUNT(XOR(M_{c1},M_{c2})) / n` | Classification A (joint) |
| $R_{\text{temp}}$ | $\mathbb{E}[\Delta\tau]^2 / \text{Var}[\Delta\tau]$ | Choix C (R3 si régulier) |
| $m_{\text{eff}}$ | `\|{c : POPCOUNT(XOR_c) > 2}\|` | Choix B (Huffman si $m_{\text{eff}} < m$) |
| $H_{\text{color}}$ | $-\sum p_c \log_2 p_c$ | Seuil N* pour Huffman (Bc) |
| $N_{\text{trans}}$ | `Σ(m_k ≠ m_{k-1})` | Choix Ba vs Bc |

---

## 11. Extension CPPM — Décomposition Monochromatique {#cppm}

La décomposition monochromatique définit $m$ processus simples $N_i^{(c)}$ pour chaque couleur $c$ :

$$N_i^{(c)}(t) = \sum_k \mathbb{I}_{\{\tau_k \leq t,\, m_k = c\}}$$

**Avantage fondamental :** Suppression totale du terme couleur $C_{\text{color}}(B)$ — la couleur est portée par l'indice, indépendamment du mode B.

### 11.1. Longueurs de Code Monochromatiques

| Code | Formule | Note |
|------|---------|------|
| $L_{\text{bin}}$ | $\sum_{c=1}^m r$ | Vecteur d'état binaire |
| $L_{\text{list}}$ | $\sum_c (\log N_c + N_c\log r)$ | Liste sans marques |
| $L_{\text{comb}}$ | $\sum_c \left(\log N_c + \log\binom{r}{N_c}\right)$ | **Optimal uniforme monochrome** |

### 11.2. Règle de Sélection Mono vs Marqué

$$\text{Choisir Mono si} \quad \sum_c \log_2\binom{r}{N_c} < \log_2\binom{r}{N} + C_{\text{color}}(B_{\text{optimal}})$$

---

## 12. Six Fonctionnalités Avancées {#fonctionnalites}

| F# | Fonctionnalité | Dimension | Coût ajouté | Condition MDL |
|----|---------------|-----------|-------------|---------------|
| **F1** | Couleurs Hiérarchiques | B, F | $N\cdot 8L$ bits (Bb) ou $N\cdot\sum H(\hat{P}_i)+\sum D_{\text{huf}_i}$ (Bc) | L niveaux actifs |
| **F2** | Arbre Spatial Quadtree | G | $\log_2(4)$ par nœud | $L4(\text{parent}) > \sum L4(\text{enfants}) + \log_2(4)$ |
| **F3** | Zoom Dynamique | G | Carte saillance $\int\hat{\alpha}\,dt$ | Latence < 100ms |
| **F4** | Extension HTML PPV | — | Custom elements `<ppv-video>` | API JS : `zoomTo()`, `query()` |
| **F5** | Requêtes PPV-QL | — | Index B-Tree / R-Tree | < 50ms sur métadonnées MDL |
| **F6** | Navigateur PPV | — | Cache LRU adaptatif | 60 fps @ 4K, < 10ms/frame |

### 12.1. Critère MDL Étendu (F1 + F2)

$$C_n^{\text{total}}(Q) = C_n^{\text{PP}}(Q) + C_{\text{color}}(B) + \underbrace{L \cdot 8}_{\text{F1 : niveaux}} + \underbrace{\log_2(4)\cdot\frac{4^d-1}{3}}_{\text{F2 : arbre}}$$

### 12.2. Requêtes PPV-QL (F5)

```sql
-- Requête temporelle (index B-Tree sur τ_k)
SELECT jump_times, colors FROM video_block(100, 200)
WHERE time BETWEEN 10 AND 20 AND intensity > 0.5;

-- Requête spatiale (index R-Tree)
SELECT process_id, intensity FROM spatial_region(circle(500, 500, 100))
WHERE color.level1 = 255 AND spatial_level >= 2;

-- Détection d'événements
DETECT events WHERE intensity_change > 0.8 WITHIN time_window(5 seconds);
```

Toutes les requêtes opèrent sur les **métadonnées MDL** ($N$, $\hat{\Lambda}$, coordonnées arbre) sans décoder les pixels.

---

## 13. Architecture 9 Agents {#architecture}

Le pipeline est organisé en 9 agents couvrant 5 phases.

| # | Agent | Phase | Dim. | Entrée | Sortie |
|---|-------|-------|------|--------|--------|
| 0 | 🎬 Extraction Bitwise | 1 | A, B | Flux vidéo | Masques $M_c$, $H_{\text{color}}$, $N_{\text{trans}}$ |
| 1 | 🔬 Classification | 1 | A | Masques + features | Type PP (5 classes) |
| 2 | 🎨 **Codage Couleur** ★ | 2 | **B** | Distribution $\hat{P}(m_k)$ | Mode B optimal, $C_{\text{color}}(B)$ |
| 3 | 🗂️ Structures + Features | 2 | A, B | Type PP + masques | 5 dataclasses avec `color_cost(B)` |
| 4 | 🔄 Précodage R1–R4b | 2 | C | Dataclasses | Repr. optimale + $L_i(B)$ pour 20 combinaisons |
| 5 | 📊 Métriques LMD | 3 | B, C | Représentation | L1–L4 corrigées, $H^2$, gain Huffman |
| 6 | 💻 Codeur | 3 | D | Métriques + processus | Bitstream avec header ABCDEFGH |
| 7 | 🌳 Couleurs + Arbre + Zoom | 4 | E, F, G | Bitstream | Hiérarchie + quadtree + API zoom |
| 8 | 🧪 Tests | 5 | — | Tous modules | Suite pytest 8 classes |

**Agent 2 (nouveau) :** Centralise toute la logique de la dimension B. Il calcule `color_cost(Ba)`, `color_cost(Bb)`, `color_cost(Bc)`, `color_cost(Bd)` et retourne le mode optimal selon $N$, $H_{\text{color}}$, $N_{\text{trans}}$ et $D_{\text{huf}}$.

---

## 14. Algorithme Global — Pipeline ABCDEFGH {#algorithme}

```
INPUT : Bloc vidéo brut (r bins × m couleurs × T frames)
│
├─ 0. EXTRACTION BITWISE
│   ├─ Masques M_c(t,p), XOR inter-frames, POPCOUNT → N_c
│   ├─ Distribution P̂(m_k) des marques → H_color, N_transitions
│   └─ 7 features bitwise → seuils cartouche
│
├─ 1. CLASSIFICATION DU TYPE (Dim. A)
│   └─ Aa/Ab/Ac/Ad/Ae selon features H_s, ρ_corr, χ²
│
├─ 2. SÉLECTION DU MODE COULEUR (Dim. B)  ← NOUVEAU
│   ├─ C_color(Ba) = log₂m + N_trans·log₂m
│   ├─ C_color(Bb) = N·log₂m          ← terme L1–L4 classiques
│   ├─ C_color(Bc) = N·H(P̂) + D_huf  ← si N > N* = D_huf/(log₂m-H)
│   ├─ C_color(Bd) = N·L*(m_k)        ← si m inconnu
│   └─ B* = argmin C_color(B)
│
├─ 3. REPRÉSENTATION TEMPORELLE (Dim. C)
│   └─ (C*, B*) = argmin_{C×B} [L_temporel_C + C_color(B)]  — 20 combinaisons
│
├─ 4. ESTIMATION MDL (Dim. D, E)
│   ├─ Pour chaque famille F ∈ {Hist, Splines, Ondel, Trig} :
│   │   L_MDL(F) = -log L_{α̂} + C_n(F) + C_color(B*)
│   └─ D* = MDL si L_MDL < L4(B*), sinon Uniforme
│
├─ 5. COULEURS HIÉRARCHIQUES + ARBRE (Dim. F, G)
│   ├─ Subdiviser quadtree si L4(parent,B*) > Σ L4(enfants,B*) + log(4)
│   └─ Activer L niveaux chromatiques si gain > overhead Huffman
│
├─ 6. HEADER ABCDEFGH (17 bits)
│   └─ Encoder Cartouche(A*,B*,C*,D*,E*,F*,G*,H)
│
└─ 7. CODAGE FINAL
    ├─ Données temporelles selon C* (R4b = (N, Index))
    └─ Données couleur selon B* :
        Ba → couleur_initiale + transitions
        Bb → N marques uniformes
        Bc → dictionnaire Huffman + N codes variables
        Bd → N codes Elias-delta
```

---

## 15. Arbre de Décision Adaptatif {#arbre-decision}

### 15.1. Seuils de Décision (7 features)

| Feature | Symbole | Seuil bas | Seuil haut | Dimension ciblée |
|---------|---------|-----------|------------|-----------------|
| Densité sauts | $N/r$ | $< 0.10$ | $> 0.80$ | C (R1 vs R4b vs R2) |
| Homogénéité spatiale | $H_s$ | $< 0.30$ | $> 0.70$ | A (Spatial) |
| Corrélation couleurs | $\rho_{\text{corr}}$ | $< 0.15$ | $> 0.80$ | A (Joint vs Marginal) |
| Régularité temporelle | $R_{\text{temp}}$ | $< 0.50$ | $> 1.50$ | C (R3) |
| Intensité moyenne | $\lambda_{\text{avg}}$ | $< 0.10$ | $> 0.50$ | D (MDL vs Uniforme) |
| Entropie couleur | $H_{\text{color}}$ | $< \log_2 m / 2$ | $> \log_2 m - 0.5$ | **B (Ba/Bb/Bc)** |
| Nb transitions | $N_{\text{trans}}$ | $< 0.05 N$ | $> 0.50 N$ | **B (Ba vs Bc)** |

### 15.2. Logique de Décision (pseudocode)

```python
def select_cartouche(bloc):
    features = extract_features_bitwise(bloc)
    N, r, m = features.N, bloc.r, bloc.m

    # Dim. A — Type processus
    if features.H_s > 0.7 and features.rho_corr > 0.8:  A = "Ad"   # Joint
    elif features.rho_corr < 0.15:                        A = "Ac"   # Marginal
    elif features.has_markov_structure():                  A = "Ae"   # Markovien
    elif can_decompose_mono(features):                     A = "Ab"   # Mono (C_color=0)
    else:                                                  A = "Aa"   # Marqué

    # Dim. B — Mode couleur (NOUVEAU)
    H = features.H_color; Nt = features.N_trans; Dh = huffman_overhead(m)
    costs = {
        "Ba": log2(m) + Nt * log2(m),
        "Bb": N * log2(m),                      # ← terme L1-L4 classiques
        "Bc": N * H + Dh,                       # si N > N* = Dh/(log2m-H)
        "Bd": N * (log2(m) + 2*log2(log2(m)))  # si m inconnu
    }
    B = min(costs, key=costs.get) if A != "Ab" else "Bb"  # Ab : C_color=0

    # Dim. C — Représentation temporelle
    Lt_costs = {R: L_temporel(R, N, r) + costs.get(B, 0) for R in ["R1","R2","R3","R4a","R4b"]}
    C = min(Lt_costs, key=Lt_costs.get)

    # Dim. D — Compression
    L_mdl  = mdl_3steps(bloc, B)
    L_unif = min(L_i(B, N, r, m) for i in ["L1","L2","L3","L4"])
    D = "Dc" if L_mdl < L_unif else "Da"

    return Cartouche(A=A, B=B, C=C, D=D, E="Eb", F=2, G=1, H=0)
```

---

## 16. Métriques de Performance {#metriques}

| Métrique | Formule / Cible | Référence |
|----------|----------------|-----------|
| **Compression vs Bb** | Gain Bc = $N(\log_2 m - H(\hat{P})) - D_{\text{huf}}$ bits | Dim. B — Huffman |
| **Seuil Huffman** | $N^* = D_{\text{huf}} / (\log_2 m - H)$ | Dim. B — Rentabilité |
| **Gain Mono** | $N\log_2 m - \sum_c \log_2\binom{r}{N_c}$ bits | Dim. A — Ab vs Aa |
| **Taux de compression** | Brut / Codé > 2.0× vs H.264 | `uniform_temp_coding.tex` |
| **Distance de Hellinger** | $H^2(P,\hat{P}_n) < 0.1$ | Théorème 1 `versatile_main.tex` |
| **Latence zoom** | < 100 ms (niveau N→N+1) | F3 — Zoom |
| **Requête temporelle** | < 50 ms (B-Tree sur $\tau_k$) | F5 — PPV-QL |
| **Requête spatiale** | < 100 ms (R-Tree région) | F5 — PPV-QL |
| **FPS rendu** | 60 fps @ 4K (WebGL 2.0) | F6 — Lecteur |
| **Cache mémoire** | < 512 MB (LRU niveaux 0–1) | F6 — Cache |
| **Décodage** | < 10 ms/frame (WebAssembly) | F6 — Lecteur |

---

## 17. Feuille de Route — 44 Semaines {#roadmap}

| Phase | Durée | Livrables | Agents |
|-------|-------|-----------|--------|
| **1 — Extraction Bitwise** | 3 sem. | Module C++ SIMD, BinaryMask, 7 features | Agent 0 |
| **2 — Classification** | 4 sem. | Classifieur 5 types, dataclasses | Agents 1–3 |
| **3 — Dim. B : Codage Couleur** ★ | 3 sem. | `color_cost(B)`, Huffman, select_color_mode | Agent 2 |
| **4 — Précodage R1–R4b** | 4 sem. | 5 représentations, 20 combinaisons C×B | Agent 4 |
| **5 — Métriques & Codage** | 5 sem. | L1–L4 corrigées, MDL 3 étapes, bitstream | Agents 5–6 |
| **6 — Couleurs Hiérarchiques (F1)** | 4 sem. | HierarchicalColor, downgrade, Huffman/niveau | Agent 7 |
| **7 — Arbre Spatial Quadtree (F2)** | 6 sem. | SpatialProcessTree, critère MDL | Agent 7 |
| **8 — Zoom + PPV-HTML (F3–F4)** | 6 sem. | ZoomController, custom elements, API JS | Agent 7 |
| **9 — Moteur PPV-QL (F5)** | 8 sem. | Parser, indexes B-Tree/R-Tree, < 50ms | Agent 7 |
| **10 — Lecteur + Tests E2E (F6)** | 8 sem. | WebGL, WebAssembly, suite pytest 8 classes | Agents 7–8 |
| **Total** | **44 sem.** | Plateforme PPV complète | 9 Agents |

---

## 18. Références {#references}

| Document | Contenu clé |
|----------|-------------|
| `Uniform_temporal.tex` (Nembé) | Modèle Aalen, modélisation vidéo par PP |
| `uniform_temp_coding.tex` (Nembé) | **Proposition 1 : L1–L4** (mode Bb), algorithmes de sélection |
| `versatile_pp.tex` (Nembé) | Représentations R1–R4b, codes universels, versatilité |
| `versatile_main.tex` (Nembé) | Théorème 1 (Hellinger), MDL, taux de convergence |

### Synthèse des corrections v6

| # | Correction | Impact |
|---|-----------|--------|
| 1 | L1–L4 valables uniquement pour mode Bb | Reformulation de toutes les formules en $L_i(B)$ |
| 2 | Dimension B : Ba/Bb/Bc/Bd | Nouvelle dimension dans le cartouche ABCDEFGH |
| 3 | Agent 2 dédié au codage couleur | Pipeline 9 agents (vs 8 précédemment) |
| 4 | Huffman optimal si $N > N^*$ | Critère de sélection B quantitatif |
| 5 | Ba séquentiel si $N_{\text{trans}} \ll N$ | Codage couleur sans per-saut overhead |
| 6 | Optimisation 20 combinaisons C×B | `select_repr_and_color()` |
| 7 | Monochromatique : $C_{\text{color}}=0$ quel que soit B | Toujours valide, indépendant de la dim. B |

---

*Document mis à jour — Version 6.0 — Mars 2026 | J. Nembé — Codage LMD Versatile / PPV*
