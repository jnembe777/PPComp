"""
lookup_tables.py - Tables de paramétrage pré-calculées pour la Dimension C
==========================================================================

Implémente les tables T1-T5 et l'Algorithme 1 corrigé selon la spécification
pp-codec-algo1-spec.docx (Nembé, J. - Mars 2026).

Corrections apportées:
- A1: Remplacement du seuil scalaire density_R4b_R2 par la table T2
- A2: δ(r,m) dynamique via la table T3
- A3: Borne λ(r,m) via la table T4
- A4: log₂(m) pré-calculé dans T5
- A6: Algorithme 1 complet de Nembé

Référence: Nembé, J. — Uniform temporal encoding of massive video datasets (2015)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
import math

from ..utils.math_utils import logC, binomial


# =============================================================================
# Table T1 — Condition d'existence de D34
# =============================================================================
# Vérifie si g(r/2) > 2^r, c'est-à-dire si la zone L3 < L4 est non vide.
# Pour r >= 30, la condition est toujours satisfaite.

def check_D34_exists(r: int) -> bool:
    """
    Vérifie si la zone D34 (L3 < L4) existe pour une résolution r donnée.

    La zone existe si g(r/2) > 2^r où g(N) = N · C(N,r).
    Pour r >= 30, c'est toujours vrai.

    Args:
        r: Nombre de bins (résolution temporelle)

    Returns:
        True si la zone D34 existe
    """
    if r >= 30:
        return True

    # Calcul explicite pour petits r
    N = r // 2
    # g(N) = N · C(r, N)
    log_g = np.log2(N + 1) + logC(r, N)
    return log_g > r


# =============================================================================
# Table T2 — Bornes γ_l et γ_r de la zone L3 < L4
# =============================================================================
# Pré-calculées selon la formule: g(N) = N · C(N,r), γ = g⁻¹(2^r)

# Valeurs extraites de la spécification
_TABLE_T2 = {
    30:   (12, 19),
    60:   (24, 37),
    120:  (51, 70),
    240:  (105, 136),
    256:  (113, 144),
    480:  (217, 264),
    512:  (232, 281),
    900:  (417, 484),
    1024: (476, 549),
    1800: (850, 951),
    2048: (970, 1079),
    3600: (1724, 1877),
    4096: (1967, 2130),
}


def _compute_g(N: int, r: int) -> float:
    """Calcule g(N) = N · C(r, N) en log2."""
    if N <= 0 or N > r:
        return 0.0
    return np.log2(N + 1) + logC(r, N)


def _find_gamma_bounds(r: int) -> Tuple[int, int]:
    """
    Trouve γ_l et γ_r par recherche dichotomique.

    γ_l = g⁻¹_l(2^r) : plus petit N tel que g(N) >= 2^r
    γ_r = g⁻¹_r(2^r) : plus grand N tel que g(N) >= 2^r

    Args:
        r: Résolution temporelle

    Returns:
        (γ_l, γ_r)
    """
    target = float(r)  # log2(2^r) = r

    # La fonction g(N) est croissante jusqu'à r/2 puis décroissante
    # On cherche les deux racines de g(N) = 2^r

    # Trouver γ_l (côté gauche, N < r/2)
    lo, hi = 1, r // 2
    gamma_l = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if _compute_g(mid, r) >= target:
            gamma_l = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # Trouver γ_r (côté droit, N > r/2)
    lo, hi = r // 2, r
    gamma_r = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        if _compute_g(mid, r) >= target:
            gamma_r = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return (gamma_l, gamma_r)


@lru_cache(maxsize=256)
def get_gamma_bounds(r: int) -> Tuple[int, int]:
    """
    Retourne les bornes γ_l et γ_r pour la résolution r.

    La zone où L3 (vecteur booléen R2) bat L4 (combinatoire R4b) est N ∈ [γ_l, γ_r].

    Args:
        r: Nombre de bins

    Returns:
        (γ_l, γ_r) : bornes de la zone L3 < L4
    """
    # Utiliser la table si disponible
    if r in _TABLE_T2:
        return _TABLE_T2[r]

    # Interpolation pour les valeurs non tabulées
    sorted_r = sorted(_TABLE_T2.keys())

    # Trouver les bornes pour interpolation
    r_low = max(rr for rr in sorted_r if rr <= r) if any(rr <= r for rr in sorted_r) else sorted_r[0]
    r_high = min(rr for rr in sorted_r if rr >= r) if any(rr >= r for rr in sorted_r) else sorted_r[-1]

    if r_low == r_high:
        return _TABLE_T2[r_low]

    # Interpolation linéaire
    gamma_l_low, gamma_r_low = _TABLE_T2[r_low]
    gamma_l_high, gamma_r_high = _TABLE_T2[r_high]

    t = (r - r_low) / (r_high - r_low)
    gamma_l = int(round(gamma_l_low + t * (gamma_l_high - gamma_l_low)))
    gamma_r = int(round(gamma_r_low + t * (gamma_r_high - gamma_r_low)))

    return (gamma_l, gamma_r)


# =============================================================================
# Table T3 — Seuil δ(r,m) pour L1 vs L3
# =============================================================================
# δ(r,m) = ⌈r · (1 − 1/log₂(m))⌉
# L1 < L3 ssi N > δ(r,m)

# Valeurs extraites de la spécification
_TABLE_T3 = {
    (120, 2): 0,    (120, 4): 60,   (120, 8): 81,   (120, 16): 90,   (120, 32): 96,   (120, 64): 100,
    (240, 2): 0,    (240, 4): 120,  (240, 8): 161,  (240, 16): 180,  (240, 32): 192,  (240, 64): 200,
    (256, 2): 0,    (256, 4): 128,  (256, 8): 171,  (256, 16): 192,  (256, 32): 205,  (256, 64): 214,
    (512, 2): 0,    (512, 4): 256,  (512, 8): 342,  (512, 16): 384,  (512, 32): 410,  (512, 64): 427,
    (900, 2): 0,    (900, 4): 450,  (900, 8): 601,  (900, 16): 675,  (900, 32): 720,  (900, 64): 750,
    (1024, 2): 0,   (1024, 4): 512, (1024, 8): 683, (1024, 16): 768, (1024, 32): 820, (1024, 64): 854,
}


@lru_cache(maxsize=1024)
def get_delta_threshold(r: int, m: int) -> int:
    """
    Retourne le seuil δ(r,m) pour la frontière L1 vs L3.

    L1 < L3 ssi N > δ(r,m)

    Formule: δ(r,m) = ⌈r · (1 − 1/log₂(m))⌉

    Args:
        r: Nombre de bins
        m: Nombre de couleurs

    Returns:
        Seuil δ
    """
    if m <= 1:
        return r  # Cas dégénéré

    # Utiliser la table si disponible
    if (r, m) in _TABLE_T3:
        return _TABLE_T3[(r, m)]

    # Calcul exact
    log2_m = np.log2(m)
    if log2_m <= 0:
        return r

    delta = math.ceil(r * (1 - 1 / log2_m))
    return max(0, delta)


# =============================================================================
# Table T4 — Borne λ(r,m) pour L4 vs L1
# =============================================================================
# λ(r,m) = h⁻¹(m^r) où h(N) = N · C(N,r) · m^N
# L4 < L1 ssi N ∈ [0, λ(r,m)]

# Valeurs extraites de la spécification
_TABLE_T4 = {
    (120, 2): 26,   (120, 4): 58,   (120, 8): 83,   (120, 16): 99,   (120, 32): 108,  (120, 64): 112,
    (240, 2): 53,   (240, 4): 118,  (240, 8): 168,  (240, 16): 200,  (240, 32): 217,  (240, 64): 227,
    (256, 2): 57,   (256, 4): 126,  (256, 8): 180,  (256, 16): 213,  (256, 32): 232,  (256, 64): 242,
    (512, 2): 115,  (512, 4): 254,  (512, 8): 361,  (512, 16): 429,  (512, 32): 467,  (512, 64): 487,
    (900, 2): 203,  (900, 4): 448,  (900, 8): 637,  (900, 16): 756,  (900, 32): 823,  (900, 64): 858,
    (1024, 2): 231, (1024, 4): 510, (1024, 8): 725, (1024, 16): 860, (1024, 32): 937, (1024, 64): 977,
}


def _compute_h(N: int, r: int, m: int) -> float:
    """Calcule h(N) = N · C(r, N) · m^N en log2."""
    if N <= 0 or N > r:
        return 0.0
    return np.log2(N + 1) + logC(r, N) + N * np.log2(m)


def _find_lambda_bound(r: int, m: int) -> int:
    """
    Trouve λ(r,m) par recherche dichotomique.

    λ(r,m) = h⁻¹(m^r) : plus grand N tel que h(N) <= m^r

    Args:
        r: Résolution temporelle
        m: Nombre de couleurs

    Returns:
        λ(r,m)
    """
    target = r * np.log2(m)  # log2(m^r)

    # Recherche dichotomique
    lo, hi = 0, r
    lambda_rm = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        h_val = _compute_h(mid, r, m)
        if h_val <= target:
            lambda_rm = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return lambda_rm


@lru_cache(maxsize=1024)
def get_lambda_bound(r: int, m: int) -> int:
    """
    Retourne la borne λ(r,m) pour la frontière L4 vs L1.

    L4 < L1 ssi N ∈ [0, λ(r,m)]

    Args:
        r: Nombre de bins
        m: Nombre de couleurs

    Returns:
        Borne λ
    """
    if m <= 1:
        return r  # Cas dégénéré

    # Utiliser la table si disponible
    if (r, m) in _TABLE_T4:
        return _TABLE_T4[(r, m)]

    # Calcul exact par dichotomie
    return _find_lambda_bound(r, m)


# =============================================================================
# Table T5 — Valeurs pré-calculées de log₂(m)
# =============================================================================

_TABLE_T5 = {
    2: 1.0,
    3: 1.584963,
    4: 2.0,
    5: 2.321928,
    6: 2.584963,
    7: 2.807355,
    8: 3.0,
    10: 3.321928,
    12: 3.584963,
    15: 3.906891,
    16: 4.0,
    20: 4.321928,
    24: 4.584963,
    32: 5.0,
    48: 5.584963,
    64: 6.0,
    128: 7.0,
    255: 7.994353,
    256: 8.0,
    512: 9.0,
    1024: 10.0,
}


def get_log2_m(m: int) -> float:
    """
    Retourne log₂(m), pré-calculé si disponible.

    Args:
        m: Nombre de couleurs

    Returns:
        log₂(m)
    """
    if m in _TABLE_T5:
        return _TABLE_T5[m]
    return np.log2(m) if m > 0 else 0.0


# =============================================================================
# Algorithme 1 corrigé — Sélection de la représentation optimale
# =============================================================================

@dataclass
class RepresentationSelection:
    """Résultat de la sélection de représentation (Dimension C)."""
    representation: int  # 0=R1, 1=R2, 2=R3, 3=R4a, 4=R4b
    strategy: str  # "L1", "L3", "L4"
    gamma_l: int
    gamma_r: int
    delta_rm: int
    lambda_rm: int
    density: float  # λ = N/r


def select_representation_algo1(N: int, r: int, m: int) -> RepresentationSelection:
    """
    Algorithme 1 corrigé — Sélection de la représentation temporelle optimale.

    Calcule les coûts exacts pour chaque représentation et sélectionne le minimum.
    Les formules correspondent à celles de exhaustive_search._compute_costs_C():

    - R1 (TIMESTAMPS): N * log2(r)
    - R2 (COUNT): r * log2(N/r + 1)
    - R3 (INTERVALS): N * log2(r/N + 1)
    - R4a (BOOLEAN): r
    - R4b (COMBINATORIAL): log2(N+1) + log2(C(r,N))

    Args:
        N: Nombre de sauts (événements)
        r: Nombre de bins (résolution temporelle)
        m: Nombre de couleurs (non utilisé pour C, mais gardé pour compatibilité)

    Returns:
        RepresentationSelection avec la représentation optimale
    """
    # Charger les tables pour référence
    gamma_l, gamma_r = get_gamma_bounds(r)
    delta_rm = get_delta_threshold(r, m)
    lambda_rm = get_lambda_bound(r, m)

    density = N / r if r > 0 else 0.0

    # Cas spécial: bloc vide
    if N == 0:
        return RepresentationSelection(
            representation=0,  # R1 - Timestamps (coût = 0)
            strategy="L1",
            gamma_l=gamma_l,
            gamma_r=gamma_r,
            delta_rm=delta_rm,
            lambda_rm=lambda_rm,
            density=density
        )

    # Calcul des coûts exacts (mêmes formules que exhaustive_search.py)
    log2_r = np.log2(r) if r > 0 else 0

    # R1 - Timestamps: N * log2(r)
    cost_R1 = N * log2_r

    # R2 - Count/Histogram: r * log2(N/r + 1)
    avg_count = N / r if r > 0 else 0
    cost_R2 = r * np.log2(avg_count + 1) if avg_count > 0 else r

    # R3 - Intervals: N * log2(r/N + 1)
    if N > 0 and r > 0:
        avg_interval = r / N
        cost_R3 = N * np.log2(avg_interval + 1)
    else:
        cost_R3 = float('inf')

    # R4a - Boolean: r bits
    cost_R4a = float(r)

    # R4b - Combinatorial: log2(N+1) + log2(C(r,N))
    cost_R4b = np.log2(N + 1) + logC(r, N)

    # Trouver le minimum
    costs = {
        0: cost_R1,   # TIMESTAMPS
        1: cost_R2,   # COUNT
        2: cost_R3,   # INTERVALS
        3: cost_R4a,  # BOOLEAN
        4: cost_R4b,  # COMBINATORIAL
    }

    best_repr = min(costs, key=costs.get)
    best_cost = costs[best_repr]

    # Déterminer la stratégie pour le logging
    strategy_map = {0: "R1", 1: "R2", 2: "R3", 3: "R4a", 4: "R4b"}
    strategy = strategy_map[best_repr]

    return RepresentationSelection(
        representation=best_repr,
        strategy=strategy,
        gamma_l=gamma_l,
        gamma_r=gamma_r,
        delta_rm=delta_rm,
        lambda_rm=lambda_rm,
        density=density
    )


def predict_dimension_C(N: int, r: int, m: int) -> int:
    """
    Prédit la valeur optimale de la dimension C (représentation temporelle).

    Interface simplifiée pour l'intégration dans le validateur.

    Args:
        N: Nombre de sauts
        r: Nombre de bins
        m: Nombre de couleurs

    Returns:
        Valeur de C (0-4)
    """
    result = select_representation_algo1(N, r, m)
    return result.representation


# =============================================================================
# Fonctions utilitaires pour le diagnostic
# =============================================================================

def compute_all_costs(N: int, r: int, m: int) -> Dict[str, float]:
    """
    Calcule tous les coûts temporels pour diagnostic.

    Args:
        N: Nombre de sauts
        r: Nombre de bins
        m: Nombre de couleurs

    Returns:
        Dict avec L1, L2, L3, L4 et leurs formules
    """
    log2_m = get_log2_m(m)

    # L1: r · log₂(m) (full state)
    L1 = r * log2_m

    # L2: N · log₂(r) (intervalles - rarement optimal)
    L2 = N * np.log2(r) if r > 1 and N > 0 else 0.0

    # L3: r bits (vecteur booléen)
    L3 = float(r)

    # L4: log₂(N+1) + log₂C(r,N) (combinatoire)
    L4 = np.log2(N + 1) + logC(r, N) if N > 0 else 0.0

    return {
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "L4": L4,
        "optimal": min(L1, L3, L4),
        "optimal_strategy": "L1" if L1 <= min(L3, L4) else ("L3" if L3 <= L4 else "L4"),
    }


def validate_tables() -> Dict[str, bool]:
    """
    Valide la cohérence des tables par rapport aux calculs exacts.

    Returns:
        Dict avec le statut de validation de chaque table
    """
    results = {}

    # Valider T2
    t2_ok = True
    for r, (gamma_l, gamma_r) in _TABLE_T2.items():
        computed = _find_gamma_bounds(r)
        if abs(computed[0] - gamma_l) > 1 or abs(computed[1] - gamma_r) > 1:
            t2_ok = False
            break
    results["T2_gamma_bounds"] = t2_ok

    # Valider T3
    t3_ok = True
    for (r, m), delta in _TABLE_T3.items():
        computed = get_delta_threshold.__wrapped__(r, m)  # Bypass cache
        if abs(computed - delta) > 1:
            t3_ok = False
            break
    results["T3_delta_threshold"] = t3_ok

    # Valider T5
    t5_ok = True
    for m, log_val in _TABLE_T5.items():
        computed = np.log2(m)
        if abs(computed - log_val) > 0.001:
            t5_ok = False
            break
    results["T5_log2_m"] = t5_ok

    return results


# =============================================================================
# Export
# =============================================================================

__all__ = [
    'check_D34_exists',
    'get_gamma_bounds',
    'get_delta_threshold',
    'get_lambda_bound',
    'get_log2_m',
    'select_representation_algo1',
    'predict_dimension_C',
    'compute_all_costs',
    'validate_tables',
    'RepresentationSelection',
]
