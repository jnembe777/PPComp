"""
math_utils.py - Utilitaires mathématiques pour le codage LMD
=============================================================

Fonctions pour:
- Calculs combinatoires (logC, binomial)
- Entropie et information
- Distance de Hellinger
- Codes universels (Elias)

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, Optional, Union
from functools import lru_cache


def log2(x: float) -> float:
    """Logarithme base 2 sécurisé (retourne 0 si x <= 0)."""
    if x <= 0:
        return 0.0
    return np.log2(x)


@lru_cache(maxsize=10000)
def logC(n: int, k: int) -> float:
    """
    Calcule log₂(C(n,k)) = log₂(n! / (k!(n-k)!))

    Utilise l'approximation de Stirling pour grands n.

    Args:
        n: Taille de l'ensemble
        k: Taille du sous-ensemble

    Returns:
        log₂ du coefficient binomial
    """
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 0.0

    # Symétrie
    k = min(k, n - k)

    # Calcul direct pour petits k
    result = 0.0
    for i in range(k):
        result += np.log2(n - i) - np.log2(i + 1)

    return result


def binomial(n: int, k: int) -> int:
    """Coefficient binomial C(n,k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def entropy(probs: Union[Dict, np.ndarray, list]) -> float:
    """
    Calcule l'entropie H(P) = -Σ p·log₂p

    Args:
        probs: Distribution de probabilité (dict, array ou list)

    Returns:
        Entropie en bits
    """
    if isinstance(probs, dict):
        values = list(probs.values())
    else:
        values = list(probs)

    H = 0.0
    for p in values:
        if p > 0:
            H -= p * np.log2(p)
    return H


def empirical_entropy(counts: np.ndarray) -> float:
    """
    Entropie empirique à partir des comptages.

    Args:
        counts: Vecteur de comptages

    Returns:
        Entropie en bits
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return entropy(probs)


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Distance de Hellinger H²(P, Q) = (1/2)·Σ(√p_i - √q_i)²

    Args:
        p, q: Distributions de probabilité

    Returns:
        H² distance (entre 0 et 1)
    """
    sqrt_p = np.sqrt(np.maximum(p, 0))
    sqrt_q = np.sqrt(np.maximum(q, 0))
    return 0.5 * np.sum((sqrt_p - sqrt_q) ** 2)


def hellinger_bound(n: int, complexity: float, bias_term: float = 0.0) -> float:
    """
    Borne de Hellinger pour l'estimateur MDL (Théorème 1).

    H²(P, P̂_n) ≤ bias + C_n(Q) / n

    Args:
        n: Taille de l'échantillon
        complexity: Complexité C_n(Q) du modèle
        bias_term: Terme de biais H²(P, Q)

    Returns:
        Borne supérieure sur H²
    """
    if n == 0:
        return bias_term
    return bias_term + complexity / n


# === Codes universels d'entiers ===

def elias_delta_length(n: int) -> int:
    """
    Longueur du code δ d'Elias pour n ≥ 1.

    L(n) = 1 + ⌊log₂n⌋ + 2·⌊log₂(1 + ⌊log₂n⌋)⌋

    Args:
        n: Entier à encoder (≥ 1)

    Returns:
        Longueur en bits
    """
    if n < 1:
        return 0
    k = int(np.floor(np.log2(n))) if n >= 1 else 0
    return 1 + k + 2 * int(np.floor(np.log2(1 + k + 1e-9)))


def elias_gamma_length(n: int) -> int:
    """
    Longueur du code γ d'Elias pour n ≥ 1.

    L(n) = 2·⌊log₂n⌋ + 1

    Args:
        n: Entier à encoder (≥ 1)

    Returns:
        Longueur en bits
    """
    if n < 1:
        return 0
    return 2 * int(np.floor(np.log2(n))) + 1


def universal_integer_length(n: int, method: str = "elias_delta") -> int:
    """
    Longueur d'un code universel d'entier.

    Args:
        n: Entier à encoder
        method: "elias_delta", "elias_gamma", ou "log_star"

    Returns:
        Longueur en bits
    """
    if method == "elias_delta":
        return elias_delta_length(n)
    elif method == "elias_gamma":
        return elias_gamma_length(n)
    elif method == "log_star":
        # log* itéré
        if n < 1:
            return 0
        total = 1
        x = n
        while x > 1:
            x = np.log2(x)
            total += int(np.ceil(x))
        return total
    else:
        return elias_delta_length(n)


# === Métriques de complexité ===

def model_complexity(k: int, n: int, method: str = "mdl") -> float:
    """
    Calcule la complexité du modèle C_n(Q).

    Args:
        k: Nombre de paramètres du modèle
        n: Taille de l'échantillon
        method: "mdl", "bic", "aic"

    Returns:
        Pénalité de complexité
    """
    if method == "mdl":
        # MDL: (k/2)·log₂(n)
        return (k / 2) * np.log2(n) if n > 1 else 0.0
    elif method == "bic":
        # BIC: (k/2)·ln(n)
        return (k / 2) * np.log(n) if n > 1 else 0.0
    elif method == "aic":
        # AIC: k
        return float(k)
    else:
        return (k / 2) * np.log2(n) if n > 1 else 0.0


def intensity_complexity(family: str, k_params: int, N: int) -> float:
    """
    Complexité pour l'estimation d'intensité α̂(t).

    C_n(F) = (k+4)·log₂N / 2

    Args:
        family: "histogram", "splines", "wavelets", "trigonometric"
        k_params: Nombre de paramètres de la famille
        N: Nombre d'observations

    Returns:
        Pénalité de complexité
    """
    if N <= 1:
        return 0.0
    return (k_params + 4) * np.log2(N) / 2


# === Calculs combinatoires pour R4b ===

def combinatorial_index(positions: np.ndarray, r: int) -> int:
    """
    Calcule l'index combinatoire d'un vecteur booléen.

    Index = rang de positions dans l'énumération lexicographique
    de C(r, N) vecteurs.

    Args:
        positions: Positions des 1 dans le vecteur booléen (triées)
        r: Longueur du vecteur

    Returns:
        Index combinatoire
    """
    N = len(positions)
    if N == 0 or N > r:
        return 0

    index = 0
    for i, pos in enumerate(positions):
        # Nombre de vecteurs avec le i-ème 1 avant la position pos
        remaining = N - i - 1
        for j in range(pos - (positions[i-1] + 1 if i > 0 else 0)):
            prev = positions[i-1] + 1 + j if i > 0 else j
            index += binomial(r - prev - 1, remaining)

    return index


def decode_combinatorial_index(index: int, N: int, r: int) -> np.ndarray:
    """
    Décode un index combinatoire en positions.

    Args:
        index: Index combinatoire
        N: Nombre de positions
        r: Longueur du vecteur

    Returns:
        Positions des 1 (triées)
    """
    positions = []
    remaining = N
    pos = 0

    for i in range(N):
        while pos < r:
            count = binomial(r - pos - 1, remaining - 1)
            if index < count:
                positions.append(pos)
                remaining -= 1
                pos += 1
                break
            index -= count
            pos += 1

    return np.array(positions)
