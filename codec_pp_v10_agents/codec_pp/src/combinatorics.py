"""
Combinatorics — Indexation combinatoire pour la représentation R4.

R4 encode le vecteur booléen b (positions des sauts) par son indice
dans l'énumération des vecteurs de r bits ayant exactement n bits à 1.

L'indice s est calculé par le système combinatoire (combinadic) :
    s = sum_{k tel que b_k=1} C(position_k, rang_k)

Le décodage inverse reconstruit b depuis (n, s, r).
"""

from math import comb
from functools import lru_cache
from typing import List, Tuple


@lru_cache(maxsize=4096)
def binomial(n: int, k: int) -> int:
    """C(n, k) avec cache."""
    if k < 0 or k > n:
        return 0
    return comb(n, k)


def bits_for_binomial(r: int, n: int) -> int:
    """Nombre de bits nécessaires pour stocker l'indice s ∈ [0, C(r,n)-1]."""
    c = binomial(r, n)
    if c <= 1:
        return 0
    return (c - 1).bit_length()


def bool_vector_to_index(bits: List[int], r: int, n: int) -> int:
    """
    Convertit un vecteur booléen de r bits avec n bits à 1
    en son indice dans l'énumération combinatoire.

    Algorithme : Combinadic ranking.
    On parcourt bits de gauche à droite (indices 0..r-1).
    Pour chaque bit à 1 en position p (le k-ème bit à 1, k=1..n),
    on ajoute C(r-1-p, n-k+1) - ... selon le système combinadique.

    Concrètement, on utilise la formule classique :
    Si les positions des 1 sont p_0 < p_1 < ... < p_{n-1},
    alors s = C(r,n) - 1 - sum_{i=0}^{n-1} C(r - 1 - p_i, n - i)
    
    Ici on utilise l'approche co-lex pour simplifier.
    """
    # Positions des bits à 1 (indices croissants)
    positions = [i for i in range(r) if bits[i] == 1]
    assert len(positions) == n, f"Attendu {n} bits à 1, trouvé {len(positions)}"

    # Ranking co-lexicographique (ordre combinadique)
    # s = sum C(p_i, i+1) pour i=0..n-1  (avec p triés croissant)
    s = 0
    for i, p in enumerate(positions):
        s += binomial(p, i + 1)
    return s


def index_to_bool_vector(s: int, r: int, n: int) -> List[int]:
    """
    Reconstruit le vecteur booléen de r bits depuis l'indice s
    et le nombre de bits à 1 = n.

    Algorithme : Combinadic unranking (co-lex).
    """
    bits = [0] * r
    k = n
    for i in range(r - 1, -1, -1):
        c = binomial(i, k)
        if c <= s:
            s -= c
            bits[i] = 1
            k -= 1
            if k == 0:
                break
    return bits


def compute_jump_vector(sequence: List[int]) -> Tuple[List[int], int]:
    """
    Depuis une séquence de couleurs (c_0, ..., c_{r-1}),
    retourne le vecteur booléen de sauts et le nombre de sauts.

    Convention : b[0] = 1 toujours (premier instant = premier "saut"),
                 b[k] = 1 si sequence[k] != sequence[k-1] pour k ≥ 1.
    """
    r = len(sequence)
    b = [0] * r
    b[0] = 1  # premier instant est toujours un "saut"
    n_jumps = 1
    for k in range(1, r):
        if sequence[k] != sequence[k - 1]:
            b[k] = 1
            n_jumps += 1
    return b, n_jumps


def bool_vector_to_int(bits: List[int]) -> int:
    """Convertit un vecteur de bits en entier (bits[0] = MSB)."""
    val = 0
    for b in bits:
        val = (val << 1) | (b & 1)
    return val


def int_to_bool_vector(val: int, r: int) -> List[int]:
    """Convertit un entier en vecteur de r bits (MSB first)."""
    bits = []
    for i in range(r - 1, -1, -1):
        bits.append((val >> i) & 1)
    return bits
