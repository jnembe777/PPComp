"""
Matrices — Construction des structures de données du codec.

Depuis la matrice vidéo brute M[t][i][j] (frames × lignes × colonnes),
on construit les matrices suivantes pour chaque pixel (i,j) :

  B[i][j][0]     = n  (nombre de sauts)
  B[i][j][1..r]  = vecteur booléen des sauts

  S[i][j]        = indice combinatoire du vecteur booléen dans C(r, n)

  MT[i][j][0]    = n  (nombre de sauts)
  MT[i][j][1..n] = dates (indices temporels) des sauts

  CB[i][j][0..n-1] = couleurs aux dates de saut (depuis B)
  CS[i][j][0..n-1] = couleurs aux dates de saut (depuis S, identique à CB)
  CMT[i][j][0..n-1]= couleurs aux dates de saut (depuis MT)

Convention : le premier instant (t=0) est toujours considéré comme un "saut".
"""

import numpy as np
from typing import Dict, Any, Tuple
from .combinatorics import (
    bool_vector_to_index, compute_jump_vector, bits_for_binomial
)


def build_all_matrices(M: np.ndarray) -> Dict[str, Any]:
    """
    Construit toutes les matrices du codec depuis M.

    Args:
        M: np.ndarray de shape (r, nl, nc) — frames × lignes × colonnes.
           Chaque élément M[t][i][j] est une valeur de couleur (entier).

    Returns:
        dict avec clés : 'B', 'S', 'MT', 'CB', 'CS', 'CMT',
                         'nl', 'nc', 'r', 'M'
    """
    r, nl, nc = M.shape

    # Pré-allouer : B est de taille variable par pixel, on utilise des listes
    B = [[None for _ in range(nc)] for _ in range(nl)]
    S = np.zeros((nl, nc), dtype=np.int64)
    MT = [[None for _ in range(nc)] for _ in range(nl)]
    CB = [[None for _ in range(nc)] for _ in range(nl)]
    CS = [[None for _ in range(nc)] for _ in range(nl)]
    CMT = [[None for _ in range(nc)] for _ in range(nl)]

    for i in range(nl):
        for j in range(nc):
            # Extraire la séquence temporelle du pixel
            seq = [int(M[t, i, j]) for t in range(r)]

            # Vecteur booléen des sauts
            bvec, n_jumps = compute_jump_vector(seq)

            # B[i][j] : [n, b_0, b_1, ..., b_{r-1}]
            B[i][j] = [n_jumps] + bvec

            # S[i][j] : indice combinatoire
            S[i][j] = bool_vector_to_index(bvec, r, n_jumps)

            # MT[i][j] : [n, t_1, t_2, ..., t_n] (dates des sauts)
            dates = [k for k in range(r) if bvec[k] == 1]
            MT[i][j] = [n_jumps] + dates

            # Couleurs aux dates de saut
            colors = [seq[k] for k in dates]
            CB[i][j] = colors
            CS[i][j] = colors  # identique par construction
            CMT[i][j] = colors

    return {
        'M': M,
        'B': B,
        'S': S,
        'MT': MT,
        'CB': CB,
        'CS': CS,
        'CMT': CMT,
        'nl': nl,
        'nc': nc,
        'r': r,
    }


def compute_representation_lengths(
    n_jumps: int, r: int, color_bits: int, duration_bits: int
) -> Tuple[int, int, int, int]:
    """
    Calcule les longueurs L1, L2, L3, L4 pour un pixel donné.

    Args:
        n_jumps: nombre de sauts (= nombre de couleurs distinctes consécutives)
        r: nombre de frames
        color_bits: bits par couleur
        duration_bits: bits pour coder un indice temporel (ceil(log2(r)))

    Returns:
        (L1, L2, L3, L4) en bits
    """
    # R1 : (c_1, ..., c_r) — toutes les couleurs
    L1 = r * color_bits

    # R2 : (n, t_1, ..., t_n, c_1, ..., c_n)
    L2 = duration_bits + n_jumps * (duration_bits + color_bits)

    # R3 : (b, c_1, ..., c_n) — vecteur booléen + couleurs
    L3 = r + n_jumps * color_bits

    # R4 : (n, s, c_1, ..., c_n) — indice combinatoire + couleurs
    s_bits = bits_for_binomial(r, n_jumps)
    L4 = duration_bits + s_bits + n_jumps * color_bits

    return L1, L2, L3, L4


def optimal_representation(
    n_jumps: int, r: int, color_bits: int, duration_bits: int
) -> Tuple[int, int]:
    """
    Retourne (rep_index, min_length) où rep_index ∈ {1,2,3,4}.
    """
    lengths = compute_representation_lengths(
        n_jumps, r, color_bits, duration_bits
    )
    min_len = min(lengths)
    best_rep = lengths.index(min_len) + 1
    return best_rep, min_len
