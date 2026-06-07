"""
process_types.py - Énumérations et types pour le cartouche ABCDEFGH
====================================================================

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

from enum import IntEnum
from typing import Literal


class ProcessType(IntEnum):
    """Dimension A - Type de processus ponctuel (3 bits, 5 valeurs)"""
    MARKED = 0          # Aa - Réel Marqué
    MONOCHROMATIC = 1   # Ab - Monochromatique (C_color = 0)
    VECTORIAL_MARG = 2  # Ac - Vectoriel Marginal
    VECTORIAL_JOINT = 3 # Ad - Vectoriel Joint
    MARKOVIAN = 4       # Ae - Markovien


class ColorMode(IntEnum):
    """Dimension B - Mode de codage couleur (2 bits, 4 valeurs)

    CORRECTION FONDAMENTALE v6:
    Les formules L1-L4 classiques ne sont valables que pour UNIFORM (Bb).
    L_i(B) = L_temporel_i + C_color(B)
    """
    SEQUENTIAL = 0   # Ba - Séquentiel: log₂m + N_trans·log₂m
    UNIFORM = 1      # Bb - Uniforme: N·log₂m (formules L1-L4 actuelles)
    HUFFMAN = 2      # Bc - Huffman: N·H(P̂) + D_huf
    ELIAS = 3        # Bd - Universel Elias: N·L*(m)


class Representation(IntEnum):
    """Dimension C - Représentation temporelle (3 bits, 5 valeurs)"""
    TIMESTAMPS = 0   # R1 - Liste des temps τ_k
    COUNT = 1        # R2 - Histogramme n_i par bin
    INTERVALS = 2    # R3 - Intervalles Δ_k
    BOOLEAN = 3      # R4a - Vecteur booléen {0,1}^r
    COMBINATORIAL = 4 # R4b - (N, Index) combinatoire - OPTIMAL


class CompressionMode(IntEnum):
    """Dimension D - Mode de compression (2 bits, 3 valeurs)"""
    UNIFORM = 0      # Da - Uniforme min(L1-L4)
    UNIVERSAL = 1    # Db - Universel entiers
    MDL = 2          # Dc - MDL Statistique 3 étapes


class IntensityFamily(IntEnum):
    """Dimension E - Famille de fonctions pour α̂(t) (2 bits, 4 valeurs)"""
    HISTOGRAM = 0    # Ea - Histogrammes adaptatifs
    SPLINES = 1      # Eb - Splines cubiques (recommandé)
    WAVELETS = 2     # Ec - Ondelettes
    TRIGONOMETRIC = 3 # Ed - Polynômes trigonométriques


class ChromaticLevel(IntEnum):
    """Dimension F - Niveau chromatique (2 bits, 3 valeurs)"""
    BITS_8 = 1       # F1 - 256 couleurs
    BITS_16 = 2      # F2 - 65536 couleurs (défaut)
    BITS_24 = 3      # F3 - 16M couleurs HDR


class SpatialResolution(IntEnum):
    """Dimension G - Résolution spatiale quadtree (2 bits, 4 valeurs)"""
    PX_16 = 0        # G0 - 16×16 pixels
    PX_8 = 1         # G1 - 8×8 pixels (défaut)
    PX_4 = 2         # G2 - 4×4 pixels
    PX_2 = 3         # G3 - 2×2 pixels (Ultra HD)


class TemporalMode(IntEnum):
    """Dimension H - Mode temporel (1 bit, 2 valeurs)"""
    CONTINUOUS = 0   # H0 - Temps continu
    DISCRETE = 1     # H1 - Temps discret


# Type aliases pour clarté
ColorModeStr = Literal["Ba", "Bb", "Bc", "Bd"]
RepresentationStr = Literal["R1", "R2", "R3", "R4a", "R4b"]
