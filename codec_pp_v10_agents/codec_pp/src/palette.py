"""
Palette — Réduction de palette par bloc et codage adaptatif selon m.

Principe fondamental (Nembé) :
  Avec le vecteur booléen (R3), la longueur est indépendante du nombre
  de sauts — le vrai paramètre est le nombre m de couleurs distinctes
  dans le bloc.

  3 régimes selon m :

  Régime 1 — m petit (m ≤ seuil_palette) :
    Codage par palette indexée.
    On stocke la palette (m × color_bits) puis chaque valeur pixel
    est remplacée par son indice sur ⌈log₂(m)⌉ bits.
    Le pipeline R1-R4 s'applique ensuite sur les indices.
    Gain : (8 - ⌈log₂(m)⌉) / 8 sur chaque symbole couleur.

    Variante "couleur par couleur" : pour chaque couleur c_k,
    encoder la carte binaire spatio-temporelle B_k[i,j,t] = 1
    si le pixel (i,j) a la couleur c_k à l'instant t.
    Coût = m × (coût de la carte binaire).

  Régime 2 — m modéré :
    Codage par transitions.
    La table des transitions (from, to) est petite même si m est
    grand, quand les transitions sont régulières.
    Chaque pixel : couleur initiale + séquence d'indices de transition.

  Régime 3 — m maximal (pire cas, chaque pixel change à chaque frame) :
    Codage R1 direct — on stocke les couleurs sans le temps.
    Avec Huffman, les couleurs fréquentes sont compressées.

Implémentation :
  La fonction palette_transform() remplace les valeurs brutes par
  des indices de palette dans le bloc AVANT le pipeline standard.
  Le coût additionnel est m × color_bits (palette) dans le bitstream.
  Le gain est la réduction de color_bits → index_bits sur TOUTES
  les couleurs encodées.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from math import ceil, log2
from collections import Counter

from .bitstream import BitWriter, BitReader


# ═══════════════════════════════════════════════════════════════
#  ANALYSE DE PALETTE
# ═══════════════════════════════════════════════════════════════

def analyze_block_palette(
    block: np.ndarray,
) -> Dict:
    """
    Analyse les couleurs distinctes dans un bloc.

    Args:
        block: shape (gop_r, bh, bw), dtype uint8

    Returns:
        dict avec :
          'm' : nombre de couleurs distinctes
          'palette' : liste triée des couleurs
          'index_bits' : ⌈log₂(m)⌉
          'color_to_idx' : mapping couleur → indice
          'idx_to_color' : mapping indice → couleur
          'savings_ratio' : gain par symbole (1 - index_bits/8)
    """
    unique_colors = sorted(set(block.flatten().tolist()))
    m = len(unique_colors)

    if m <= 1:
        index_bits = 1  # minimum 1 bit
    else:
        index_bits = max(1, ceil(log2(m)))

    color_to_idx = {c: i for i, c in enumerate(unique_colors)}
    idx_to_color = {i: c for i, c in enumerate(unique_colors)}

    savings = 1.0 - index_bits / 8.0 if m < 256 else 0.0

    return {
        'm': m,
        'palette': unique_colors,
        'index_bits': index_bits,
        'color_to_idx': color_to_idx,
        'idx_to_color': idx_to_color,
        'savings_ratio': savings,
    }


# ═══════════════════════════════════════════════════════════════
#  TRANSFORMATION PALETTE (ENCODEUR)
# ═══════════════════════════════════════════════════════════════

def palette_transform(
    block: np.ndarray,
    pal_info: Dict,
) -> np.ndarray:
    """
    Remplace les valeurs brutes par des indices de palette.

    Args:
        block: shape (gop_r, bh, bw), dtype uint8
        pal_info: résultat de analyze_block_palette()

    Returns:
        indexed_block: shape (gop_r, bh, bw), dtype uint8
                       valeurs dans [0, m-1]
    """
    mapping = pal_info['color_to_idx']
    # Vectorisé via lookup table
    lut = np.zeros(256, dtype=np.uint8)
    for color, idx in mapping.items():
        lut[color] = idx
    return lut[block]


def inverse_palette_transform(
    indexed_block: np.ndarray,
    pal_info: Dict,
) -> np.ndarray:
    """
    Reconstruit les valeurs brutes depuis les indices de palette.
    """
    palette = pal_info['palette']
    lut = np.zeros(256, dtype=np.uint8)
    for idx, color in enumerate(palette):
        lut[idx] = color
    return lut[indexed_block]


# ═══════════════════════════════════════════════════════════════
#  ESTIMATION DL AVEC PALETTE
# ═══════════════════════════════════════════════════════════════

def estimate_dl_palette(
    pal_info: Dict,
    n_pixels: int,
    gop_r: int,
    color_bits: int,
    gop_duration_bits: int,
    n_jumps_per_pixel: List[int],
) -> Tuple[float, bool]:
    """
    Estime la longueur de description avec et sans palette.

    Calcule le coût palette vs le coût fixe pour déterminer
    si la transformation vaut le coup.

    Args:
        pal_info: résultat de analyze_block_palette()
        n_pixels: nombre de pixels dans le bloc
        gop_r: frames dans le GOP
        n_jumps_per_pixel: nombre de sauts par pixel

    Returns:
        (dl_with_palette, use_palette)
    """
    m = pal_info['m']
    index_bits = pal_info['index_bits']

    if m >= 128:
        # Trop de couleurs, la palette n'aide pas
        return float('inf'), False

    # Coût de la palette dans le bitstream
    # 1 bit flag + Elias gamma(m) + m × color_bits
    palette_overhead = 1 + (2 * max(m.bit_length(), 1) - 1) + m * color_bits

    # Total symboles couleur émis (somme des n_jumps)
    total_color_symbols = sum(n_jumps_per_pixel)

    # Coût avec palette : palette + symboles × index_bits
    # (le Huffman opère ensuite sur les index_bits, pas color_bits)
    dl_with = palette_overhead + total_color_symbols * index_bits

    # Coût sans palette : symboles × color_bits
    dl_without = total_color_symbols * color_bits

    use = dl_with < dl_without
    return dl_with if use else dl_without, use


# ═══════════════════════════════════════════════════════════════
#  CODAGE "COULEUR PAR COULEUR"
# ═══════════════════════════════════════════════════════════════

def estimate_dl_per_color(
    block: np.ndarray,
    pal_info: Dict,
    gop_r: int,
) -> float:
    """
    Estime le DL du codage couleur par couleur.

    Pour chaque couleur c_k dans la palette :
      - Construire le masque binaire M_k[i,j,t] = 1 si pixel=c_k
      - Le masque a n_k positions à 1 sur (bh × bw × r) positions
      - Coût = ⌈log₂(C(bh×bw×r, n_k))⌉ (indexation combinatoire)

    Coût total = m × color_bits (palette) + Σ coût_masque(c_k)
    """
    m = pal_info['m']
    palette = pal_info['palette']
    gop_r_val, bh, bw = block.shape
    total_positions = bh * bw * gop_r_val

    if m > 32:
        return float('inf')

    total = m * 8  # palette

    flat = block.flatten()
    for color in palette:
        n_k = int(np.sum(flat == color))
        # Coût combinatoire : ⌈log₂(C(total_positions, n_k))⌉
        # Approximation : n_k × log₂(total_positions/n_k) + ...
        # Utilisons l'entropie binaire × total_positions
        if n_k == 0 or n_k == total_positions:
            cost_k = 1  # trivial
        else:
            p = n_k / total_positions
            h_bin = -p * log2(p) - (1 - p) * log2(1 - p)
            cost_k = int(h_bin * total_positions) + 1
        total += cost_k

    return total


# ═══════════════════════════════════════════════════════════════
#  SÉRIALISATION PALETTE
# ═══════════════════════════════════════════════════════════════

def write_palette(
    writer: BitWriter,
    pal_info: Dict,
    color_bits: int,
) -> int:
    """
    Écrit la palette dans le bitstream.
    Format : [m : Elias gamma] [c_0 : color_bits] ... [c_{m-1} : color_bits]

    Returns: bits écrits
    """
    m = pal_info['m']
    bits = writer.write_elias_gamma(m)
    for color in pal_info['palette']:
        writer.write_bits(color, color_bits)
        bits += color_bits
    return bits


def read_palette(
    reader: BitReader,
    color_bits: int,
) -> Dict:
    """
    Lit la palette depuis le bitstream et reconstruit pal_info.
    """
    m = reader.read_elias_gamma()
    palette = [reader.read_bits(color_bits) for _ in range(m)]

    index_bits = max(1, ceil(log2(m))) if m > 1 else 1
    color_to_idx = {c: i for i, c in enumerate(palette)}
    idx_to_color = {i: c for i, c in enumerate(palette)}

    return {
        'm': m,
        'palette': palette,
        'index_bits': index_bits,
        'color_to_idx': color_to_idx,
        'idx_to_color': idx_to_color,
    }
