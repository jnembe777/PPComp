"""
Intensity — Codage par fonctions d'intensité d'entrée dans chaque état.

Reformulation fondamentale (Nembé) :

Au lieu de coder chaque pixel individuellement, on code la STRUCTURE
du processus ponctuel au niveau du bloc :

1. PALETTE DU BLOC : les m couleurs distinctes {c_0, ..., c_{m-1}}
   Coût : m × color_bits

2. FONCTIONS D'INTENSITÉ : pour chaque état c_k (k=0..m-1),
   la fonction d'intensité λ_k(t) encode les instants où un pixel
   ENTRE dans l'état c_k. C'est un vecteur booléen de r bits :
     λ_k[t] = 1  si au moins un pixel du bloc entre dans l'état c_k
               à l'instant t

   On stocke m vecteurs de r bits = m×r bits au total.
   C'est la CARTE D'ACTIVITÉ de chaque couleur.

   Subtilité : on utilise m fonctions (entrée dans chaque état)
   au lieu de m² fonctions (transitions entre paires d'états).
   Gain : m vs m² quand m est modéré.

3. TRAJECTOIRES PIXELIQUES : pour chaque pixel (i,j), sa trajectoire
   est une séquence d'indices dans la palette : (k_0, k_1, ..., k_{n-1})
   aux dates de saut. Mais comme les dates de saut sont DÉJÀ encodées
   dans les fonctions d'intensité, le pixel n'a besoin de stocker que :
     - Le sous-ensemble des états qu'il visite (masque sur m bits)
     - L'ordre dans lequel il les visite

   Si le pixel visite p états parmi les m possibles, le coût est :
     ⌈log₂(C(m, p))⌉ + ⌈log₂(p!)⌉ bits  (ou moins avec des tricks)

   En pratique, beaucoup de pixels partagent les MÊMES sous-ensembles
   d'états → on peut factoriser.

4. FENÊTRES ADAPTATIVES : au lieu d'un GOP fixe de taille r,
   on segmente le temps en fenêtres de taille variable r_k telles que :
     - Le nombre de couleurs distinctes (NCD) dans la fenêtre est minimal
     - Le nombre moyen de sauts N s'éloigne de r/2
     - Critère MDL : on cherche la segmentation qui minimise
       la longueur de description totale

   Les fenêtres sont encodées par leurs bornes (suite croissante).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from math import ceil, log2, factorial, comb
from collections import Counter

from .bitstream import BitWriter, BitReader
from .combinatorics import (
    bool_vector_to_index, index_to_bool_vector,
    bits_for_binomial, compute_jump_vector,
)


# ═══════════════════════════════════════════════════════════════
#  1. ANALYSE DE LA STRUCTURE DU BLOC
# ═══════════════════════════════════════════════════════════════

def analyze_block_structure(block: np.ndarray) -> Dict:
    """
    Analyse complète de la structure du processus ponctuel dans un bloc.

    Args:
        block: shape (r, bh, bw), dtype uint8

    Returns:
        dict avec :
          'm' : nombre de couleurs distinctes
          'palette' : liste triée des m couleurs
          'color_to_idx' : mapping couleur → indice
          'intensity' : ndarray (m, r) booléen — λ_k[t]
          'pixel_trajectories' : liste de (bh×bw) tuples d'indices
          'n_jumps_per_pixel' : nombre de sauts par pixel
          'ncd' : nombre de couleurs distinctes (= m)
          'mean_jumps' : nombre moyen de sauts
          'efficiency_score' : 1 - |2N/r - 1| (0=optimal, 1=pire)
    """
    r, bh, bw = block.shape
    n_pixels = bh * bw

    # Palette
    unique_colors = sorted(set(block.flatten().tolist()))
    m = len(unique_colors)
    color_to_idx = {c: i for i, c in enumerate(unique_colors)}
    idx_bits = max(1, ceil(log2(m))) if m > 1 else 1

    # Convertir le bloc en indices
    lut = np.zeros(256, dtype=np.uint8)
    for c, idx in color_to_idx.items():
        lut[c] = idx
    block_idx = lut[block]  # (r, bh, bw) en indices 0..m-1

    # Fonctions d'intensité : λ_k[t] = 1 si un pixel entre dans l'état k à t
    intensity = np.zeros((m, r), dtype=np.uint8)

    # Trajectoires par pixel
    trajectories = []
    n_jumps_list = []
    state_masks = []  # masque m bits des états visités par chaque pixel

    for i in range(bh):
        for j in range(bw):
            seq = block_idx[:, i, j]  # séquence d'indices

            # Premier instant : entrée dans l'état initial
            intensity[seq[0], 0] = 1

            # Sauts
            traj = [int(seq[0])]
            n_jumps = 1
            for t in range(1, r):
                if seq[t] != seq[t - 1]:
                    intensity[seq[t], t] = 1
                    traj.append(int(seq[t]))
                    n_jumps += 1

            trajectories.append(tuple(traj))
            n_jumps_list.append(n_jumps)

            # Masque des états visités
            mask = 0
            for k in set(traj):
                mask |= (1 << k)
            state_masks.append(mask)

    mean_jumps = np.mean(n_jumps_list)
    # Efficacité : distance à r/2 normalisée
    eff = abs(2 * mean_jumps / r - 1) if r > 0 else 0

    return {
        'm': m,
        'palette': unique_colors,
        'color_to_idx': color_to_idx,
        'idx_bits': idx_bits,
        'intensity': intensity,
        'trajectories': trajectories,
        'n_jumps_list': n_jumps_list,
        'state_masks': state_masks,
        'ncd': m,
        'mean_jumps': mean_jumps,
        'efficiency_score': eff,
        'r': r,
        'n_pixels': n_pixels,
    }


# ═══════════════════════════════════════════════════════════════
#  2. ESTIMATION DL PAR FONCTIONS D'INTENSITÉ
# ═══════════════════════════════════════════════════════════════

def estimate_dl_intensity(info: Dict, color_bits: int) -> float:
    """
    Estime la longueur de description du codage palette + R4 indices.

    Structure :
      [m : Elias gamma]
      [palette : m × color_bits]
      Par pixel :
        [n_jumps : Elias gamma]
        [jump_index : ⌈log₂(C(r, n))⌉ bits]
        [couleurs : n × idx_bits]
    """
    m = info['m']
    r = info['r']
    n_pixels = info['n_pixels']
    idx_bits = info['idx_bits']

    # Palette
    dl = (2 * max(m.bit_length(), 1) - 1) + m * color_bits

    # Par pixel
    for px_idx in range(n_pixels):
        n = info['n_jumps_list'][px_idx]
        # Elias gamma(n)
        dl += (2 * max(n.bit_length(), 1) - 1)
        # Indice combinatoire C(r, n)
        dl += bits_for_binomial(r, n)
        # Couleurs : n × idx_bits (au lieu de n × color_bits)
        dl += n * idx_bits

    return dl


def estimate_dl_pixel_by_pixel(info: Dict, color_bits: int) -> float:
    """
    Estime le DL du codage R4 pixel par pixel (ancien mode) pour comparaison.
    """
    r = info['r']
    n_pixels = info['n_pixels']
    duration_bits = max(1, ceil(log2(max(r, 2))))

    dl = 0
    for px_idx in range(n_pixels):
        n = info['n_jumps_list'][px_idx]
        # R4 : Elias(n) + ⌈log₂(C(r, n))⌉ + n × color_bits
        dl += (2 * max(n.bit_length(), 1) - 1)
        dl += bits_for_binomial(r, n)
        dl += n * color_bits

    return dl


# ═══════════════════════════════════════════════════════════════
#  3. ENCODAGE PAR INTENSITÉ
# ═══════════════════════════════════════════════════════════════

def encode_block_intensity(
    writer: BitWriter,
    info: Dict,
    color_bits: int,
) -> int:
    """
    Encode un bloc via palette + R4 sur indices réduits.

    Structure :
      [m : Elias gamma]
      [palette : m × color_bits]
      Pour chaque pixel :
        [n_jumps : Elias gamma]
        [jump_index : ⌈log₂(C(r, n))⌉ bits]  (R4 du vecteur booléen)
        [couleurs : n × idx_bits]              (indices dans la palette)

    Le gain vient du remplacement de color_bits (8) par idx_bits (⌈log₂(m)⌉).
    """
    m = info['m']
    r = info['r']
    idx_bits = info['idx_bits']
    bits = 0

    # Palette
    bits += writer.write_elias_gamma(m)
    for c in info['palette']:
        writer.write_bits(c, color_bits)
        bits += color_bits

    # Trajectoires par pixel : R4 avec indices réduits
    for px_idx in range(info['n_pixels']):
        traj = info['trajectories'][px_idx]
        n_jumps = info['n_jumps_list'][px_idx]

        # Nombre de sauts
        bits += writer.write_elias_gamma(n_jumps)

        # Vecteur booléen des sauts via indice combinatoire (R4)
        # Reconstruire le bool vector depuis la trajectoire
        # On a besoin des dates de saut, pas juste la trajectoire
        # Les dates sont dans l'analyse du bloc
        mask = info['state_masks'][px_idx]

        # Encoder l'indice combinatoire du vecteur de sauts
        s_bits = bits_for_binomial(r, n_jumps)
        if s_bits > 0:
            # Recalculer le bool vector pour ce pixel
            # On doit le faire depuis le bloc original
            # Il est stocké implicitement — on passe par les séquences
            pass  # sera calculé ci-dessous

        # Encoder les couleurs comme indices de palette
        for state in traj:
            writer.write_bits(state, idx_bits)
            bits += idx_bits

    return bits


def encode_block_intensity_v2(
    writer: BitWriter,
    block: np.ndarray,
    info: Dict,
    color_bits: int,
) -> int:
    """
    Version corrigée : encode le bloc avec palette + R4 sur indices.

    Accède au bloc original pour reconstruire les vecteurs booléens.
    """
    m = info['m']
    r = info['r']
    idx_bits = info['idx_bits']
    color_to_idx = info['color_to_idx']
    bh = block.shape[1]
    bw = block.shape[2]
    bits = 0

    # Palette
    bits += writer.write_elias_gamma(m)
    for c in info['palette']:
        writer.write_bits(c, color_bits)
        bits += color_bits

    # Par pixel : jump vector (R4) + couleurs aux sauts (idx_bits)
    for i in range(bh):
        for j in range(bw):
            seq = block[:, i, j]

            # Construire le vecteur booléen
            bvec = [0] * r
            bvec[0] = 1
            n_jumps = 1
            for t in range(1, r):
                if seq[t] != seq[t - 1]:
                    bvec[t] = 1
                    n_jumps += 1

            # n_jumps
            bits += writer.write_elias_gamma(n_jumps)

            # Indice combinatoire du vecteur booléen
            s_bits = bits_for_binomial(r, n_jumps)
            if s_bits > 0:
                s_idx = bool_vector_to_index(bvec, r, n_jumps)
                writer.write_bits(s_idx, s_bits)
                bits += s_bits

            # Couleurs aux dates de saut (en indices palette)
            for t in range(r):
                if bvec[t] == 1:
                    cidx = color_to_idx[int(seq[t])]
                    writer.write_bits(cidx, idx_bits)
                    bits += idx_bits

    return bits


def decode_block_intensity(
    reader: BitReader,
    r: int,
    bh: int, bw: int,
    color_bits: int,
) -> np.ndarray:
    """
    Décode un bloc encodé par palette + R4 sur indices réduits.
    """
    # Palette
    m = reader.read_elias_gamma()
    palette = [reader.read_bits(color_bits) for _ in range(m)]
    idx_bits = max(1, ceil(log2(m))) if m > 1 else 1

    block = np.zeros((r, bh, bw), dtype=np.uint8)

    for i in range(bh):
        for j in range(bw):
            n_jumps = reader.read_elias_gamma()

            # Indice combinatoire → vecteur booléen
            s_bits = bits_for_binomial(r, n_jumps)
            if s_bits > 0:
                s_idx = reader.read_bits(s_bits)
                bvec = index_to_bool_vector(s_idx, r, n_jumps)
            else:
                bvec = [1] + [0] * (r - 1)

            # Couleurs aux dates de saut
            colors_at_jumps = []
            for t in range(r):
                if bvec[t] == 1:
                    cidx = reader.read_bits(idx_bits)
                    colors_at_jumps.append(palette[cidx])

            # Reconstruire la séquence
            color_idx = 0
            current_color = 0
            for t in range(r):
                if bvec[t] == 1:
                    current_color = colors_at_jumps[color_idx]
                    color_idx += 1
                block[t, i, j] = current_color

    return block


# ═══════════════════════════════════════════════════════════════
#  4. FENÊTRES TEMPORELLES ADAPTATIVES
# ═══════════════════════════════════════════════════════════════

def find_adaptive_windows(
    plane: np.ndarray,
    block_i0: int, block_i1: int,
    block_j0: int, block_j1: int,
    min_window: int = 4,
    max_window: int = 128,
    color_bits: int = 8,
) -> List[Tuple[int, int, Dict]]:
    """
    Segmente l'axe temporel en fenêtres adaptatives qui minimisent
    la longueur de description totale.

    Critères de segmentation :
      - Minimiser NCD (nombre de couleurs distinctes) par fenêtre
      - Préférer N loin de r/2
      - Critère MDL global

    Algorithme glouton :
      1. Commencer à t=0
      2. Étendre la fenêtre tant que NCD reste faible
      3. Quand NCD dépasse un seuil ou que N approche r/2, couper
      4. Recommencer depuis la nouvelle position

    Args:
        plane: shape (r_total, nl, nc)
        block_i0..j1: coordonnées du bloc
        min_window: taille minimale d'une fenêtre
        max_window: taille maximale

    Returns:
        Liste de (t_start, t_end, analysis_info) pour chaque fenêtre
    """
    r_total = plane.shape[0]
    bh = block_i1 - block_i0
    bw = block_j1 - block_j0
    n_pixels = bh * bw

    windows = []
    t = 0

    while t < r_total:
        best_end = min(t + min_window, r_total)
        best_dl = float('inf')
        best_info = None

        # Essayer des fenêtres de taille croissante
        for end in range(t + min_window, min(t + max_window + 1, r_total + 1)):
            block = plane[t:end, block_i0:block_i1, block_j0:block_j1]
            r_win = end - t

            # Analyse rapide
            unique = set(block.flatten().tolist())
            m = len(unique)

            # Nombre moyen de sauts
            total_jumps = 0
            for i in range(bh):
                for j in range(bw):
                    seq = block[:, i, j]
                    for tt in range(1, r_win):
                        if seq[tt] != seq[tt - 1]:
                            total_jumps += 1
            mean_jumps = (total_jumps / n_pixels) + 1

            # Score d'efficacité
            # Proche de 0 ou r → bon, proche de r/2 → mauvais
            eff = abs(2 * mean_jumps / r_win - 1)

            # DL estimé rapide
            idx_bits = max(1, ceil(log2(m))) if m > 1 else 1
            # Palette + intensité + trajectoires (estimation)
            dl_palette = m * color_bits
            dl_intensity = m * r_win  # m vecteurs de r bits
            dl_traj = n_pixels * mean_jumps * max(1, ceil(log2(max(m, 2))))
            dl_total = dl_palette + dl_intensity + dl_traj

            # Bonus pour m petit et efficacité haute
            dl_adjusted = dl_total / max(eff + 0.1, 0.1)

            if dl_total < best_dl:
                best_dl = dl_total
                best_end = end
                best_info = {
                    'm': m, 'r_win': r_win,
                    'mean_jumps': mean_jumps,
                    'eff': eff,
                    'dl_estimate': dl_total,
                }

            # Arrêter si m explose ou efficacité chute
            if m > 128 and end > t + min_window:
                break

        windows.append((t, best_end, best_info))
        t = best_end

    return windows


def encode_window_boundaries(
    writer: BitWriter,
    windows: List[Tuple[int, int, Dict]],
    r_total: int,
) -> int:
    """
    Encode les bornes des fenêtres adaptatives.
    Format : [nb_windows : Elias gamma] [r_0, r_1, ..., r_{n-1} : Elias gamma]
    """
    n = len(windows)
    bits = writer.write_elias_gamma(n)
    for t_start, t_end, _ in windows:
        bits += writer.write_elias_gamma(t_end - t_start)
    return bits


def decode_window_boundaries(
    reader: BitReader,
) -> List[Tuple[int, int]]:
    """Décode les bornes des fenêtres."""
    n = reader.read_elias_gamma()
    windows = []
    t = 0
    for _ in range(n):
        r_win = reader.read_elias_gamma()
        windows.append((t, t + r_win))
        t += r_win
    return windows


# ═══════════════════════════════════════════════════════════════
#  5. MESURE DU GAIN : INTENSITÉ vs PIXEL-PAR-PIXEL
# ═══════════════════════════════════════════════════════════════

def compare_encodings(block: np.ndarray, color_bits: int = 8) -> Dict:
    """
    Compare le DL du codage par intensité vs pixel par pixel.

    Returns:
        dict avec dl_intensity, dl_pixel, gain_ratio, analysis
    """
    info = analyze_block_structure(block)

    dl_intensity = estimate_dl_intensity(info, color_bits)
    dl_pixel = estimate_dl_pixel_by_pixel(info, color_bits)

    gain = (1 - dl_intensity / dl_pixel) * 100 if dl_pixel > 0 else 0

    return {
        'dl_intensity': dl_intensity,
        'dl_pixel': dl_pixel,
        'gain_pct': gain,
        'm': info['m'],
        'mean_jumps': info['mean_jumps'],
        'efficiency': info['efficiency_score'],
        'r': info['r'],
    }
