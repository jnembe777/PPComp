"""
Prediction — Prédiction spatiale intra-frame.

Pour chaque frame du GOP, chaque pixel est prédit depuis ses voisins
spatiaux déjà décodés. Le résidu (valeur - prédiction) mod 256 est
stocké à la place de la valeur brute. Le résidu est dans [0, 255]
et sa distribution est concentrée autour de 0 (mod 256), ce qui
améliore fortement le Huffman et réduit les sauts entre frames.

4 modes de prédiction par bloc :
  Mode 0 : Bypass        — pas de prédiction (valeur brute)
  Mode 1 : Left          — pixel[i][j-1]  (bord gauche → 0)
  Mode 2 : Top           — pixel[i-1][j]  (bord haut → 0)
  Mode 3 : DC            — (left + top) // 2

Le mode est choisi par bloc en estimant l'entropie des résidus.
Le mode coûte 2 bits par bloc dans le bitstream.

Résidu modulo 256 :
  encode : residual = (value - prediction) & 0xFF
  decode : value    = (residual + prediction) & 0xFF
  → Toujours dans [0, 255], compatible avec tout le pipeline 8 bits.
"""

import numpy as np
from typing import Tuple
from math import log2
from collections import Counter


# ── Modes ──────────────────────────────────────────────────────

PRED_BYPASS = 0
PRED_LEFT   = 1
PRED_TOP    = 2
PRED_DC     = 3

PRED_NAMES = {0: "Bypass", 1: "Left", 2: "Top", 3: "DC"}
NUM_PRED_MODES = 4


# ═══════════════════════════════════════════════════════════════
#  APPLICATION DE LA PRÉDICTION (ENCODEUR)
# ═══════════════════════════════════════════════════════════════

def apply_prediction_frame(frame: np.ndarray, mode: int) -> np.ndarray:
    """
    Applique la prédiction à une frame et retourne les résidus.

    Args:
        frame: shape (nl, nc), dtype uint8
        mode: PRED_BYPASS, PRED_LEFT, PRED_TOP, PRED_DC

    Returns:
        residuals: shape (nl, nc), dtype uint8
                   residual = (value - prediction) & 0xFF
    """
    if mode == PRED_BYPASS:
        return frame.copy()

    nl, nc = frame.shape
    pred = np.zeros((nl, nc), dtype=np.int32)

    if mode == PRED_LEFT:
        # pred[i][j] = frame[i][j-1], bord gauche = 0
        pred[:, 1:] = frame[:, :-1].astype(np.int32)

    elif mode == PRED_TOP:
        # pred[i][j] = frame[i-1][j], bord haut = 0
        pred[1:, :] = frame[:-1, :].astype(np.int32)

    elif mode == PRED_DC:
        # pred[i][j] = (left + top) // 2
        left = np.zeros((nl, nc), dtype=np.int32)
        left[:, 1:] = frame[:, :-1].astype(np.int32)
        top = np.zeros((nl, nc), dtype=np.int32)
        top[1:, :] = frame[:-1, :].astype(np.int32)
        pred = (left + top) >> 1

    residuals = (frame.astype(np.int32) - pred) & 0xFF
    return residuals.astype(np.uint8)


def invert_prediction_frame(residuals: np.ndarray, mode: int) -> np.ndarray:
    """
    Inverse la prédiction : reconstruit la frame depuis les résidus.

    Le décodeur reconstruit pixel par pixel (raster scan) car la
    prédiction dépend des pixels déjà reconstruits.

    Args:
        residuals: shape (nl, nc), dtype uint8
        mode: mode de prédiction

    Returns:
        frame: shape (nl, nc), dtype uint8
    """
    if mode == PRED_BYPASS:
        return residuals.copy()

    nl, nc = residuals.shape
    frame = np.zeros((nl, nc), dtype=np.uint8)

    for i in range(nl):
        for j in range(nc):
            if mode == PRED_LEFT:
                pred = int(frame[i, j - 1]) if j > 0 else 0
            elif mode == PRED_TOP:
                pred = int(frame[i - 1, j]) if i > 0 else 0
            elif mode == PRED_DC:
                left = int(frame[i, j - 1]) if j > 0 else 0
                top = int(frame[i - 1, j]) if i > 0 else 0
                pred = (left + top) >> 1
            else:
                pred = 0

            frame[i, j] = (int(residuals[i, j]) + pred) & 0xFF

    return frame


# ═══════════════════════════════════════════════════════════════
#  APPLICATION SUR UN GOP COMPLET
# ═══════════════════════════════════════════════════════════════

def apply_prediction_gop(
    gop: np.ndarray,
    mode: int,
) -> np.ndarray:
    """
    Applique la prédiction frame par frame sur un GOP.

    Args:
        gop: shape (gop_r, nl, nc), dtype uint8
        mode: mode de prédiction

    Returns:
        residuals_gop: shape (gop_r, nl, nc), dtype uint8
    """
    gop_r = gop.shape[0]
    residuals = np.zeros_like(gop)

    for t in range(gop_r):
        residuals[t] = apply_prediction_frame(gop[t], mode)

    return residuals


def invert_prediction_gop(
    residuals_gop: np.ndarray,
    mode: int,
) -> np.ndarray:
    """
    Inverse la prédiction sur un GOP complet.
    """
    gop_r = residuals_gop.shape[0]
    gop = np.zeros_like(residuals_gop)

    for t in range(gop_r):
        gop[t] = invert_prediction_frame(residuals_gop[t], mode)

    return gop


# ═══════════════════════════════════════════════════════════════
#  CHOIX DU MODE OPTIMAL
# ═══════════════════════════════════════════════════════════════

def _estimate_entropy(data: np.ndarray) -> float:
    """Estime l'entropie de Shannon en bits/symbole."""
    flat = data.flatten()
    n = len(flat)
    if n == 0:
        return 8.0
    counts = Counter(flat.tolist())
    entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            entropy -= p * log2(p)
    return entropy


def choose_best_mode(
    gop: np.ndarray,
    block_i0: int, block_i1: int,
    block_j0: int, block_j1: int,
) -> Tuple[int, float]:
    """
    Choisit le mode de prédiction optimal pour un bloc.

    Extrait la zone [i0:i1, j0:j1] de chaque frame, applique chaque
    mode de prédiction, mesure l'entropie des résidus, et retourne
    le mode avec l'entropie la plus basse.

    Args:
        gop: shape (gop_r, nl, nc)
        block_i0..j1: coordonnées du bloc

    Returns:
        (best_mode, best_entropy)
    """
    gop_r = gop.shape[0]
    block = gop[:, block_i0:block_i1, block_j0:block_j1].copy()

    best_mode = PRED_BYPASS
    best_entropy = 8.0  # pire cas

    for mode in range(NUM_PRED_MODES):
        # Appliquer la prédiction sur le sous-bloc
        # Note : pour Left/Top, les prédictions aux bords du bloc
        # utilisent 0 (pas les pixels voisins hors bloc).
        # C'est une simplification — en H.264, on utiliserait les
        # vrais voisins. Ici ça suffit pour l'estimation MDL.
        residuals = np.zeros_like(block)
        for t in range(gop_r):
            residuals[t] = apply_prediction_frame(block[t], mode)

        ent = _estimate_entropy(residuals)
        if ent < best_entropy:
            best_entropy = ent
            best_mode = mode

    return best_mode, best_entropy


def choose_best_mode_plane(
    gop: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """
    Choisit le mode optimal pour chaque bloc du plan.

    Returns:
        mode_map: shape (nb_blocks_i, nb_blocks_j), dtype int
    """
    from math import ceil
    gop_r, nl, nc = gop.shape
    nb_bi = ceil(nl / block_size)
    nb_bj = ceil(nc / block_size)

    mode_map = np.zeros((nb_bi, nb_bj), dtype=np.int32)

    for bi in range(nb_bi):
        for bj in range(nb_bj):
            i0 = bi * block_size
            i1 = min(i0 + block_size, nl)
            j0 = bj * block_size
            j1 = min(j0 + block_size, nc)

            mode, _ = choose_best_mode(gop, i0, i1, j0, j1)
            mode_map[bi, bj] = mode

    return mode_map


def apply_prediction_by_blocks(
    gop: np.ndarray,
    mode_map: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """
    Applique la prédiction bloc par bloc avec le mode choisi.

    Returns:
        residuals: shape (gop_r, nl, nc)
    """
    from math import ceil
    gop_r, nl, nc = gop.shape
    nb_bi = ceil(nl / block_size)
    nb_bj = ceil(nc / block_size)

    residuals = np.zeros_like(gop)

    for bi in range(nb_bi):
        for bj in range(nb_bj):
            i0 = bi * block_size
            i1 = min(i0 + block_size, nl)
            j0 = bj * block_size
            j1 = min(j0 + block_size, nc)

            mode = int(mode_map[bi, bj])
            block = gop[:, i0:i1, j0:j1]

            for t in range(gop_r):
                residuals[t, i0:i1, j0:j1] = apply_prediction_frame(
                    block[t], mode
                )

    return residuals


def invert_prediction_by_blocks(
    residuals: np.ndarray,
    mode_map: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """
    Inverse la prédiction bloc par bloc.
    """
    from math import ceil
    gop_r, nl, nc = residuals.shape
    nb_bi = ceil(nl / block_size)
    nb_bj = ceil(nc / block_size)

    gop = np.zeros_like(residuals)

    for bi in range(nb_bi):
        for bj in range(nb_bj):
            i0 = bi * block_size
            i1 = min(i0 + block_size, nl)
            j0 = bj * block_size
            j1 = min(j0 + block_size, nc)

            mode = int(mode_map[bi, bj])
            block_res = residuals[:, i0:i1, j0:j1]

            for t in range(gop_r):
                gop[t, i0:i1, j0:j1] = invert_prediction_frame(
                    block_res[t], mode
                )

    return gop
