"""
Colorspace — Conversion RGB ↔ YCbCr et sous-échantillonnage chrominance.

Le codec traite 3 plans séparément :
  - Y  (luminance)   : résolution complète nl × nc
  - Cb (chrominance)  : résolution réduite (nl//2) × (nc//2)  en 4:2:0
  - Cr (chrominance)  : résolution réduite (nl//2) × (nc//2)  en 4:2:0

La conversion est entière (BT.601) pour garantir la réversibilité
sur les entiers 0-255 :

  Y  =  ( 66*R + 129*G +  25*B + 128) >> 8  + 16
  Cb = (-38*R -  74*G + 112*B + 128) >> 8  + 128
  Cr = (112*R -  94*G -  18*B + 128) >> 8  + 128

Inverse :
  C  = Y  - 16
  D  = Cb - 128
  E  = Cr - 128
  R  = clip((298*C + 409*E + 128) >> 8)
  G  = clip((298*C - 100*D - 208*E + 128) >> 8)
  B  = clip((298*C + 516*D + 128) >> 8)

Sous-échantillonnage 4:2:0 : moyenne 2×2 sur Cb et Cr.
Suréchantillonnage : nearest-neighbor (rapide) ou bilinéaire.
"""

import numpy as np
from typing import Tuple

# ── Modes de sous-échantillonnage ───────────────────────────────
SUBSAMPLE_444 = 0   # Pas de sous-échantillonnage (lossless chroma)
SUBSAMPLE_420 = 1   # 4:2:0 — divise chroma par 2 dans chaque dimension


def rgb_to_ycbcr(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convertit RGB en YCbCr (BT.601, entier).

    Args:
        rgb: shape (..., 3), dtype uint8, canaux R, G, B

    Returns:
        (Y, Cb, Cr) chacun de shape (...), dtype uint8
    """
    R = rgb[..., 0].astype(np.int32)
    G = rgb[..., 1].astype(np.int32)
    B = rgb[..., 2].astype(np.int32)

    Y  = (( 66 * R + 129 * G +  25 * B + 128) >> 8) +  16
    Cb = ((-38 * R -  74 * G + 112 * B + 128) >> 8) + 128
    Cr = ((112 * R -  94 * G -  18 * B + 128) >> 8) + 128

    Y  = np.clip(Y, 0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)

    return Y, Cb, Cr


def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray,
                 Cr: np.ndarray) -> np.ndarray:
    """
    Convertit YCbCr en RGB (BT.601, entier).

    Args:
        Y, Cb, Cr: shape (...), dtype uint8

    Returns:
        rgb: shape (..., 3), dtype uint8
    """
    C = Y.astype(np.int32) - 16
    D = Cb.astype(np.int32) - 128
    E = Cr.astype(np.int32) - 128

    R = (298 * C           + 409 * E + 128) >> 8
    G = (298 * C - 100 * D - 208 * E + 128) >> 8
    B = (298 * C + 516 * D           + 128) >> 8

    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    rgb = np.stack([R, G, B], axis=-1)
    return rgb


def subsample_420(plane: np.ndarray) -> np.ndarray:
    """
    Sous-échantillonne un plan 2D ou 3D par facteur 2 dans chaque
    dimension spatiale (moyenne 2×2).

    Args:
        plane: shape (nl, nc) ou (r, nl, nc), dtype uint8

    Returns:
        shape (nl//2, nc//2) ou (r, nl//2, nc//2), dtype uint8
    """
    if plane.ndim == 2:
        nl, nc = plane.shape
        # Padding si dimensions impaires
        pad_nl = nl % 2
        pad_nc = nc % 2
        if pad_nl or pad_nc:
            plane = np.pad(plane, ((0, pad_nl), (0, pad_nc)), mode='edge')
        p = plane.astype(np.int32)
        sub = (p[0::2, 0::2] + p[1::2, 0::2] +
               p[0::2, 1::2] + p[1::2, 1::2] + 2) >> 2
        return sub.astype(np.uint8)

    elif plane.ndim == 3:
        r, nl, nc = plane.shape
        pad_nl = nl % 2
        pad_nc = nc % 2
        if pad_nl or pad_nc:
            plane = np.pad(plane,
                           ((0, 0), (0, pad_nl), (0, pad_nc)),
                           mode='edge')
        p = plane.astype(np.int32)
        sub = (p[:, 0::2, 0::2] + p[:, 1::2, 0::2] +
               p[:, 0::2, 1::2] + p[:, 1::2, 1::2] + 2) >> 2
        return sub.astype(np.uint8)

    else:
        raise ValueError(f"Dimensions non supportées : {plane.ndim}")


def upsample_420(sub: np.ndarray, target_nl: int,
                 target_nc: int) -> np.ndarray:
    """
    Suréchantillonne un plan chrominance (nearest neighbor).

    Args:
        sub: shape (nl_sub, nc_sub) ou (r, nl_sub, nc_sub)
        target_nl, target_nc: dimensions cibles

    Returns:
        shape (target_nl, target_nc) ou (r, target_nl, target_nc)
    """
    if sub.ndim == 2:
        up = np.repeat(np.repeat(sub, 2, axis=0), 2, axis=1)
        return up[:target_nl, :target_nc]
    elif sub.ndim == 3:
        up = np.repeat(np.repeat(sub, 2, axis=1), 2, axis=2)
        return up[:, :target_nl, :target_nc]
    else:
        raise ValueError(f"Dimensions non supportées : {sub.ndim}")


def split_video_planes(
    M_rgb: np.ndarray,
    subsampling: int = SUBSAMPLE_420
) -> dict:
    """
    Découpe une vidéo RGB en 3 plans Y, Cb, Cr.

    Args:
        M_rgb: shape (r, nl, nc, 3), dtype uint8
        subsampling: SUBSAMPLE_444 ou SUBSAMPLE_420

    Returns:
        dict avec 'Y', 'Cb', 'Cr', 'nl', 'nc', 'r',
             'nl_c', 'nc_c' (dimensions chrominance),
             'subsampling'
    """
    r, nl, nc, _ = M_rgb.shape

    # Conversion frame par frame pour gérer la mémoire
    Y_all = np.zeros((r, nl, nc), dtype=np.uint8)
    Cb_all = np.zeros((r, nl, nc), dtype=np.uint8)
    Cr_all = np.zeros((r, nl, nc), dtype=np.uint8)

    for t in range(r):
        Y_all[t], Cb_all[t], Cr_all[t] = rgb_to_ycbcr(M_rgb[t])

    if subsampling == SUBSAMPLE_420:
        Cb_sub = subsample_420(Cb_all)
        Cr_sub = subsample_420(Cr_all)
        nl_c = Cb_sub.shape[1]
        nc_c = Cb_sub.shape[2]
    else:
        Cb_sub = Cb_all
        Cr_sub = Cr_all
        nl_c = nl
        nc_c = nc

    return {
        'Y': Y_all,
        'Cb': Cb_sub,
        'Cr': Cr_sub,
        'nl': nl,
        'nc': nc,
        'r': r,
        'nl_c': nl_c,
        'nc_c': nc_c,
        'subsampling': subsampling,
    }


def merge_video_planes(
    Y: np.ndarray,
    Cb: np.ndarray,
    Cr: np.ndarray,
    nl: int,
    nc: int,
    subsampling: int = SUBSAMPLE_420
) -> np.ndarray:
    """
    Reconstruit une vidéo RGB depuis les 3 plans.

    Args:
        Y:  shape (r, nl, nc)
        Cb: shape (r, nl_c, nc_c)
        Cr: shape (r, nl_c, nc_c)
        nl, nc: dimensions luma (target)
        subsampling: mode utilisé

    Returns:
        M_rgb: shape (r, nl, nc, 3), dtype uint8
    """
    r = Y.shape[0]

    if subsampling == SUBSAMPLE_420:
        Cb_full = upsample_420(Cb, nl, nc)
        Cr_full = upsample_420(Cr, nl, nc)
    else:
        Cb_full = Cb
        Cr_full = Cr

    M_rgb = np.zeros((r, nl, nc, 3), dtype=np.uint8)
    for t in range(r):
        M_rgb[t] = ycbcr_to_rgb(Y[t], Cb_full[t], Cr_full[t])

    return M_rgb


def generate_synthetic_color_video(
    nl: int = 8,
    nc: int = 8,
    r: int = 16,
    n_colors: int = 4,
    change_prob: float = 0.15,
    seed: int = 42
) -> np.ndarray:
    """
    Génère une vidéo synthétique RGB pour les tests.

    Les couleurs sont choisies dans une palette aléatoire fixe
    pour obtenir des patterns réalistes avec des sauts nets.

    Returns:
        M_rgb: shape (r, nl, nc, 3), dtype uint8
    """
    rng = np.random.RandomState(seed)

    # Palette de n_colors couleurs aléatoires
    palette = rng.randint(0, 256, (n_colors, 3), dtype=np.uint8)

    # Indices de couleur par pixel
    idx = np.zeros((r, nl, nc), dtype=np.int32)
    idx[0] = rng.randint(0, n_colors, (nl, nc))

    for t in range(1, r):
        change_mask = rng.random((nl, nc)) < change_prob
        new_idx = rng.randint(0, n_colors, (nl, nc))
        idx[t] = np.where(change_mask, new_idx, idx[t - 1])

    # Convertir en RGB
    M_rgb = palette[idx]  # broadcasting magique
    return M_rgb.astype(np.uint8)
