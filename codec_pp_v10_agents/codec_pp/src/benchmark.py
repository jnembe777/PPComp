"""
Benchmark — Framework de benchmark pour le codec PPV.

Génère des vidéos synthétiques réalistes, encode/décode avec le codec PPV
dans plusieurs configurations, mesure les métriques de qualité et performance,
et produit un rapport comparatif.

Métriques :
  - Taux de compression (ratio, bits/pixel/frame)
  - PSNR / SSIM (qualité pour le mode couleur, lossy via 4:2:0)
  - Temps d'encodage / décodage
  - Distribution des types de processus et représentations

Vidéos synthétiques réalistes :
  - "static_bg"   : fond statique + objet mobile (surveillance)
  - "gradient"    : dégradés lents (ciel, eau)
  - "animation"   : zones plates avec changements nets (cartoon)
  - "noise"       : bruit modéré (capteur bas de gamme)
  - "periodic"    : pattern oscillant (machines, indicateurs)
"""

import numpy as np
import time
import json
from typing import Dict, List, Any, Optional
from math import log10

from .encoder import PPVEncoder
from .decoder import PPVDecoder
from .video_io import generate_synthetic_video
from .colorspace import (
    generate_synthetic_color_video,
    split_video_planes, merge_video_planes,
    SUBSAMPLE_420,
)


# ═══════════════════════════════════════════════════════════════
#  GÉNÉRATEURS DE VIDÉOS RÉALISTES
# ═══════════════════════════════════════════════════════════════

def gen_static_bg(nl=64, nc=64, r=64, seed=42):
    """Fond statique avec un petit rectangle mobile."""
    rng = np.random.RandomState(seed)
    bg = rng.randint(40, 80, (nl, nc), dtype=np.uint8)
    M = np.tile(bg, (r, 1, 1)).copy()

    # Rectangle mobile (16×16)
    obj_h, obj_w = min(16, nl // 2), min(16, nc // 2)
    obj_color = 200
    for t in range(r):
        y = int((nl - obj_h) * (0.5 + 0.4 * np.sin(2 * np.pi * t / r)))
        x = int((nc - obj_w) * t / r)
        M[t, y:y + obj_h, x:x + obj_w] = obj_color

    return M, "static_bg", "Fond statique + objet mobile"


def gen_gradient(nl=64, nc=64, r=64, seed=42):
    """Dégradé lent qui évolue dans le temps (ciel/eau)."""
    t_arr = np.arange(r).reshape(r, 1, 1) / r
    i_arr = np.arange(nl).reshape(1, nl, 1) / nl
    val = 50 + 150 * i_arr + 20 * np.sin(2 * np.pi * t_arr)
    M = np.clip(val, 0, 255).astype(np.uint8)
    M = np.broadcast_to(M, (r, nl, nc)).copy()
    return M, "gradient", "Dégradé vertical lent"


def gen_animation(nl=64, nc=64, r=64, seed=42):
    """Zones plates avec changements nets (cartoon/animation)."""
    rng = np.random.RandomState(seed)
    palette = [30, 80, 130, 180, 220]
    M = np.zeros((r, nl, nc), dtype=np.uint8)

    # Découper en 4×4 zones
    zone_h = nl // 4
    zone_w = nc // 4
    for zi in range(4):
        for zj in range(4):
            color_idx = rng.randint(0, len(palette))
            change_frames = sorted(rng.choice(r, size=3, replace=False))
            for t in range(r):
                if t in change_frames:
                    color_idx = (color_idx + 1) % len(palette)
                y0, y1 = zi * zone_h, (zi + 1) * zone_h
                x0, x1 = zj * zone_w, (zj + 1) * zone_w
                M[t, y0:y1, x0:x1] = palette[color_idx]

    return M, "animation", "Zones plates, sauts nets (cartoon)"


def gen_noise(nl=64, nc=64, r=64, seed=42):
    """Signal bruité modéré — pire cas pour le codec."""
    rng = np.random.RandomState(seed)
    base = rng.randint(60, 160, (nl, nc), dtype=np.uint8)
    M = np.zeros((r, nl, nc), dtype=np.uint8)
    for t in range(r):
        noise = rng.randint(-15, 16, (nl, nc))
        M[t] = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
    return M, "noise", "Signal + bruit modéré"


def gen_periodic(nl=64, nc=64, r=64, seed=42):
    """Pattern oscillant périodique (machines, oscilloscope)."""
    t_arr = np.arange(r).reshape(r, 1, 1)
    i_arr = np.arange(nl).reshape(1, nl, 1)
    j_arr = np.arange(nc).reshape(1, 1, nc)
    val = 128 + 60 * np.sin(2 * np.pi * (t_arr / 8 + i_arr / 16 + j_arr / 16))
    M = np.clip(val, 0, 255).astype(np.uint8)
    return M, "periodic", "Oscillation périodique"


def gen_color_static_bg(nl=64, nc=64, r=64, seed=42):
    """Fond statique couleur + objet mobile coloré."""
    rng = np.random.RandomState(seed)
    bg = np.zeros((nl, nc, 3), dtype=np.uint8)
    bg[:, :, 0] = rng.randint(20, 60, (nl, nc))
    bg[:, :, 1] = rng.randint(80, 120, (nl, nc))
    bg[:, :, 2] = rng.randint(40, 80, (nl, nc))

    M = np.tile(bg, (r, 1, 1, 1)).copy()

    obj_h, obj_w = min(12, nl // 2), min(12, nc // 2)
    for t in range(r):
        y = int((nl - obj_h) * (0.5 + 0.3 * np.sin(2 * np.pi * t / r)))
        x = int((nc - obj_w) * t / r)
        M[t, y:y + obj_h, x:x + obj_w] = [220, 50, 50]

    return M, "color_static_bg", "Fond statique couleur + objet"


ALL_GENERATORS_GRAY = [gen_static_bg, gen_gradient, gen_animation,
                       gen_noise, gen_periodic]

ALL_GENERATORS_COLOR = [gen_color_static_bg]


# ═══════════════════════════════════════════════════════════════
#  MÉTRIQUES
# ═══════════════════════════════════════════════════════════════

def compute_psnr(original: np.ndarray, decoded: np.ndarray) -> float:
    """PSNR entre deux vidéos (dB)."""
    mse = np.mean((original.astype(float) - decoded.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * log10(255.0 ** 2 / mse)


def compute_ssim_frame(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM simplifié pour une frame (Wang et al. 2004)."""
    a = a.astype(float)
    b = b.astype(float)
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a2 = np.var(a)
    sigma_b2 = np.var(b)
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return num / den


def compute_ssim(original: np.ndarray, decoded: np.ndarray) -> float:
    """SSIM moyen sur toutes les frames."""
    if original.ndim == 4:
        # Couleur : calculer sur la luminance (canal moyen)
        orig_y = np.mean(original.astype(float), axis=-1)
        dec_y = np.mean(decoded.astype(float), axis=-1)
    else:
        orig_y = original.astype(float)
        dec_y = decoded.astype(float)

    r = orig_y.shape[0]
    ssim_vals = [compute_ssim_frame(orig_y[t], dec_y[t]) for t in range(r)]
    return np.mean(ssim_vals)


# ═══════════════════════════════════════════════════════════════
#  BENCHMARK ENGINE
# ═══════════════════════════════════════════════════════════════

def benchmark_single(
    M: np.ndarray,
    name: str,
    gop_size: int = 32,
    block_size: int = 8,
    use_huffman: bool = True,
    color_bits: int = 8,
) -> Dict[str, Any]:
    """Benchmark une vidéo avec une configuration donnée."""
    import tempfile
    ppv_path = tempfile.mktemp(suffix='.ppv')

    is_color = (M.ndim == 4)
    grayscale = not is_color

    # Encode
    enc = PPVEncoder(gop_size=gop_size, block_size=block_size,
                     use_huffman=use_huffman, verbose=False)
    stats = enc.encode(M=M, output_path=ppv_path, color_bits=color_bits)

    # Decode
    dec = PPVDecoder(verbose=False)
    M_dec, meta = dec.decode(ppv_path)

    # Métriques
    if grayscale:
        lossless = np.array_equal(M, M_dec)
        psnr = compute_psnr(M, M_dec)
        ssim = compute_ssim(M, M_dec)
    else:
        # Pour la couleur, comparer vs la référence YCbCr (pas RGB direct)
        planes_orig = split_video_planes(M, SUBSAMPLE_420)
        M_ref = merge_video_planes(
            planes_orig['Y'], planes_orig['Cb'], planes_orig['Cr'],
            planes_orig['nl'], planes_orig['nc'], SUBSAMPLE_420
        )
        lossless = np.array_equal(M_ref, M_dec)
        psnr = compute_psnr(M, M_dec)  # vs original RGB
        ssim = compute_ssim(M, M_dec)

    shape = M.shape
    if is_color:
        r, nl, nc, _ = shape
        raw_bits = r * nl * nc * 3 * color_bits
    else:
        r, nl, nc = shape
        raw_bits = r * nl * nc * color_bits

    bpp = stats['compressed_size_bytes'] * 8 / (r * nl * nc)

    import os
    os.remove(ppv_path)

    return {
        'name': name,
        'resolution': f"{nl}×{nc}",
        'frames': r,
        'is_color': is_color,
        'gop_size': gop_size,
        'block_size': block_size,
        'use_huffman': use_huffman,
        'raw_bytes': raw_bits // 8,
        'compressed_bytes': stats['compressed_size_bytes'],
        'ratio': stats['compression_ratio'],
        'savings_pct': stats['savings_percent'],
        'bpp': bpp,
        'psnr': psnr,
        'ssim': ssim,
        'lossless': lossless,
        'encode_time_ms': stats['encode_time_s'] * 1000,
        'decode_time_ms': meta['decode_time_s'] * 1000,
        'rep_counts': stats['rep_counts'],
        'mono_count': stats['mono_count'],
        'proc_spatial': stats.get('proc_spatial_count', 0),
        'proc_markov': stats.get('proc_markov_count', 0),
    }


def run_full_benchmark(
    nl: int = 64,
    nc: int = 64,
    r: int = 64,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Exécute le benchmark complet sur toutes les vidéos synthétiques.

    Returns:
        Liste de résultats
    """
    results = []

    if verbose:
        print(f"\n{'═' * 80}")
        print(f"  BENCHMARK CODEC PPV — {nl}×{nc}, {r} frames")
        print(f"{'═' * 80}")

    # ── Vidéos grises ──────────────────────────────────────────
    for gen_fn in ALL_GENERATORS_GRAY:
        M, name, desc = gen_fn(nl=nl, nc=nc, r=r)

        if verbose:
            print(f"\n  ── {desc} ({name}) ──")

        # Config 1 : sans Huffman
        res_no_h = benchmark_single(M, f"{name}_no_huff",
                                     use_huffman=False)
        # Config 2 : avec Huffman
        res_huff = benchmark_single(M, f"{name}_huffman",
                                     use_huffman=True)
        # Config 3 : blocs 4×4
        res_b4 = benchmark_single(M, f"{name}_block4",
                                   block_size=4, use_huffman=True)

        results.extend([res_no_h, res_huff, res_b4])

        if verbose:
            print(f"    {'Config':<20s}  {'Ratio':>6s}  {'bpp':>6s}  "
                  f"{'PSNR':>6s}  {'SSIM':>5s}  {'Enc ms':>7s}  "
                  f"{'Lossless':>8s}")
            for res in [res_no_h, res_huff, res_b4]:
                tag = res['name'].split('_')[-1]
                ll = "✓" if res['lossless'] else "✗"
                ps = f"{res['psnr']:.1f}" if res['psnr'] != float('inf') else "∞"
                print(f"    {tag:<20s}  {res['ratio']:>5.1f}x  "
                      f"{res['bpp']:>5.2f}  {ps:>6s}  "
                      f"{res['ssim']:.3f}  {res['encode_time_ms']:>6.1f}  "
                      f"{'   ' + ll:>8s}")

    # ── Vidéo couleur ──────────────────────────────────────────
    for gen_fn in ALL_GENERATORS_COLOR:
        M_rgb, name, desc = gen_fn(nl=nl, nc=nc, r=r)

        if verbose:
            print(f"\n  ── {desc} ({name}) ──")

        res_color = benchmark_single(M_rgb, f"{name}_color",
                                      use_huffman=True)
        results.append(res_color)

        if verbose:
            ll = "✓" if res_color['lossless'] else "✗"
            ps = f"{res_color['psnr']:.1f}"
            print(f"    {'huffman+color':<20s}  {res_color['ratio']:>5.1f}x  "
                  f"{res_color['bpp']:>5.2f}  {ps:>6s}  "
                  f"{res_color['ssim']:.3f}  "
                  f"{res_color['encode_time_ms']:>6.1f}  "
                  f"{'   ' + ll:>8s}")

    # ── Tableau récapitulatif ──────────────────────────────────
    if verbose:
        print(f"\n{'═' * 80}")
        print(f"  RÉCAPITULATIF — Meilleure config par vidéo")
        print(f"{'─' * 80}")
        print(f"  {'Vidéo':<22s}  {'Ratio':>6s}  {'bpp':>6s}  "
              f"{'PSNR':>7s}  {'Économie':>8s}  {'Type dominant':>14s}")
        print(f"{'─' * 80}")

        # Grouper par vidéo de base
        by_base = {}
        for res in results:
            base = res['name'].rsplit('_', 1)[0]
            if base not in by_base or res['ratio'] > by_base[base]['ratio']:
                by_base[base] = res

        for base, res in by_base.items():
            ps = f"{res['psnr']:.1f}" if res['psnr'] != float('inf') else "∞"
            # Type dominant
            reps = res['rep_counts']
            mono = res['mono_count']
            spatial = res['proc_spatial']
            total = sum(reps.values()) + mono + spatial
            if total > 0:
                if mono > total * 0.5:
                    dom = "MONO"
                elif spatial > total * 0.3:
                    dom = "SPATIAL"
                elif reps[4] > total * 0.5:
                    dom = "R4"
                else:
                    dom = f"R{max(reps, key=reps.get)}"
            else:
                dom = "-"

            print(f"  {base:<22s}  {res['ratio']:>5.1f}x  "
                  f"{res['bpp']:>5.2f}  {ps:>7s}  "
                  f"{res['savings_pct']:>6.1f}%  {dom:>14s}")

        print(f"{'═' * 80}")

    return results


def generate_report(results: List[Dict[str, Any]], path: str = None) -> str:
    """Génère un rapport texte ou JSON."""
    report = {
        'codec': 'PPV v5',
        'modules': [
            'Processus ponctuels marqués (R1-R4)',
            'Classification MDL (Mono/PP/Spatial/Markov)',
            'Huffman adaptatif par bloc',
            'YCbCr 4:2:0',
            'Macroblocs adaptatifs',
        ],
        'results': results,
        'summary': {
            'avg_ratio': np.mean([r['ratio'] for r in results]),
            'max_ratio': max(r['ratio'] for r in results),
            'avg_bpp': np.mean([r['bpp'] for r in results]),
            'all_lossless_gray': all(
                r['lossless'] for r in results if not r['is_color']
            ),
        }
    }

    if path:
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    return json.dumps(report, indent=2, default=str)
