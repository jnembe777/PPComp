#!/usr/bin/env python3
"""
Comparaison des deux encodeurs sur 3 vidéos réelles.

Vidéos :
  1. vtest.avi     — Surveillance (piétons, 768×576, fond statique + mouvement)
  2. Megamind.avi  — Film animation (720×528, couleurs vives, sauts nets)
  3. tree.avi      — Nature (320×240, textures, mouvement lent)

Pour chaque vidéo, on compare :
  A) Encodeur exhaustif MDL (classify_block sur chaque bloc)
  B) Encodeur heuristique M6 (arbre de décision)

En mode gris et en mode couleur, sur un extrait de 64 frames.
"""

import sys, os, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codec_pp.src.encoder import PPVEncoder
from codec_pp.src.decoder import PPVDecoder
from codec_pp.src.heuristic import DecisionHeuristic
from codec_pp.src.colorspace import split_video_planes, merge_video_planes, SUBSAMPLE_420


def load_video_clip(path, max_frames=64, target_w=None, target_h=None,
                    grayscale=True):
    """Charge un extrait vidéo."""
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if target_w and target_h:
            frame = cv2.resize(frame, (target_w, target_h))
        if grayscale:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    M = np.array(frames, dtype=np.uint8)
    return M, fps


def compute_psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


def run_comparison(M, name, fps, heuristic, ppv_path, is_color=False):
    """Encode/décode avec les deux modes et compare."""

    # ── Exhaustif ──────────────────────────────────────────────
    enc_exh = PPVEncoder(gop_size=32, block_size=8,
                         use_huffman=True, verbose=False)
    t0 = time.time()
    s_exh = enc_exh.encode(M=M, output_path=ppv_path,
                           color_bits=8, fps=fps)
    t_enc_exh = (time.time() - t0) * 1000

    dec = PPVDecoder(verbose=False)
    t0 = time.time()
    M_exh, meta_exh = dec.decode(ppv_path)
    t_dec_exh = (time.time() - t0) * 1000

    # ── Heuristique ────────────────────────────────────────────
    enc_fast = PPVEncoder(gop_size=32, block_size=8,
                          use_huffman=True, verbose=False,
                          heuristic=heuristic)
    t0 = time.time()
    s_fast = enc_fast.encode(M=M, output_path=ppv_path,
                             color_bits=8, fps=fps)
    t_enc_fast = (time.time() - t0) * 1000

    t0 = time.time()
    M_fast, meta_fast = dec.decode(ppv_path)
    t_dec_fast = (time.time() - t0) * 1000

    # ── Vérification lossless ──────────────────────────────────
    if is_color:
        planes_orig = split_video_planes(M, SUBSAMPLE_420)
        M_ref = merge_video_planes(
            planes_orig['Y'], planes_orig['Cb'], planes_orig['Cr'],
            planes_orig['nl'], planes_orig['nc'], SUBSAMPLE_420
        )
        exh_lossless = np.array_equal(M_ref, M_exh)
        fast_lossless = np.array_equal(M_ref, M_fast)
        psnr_exh = compute_psnr(M, M_exh)
        psnr_fast = compute_psnr(M, M_fast)
    else:
        exh_lossless = np.array_equal(M, M_exh)
        fast_lossless = np.array_equal(M, M_fast)
        psnr_exh = float('inf') if exh_lossless else compute_psnr(M, M_exh)
        psnr_fast = float('inf') if fast_lossless else compute_psnr(M, M_fast)

    # ── Résultats ──────────────────────────────────────────────
    shape = M.shape
    if is_color:
        r, nl, nc = shape[0], shape[1], shape[2]
        raw_bytes = r * nl * nc * 3
    else:
        r, nl, nc = shape
        raw_bytes = r * nl * nc

    return {
        'name': name,
        'resolution': f"{nl}×{nc}",
        'frames': r,
        'raw_bytes': raw_bytes,
        'is_color': is_color,
        'exh': {
            'ratio': s_exh['compression_ratio'],
            'savings': s_exh['savings_percent'],
            'size': s_exh['compressed_size_bytes'],
            'enc_ms': t_enc_exh,
            'dec_ms': t_dec_exh,
            'lossless': exh_lossless,
            'psnr': psnr_exh,
            'reps': s_exh['rep_counts'],
            'mono': s_exh['mono_count'],
            'spatial': s_exh.get('proc_spatial_count', 0),
        },
        'fast': {
            'ratio': s_fast['compression_ratio'],
            'savings': s_fast['savings_percent'],
            'size': s_fast['compressed_size_bytes'],
            'enc_ms': t_enc_fast,
            'dec_ms': t_dec_fast,
            'lossless': fast_lossless,
            'psnr': psnr_fast,
            'reps': s_fast['rep_counts'],
            'mono': s_fast['mono_count'],
            'spatial': s_fast.get('proc_spatial_count', 0),
        },
    }


def print_result(res):
    """Affiche les résultats d'une comparaison."""
    e = res['exh']
    f = res['fast']
    clr = "RGB 4:2:0" if res['is_color'] else "Gris"
    speedup = e['enc_ms'] / f['enc_ms'] if f['enc_ms'] > 0 else 0
    ratio_diff = (f['ratio'] / e['ratio'] - 1) * 100 if e['ratio'] > 0 else 0

    ps_e = f"{e['psnr']:.1f}" if e['psnr'] != float('inf') else "∞ (lossless)"
    ps_f = f"{f['psnr']:.1f}" if f['psnr'] != float('inf') else "∞ (lossless)"

    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  {res['name']:<30s} {res['resolution']} × {res['frames']}fr  {clr}")
    print(f"  │  Brut : {res['raw_bytes']:,} octets")
    print(f"  ├──────────────────────┬──────────────────────┬──────────────────┤")
    print(f"  │                      │  EXHAUSTIF           │  HEURISTIQUE M6  │")
    print(f"  ├──────────────────────┼──────────────────────┼──────────────────┤")
    print(f"  │  Taille compressée   │  {e['size']:>8,} B           │  {f['size']:>8,} B        │")
    print(f"  │  Ratio               │  {e['ratio']:>8.1f}x           │  {f['ratio']:>8.1f}x        │")
    print(f"  │  Économie            │  {e['savings']:>7.1f}%            │  {f['savings']:>7.1f}%         │")
    print(f"  │  PSNR                │  {ps_e:>18s}  │  {ps_f:>15s}  │")
    print(f"  │  Lossless            │  {'✓' if e['lossless'] else '✗':>5s}                │  {'✓' if f['lossless'] else '✗':>5s}              │")
    print(f"  │  Encodage            │  {e['enc_ms']:>7.0f} ms           │  {f['enc_ms']:>7.0f} ms        │")
    print(f"  │  Décodage            │  {e['dec_ms']:>7.0f} ms           │  {f['dec_ms']:>7.0f} ms        │")
    print(f"  ├──────────────────────┼──────────────────────┴──────────────────┤")
    print(f"  │  Speedup encodage    │  {speedup:.2f}x                                  │")
    print(f"  │  Perte de ratio      │  {ratio_diff:+.1f}%                                  │")
    print(f"  ├──────────────────────┼─────────────────────────────────────────┤")

    # Distribution
    total_e = sum(e['reps'].values()) + e['mono'] + e['spatial']
    total_f = sum(f['reps'].values()) + f['mono'] + f['spatial']

    def pct(v, t): return f"{v/t*100:.0f}%" if t > 0 else "0%"

    print(f"  │  Distribution (exh)  │  Mono={e['mono']}  "
          f"Spatial={e['spatial']}  "
          f"R4={e['reps'][4]}  R1={e['reps'][1]}")
    print(f"  │  Distribution (fast) │  Mono={f['mono']}  "
          f"Spatial={f['spatial']}  "
          f"R4={f['reps'][4]}  R1={f['reps'][1]}")
    print(f"  └──────────────────────┴─────────────────────────────────────────┘")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  COMPARAISON SUR 3 VIDÉOS RÉELLES — Exhaustif vs M6        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── 1. Entraîner l'heuristique ─────────────────────────────
    print("\n  Entraînement de l'heuristique M6...")
    h = DecisionHeuristic.train(block_size=8, color_bits=8, verbose=False)
    print(f"  Précision proc: {h.accuracy_proc:.1%}  "
          f"pred: {h.accuracy_pred:.1%}")

    # ── 2. Charger les vidéos ──────────────────────────────────
    base = "/home/claude/opencv_repo/samples/data"
    videos = [
        (f"{base}/vtest.avi",    "Surveillance (piétons)", 128, 96),
        (f"{base}/Megamind.avi", "Film animation",         128, 96),
        (f"{base}/tree.avi",     "Nature (arbre)",         128, 96),
    ]

    ppv_path = "/tmp/real_video_test.ppv"
    all_results = []

    for path, desc, tw, th in videos:
        print(f"\n{'═' * 65}")
        print(f"  Vidéo : {desc}  ({os.path.basename(path)})")

        # ── Mode gris ──────────────────────────────────────
        M_gray, fps = load_video_clip(path, max_frames=64,
                                       target_w=tw, target_h=th,
                                       grayscale=True)
        res_gray = run_comparison(M_gray, f"{desc} [gris]",
                                   fps, h, ppv_path, is_color=False)
        print_result(res_gray)
        all_results.append(res_gray)

        # ── Mode couleur ───────────────────────────────────
        M_color, fps = load_video_clip(path, max_frames=64,
                                        target_w=tw, target_h=th,
                                        grayscale=False)
        res_color = run_comparison(M_color, f"{desc} [couleur]",
                                    fps, h, ppv_path, is_color=True)
        print_result(res_color)
        all_results.append(res_color)

    # ── 3. Tableau récapitulatif ───────────────────────────────
    print(f"\n\n{'═' * 90}")
    print(f"  TABLEAU RÉCAPITULATIF")
    print(f"{'─' * 90}")
    print(f"  {'Vidéo':<32s}  {'Exh ratio':>9s}  {'Fast ratio':>10s}  "
          f"{'Speedup':>7s}  {'Perte':>6s}  {'Lossless':>8s}")
    print(f"{'─' * 90}")

    total_exh_ms = 0
    total_fast_ms = 0

    for res in all_results:
        e = res['exh']
        f = res['fast']
        sp = e['enc_ms'] / f['enc_ms'] if f['enc_ms'] > 0 else 0
        rd = (f['ratio'] / e['ratio'] - 1) * 100 if e['ratio'] > 0 else 0
        ll = "✓/✓" if (e['lossless'] and f['lossless']) else "✗"
        total_exh_ms += e['enc_ms']
        total_fast_ms += f['enc_ms']

        print(f"  {res['name']:<32s}  {e['ratio']:>8.1f}x  {f['ratio']:>9.1f}x  "
              f"{sp:>6.2f}x  {rd:>+5.1f}%  {'   '+ll:>8s}")

    sp_total = total_exh_ms / total_fast_ms if total_fast_ms > 0 else 0
    print(f"{'─' * 90}")
    print(f"  {'TOTAL':>32s}  {'':>9s}  {'':>10s}  "
          f"{sp_total:>6.2f}x")
    print(f"{'═' * 90}")

    if os.path.exists(ppv_path):
        os.remove(ppv_path)

    all_ok = all(r['exh']['lossless'] and r['fast']['lossless']
                 for r in all_results if not r['is_color'])
    print(f"\n  Tous les modes gris sont lossless : {'✓' if all_ok else '✗'}")
