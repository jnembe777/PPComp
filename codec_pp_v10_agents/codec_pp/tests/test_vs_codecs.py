#!/usr/bin/env python3
"""
Comparaison PPV Codec vs H.264, H.265, VP9 sur 3 vidéos réelles.

Pour chaque vidéo et chaque codec :
  - Mode LOSSLESS (CRF 0 / lossless flag)
  - Mode LOSSY qualité haute (CRF 18-23, qualité visuelle quasi-identique)

Métriques :
  - Taille compressée (octets)
  - Ratio de compression
  - PSNR (dB)
  - Temps d'encodage (ms)
  - Bits par pixel par frame (bpp)
"""

import sys, os, time, subprocess, tempfile
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codec_pp.src.encoder import PPVEncoder
from codec_pp.src.decoder import PPVDecoder
from codec_pp.src.heuristic import DecisionHeuristic


def load_clip(path, max_frames=64, tw=128, th=96):
    """Charge un extrait vidéo en gris et couleur."""
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    frames_gray = []
    frames_bgr = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (tw, th))
        frames_bgr.append(frame)
        frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    M_gray = np.array(frames_gray, dtype=np.uint8)
    M_bgr = np.array(frames_bgr, dtype=np.uint8)
    return M_gray, M_bgr, fps


def save_raw_yuv(M_gray, path):
    """Sauvegarde en YUV brut pour FFmpeg."""
    with open(path, 'wb') as f:
        for t in range(M_gray.shape[0]):
            f.write(M_gray[t].tobytes())


def save_raw_avi(M_bgr, path, fps):
    """Sauvegarde en AVI non compressé pour FFmpeg."""
    r, h, w, _ = M_bgr.shape
    fourcc = cv2.VideoWriter_fourcc(*'HFYU')  # HuffYUV lossless
    out = cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=True)
    for t in range(r):
        out.write(M_bgr[t])
    out.release()


def compute_psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)


def ffmpeg_encode(input_path, output_path, codec, mode, fps, w, h,
                  is_gray=True):
    """Encode avec FFmpeg et retourne (taille, temps_ms)."""
    if is_gray:
        input_args = [
            '-f', 'rawvideo', '-pix_fmt', 'gray',
            '-s', f'{w}x{h}', '-r', str(fps),
            '-i', input_path
        ]
    else:
        input_args = ['-i', input_path]

    if codec == 'h264':
        if mode == 'lossless':
            codec_args = ['-c:v', 'libx264', '-qp', '0', '-preset', 'medium']
        else:
            codec_args = ['-c:v', 'libx264', '-crf', '18', '-preset', 'medium']
        if is_gray:
            codec_args += ['-pix_fmt', 'yuv420p']

    elif codec == 'h265':
        if mode == 'lossless':
            codec_args = ['-c:v', 'libx265', '-x265-params', 'lossless=1',
                          '-preset', 'medium']
        else:
            codec_args = ['-c:v', 'libx265', '-crf', '20', '-preset', 'medium']
        if is_gray:
            codec_args += ['-pix_fmt', 'yuv420p']

    elif codec == 'vp9':
        if mode == 'lossless':
            codec_args = ['-c:v', 'libvpx-vp9', '-lossless', '1']
        else:
            codec_args = ['-c:v', 'libvpx-vp9', '-crf', '23', '-b:v', '0']
        if is_gray:
            codec_args += ['-pix_fmt', 'yuv420p']

    cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error'] + \
          input_args + codec_args + ['-an', output_path]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    t_ms = (time.time() - t0) * 1000

    if result.returncode != 0:
        err = result.stderr.decode()[:200]
        return None, t_ms, err

    size = os.path.getsize(output_path)
    return size, t_ms, None


def ffmpeg_decode_gray(encoded_path, w, h, n_frames):
    """Décode avec FFmpeg et retourne la vidéo grise."""
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', encoded_path,
        '-pix_fmt', 'gray', '-f', 'rawvideo', 'pipe:1'
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        return None
    raw = result.stdout
    expected = n_frames * h * w
    if len(raw) < expected:
        n_frames = len(raw) // (h * w)
    M = np.frombuffer(raw[:n_frames * h * w], dtype=np.uint8)
    return M.reshape(n_frames, h, w)


def run_full_comparison():
    base = "/home/claude/opencv_repo/samples/data"
    videos = [
        (f"{base}/vtest.avi",    "Surveillance"),
        (f"{base}/Megamind.avi", "Animation"),
        (f"{base}/tree.avi",     "Nature"),
    ]

    tw, th = 128, 96
    max_frames = 64

    # Train heuristic
    print("  Entraînement M6...", end=" ", flush=True)
    heuristic = DecisionHeuristic.train(block_size=8, verbose=False)
    print(f"OK (proc={heuristic.accuracy_proc:.0%})")

    codecs_lossless = ['ppv_exh', 'ppv_fast', 'h264', 'h265', 'vp9']
    codecs_lossy = ['h264', 'h265', 'vp9']

    all_results = []

    for vpath, vname in videos:
        M_gray, M_bgr, fps = load_clip(vpath, max_frames, tw, th)
        r, h, w = M_gray.shape
        raw_bytes = r * h * w
        raw_bpp = 8.0

        print(f"\n{'═' * 85}")
        print(f"  {vname} — {w}×{h}, {r} frames, {fps} fps, "
              f"brut={raw_bytes:,} octets")
        print(f"{'═' * 85}")

        # Fichiers temporaires
        yuv_path = tempfile.mktemp(suffix='.yuv')
        avi_path = tempfile.mktemp(suffix='.avi')
        ppv_path = tempfile.mktemp(suffix='.ppv')
        save_raw_yuv(M_gray, yuv_path)

        results_video = {'name': vname, 'raw_bytes': raw_bytes,
                         'resolution': f"{w}×{h}", 'frames': r}

        # ── MODE LOSSLESS ──────────────────────────────────────
        print(f"\n  {'LOSSLESS':>12s}  {'Taille':>10s}  {'Ratio':>7s}  "
              f"{'bpp':>6s}  {'Enc ms':>8s}  {'PSNR':>8s}")
        print(f"  {'─' * 65}")

        lossless_results = {}

        # PPV Exhaustif
        enc = PPVEncoder(gop_size=32, block_size=8,
                         use_huffman=True, verbose=False)
        t0 = time.time()
        stats = enc.encode(M=M_gray, output_path=ppv_path, color_bits=8)
        t_ppv = (time.time() - t0) * 1000
        dec = PPVDecoder(verbose=False)
        M_dec, _ = dec.decode(ppv_path)
        psnr_ppv = compute_psnr(M_gray, M_dec)
        bpp_ppv = stats['compressed_size_bytes'] * 8 / (r * h * w)
        lossless_results['ppv_exh'] = {
            'size': stats['compressed_size_bytes'],
            'ratio': stats['compression_ratio'],
            'bpp': bpp_ppv, 'enc_ms': t_ppv,
            'psnr': psnr_ppv,
        }
        ps = "∞" if psnr_ppv == float('inf') else f"{psnr_ppv:.1f}"
        print(f"  {'PPV Exhaustif':>12s}  {stats['compressed_size_bytes']:>9,}B  "
              f"{stats['compression_ratio']:>6.2f}x  {bpp_ppv:>5.2f}  "
              f"{t_ppv:>7.0f}ms  {ps:>8s}")

        # PPV Heuristique
        enc_fast = PPVEncoder(gop_size=32, block_size=8,
                              use_huffman=True, verbose=False,
                              heuristic=heuristic)
        t0 = time.time()
        stats_f = enc_fast.encode(M=M_gray, output_path=ppv_path, color_bits=8)
        t_ppv_f = (time.time() - t0) * 1000
        M_dec_f, _ = dec.decode(ppv_path)
        psnr_ppv_f = compute_psnr(M_gray, M_dec_f)
        bpp_ppv_f = stats_f['compressed_size_bytes'] * 8 / (r * h * w)
        lossless_results['ppv_fast'] = {
            'size': stats_f['compressed_size_bytes'],
            'ratio': stats_f['compression_ratio'],
            'bpp': bpp_ppv_f, 'enc_ms': t_ppv_f,
            'psnr': psnr_ppv_f,
        }
        ps_f = "∞" if psnr_ppv_f == float('inf') else f"{psnr_ppv_f:.1f}"
        print(f"  {'PPV Fast M6':>12s}  {stats_f['compressed_size_bytes']:>9,}B  "
              f"{stats_f['compression_ratio']:>6.2f}x  {bpp_ppv_f:>5.2f}  "
              f"{t_ppv_f:>7.0f}ms  {ps_f:>8s}")

        # H.264, H.265, VP9 lossless
        for codec in ['h264', 'h265', 'vp9']:
            ext = {'h264': '.mp4', 'h265': '.mp4', 'vp9': '.webm'}[codec]
            out_path = tempfile.mktemp(suffix=ext)
            size, t_ms, err = ffmpeg_encode(
                yuv_path, out_path, codec, 'lossless', fps, w, h
            )
            if size is None:
                print(f"  {codec.upper():>12s}  {'ERREUR':>10s}  {err[:40]}")
                lossless_results[codec] = None
                continue

            # Décoder et mesurer PSNR
            M_ff = ffmpeg_decode_gray(out_path, w, h, r)
            if M_ff is not None and M_ff.shape == M_gray.shape:
                psnr_ff = compute_psnr(M_gray, M_ff)
            else:
                psnr_ff = -1

            ratio_ff = raw_bytes / size if size > 0 else 0
            bpp_ff = size * 8 / (r * h * w)
            lossless_results[codec] = {
                'size': size, 'ratio': ratio_ff,
                'bpp': bpp_ff, 'enc_ms': t_ms, 'psnr': psnr_ff,
            }
            ps_ff = "∞" if psnr_ff == float('inf') else f"{psnr_ff:.1f}"
            label = {'h264': 'H.264', 'h265': 'H.265', 'vp9': 'VP9'}[codec]
            print(f"  {label:>12s}  {size:>9,}B  "
                  f"{ratio_ff:>6.2f}x  {bpp_ff:>5.2f}  "
                  f"{t_ms:>7.0f}ms  {ps_ff:>8s}")

            os.remove(out_path)

        # ── MODE LOSSY (FFmpeg uniquement, PPV est toujours lossless) ──
        print(f"\n  {'LOSSY':>12s}  {'Taille':>10s}  {'Ratio':>7s}  "
              f"{'bpp':>6s}  {'Enc ms':>8s}  {'PSNR':>8s}")
        print(f"  {'─' * 65}")

        lossy_results = {}
        for codec in ['h264', 'h265', 'vp9']:
            ext = {'h264': '.mp4', 'h265': '.mp4', 'vp9': '.webm'}[codec]
            out_path = tempfile.mktemp(suffix=ext)
            size, t_ms, err = ffmpeg_encode(
                yuv_path, out_path, codec, 'lossy', fps, w, h
            )
            if size is None:
                print(f"  {codec.upper():>12s}  {'ERREUR':>10s}")
                continue

            M_ff = ffmpeg_decode_gray(out_path, w, h, r)
            if M_ff is not None and M_ff.shape == M_gray.shape:
                psnr_ff = compute_psnr(M_gray, M_ff)
            else:
                psnr_ff = -1

            ratio_ff = raw_bytes / size if size > 0 else 0
            bpp_ff = size * 8 / (r * h * w)
            lossy_results[codec] = {
                'size': size, 'ratio': ratio_ff,
                'bpp': bpp_ff, 'enc_ms': t_ms, 'psnr': psnr_ff,
            }
            label = {'h264': 'H.264', 'h265': 'H.265', 'vp9': 'VP9'}[codec]
            print(f"  {label:>12s}  {size:>9,}B  "
                  f"{ratio_ff:>6.2f}x  {bpp_ff:>5.2f}  "
                  f"{t_ms:>7.0f}ms  {psnr_ff:>7.1f}dB")

            os.remove(out_path)

        results_video['lossless'] = lossless_results
        results_video['lossy'] = lossy_results
        all_results.append(results_video)

        # Cleanup
        os.remove(yuv_path)

    # ── TABLEAU RÉCAPITULATIF ──────────────────────────────────
    print(f"\n\n{'═' * 90}")
    print(f"  RÉCAPITULATIF — LOSSLESS (gris, {tw}×{th}, {max_frames} frames)")
    print(f"{'─' * 90}")
    print(f"  {'Vidéo':<15s}  {'PPV Exh':>10s}  {'PPV M6':>10s}  "
          f"{'H.264':>10s}  {'H.265':>10s}  {'VP9':>10s}")
    print(f"{'─' * 90}")

    for res in all_results:
        ll = res['lossless']
        parts = []
        for c in ['ppv_exh', 'ppv_fast', 'h264', 'h265', 'vp9']:
            if ll.get(c):
                parts.append(f"{ll[c]['ratio']:>6.1f}x")
            else:
                parts.append(f"{'N/A':>7s}")

        print(f"  {res['name']:<15s}  {'  '.join(parts)}")

    print(f"{'─' * 90}")

    # bpp
    print(f"\n  {'bpp (bits/pixel)'}")
    print(f"  {'Vidéo':<15s}  {'PPV Exh':>10s}  {'PPV M6':>10s}  "
          f"{'H.264':>10s}  {'H.265':>10s}  {'VP9':>10s}")
    print(f"{'─' * 90}")
    for res in all_results:
        ll = res['lossless']
        parts = []
        for c in ['ppv_exh', 'ppv_fast', 'h264', 'h265', 'vp9']:
            if ll.get(c):
                parts.append(f"{ll[c]['bpp']:>7.2f}")
            else:
                parts.append(f"{'N/A':>7s}")
        print(f"  {res['name']:<15s}  {'   '.join(parts)}")

    print(f"{'═' * 90}")

    # Temps
    print(f"\n  {'Temps encodage (ms)'}")
    print(f"  {'Vidéo':<15s}  {'PPV Exh':>10s}  {'PPV M6':>10s}  "
          f"{'H.264':>10s}  {'H.265':>10s}  {'VP9':>10s}")
    print(f"{'─' * 90}")
    for res in all_results:
        ll = res['lossless']
        parts = []
        for c in ['ppv_exh', 'ppv_fast', 'h264', 'h265', 'vp9']:
            if ll.get(c):
                parts.append(f"{ll[c]['enc_ms']:>7.0f}")
            else:
                parts.append(f"{'N/A':>7s}")
        print(f"  {res['name']:<15s}  {'   '.join(parts)}")

    print(f"{'═' * 90}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  COMPARAISON PPV vs H.264 / H.265 / VP9 — 3 vidéos réelles     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    run_full_comparison()
