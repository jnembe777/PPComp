#!/usr/bin/env python3
"""
Test du codage par fonctions d'intensité + fenêtres adaptatives.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codec_pp.src.intensity import (
    analyze_block_structure, estimate_dl_intensity, estimate_dl_pixel_by_pixel,
    encode_block_intensity_v2, decode_block_intensity,
    compare_encodings, find_adaptive_windows,
)
from codec_pp.src.bitstream import BitWriter, BitReader


def test_roundtrip_intensity():
    """Encode/décode par intensité et vérifie lossless."""
    print("\n═══ Test Roundtrip Intensité ═══")

    scenarios = [
        ("Constant",        lambda: np.full((16, 4, 4), 42, dtype=np.uint8)),
        ("2 couleurs alt",  lambda: _gen_alternating(16, 4, 4, [10, 200], 4)),
        ("3 couleurs",      lambda: _gen_alternating(16, 4, 4, [10, 100, 200], 5)),
        ("Palette 4",       lambda: _gen_random_palette(32, 8, 8, 4, 0.1, 42)),
        ("Palette 8",       lambda: _gen_random_palette(32, 8, 8, 8, 0.15, 43)),
        ("Palette 16",      lambda: _gen_random_palette(32, 8, 8, 16, 0.1, 44)),
        ("Spatial sync",    lambda: _gen_spatial_sync(32, 8, 8, [10, 50, 90, 130])),
    ]

    all_ok = True
    for name, gen_fn in scenarios:
        block = gen_fn()
        r, bh, bw = block.shape

        # Encode
        w = BitWriter()
        info = analyze_block_structure(block)
        encode_block_intensity_v2(w, block, info, color_bits=8)
        data = w.flush()

        # Decode
        reader = BitReader(data)
        block_dec = decode_block_intensity(reader, r, bh, bw, color_bits=8)

        ok = np.array_equal(block, block_dec)
        all_ok = all_ok and ok
        bits = len(data) * 8
        bpp = bits / (r * bh * bw)

        status = "✓" if ok else "✗"
        if not ok:
            diff = np.sum(block != block_dec)
            print(f"  {status}  {name:<20s}  m={info['m']:>3}  "
                  f"ERREUR: {diff} pixels différents")
        else:
            print(f"  {status}  {name:<20s}  m={info['m']:>3}  "
                  f"N̄={info['mean_jumps']:.1f}  "
                  f"{bits:>6} bits  {bpp:.2f} bpp")

    return all_ok


def test_gain_analysis():
    """Compare DL intensité vs pixel-par-pixel sur différents scénarios."""
    print("\n═══ Analyse du Gain : Intensité vs Pixel-par-Pixel ═══")

    scenarios = [
        ("m=2, N=4/16",   _gen_random_palette(16, 8, 8, 2, 0.1, 50)),
        ("m=4, N=4/32",   _gen_random_palette(32, 8, 8, 4, 0.05, 51)),
        ("m=4, N=16/32",  _gen_random_palette(32, 8, 8, 4, 0.5, 52)),
        ("m=8, N=4/32",   _gen_random_palette(32, 8, 8, 8, 0.05, 53)),
        ("m=8, N=16/32",  _gen_random_palette(32, 8, 8, 8, 0.5, 54)),
        ("m=16, N=4/32",  _gen_random_palette(32, 8, 8, 16, 0.05, 55)),
        ("m=16, N=16/32", _gen_random_palette(32, 8, 8, 16, 0.5, 56)),
        ("m=32, N=4/64",  _gen_random_palette(64, 8, 8, 32, 0.05, 57)),
        ("m=64, N=4/64",  _gen_random_palette(64, 8, 8, 64, 0.05, 58)),
        ("m=128, N=4/64", _gen_random_palette(64, 8, 8, 128, 0.05, 59)),
        ("Spatial sync",  _gen_spatial_sync(32, 8, 8, [10, 50, 90, 130])),
        ("Worst: m=256",  _gen_random_palette(32, 8, 8, 256, 0.5, 60)),
    ]

    print(f"  {'Scénario':<20s}  {'m':>4}  {'N̄':>5}  {'eff':>4}  "
          f"{'DL pixel':>9}  {'DL intens':>9}  {'Gain':>7}")
    print(f"  {'─' * 72}")

    for name, block in scenarios:
        res = compare_encodings(block, color_bits=8)
        eff_str = f"{res['efficiency']:.2f}"
        gain_str = f"{res['gain_pct']:+.1f}%"
        marker = " ◀" if res['gain_pct'] > 5 else ""
        print(f"  {name:<20s}  {res['m']:>4}  {res['mean_jumps']:>5.1f}  "
              f"{eff_str:>4}  {res['dl_pixel']:>9.0f}  "
              f"{res['dl_intensity']:>9.0f}  {gain_str:>7}{marker}")


def test_adaptive_windows():
    """Test des fenêtres temporelles adaptatives."""
    print("\n═══ Test Fenêtres Adaptatives ═══")

    # Vidéo avec 2 régimes : zone calme (m=2) puis zone agitée (m=16)
    r_total = 128
    bh, bw = 8, 8
    rng = np.random.RandomState(42)

    # Zone 1 (frames 0-63) : 2 couleurs, peu de sauts
    zone1 = np.zeros((64, bh, bw), dtype=np.uint8)
    zone1[:32] = 40
    zone1[32:] = 200

    # Zone 2 (frames 64-127) : 16 couleurs, beaucoup de sauts
    palette_z2 = rng.randint(0, 256, 16, dtype=np.uint8)
    zone2 = np.zeros((64, bh, bw), dtype=np.uint8)
    zone2[0] = palette_z2[rng.randint(0, 16, (bh, bw))]
    for t in range(1, 64):
        change = rng.random((bh, bw)) < 0.3
        new = palette_z2[rng.randint(0, 16, (bh, bw))]
        zone2[t] = np.where(change, new, zone2[t-1])

    plane = np.concatenate([zone1, zone2], axis=0)
    full_plane = np.zeros((r_total, 16, 16), dtype=np.uint8)
    full_plane[:, :bh, :bw] = plane

    windows = find_adaptive_windows(
        full_plane, 0, bh, 0, bw,
        min_window=8, max_window=64, color_bits=8
    )

    print(f"  {len(windows)} fenêtres trouvées :")
    total_dl = 0
    for t_start, t_end, info in windows:
        dl = info['dl_estimate']
        total_dl += dl
        print(f"    [{t_start:>3}-{t_end:>3}]  r={info['r_win']:>3}  "
              f"m={info['m']:>3}  N̄={info['mean_jumps']:.1f}  "
              f"eff={info['eff']:.2f}  DL={dl:,.0f}")

    # Comparer avec fenêtre fixe de 128
    block_full = plane
    res_full = compare_encodings(block_full, color_bits=8)
    print(f"\n  Fenêtre fixe r=128 : DL_pixel={res_full['dl_pixel']:.0f}  "
          f"DL_intens={res_full['dl_intensity']:.0f}  m={res_full['m']}")
    print(f"  Fenêtres adaptatives : DL_total={total_dl:.0f}")

    if total_dl < res_full['dl_intensity']:
        gain = (1 - total_dl / res_full['dl_intensity']) * 100
        print(f"  → Gain adaptatif : {gain:.1f}% vs fenêtre fixe")
    else:
        print(f"  → Fenêtre fixe plus efficace sur ce cas")


def test_real_video_analysis():
    """Analyse du gain sur vidéo réelle si disponible."""
    video_path = "/home/claude/opencv_repo/samples/data/vtest.avi"
    if not os.path.exists(video_path):
        print("\n  (vidéo réelle non disponible, skip)")
        return

    print("\n═══ Analyse Vidéo Réelle (Surveillance) ═══")

    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(64):
        ret, f = cap.read()
        if not ret: break
        f = cv2.resize(f, (128, 96))
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    M = np.array(frames, dtype=np.uint8)

    # Analyser bloc par bloc
    gains = []
    m_values = []
    bs = 8
    for bi in range(0, 96, bs):
        for bj in range(0, 128, bs):
            block = M[:, bi:bi+bs, bj:bj+bs]
            res = compare_encodings(block, color_bits=8)
            gains.append(res['gain_pct'])
            m_values.append(res['m'])

    gains = np.array(gains)
    m_values = np.array(m_values)

    print(f"  Blocs analysés : {len(gains)}")
    print(f"  m moyen : {m_values.mean():.0f}  "
          f"médian : {np.median(m_values):.0f}  "
          f"min : {m_values.min()}  max : {m_values.max()}")
    print(f"  Gain moyen intensité vs pixel : {gains.mean():.1f}%")
    print(f"  Blocs où intensité gagne : "
          f"{np.sum(gains > 0)}/{len(gains)} "
          f"({np.sum(gains > 0)/len(gains)*100:.0f}%)")
    print(f"  Blocs où gain > 10% : "
          f"{np.sum(gains > 10)}/{len(gains)} "
          f"({np.sum(gains > 10)/len(gains)*100:.0f}%)")

    # Distribution par tranche de m
    for m_max in [4, 8, 16, 32, 64, 128, 256]:
        mask = m_values <= m_max
        if np.sum(mask) > 0:
            print(f"    m ≤ {m_max:>3} : {np.sum(mask):>3} blocs, "
                  f"gain moyen = {gains[mask].mean():+.1f}%")


# ── Générateurs ────────────────────────────────────────────────

def _gen_alternating(r, bh, bw, colors, period):
    block = np.zeros((r, bh, bw), dtype=np.uint8)
    for t in range(r):
        idx = (t // period) % len(colors)
        block[t] = colors[idx]
    return block

def _gen_random_palette(r, bh, bw, n_colors, change_prob, seed):
    rng = np.random.RandomState(seed)
    palette = rng.randint(0, 256, n_colors, dtype=np.uint8)
    block = np.zeros((r, bh, bw), dtype=np.uint8)
    block[0] = palette[rng.randint(0, n_colors, (bh, bw))]
    for t in range(1, r):
        change = rng.random((bh, bw)) < change_prob
        new = palette[rng.randint(0, n_colors, (bh, bw))]
        block[t] = np.where(change, new, block[t-1])
    return block

def _gen_spatial_sync(r, bh, bw, colors):
    block = np.zeros((r, bh, bw), dtype=np.uint8)
    rng = np.random.RandomState(77)
    seg_len = r // len(colors)
    for i in range(bh):
        for j in range(bw):
            for si, c in enumerate(colors):
                t0 = si * seg_len
                t1 = (si + 1) * seg_len if si < len(colors) - 1 else r
                block[t0:t1, i, j] = c + rng.randint(-2, 3)
    return np.clip(block, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CODAGE PAR FONCTIONS D'INTENSITÉ + FENÊTRES ADAPTATIVES ║")
    print("╚══════════════════════════════════════════════════════════╝")

    ok = test_roundtrip_intensity()
    test_gain_analysis()
    test_adaptive_windows()
    test_real_video_analysis()

    if ok:
        print(f"\n  ✓ Roundtrip lossless validé")
    else:
        print(f"\n  ✗ ERREUR roundtrip")
        sys.exit(1)
