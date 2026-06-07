#!/usr/bin/env python3
"""
Test Heuristic M6 — Entraînement + comparaison exhaustif vs fast.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codec_pp.src.heuristic import DecisionHeuristic
from codec_pp.src.encoder import PPVEncoder
from codec_pp.src.decoder import PPVDecoder
from codec_pp.src.video_io import generate_synthetic_video
from codec_pp.src.benchmark import (
    gen_static_bg, gen_gradient, gen_animation,
    gen_noise, gen_periodic,
)


def test_train_heuristic():
    """Entraîne le classifieur M6 et mesure la précision."""
    print("╔════════════════════════════════════════════════════╗")
    print("║  PHASE 4 — ENTRAÎNEMENT HEURISTIQUE M6             ║")
    print("╚════════════════════════════════════════════════════╝")

    h = DecisionHeuristic.train(
        block_size=8, color_bits=8, max_depth=8, verbose=True
    )

    # Sauvegarder
    model_path = "/tmp/heuristic_m6.json"
    h.save(model_path)
    print(f"\n  Modèle sauvegardé : {model_path}")

    # Recharger et vérifier
    h2 = DecisionHeuristic.load(model_path)
    print(f"  Rechargé : proc={h2.accuracy_proc:.1%}, "
          f"pred={h2.accuracy_pred:.1%}")

    return h


def test_exhaustive_vs_fast(heuristic):
    """Compare encodage exhaustif vs heuristique sur plusieurs vidéos."""
    print(f"\n{'═' * 70}")
    print(f"  COMPARAISON : EXHAUSTIF vs HEURISTIQUE M6")
    print(f"{'═' * 70}")

    ppv_path = "/tmp/test_heuristic.ppv"

    test_videos = [
        ("static_bg",  gen_static_bg(nl=64, nc=64, r=32)[0]),
        ("gradient",   gen_gradient(nl=64, nc=64, r=32)[0]),
        ("animation",  gen_animation(nl=64, nc=64, r=32)[0]),
        ("noise",      gen_noise(nl=64, nc=64, r=32)[0]),
        ("periodic",   gen_periodic(nl=64, nc=64, r=32)[0]),
        ("random_lo",  generate_synthetic_video(
            nl=64, nc=64, r=32, n_colors=4, change_prob=0.05, seed=99)),
        ("random_hi",  generate_synthetic_video(
            nl=64, nc=64, r=32, n_colors=16, change_prob=0.20, seed=99)),
    ]

    print(f"\n  {'Vidéo':<14s}  "
          f"{'Exh ratio':>9s}  {'Exh ms':>7s}  "
          f"{'Fast ratio':>10s}  {'Fast ms':>7s}  "
          f"{'Speedup':>7s}  {'Ratio diff':>10s}  "
          f"{'Lossless':>8s}")
    print(f"  {'─' * 88}")

    total_exh_time = 0
    total_fast_time = 0
    all_lossless = True

    for name, M in test_videos:
        # ── Exhaustif ──────────────────────────────────────
        enc_exh = PPVEncoder(
            gop_size=32, block_size=8,
            use_huffman=True, verbose=False
        )
        t0 = time.time()
        s_exh = enc_exh.encode(M=M, output_path=ppv_path, color_bits=8)
        t_exh = (time.time() - t0) * 1000

        dec = PPVDecoder(verbose=False)
        M_exh, _ = dec.decode(ppv_path)
        exh_ok = np.array_equal(M, M_exh)

        # ── Heuristique ────────────────────────────────────
        enc_fast = PPVEncoder(
            gop_size=32, block_size=8,
            use_huffman=True, verbose=False,
            heuristic=heuristic
        )
        t0 = time.time()
        s_fast = enc_fast.encode(M=M, output_path=ppv_path, color_bits=8)
        t_fast = (time.time() - t0) * 1000

        M_fast, _ = dec.decode(ppv_path)
        fast_ok = np.array_equal(M, M_fast)

        # ── Comparaison ────────────────────────────────────
        speedup = t_exh / t_fast if t_fast > 0 else 0
        ratio_diff = (s_fast['compression_ratio'] / s_exh['compression_ratio']
                      - 1) * 100
        lossless = exh_ok and fast_ok
        all_lossless = all_lossless and lossless

        total_exh_time += t_exh
        total_fast_time += t_fast

        ll = "✓" if lossless else "✗"
        print(f"  {name:<14s}  "
              f"{s_exh['compression_ratio']:>8.1f}x  {t_exh:>6.0f}ms  "
              f"{s_fast['compression_ratio']:>9.1f}x  {t_fast:>6.0f}ms  "
              f"{speedup:>6.2f}x  {ratio_diff:>+8.1f}%  "
              f"{'   ' + ll:>8s}")

    print(f"  {'─' * 88}")
    global_speedup = total_exh_time / total_fast_time if total_fast_time > 0 else 0
    print(f"  {'TOTAL':<14s}  "
          f"{'':>9s}  {total_exh_time:>6.0f}ms  "
          f"{'':>10s}  {total_fast_time:>6.0f}ms  "
          f"{global_speedup:>6.2f}x")
    print(f"{'═' * 70}")

    if os.path.exists(ppv_path):
        os.remove(ppv_path)

    return all_lossless


if __name__ == "__main__":
    h = test_train_heuristic()
    ok = test_exhaustive_vs_fast(h)

    # Sauvegarder le modèle dans outputs
    h.save("/mnt/user-data/outputs/heuristic_m6.json")

    if ok:
        print(f"\n  ✓ PHASE 4 COMPLÈTE — Heuristique M6 opérationnelle")
        print(f"    Précision proc : {h.accuracy_proc:.1%}")
        print(f"    Précision pred : {h.accuracy_pred:.1%}")
    else:
        print(f"\n  ✗ ERREUR : certains roundtrips ne sont pas lossless")
        sys.exit(1)
