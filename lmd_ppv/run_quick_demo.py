#!/usr/bin/env python3
"""
run_quick_demo.py - Demo rapide du benchmark LMD-PPV
=====================================================

Benchmark simplifie utilisant les videos locales existantes.
Ne necessite pas de telechargement externe.

Usage:
    python run_quick_demo.py
    python run_quick_demo.py --video sample_video.mp4
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))


def find_local_videos(base_dir: Path) -> list:
    """Trouve les videos locales."""
    videos = []
    extensions = ['*.mp4', '*.y4m', '*.avi', '*.mkv']

    for ext in extensions:
        videos.extend(base_dir.glob(ext))
        videos.extend(base_dir.glob(f'**/{ext}'))

    # Deduplicate et filtrer les gros fichiers
    seen = set()
    result = []
    for v in videos:
        if v.name not in seen and v.stat().st_size < 100_000_000:  # < 100MB
            seen.add(v.name)
            result.append(v)

    return result[:5]  # Max 5 videos


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Demo rapide du benchmark')
    parser.add_argument('--video', type=Path, help='Video specifique a tester')
    parser.add_argument('--output', type=Path, default=Path('./demo_results'),
                        help='Repertoire de sortie')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  LMD-PPV QUICK DEMO")
    print("=" * 60)

    # Trouver les videos
    if args.video:
        if args.video.exists():
            videos = [args.video]
        else:
            print(f"[ERREUR] Video non trouvee: {args.video}")
            return 1
    else:
        videos = find_local_videos(Path(__file__).parent)

    if not videos:
        print("[ERREUR] Aucune video trouvee.")
        print("  Placez une video .mp4 dans le dossier ou utilisez --video")
        return 1

    print(f"\n  Videos trouvees: {len(videos)}")
    for v in videos:
        print(f"    - {v.name} ({v.stat().st_size // 1024} KB)")

    # Creer le repertoire de sortie
    args.output.mkdir(parents=True, exist_ok=True)

    results = []
    total_start = time.time()

    # Test 1: Encodage LMD-PPV
    print("\n[1/3] Test d'encodage LMD-PPV...")
    print("-" * 40)

    try:
        from src.video.fast_encoder import FastVideoEncoder

        for video_path in videos:
            print(f"\n  Encodage: {video_path.name}")
            start = time.time()

            encoder = FastVideoEncoder(block_size=16, block_frames=32, n_colors=256)
            stats = encoder.encode(str(video_path), max_frames=100)

            elapsed = time.time() - start

            # stats est un FastEncodingStats (dataclass), pas un dict
            result = {
                'video': video_path.name,
                'test': 'encode_lmd',
                'time_sec': elapsed,
                'n_frames': getattr(stats, 'total_frames', 0),
                'compression_ratio': getattr(stats, 'compression_ratio', 0),
            }
            results.append(result)

            print(f"    Temps: {elapsed:.2f}s")
            print(f"    Frames: {result['n_frames']}")
            print(f"    Ratio: {result['compression_ratio']:.1f}x")

    except Exception as e:
        print(f"  [WARN] Erreur encodage: {e}")

    # Test 2: Features et classification
    print("\n[2/3] Test de classification...")
    print("-" * 40)

    try:
        from src.core.features import BlockFeatures
        from src.agents.agent_1_classification import ClassificationAgent
        import numpy as np

        agent = ClassificationAgent()
        np.random.seed(42)

        # Generer des features synthetiques
        test_features = []
        for _ in range(100):
            features = BlockFeatures(
                N=np.random.randint(10, 200),
                r=256,
                m=np.random.randint(4, 64),
                H_s=np.random.random(),
                rho_corr=np.random.random(),
                H_color=np.random.random() * 6,
                N_trans=np.random.randint(5, 100)
            )
            test_features.append(features)

        # Classifier
        start = time.time()
        classifications = []
        for f in test_features:
            result = agent.classify(f)
            classifications.append(result.process_type)

        elapsed = time.time() - start

        # Statistiques
        from collections import Counter
        type_counts = Counter(classifications)

        print(f"  Blocs classifies: {len(test_features)}")
        print(f"  Temps: {elapsed:.3f}s ({len(test_features)/elapsed:.0f} blocs/s)")
        print(f"  Distribution:")
        for t, count in sorted(type_counts.items()):
            print(f"    Type {t}: {count} ({count/len(test_features):.1%})")

        results.append({
            'test': 'classification',
            'n_blocks': len(test_features),
            'time_sec': elapsed,
            'distribution': dict(type_counts),
        })

    except Exception as e:
        print(f"  [WARN] Erreur classification: {e}")

    # Test 3: Metriques de qualite
    print("\n[3/3] Test de metriques...")
    print("-" * 40)

    try:
        from benchmark.metrics.quality import compute_psnr, compute_ssim
        import numpy as np

        # Generer des images de test
        img1 = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        img2 = img1.copy()
        img2[:128, :128] = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

        start = time.time()
        psnr = compute_psnr(img1, img2)
        ssim = compute_ssim(img1, img2)
        elapsed = time.time() - start

        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
        print(f"  Temps: {elapsed:.3f}s")

        results.append({
            'test': 'quality_metrics',
            'psnr': psnr,
            'ssim': ssim,
            'time_sec': elapsed,
        })

    except Exception as e:
        print(f"  [WARN] Erreur metriques: {e}")

    # Resume
    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("  RESUME")
    print("=" * 60)
    print(f"\n  Tests executes: {len(results)}")
    print(f"  Temps total: {total_time:.2f}s")

    # Sauvegarder les resultats
    output_file = args.output / 'demo_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_time_sec': total_time,
            'results': results,
        }, f, indent=2)

    print(f"\n  Resultats sauvegardes: {output_file}")
    print("\n  Pour le benchmark complet, lancez:")
    print("    python run_full_benchmark.py --quick --skip-download")
    print("=" * 60 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
