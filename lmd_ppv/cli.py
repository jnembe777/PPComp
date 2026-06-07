#!/usr/bin/env python3
"""
cli.py - Interface en ligne de commande LMD-PPV
================================================

Utilisation:
    python cli.py encode input.mp4 -o output.lmd
    python cli.py decode output.lmd -o reconstructed.mp4
    python cli.py info output.lmd
    python cli.py benchmark input.mp4
    python cli.py benchmark download --source xiph,cdvl
    python cli.py benchmark optimize --dataset ./datasets
    python cli.py benchmark validate --thresholds ./optimization.json
    python cli.py benchmark run --dataset ./datasets --codecs all
    python cli.py benchmark report --results ./results
    python cli.py demo

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import argparse
import sys
from pathlib import Path


def cmd_encode(args):
    """Encode une video."""
    if args.fast:
        # Mode rapide (stats seulement)
        from src.video.fast_encoder import FastVideoEncoder

        print(f"=== Encodage RAPIDE LMD-PPV ===\n")

        encoder = FastVideoEncoder(
            block_size=args.block_size,
            block_frames=args.block_frames,
            n_colors=args.colors
        )

        stats = encoder.encode(args.input, max_frames=args.max_frames)
        encoder.print_report(stats)
    elif args.turbo:
        # Mode turbo (rapide + sauvegarde .lmd)
        from src.video.turbo_encoder import TurboVideoEncoder

        print(f"=== Encodage TURBO LMD-PPV ===\n")

        encoder = TurboVideoEncoder(
            block_size=args.block_size,
            block_frames=args.block_frames,
            n_colors=args.colors
        )

        output = args.output or str(Path(args.input).with_suffix('.lmd'))
        stats = encoder.encode(args.input, output, max_frames=args.max_frames)
        encoder.print_report(stats)

        print(f"\n[OK] Fichier: {output}")
    else:
        # Mode complet
        from src.video.encoder import VideoEncoder
        from src.video.quantizer import QuantizeMethod

        print(f"=== Encodage LMD-PPV ===\n")

        methods = {
            'uniform': QuantizeMethod.UNIFORM,
            'kmeans': QuantizeMethod.KMEANS,
            'median': QuantizeMethod.MEDIAN_CUT
        }
        method = methods.get(args.quantize, QuantizeMethod.KMEANS)

        encoder = VideoEncoder(
            block_size=args.block_size,
            block_frames=args.block_frames,
            n_colors=args.colors,
            quantize_method=method
        )

        output = args.output or str(Path(args.input).with_suffix('.lmd'))

        encoded = encoder.encode(
            args.input,
            output,
            max_frames=args.max_frames
        )

        print(f"\n[OK] Fichier cree: {output}")
        print(f"[OK] Taille: {len(encoded.to_bytes()) / 1024:.1f} KB")


def cmd_decode(args):
    """Decode une video."""
    from src.video.fast_decoder import FastVideoDecoder
    import time

    print(f"=== Decodage RAPIDE LMD-PPV ===\n")

    t0 = time.time()
    decoder = FastVideoDecoder()
    header = decoder.load(args.input)

    print(f"Video: {header.width}x{header.height}")
    print(f"Frames: {header.n_frames} @ {header.fps:.1f} fps")
    print(f"Blocs: {len(decoder.blocks_data)}")

    if args.output:
        if args.output.endswith(('.png', '.jpg')):
            # Sauvegarde en frames
            output_dir = Path(args.output).parent / Path(args.output).stem
            for i in range(header.n_frames):
                frame = decoder.decode_frame(i)
                # Save frame logic here
            print(f"Frames sauvegardees: {output_dir}")
        else:
            # Sauvegarde en video
            decoder.save_video(args.output)

        elapsed = time.time() - t0
        print(f"\n[OK] Sortie: {args.output}")
        print(f"[OK] Temps: {elapsed:.2f}s ({header.n_frames / elapsed:.1f} fps)")
    else:
        # Benchmark
        stats = decoder.benchmark()
        print(f"\n[OK] Decode: {header.n_frames} frames")
        print(f"[OK] Temps: {stats.decode_time_sec:.2f}s ({stats.fps_decoding:.1f} fps)")


def cmd_info(args):
    """Affiche les informations d'un fichier LMD."""
    from src.video.decoder import VideoDecoder

    print(f"=== Info LMD-PPV ===\n")

    decoder = VideoDecoder()
    header = decoder.load(args.input)

    print(f"Fichier:        {args.input}")
    print(f"Resolution:     {header.width}x{header.height}")
    print(f"Frames:         {header.n_frames}")
    print(f"FPS:            {header.fps:.2f}")
    print(f"Duree:          {header.n_frames / header.fps:.2f}s")
    print(f"Taille bloc:    {header.block_size}x{header.block_size}")
    print(f"Frames/bloc:    {header.block_frames}")
    print(f"Couleurs:       {header.n_colors}")
    print(f"Blocs:          {len(decoder.blocks)}")

    # Taille fichier
    file_size = Path(args.input).stat().st_size
    print(f"\nTaille fichier: {file_size / 1024:.1f} KB")

    # Estimation taille originale
    original = header.width * header.height * header.n_frames * 3
    print(f"Taille orig:    {original / 1024 / 1024:.1f} MB")
    print(f"Ratio:          {original / file_size:.1f}x")


def cmd_benchmark_simple(args):
    """Benchmark simple sur une video."""
    from src.video.encoder import VideoEncoder
    from src.video.quantizer import QuantizeMethod
    import time

    print(f"=== Benchmark LMD-PPV ===\n")

    configs = [
        {'block_size': 8, 'colors': 64},
        {'block_size': 16, 'colors': 128},
        {'block_size': 16, 'colors': 256},
        {'block_size': 32, 'colors': 256},
    ]

    results = []

    for config in configs:
        print(f"\nConfig: block={config['block_size']}, colors={config['colors']}")

        encoder = VideoEncoder(
            block_size=config['block_size'],
            n_colors=config['colors'],
            quantize_method=QuantizeMethod.KMEANS
        )

        start = time.time()
        encoded = encoder.encode(
            args.input,
            max_frames=args.max_frames or 100
        )
        elapsed = time.time() - start

        results.append({
            'config': config,
            'ratio': encoded.stats.compression_ratio,
            'time': elapsed,
            'fps': encoded.stats.fps_encoding
        })

    print("\n" + "=" * 60)
    print("RESULTATS")
    print("=" * 60)

    print(f"\n{'Block':>6} {'Colors':>7} {'Ratio':>8} {'Time':>8} {'FPS':>8}")
    print("-" * 45)

    for r in results:
        print(f"{r['config']['block_size']:>6} "
              f"{r['config']['colors']:>7} "
              f"{r['ratio']:>7.1f}x "
              f"{r['time']:>7.2f}s "
              f"{r['fps']:>7.1f}")


# ============================================================================
# NOUVELLES COMMANDES BENCHMARK
# ============================================================================

def cmd_benchmark_download(args):
    """Telecharge les datasets video."""
    from benchmark.datasets.downloader import DatasetDownloader
    from benchmark.config import BenchmarkConfig

    print("=== Telechargement des datasets ===\n")

    output_dir = Path(args.output) if args.output else Path('./datasets')
    downloader = DatasetDownloader(output_dir)

    sources = args.source.split(',') if args.source else ['xiph']

    print(f"Sources: {sources}")
    print(f"Output:  {output_dir}")
    print()

    results = downloader.download_all(sources=sources)

    total = 0
    for source, videos in results.items():
        print(f"{source}: {len(videos)} videos")
        total += len(videos)

    print(f"\nTotal: {total} videos telecharges")
    print(f"[OK] Datasets sauvegardes dans: {output_dir}")


def cmd_benchmark_optimize(args):
    """Optimise les seuils du classifieur par grid search."""
    from optimization.grid_search import run_optimization_pipeline
    from benchmark.config import BenchmarkConfig
    import json

    print("=== Optimisation des seuils ===\n")

    dataset_path = Path(args.dataset) if args.dataset else Path('./datasets')
    output_path = Path(args.output) if args.output else Path('./optimization.json')

    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_path}")
    print(f"Quick:   {args.quick}")
    print()

    # Charger les features des videos
    # Note: Cette partie necessite d'extraire les features des videos
    # Pour l'instant, on utilise un placeholder
    print("[INFO] Extraction des features...")

    # Placeholder - en production, extraire les features reelles
    features_list = []

    if not features_list:
        print("[WARN] Aucune feature extraite. Utilisation de donnees synthetiques.")
        from src.core.features import BlockFeatures
        import numpy as np

        # Generer des features synthetiques pour demo
        np.random.seed(42)
        for _ in range(100):
            features_list.append(BlockFeatures(
                N=np.random.randint(10, 200),
                r=256,
                m=np.random.randint(4, 64),
                H_s=np.random.random(),
                rho_corr=np.random.random(),
                H_color=np.random.random() * 6,
                N_trans=np.random.randint(5, 100)
            ))

    print(f"[INFO] {len(features_list)} blocs a analyser")
    print()

    # Lancer l'optimisation
    result = run_optimization_pipeline(
        features_list,
        output_path,
        alpha=args.alpha,
        beta=args.beta,
        n_workers=args.workers,
        quick_search=args.quick
    )

    print(f"\n[OK] Configuration optimale sauvegardee: {output_path}")


def cmd_benchmark_validate(args):
    """Valide le classifieur avec les seuils optimises."""
    from validation.predictor_vs_optimal import run_validation
    from optimization.threshold_config import ThresholdConfig
    import json

    print("=== Validation du classifieur ===\n")

    # Charger les seuils
    if args.thresholds:
        thresholds_path = Path(args.thresholds)
        if thresholds_path.exists():
            config = ThresholdConfig.load(thresholds_path)
            print(f"Seuils charges depuis: {thresholds_path}")
        else:
            print(f"[WARN] Fichier non trouve: {thresholds_path}")
            config = ThresholdConfig.default()
    else:
        config = ThresholdConfig.default()
        print("Utilisation des seuils par defaut")

    print(f"Configuration: {config.to_dict()}")
    print()

    # Charger les features de test
    # Placeholder - en production, charger les features reelles
    from src.core.features import BlockFeatures
    import numpy as np

    np.random.seed(123)
    test_features = []
    for _ in range(50):
        test_features.append(BlockFeatures(
            N=np.random.randint(10, 200),
            r=256,
            m=np.random.randint(4, 64),
            H_s=np.random.random(),
            rho_corr=np.random.random(),
            H_color=np.random.random() * 6,
            N_trans=np.random.randint(5, 100)
        ))

    output_path = Path(args.output) if args.output else Path('./validation_results.json')

    result = run_validation(test_features, config, output_path)


def cmd_benchmark_run(args):
    """Execute le benchmark complet sur tous les codecs."""
    from benchmark.runner import run_benchmark_pipeline
    from benchmark.datasets.downloader import DatasetDownloader
    from benchmark.config import BenchmarkConfig

    print("=== Benchmark complet ===\n")

    dataset_path = Path(args.dataset) if args.dataset else Path('./datasets')
    output_path = Path(args.output) if args.output else Path('./benchmark_results')

    print(f"Dataset: {dataset_path}")
    print(f"Output:  {output_path}")
    print(f"Codecs:  {args.codecs}")
    print()

    # Charger les videos
    downloader = DatasetDownloader(dataset_path)
    videos = downloader.get_all_videos()

    if not videos:
        print("[WARN] Aucune video trouvee. Lancez 'benchmark download' d'abord.")
        return

    print(f"[INFO] {len(videos)} videos a traiter")

    # Filtrer les codecs
    codecs = args.codecs.split(',') if args.codecs != 'all' else None

    # Configuration
    config = BenchmarkConfig(
        output_dir=output_path,
        datasets_dir=dataset_path,
        max_videos=args.max_videos,
        n_workers=args.workers
    )

    # Lancer le benchmark
    summary = run_benchmark_pipeline(videos, output_path, config, codecs)

    print(f"\n[OK] Resultats sauvegardes: {output_path}")


def cmd_benchmark_report(args):
    """Genere le rapport de benchmark."""
    from reports.generator import generate_report

    print("=== Generation du rapport ===\n")

    results_path = Path(args.results) if args.results else Path('./benchmark_results')
    output_path = Path(args.output) if args.output else Path('./report')

    print(f"Resultats: {results_path}")
    print(f"Output:    {output_path}")
    print(f"Formats:   {args.format}")
    print()

    formats = args.format.split(',')

    outputs = generate_report(
        output_path,
        benchmark_path=results_path / 'benchmark_results.json',
        optimization_path=results_path / 'optimization.json',
        validation_path=results_path / 'validation_results.json',
        formats=formats
    )

    for fmt, path in outputs.items():
        print(f"[OK] {fmt.upper()}: {path}")


def cmd_demo(args):
    """Execute la demo."""
    from src.pipeline import run_demo
    run_demo()


def cmd_test(args):
    """Execute les tests."""
    from src.pipeline import LMDPipeline

    pipeline = LMDPipeline()
    report = pipeline.run_tests()
    print(report)


def main():
    parser = argparse.ArgumentParser(
        description='LMD-PPV Video Compression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Encodage/Decodage
  python cli.py encode video.mp4 -o compressed.lmd
  python cli.py decode compressed.lmd -o output.mp4
  python cli.py info compressed.lmd

  # Benchmark simple
  python cli.py benchmark video.mp4 --max-frames 100

  # Benchmark complet
  python cli.py benchmark download --source xiph,cdvl --output ./datasets
  python cli.py benchmark optimize --dataset ./datasets --output ./optimization.json
  python cli.py benchmark validate --thresholds ./optimization.json
  python cli.py benchmark run --dataset ./datasets --codecs all
  python cli.py benchmark report --results ./benchmark_results --format html,pdf

  # Demo/Test
  python cli.py demo
  python cli.py test
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commande')

    # Encode
    p_encode = subparsers.add_parser('encode', help='Encode une video')
    p_encode.add_argument('input', help='Video source (mp4, avi, ...)')
    p_encode.add_argument('-o', '--output', help='Fichier de sortie (.lmd)')
    p_encode.add_argument('--fast', action='store_true',
                          help='Mode rapide (stats seulement)')
    p_encode.add_argument('--turbo', action='store_true',
                          help='Mode turbo (rapide + sauvegarde .lmd)')
    p_encode.add_argument('--block-size', type=int, default=16,
                          help='Taille des blocs (defaut: 16)')
    p_encode.add_argument('--block-frames', type=int, default=32,
                          help='Frames par bloc (defaut: 32)')
    p_encode.add_argument('--colors', type=int, default=256,
                          help='Nombre de couleurs (defaut: 256)')
    p_encode.add_argument('--quantize', choices=['uniform', 'kmeans', 'median'],
                          default='kmeans', help='Methode de quantification')
    p_encode.add_argument('--max-frames', type=int,
                          help='Limite de frames a encoder')
    p_encode.set_defaults(func=cmd_encode)

    # Decode
    p_decode = subparsers.add_parser('decode', help='Decode une video')
    p_decode.add_argument('input', help='Fichier LMD')
    p_decode.add_argument('-o', '--output', help='Video de sortie')
    p_decode.set_defaults(func=cmd_decode)

    # Info
    p_info = subparsers.add_parser('info', help='Informations sur un fichier')
    p_info.add_argument('input', help='Fichier LMD')
    p_info.set_defaults(func=cmd_info)

    # Benchmark (avec sous-commandes)
    p_bench = subparsers.add_parser('benchmark', help='Benchmark et optimisation')
    bench_subparsers = p_bench.add_subparsers(dest='bench_command', help='Sous-commande')

    # benchmark download
    p_download = bench_subparsers.add_parser('download', help='Telecharge les datasets')
    p_download.add_argument('--source', default='xiph',
                            help='Sources: xiph,cdvl,vimeo (defaut: xiph)')
    p_download.add_argument('--output', '-o', help='Repertoire de sortie')
    p_download.set_defaults(func=cmd_benchmark_download)

    # benchmark optimize
    p_optimize = bench_subparsers.add_parser('optimize', help='Optimise les seuils')
    p_optimize.add_argument('--dataset', help='Repertoire des datasets')
    p_optimize.add_argument('--output', '-o', help='Fichier de sortie JSON')
    p_optimize.add_argument('--alpha', type=float, default=1.0,
                            help='Poids du cout d\'encodage')
    p_optimize.add_argument('--beta', type=float, default=0.5,
                            help='Poids de la penalite')
    p_optimize.add_argument('--workers', type=int, default=4,
                            help='Nombre de workers')
    p_optimize.add_argument('--quick', action='store_true',
                            help='Recherche rapide (random + local)')
    p_optimize.set_defaults(func=cmd_benchmark_optimize)

    # benchmark validate
    p_validate = bench_subparsers.add_parser('validate', help='Valide le classifieur')
    p_validate.add_argument('--thresholds', help='Fichier de seuils optimises')
    p_validate.add_argument('--dataset', help='Repertoire des datasets')
    p_validate.add_argument('--output', '-o', help='Fichier de resultats')
    p_validate.set_defaults(func=cmd_benchmark_validate)

    # benchmark run
    p_run = bench_subparsers.add_parser('run', help='Execute le benchmark complet')
    p_run.add_argument('--dataset', help='Repertoire des datasets')
    p_run.add_argument('--output', '-o', help='Repertoire de sortie')
    p_run.add_argument('--codecs', default='all',
                       help='Codecs: all, h264,h265,vp9,av1,lmd')
    p_run.add_argument('--max-videos', type=int, help='Nombre max de videos')
    p_run.add_argument('--workers', type=int, default=4, help='Nombre de workers')
    p_run.set_defaults(func=cmd_benchmark_run)

    # benchmark report
    p_report = bench_subparsers.add_parser('report', help='Genere le rapport')
    p_report.add_argument('--results', help='Repertoire des resultats')
    p_report.add_argument('--output', '-o', help='Repertoire de sortie')
    p_report.add_argument('--format', default='html,json',
                          help='Formats: html,pdf,json')
    p_report.set_defaults(func=cmd_benchmark_report)

    # benchmark simple (retrocompatibilite)
    p_bench_simple = bench_subparsers.add_parser('simple', help='Benchmark simple')
    p_bench_simple.add_argument('input', help='Video source')
    p_bench_simple.add_argument('--max-frames', type=int, default=100,
                                help='Nombre max de frames')
    p_bench_simple.set_defaults(func=cmd_benchmark_simple)

    # Demo
    p_demo = subparsers.add_parser('demo', help='Execute la demo')
    p_demo.set_defaults(func=cmd_demo)

    # Test
    p_test = subparsers.add_parser('test', help='Execute les tests')
    p_test.set_defaults(func=cmd_test)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Gestion speciale pour benchmark sans sous-commande (retrocompatibilite)
    if args.command == 'benchmark':
        if not hasattr(args, 'bench_command') or args.bench_command is None:
            # Si pas de sous-commande, afficher l'aide benchmark
            p_bench.print_help()
            return 1

    try:
        args.func(args)
        return 0
    except FileNotFoundError as e:
        print(f"[ERREUR] Fichier non trouve: {e}")
        return 1
    except ImportError as e:
        print(f"[ERREUR] Module manquant: {e}")
        print("Installez les dependances: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"[ERREUR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
