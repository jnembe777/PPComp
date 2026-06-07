#!/usr/bin/env python3
"""
check_benchmark_setup.py - Verification de l'installation
==========================================================

Verifie que tous les modules du benchmark sont correctement installes.

Usage:
    python check_benchmark_setup.py
"""

import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))


def check_module(name: str, import_path: str) -> bool:
    """Verifie qu'un module peut etre importe."""
    try:
        exec(f"from {import_path} import *")
        return True
    except ImportError as e:
        print(f"  [ERREUR] {e}")
        return False
    except Exception as e:
        print(f"  [WARN] Import partiel: {e}")
        return True


def check_dependency(name: str) -> bool:
    """Verifie qu'une dependance externe est installee."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def main():
    print("=" * 60)
    print("  VERIFICATION DE L'INSTALLATION DU BENCHMARK")
    print("=" * 60)
    print()

    errors = []
    warnings = []

    # 1. Dependances Python
    print("[1/5] Dependances Python externes...")
    deps = {
        'numpy': 'numpy',
        'cv2 (OpenCV)': 'cv2',
        'tqdm': 'tqdm',
        'requests': 'requests',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'scikit-image': 'skimage',
    }

    for name, module in deps.items():
        if check_dependency(module):
            print(f"  [OK] {name}")
        else:
            print(f"  [MANQUANT] {name}")
            if name in ['numpy', 'tqdm', 'requests']:
                errors.append(f"Dependance requise: {name}")
            else:
                warnings.append(f"Dependance optionnelle: {name}")

    # 2. Modules core
    print("\n[2/5] Modules core...")
    core_modules = [
        ('BlockFeatures', 'src.core.features'),
        ('Cartouche', 'src.core.cartouche'),
        ('ProcessType', 'src.core.process_types'),
    ]

    for name, path in core_modules:
        print(f"  Checking {name}...", end=' ')
        if check_module(name, path):
            print("[OK]")
        else:
            errors.append(f"Module core: {path}")

    # 3. Modules benchmark
    print("\n[3/5] Modules benchmark...")
    bench_modules = [
        ('BenchmarkConfig', 'benchmark.config'),
        ('DatasetDownloader', 'benchmark.datasets.downloader'),
        ('BenchmarkRunner', 'benchmark.runner'),
        ('ResultsStore', 'benchmark.results'),
        ('QualityMetrics', 'benchmark.metrics.quality'),
    ]

    for name, path in bench_modules:
        print(f"  Checking {name}...", end=' ')
        if check_module(name, path):
            print("[OK]")
        else:
            errors.append(f"Module benchmark: {path}")

    # 4. Modules optimisation/validation
    print("\n[4/5] Modules optimisation et validation...")
    opt_modules = [
        ('ThresholdConfig', 'optimization.threshold_config'),
        ('GridSearch', 'optimization.grid_search'),
        ('PredictorValidator', 'validation.predictor_vs_optimal'),
    ]

    for name, path in opt_modules:
        print(f"  Checking {name}...", end=' ')
        if check_module(name, path):
            print("[OK]")
        else:
            errors.append(f"Module optimisation: {path}")

    # 5. Modules rapports
    print("\n[5/5] Modules rapports...")
    report_modules = [
        ('ReportGenerator', 'reports.generator'),
        ('ChartGenerator', 'reports.charts'),
    ]

    for name, path in report_modules:
        print(f"  Checking {name}...", end=' ')
        if check_module(name, path):
            print("[OK]")
        else:
            errors.append(f"Module rapport: {path}")

    # 6. Outils externes
    print("\n[BONUS] Outils externes...")
    import shutil

    external_tools = ['ffmpeg', 'ffprobe']
    for tool in external_tools:
        if shutil.which(tool):
            print(f"  [OK] {tool}")
        else:
            print(f"  [MANQUANT] {tool} (requis pour le benchmark complet)")
            warnings.append(f"Outil externe: {tool}")

    # Resume
    print("\n" + "=" * 60)
    print("  RESUME")
    print("=" * 60)

    if not errors and not warnings:
        print("\n  [OK] Tous les modules sont installes correctement!")
        print("\n  Vous pouvez lancer le benchmark avec:")
        print("    python run_full_benchmark.py --quick")
        return 0
    else:
        if errors:
            print(f"\n  [ERREUR] {len(errors)} erreur(s) critique(s):")
            for e in errors:
                print(f"    - {e}")

        if warnings:
            print(f"\n  [WARN] {len(warnings)} avertissement(s):")
            for w in warnings:
                print(f"    - {w}")

        print("\n  Pour installer les dependances manquantes:")
        print("    pip install numpy opencv-python tqdm requests matplotlib pandas scikit-image")
        print("\n  Pour FFmpeg:")
        print("    Windows: winget install ffmpeg")
        print("    Linux: sudo apt install ffmpeg")
        print("    macOS: brew install ffmpeg")

        return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
