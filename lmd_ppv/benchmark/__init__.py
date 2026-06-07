"""
benchmark - Module de Benchmark LMD-PPV
========================================

Comparaison LMD-PPV vs H.264/H.265/VP9/AV1 sur datasets standards.

Modules:
- datasets: Telechargement et gestion des datasets video
- codecs: Wrappers FFmpeg pour tous les codecs
- metrics: PSNR, SSIM, ratio compression, vitesse
- runner: Orchestrateur de benchmark
- results: Stockage et analyse des resultats

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from .config import BenchmarkConfig

__all__ = ['BenchmarkConfig']
