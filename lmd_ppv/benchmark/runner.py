"""
runner.py - Orchestrateur principal du benchmark
=================================================

Execute le benchmark complet:
- Pour chaque video
- Pour chaque codec (H.264, H.265, VP9, AV1, LMD-PPV)
- Pour chaque niveau de qualite

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import BenchmarkConfig
from .codecs import CodecBase, CodecFactory, EncodeResult
from .metrics import QualityMetrics, PerformanceMetrics, compute_quality
from .metrics.comparator import CodecComparator, CodecResult
from .datasets.downloader import VideoInfo


@dataclass
class BenchmarkRun:
    """Configuration d'un run de benchmark."""
    video: VideoInfo
    codec: CodecBase
    quality: any
    output_dir: Path


@dataclass
class BenchmarkResult:
    """Resultat d'un benchmark complet."""
    video_name: str
    codec_name: str
    quality_param: any

    # Encodage
    encode_result: EncodeResult

    # Metriques
    quality_metrics: Optional[QualityMetrics] = None
    performance_metrics: Optional[PerformanceMetrics] = None

    # Temps total
    total_time_sec: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'video_name': self.video_name,
            'codec_name': self.codec_name,
            'quality_param': str(self.quality_param),
            'encode_success': self.encode_result.success,
            'encode_time_sec': self.encode_result.encode_time_sec,
            'input_size_bytes': self.encode_result.input_size_bytes,
            'output_size_bytes': self.encode_result.output_size_bytes,
            'compression_ratio': self.encode_result.compression_ratio,
            'bitrate_kbps': self.encode_result.bitrate_kbps,
            'psnr': self.quality_metrics.psnr if self.quality_metrics else 0,
            'ssim': self.quality_metrics.ssim if self.quality_metrics else 0,
            'total_time_sec': self.total_time_sec,
        }


@dataclass
class BenchmarkSummary:
    """Resume du benchmark."""
    n_videos: int
    n_codecs: int
    n_quality_levels: int
    total_runs: int
    successful_runs: int
    total_time_sec: float

    results: List[BenchmarkResult] = field(default_factory=list)
    comparator: Optional[CodecComparator] = None

    def to_dict(self) -> Dict:
        return {
            'n_videos': self.n_videos,
            'n_codecs': self.n_codecs,
            'n_quality_levels': self.n_quality_levels,
            'total_runs': self.total_runs,
            'successful_runs': self.successful_runs,
            'success_rate': self.successful_runs / self.total_runs if self.total_runs > 0 else 0,
            'total_time_sec': self.total_time_sec,
            'results': [r.to_dict() for r in self.results],
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))


class BenchmarkRunner:
    """
    Orchestrateur du benchmark.

    Execute les benchmarks sur tous les codecs et videos.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """
        Initialise le runner.

        Args:
            config: Configuration du benchmark
            progress_callback: Callback (current, total, message)
        """
        self.config = config
        self.progress_callback = progress_callback

        # Creer les codecs
        self.codecs = CodecFactory.create_all(include_lmd=True)

        # Repertoires
        self.output_dir = config.output_dir
        self.temp_dir = config.cache_dir / 'temp'
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def run_single(
        self,
        video: VideoInfo,
        codec: CodecBase,
        quality: any,
        compute_metrics: bool = True
    ) -> BenchmarkResult:
        """
        Execute un benchmark unique.

        Args:
            video: Video a tester
            codec: Codec a utiliser
            quality: Parametre de qualite
            compute_metrics: Calculer PSNR/SSIM

        Returns:
            BenchmarkResult
        """
        start_time = time.time()

        # Chemins
        video_path = Path(video.path)
        output_ext = codec.get_output_extension()
        output_name = f"{video.name}_{codec.name}_{quality}{output_ext}"
        output_path = self.temp_dir / output_name

        # Encoder
        encode_result = codec.encode(video_path, output_path, quality)

        quality_metrics = None
        if encode_result.success and compute_metrics:
            # Calculer les metriques de qualite
            try:
                # Decoder vers un format comparable
                decoded_path = self.temp_dir / f"{output_name}_decoded.y4m"

                if codec.name == 'LMD-PPV':
                    # Decoder LMD vers frames
                    decode_result = codec.decode(output_path, decoded_path)
                    if decode_result.success:
                        quality_metrics = compute_quality(
                            video_path, decoded_path, method='numpy'
                        )
                else:
                    # FFmpeg decode
                    quality_metrics = compute_quality(
                        video_path, output_path, method='ffmpeg'
                    )

                # Nettoyer
                if decoded_path.exists():
                    decoded_path.unlink()

            except Exception as e:
                quality_metrics = QualityMetrics(psnr=0, ssim=0)

        total_time = time.time() - start_time

        # Nettoyer le fichier encode
        if output_path.exists():
            output_path.unlink()

        return BenchmarkResult(
            video_name=video.name,
            codec_name=codec.name,
            quality_param=quality,
            encode_result=encode_result,
            quality_metrics=quality_metrics,
            total_time_sec=total_time
        )

    def run_video(
        self,
        video: VideoInfo,
        codecs: Optional[Dict[str, CodecBase]] = None,
        qualities: Optional[Dict[str, List]] = None
    ) -> List[BenchmarkResult]:
        """
        Execute le benchmark sur une video pour tous les codecs.

        Args:
            video: Video a tester
            codecs: Codecs a utiliser (defaut: tous)
            qualities: Niveaux de qualite par codec

        Returns:
            Liste de BenchmarkResult
        """
        codecs = codecs or self.codecs

        # Qualites par defaut
        default_qualities = {
            'h264': [18, 23, 28],
            'h265': [18, 23, 28],
            'vp9': [20, 30, 40],
            'av1': [25, 35, 45],
            'lmd': [None],  # Adaptatif
        }
        qualities = qualities or default_qualities

        results = []

        for codec_name, codec in codecs.items():
            codec_qualities = qualities.get(codec_name.lower(), [None])

            for quality in codec_qualities:
                result = self.run_single(video, codec, quality)
                results.append(result)

        return results

    def run_all(
        self,
        videos: List[VideoInfo],
        codecs: Optional[Dict[str, CodecBase]] = None,
        parallel: bool = True
    ) -> BenchmarkSummary:
        """
        Execute le benchmark complet.

        Args:
            videos: Liste des videos
            codecs: Codecs a utiliser
            parallel: Execution parallele

        Returns:
            BenchmarkSummary
        """
        codecs = codecs or self.codecs
        start_time = time.time()

        # Limiter si configure
        if self.config.max_videos:
            videos = videos[:self.config.max_videos]

        all_results = []
        comparator = CodecComparator()
        comparator.set_reference('h265')  # H.265 comme reference

        total_runs = len(videos) * sum(
            len([18, 23, 28]) if 'h264' in c.lower() or 'h265' in c.lower()
            else len([20, 30, 40]) if 'vp9' in c.lower()
            else len([25, 35, 45]) if 'av1' in c.lower()
            else 1
            for c in codecs.keys()
        )

        current = 0

        for video in tqdm(videos, desc="Benchmarking videos"):
            video_results = self.run_video(video, codecs)
            all_results.extend(video_results)

            # Ajouter au comparateur
            for result in video_results:
                if result.encode_result.success:
                    perf = PerformanceMetrics.from_encode_result(result.encode_result)
                    codec_result = CodecResult(
                        codec_name=result.codec_name,
                        video_name=result.video_name,
                        quality_param=result.quality_param,
                        quality=result.quality_metrics or QualityMetrics(psnr=0, ssim=0),
                        performance=perf,
                        success=True
                    )
                    comparator.add_result(codec_result)

            current += len(video_results)
            if self.progress_callback:
                self.progress_callback(current, total_runs, f"Completed {video.name}")

        total_time = time.time() - start_time
        successful = sum(1 for r in all_results if r.encode_result.success)

        return BenchmarkSummary(
            n_videos=len(videos),
            n_codecs=len(codecs),
            n_quality_levels=3,
            total_runs=len(all_results),
            successful_runs=successful,
            total_time_sec=total_time,
            results=all_results,
            comparator=comparator
        )

    def run_lmd_only(
        self,
        videos: List[VideoInfo],
        configurations: Optional[List[Dict]] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark uniquement LMD-PPV avec differentes configurations.

        Args:
            videos: Videos a tester
            configurations: Configurations LMD a tester

        Returns:
            Liste de BenchmarkResult
        """
        from .codecs.lmd_wrapper import LMDCodec

        default_configs = [
            {'block_size': 8, 'n_colors': 64},
            {'block_size': 16, 'n_colors': 128},
            {'block_size': 16, 'n_colors': 256},
            {'block_size': 32, 'n_colors': 256},
        ]
        configurations = configurations or default_configs

        results = []

        for video in tqdm(videos, desc="LMD-PPV Benchmark"):
            for config in configurations:
                codec = LMDCodec(**config)
                result = self.run_single(video, codec, config)
                results.append(result)

        return results


def run_benchmark_pipeline(
    videos: List[VideoInfo],
    output_dir: Path,
    config: Optional[BenchmarkConfig] = None,
    codecs: Optional[List[str]] = None
) -> BenchmarkSummary:
    """
    Pipeline complet de benchmark.

    Args:
        videos: Liste des videos
        output_dir: Repertoire de sortie
        config: Configuration (optionnel)
        codecs: Liste des codecs a tester

    Returns:
        BenchmarkSummary
    """
    config = config or BenchmarkConfig(output_dir=output_dir)
    runner = BenchmarkRunner(config)

    # Filtrer les codecs
    if codecs:
        runner.codecs = {
            k: v for k, v in runner.codecs.items()
            if k.lower() in [c.lower() for c in codecs]
        }

    print(f"Starting benchmark with {len(videos)} videos and {len(runner.codecs)} codecs")
    print(f"Codecs: {list(runner.codecs.keys())}")

    # Executer
    summary = runner.run_all(videos)

    # Sauvegarder
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.save(output_dir / 'benchmark_results.json')

    if summary.comparator:
        summary.comparator.save(output_dir / 'comparison_results.json')

    print(f"\nBenchmark complete!")
    print(f"  Videos: {summary.n_videos}")
    print(f"  Total runs: {summary.total_runs}")
    print(f"  Successful: {summary.successful_runs} ({summary.successful_runs/summary.total_runs:.1%})")
    print(f"  Total time: {summary.total_time_sec/60:.1f} minutes")
    print(f"\nResults saved to: {output_dir}")

    return summary
