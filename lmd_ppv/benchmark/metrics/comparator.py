"""
comparator.py - Comparaison cross-codec
========================================

Compare les performances de differents codecs sur un meme dataset.

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import json

from .quality import QualityMetrics
from .performance import PerformanceMetrics, AggregatePerformance, calculate_bd_rate


@dataclass
class CodecResult:
    """Resultats d'un codec sur une video."""
    codec_name: str
    video_name: str
    quality_param: any
    quality: QualityMetrics
    performance: PerformanceMetrics
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'codec_name': self.codec_name,
            'video_name': self.video_name,
            'quality_param': str(self.quality_param),
            'quality': self.quality.to_dict(),
            'performance': self.performance.to_dict(),
            'success': self.success,
            'error': self.error
        }


@dataclass
class ComparisonResult:
    """Resultat de comparaison entre codecs."""
    video_name: str
    codecs: Dict[str, CodecResult]

    # Meilleur codec par metrique
    best_psnr: Optional[str] = None
    best_ssim: Optional[str] = None
    best_compression: Optional[str] = None
    best_encode_speed: Optional[str] = None

    def __post_init__(self):
        """Determine les meilleurs codecs."""
        if not self.codecs:
            return

        valid = {k: v for k, v in self.codecs.items() if v.success}

        if valid:
            self.best_psnr = max(valid, key=lambda k: valid[k].quality.psnr)
            self.best_ssim = max(valid, key=lambda k: valid[k].quality.ssim)
            self.best_compression = max(
                valid,
                key=lambda k: valid[k].performance.compression_ratio
            )
            self.best_encode_speed = max(
                valid,
                key=lambda k: valid[k].performance.encode_fps
            )

    def to_dict(self) -> Dict:
        return {
            'video_name': self.video_name,
            'codecs': {k: v.to_dict() for k, v in self.codecs.items()},
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'best_compression': self.best_compression,
            'best_encode_speed': self.best_encode_speed
        }


@dataclass
class AggregateComparison:
    """Comparaison agregee sur tout le dataset."""
    n_videos: int
    codec_stats: Dict[str, AggregatePerformance]
    codec_quality: Dict[str, Dict[str, float]]  # codec -> {psnr_avg, ssim_avg, ...}

    # Classements
    ranking_psnr: List[str] = field(default_factory=list)
    ranking_ssim: List[str] = field(default_factory=list)
    ranking_compression: List[str] = field(default_factory=list)
    ranking_speed: List[str] = field(default_factory=list)

    # BD-Rates vs reference
    bd_rates: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'n_videos': self.n_videos,
            'codec_stats': {k: v.to_dict() for k, v in self.codec_stats.items()},
            'codec_quality': self.codec_quality,
            'ranking_psnr': self.ranking_psnr,
            'ranking_ssim': self.ranking_ssim,
            'ranking_compression': self.ranking_compression,
            'ranking_speed': self.ranking_speed,
            'bd_rates': self.bd_rates
        }


class CodecComparator:
    """
    Compare les performances de plusieurs codecs.

    Calcule les metriques de qualite et performance pour chaque codec
    et genere des comparaisons agregees.
    """

    def __init__(self):
        self.results: Dict[str, Dict[str, CodecResult]] = {}  # video -> codec -> result
        self.reference_codec: Optional[str] = None

    def add_result(self, result: CodecResult) -> None:
        """Ajoute un resultat de codec."""
        if result.video_name not in self.results:
            self.results[result.video_name] = {}
        self.results[result.video_name][result.codec_name] = result

    def set_reference(self, codec_name: str) -> None:
        """Definit le codec de reference pour les BD-Rates."""
        self.reference_codec = codec_name

    def get_comparison(self, video_name: str) -> Optional[ComparisonResult]:
        """Retourne la comparaison pour une video."""
        if video_name not in self.results:
            return None

        return ComparisonResult(
            video_name=video_name,
            codecs=self.results[video_name]
        )

    def get_aggregate(self) -> AggregateComparison:
        """Calcule les statistiques agregees."""
        # Collecter les resultats par codec
        codec_results: Dict[str, List[CodecResult]] = {}

        for video_results in self.results.values():
            for codec_name, result in video_results.items():
                if result.success:
                    if codec_name not in codec_results:
                        codec_results[codec_name] = []
                    codec_results[codec_name].append(result)

        # Calculer les statistiques
        codec_stats = {}
        codec_quality = {}

        for codec_name, results in codec_results.items():
            # Performance agregee
            perf_list = [r.performance for r in results]
            codec_stats[codec_name] = AggregatePerformance.from_metrics_list(perf_list)

            # Qualite moyenne
            psnr_values = [r.quality.psnr for r in results if r.quality.psnr > 0]
            ssim_values = [r.quality.ssim for r in results if r.quality.ssim > 0]

            codec_quality[codec_name] = {
                'psnr_avg': np.mean(psnr_values) if psnr_values else 0,
                'psnr_std': np.std(psnr_values) if psnr_values else 0,
                'ssim_avg': np.mean(ssim_values) if ssim_values else 0,
                'ssim_std': np.std(ssim_values) if ssim_values else 0,
                'n_videos': len(results)
            }

        # Classements
        ranking_psnr = sorted(
            codec_quality.keys(),
            key=lambda k: codec_quality[k]['psnr_avg'],
            reverse=True
        )
        ranking_ssim = sorted(
            codec_quality.keys(),
            key=lambda k: codec_quality[k]['ssim_avg'],
            reverse=True
        )
        ranking_compression = sorted(
            codec_stats.keys(),
            key=lambda k: codec_stats[k].avg_compression_ratio,
            reverse=True
        )
        ranking_speed = sorted(
            codec_stats.keys(),
            key=lambda k: codec_stats[k].avg_encode_fps,
            reverse=True
        )

        # BD-Rates
        bd_rates = self._calculate_bd_rates(codec_results)

        return AggregateComparison(
            n_videos=len(self.results),
            codec_stats=codec_stats,
            codec_quality=codec_quality,
            ranking_psnr=ranking_psnr,
            ranking_ssim=ranking_ssim,
            ranking_compression=ranking_compression,
            ranking_speed=ranking_speed,
            bd_rates=bd_rates
        )

    def _calculate_bd_rates(
        self,
        codec_results: Dict[str, List[CodecResult]]
    ) -> Dict[str, float]:
        """Calcule les BD-Rates par rapport au codec de reference."""
        bd_rates = {}

        if not self.reference_codec or self.reference_codec not in codec_results:
            return bd_rates

        ref_results = codec_results[self.reference_codec]
        ref_bitrates = [r.performance.bitrate_kbps for r in ref_results]
        ref_psnr = [r.quality.psnr for r in ref_results]

        for codec_name, results in codec_results.items():
            if codec_name == self.reference_codec:
                bd_rates[codec_name] = 0.0
                continue

            bitrates = [r.performance.bitrate_kbps for r in results]
            psnr = [r.quality.psnr for r in results]

            bd_rate = calculate_bd_rate(ref_bitrates, ref_psnr, bitrates, psnr)
            bd_rates[codec_name] = bd_rate

        return bd_rates

    def get_rate_distortion_data(
        self,
        metric: str = 'psnr'
    ) -> Dict[str, Tuple[List[float], List[float]]]:
        """
        Retourne les donnees rate-distortion pour chaque codec.

        Args:
            metric: 'psnr' ou 'ssim'

        Returns:
            Dict codec -> (bitrates, qualities)
        """
        data = {}

        # Collecter par codec
        codec_points: Dict[str, List[Tuple[float, float]]] = {}

        for video_results in self.results.values():
            for codec_name, result in video_results.items():
                if not result.success:
                    continue

                bitrate = result.performance.bitrate_kbps
                quality = result.quality.psnr if metric == 'psnr' else result.quality.ssim

                if codec_name not in codec_points:
                    codec_points[codec_name] = []
                codec_points[codec_name].append((bitrate, quality))

        # Trier par bitrate et separer
        for codec_name, points in codec_points.items():
            points.sort(key=lambda x: x[0])
            bitrates = [p[0] for p in points]
            qualities = [p[1] for p in points]
            data[codec_name] = (bitrates, qualities)

        return data

    def save(self, path: Path) -> None:
        """Sauvegarde les resultats."""
        data = {
            'reference_codec': self.reference_codec,
            'results': {
                video: {
                    codec: result.to_dict()
                    for codec, result in codecs.items()
                }
                for video, codecs in self.results.items()
            }
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> 'CodecComparator':
        """Charge les resultats."""
        data = json.loads(path.read_text())

        comparator = cls()
        comparator.reference_codec = data.get('reference_codec')

        for video, codecs in data.get('results', {}).items():
            for codec, result_data in codecs.items():
                result = CodecResult(
                    codec_name=result_data['codec_name'],
                    video_name=result_data['video_name'],
                    quality_param=result_data['quality_param'],
                    quality=QualityMetrics(**result_data['quality']),
                    performance=PerformanceMetrics(**result_data['performance']),
                    success=result_data['success'],
                    error=result_data.get('error')
                )
                comparator.add_result(result)

        return comparator
