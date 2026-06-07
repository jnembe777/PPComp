"""
results.py - Stockage et analyse des resultats
===============================================

Gestion persistante des resultats de benchmark.

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


@dataclass
class ResultEntry:
    """Entree de resultat individuelle."""
    timestamp: str
    video_name: str
    codec_name: str
    quality_param: str

    # Compression
    input_size: int
    output_size: int
    compression_ratio: float
    bitrate_kbps: float

    # Qualite
    psnr: float
    ssim: float

    # Performance
    encode_time: float
    decode_time: float
    encode_fps: float

    # Video info
    width: int = 0
    height: int = 0
    n_frames: int = 0
    duration: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'video_name': self.video_name,
            'codec_name': self.codec_name,
            'quality_param': self.quality_param,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'compression_ratio': self.compression_ratio,
            'bitrate_kbps': self.bitrate_kbps,
            'psnr': self.psnr,
            'ssim': self.ssim,
            'encode_time': self.encode_time,
            'decode_time': self.decode_time,
            'encode_fps': self.encode_fps,
            'width': self.width,
            'height': self.height,
            'n_frames': self.n_frames,
            'duration': self.duration,
        }


class ResultsStore:
    """
    Stockage persistant des resultats.

    Permet d'accumuler les resultats au fil du temps
    et de les analyser.
    """

    def __init__(self, store_path: Path):
        """
        Initialise le store.

        Args:
            store_path: Chemin du fichier de stockage
        """
        self.store_path = Path(store_path)
        self.entries: List[ResultEntry] = []

        if self.store_path.exists():
            self.load()

    def add(self, entry: ResultEntry) -> None:
        """Ajoute une entree."""
        self.entries.append(entry)

    def add_from_benchmark_result(self, result) -> None:
        """Ajoute depuis un BenchmarkResult."""
        entry = ResultEntry(
            timestamp=datetime.now().isoformat(),
            video_name=result.video_name,
            codec_name=result.codec_name,
            quality_param=str(result.quality_param),
            input_size=result.encode_result.input_size_bytes,
            output_size=result.encode_result.output_size_bytes,
            compression_ratio=result.encode_result.compression_ratio,
            bitrate_kbps=result.encode_result.bitrate_kbps,
            psnr=result.quality_metrics.psnr if result.quality_metrics else 0,
            ssim=result.quality_metrics.ssim if result.quality_metrics else 0,
            encode_time=result.encode_result.encode_time_sec,
            decode_time=0,
            encode_fps=result.encode_result.encode_fps,
            width=result.encode_result.width,
            height=result.encode_result.height,
            n_frames=result.encode_result.n_frames,
            duration=result.encode_result.duration_sec,
        )
        self.add(entry)

    def save(self) -> None:
        """Sauvegarde les resultats."""
        data = [e.to_dict() for e in self.entries]
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        """Charge les resultats."""
        if not self.store_path.exists():
            return

        data = json.loads(self.store_path.read_text())
        self.entries = [
            ResultEntry(**d) for d in data
        ]

    def to_dataframe(self):
        """Convertit en DataFrame pandas (si disponible)."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas requis pour to_dataframe(). Installez avec: pip install pandas")
        return pd.DataFrame([e.to_dict() for e in self.entries])

    def to_list(self) -> List[Dict]:
        """Convertit en liste de dictionnaires (alternative sans pandas)."""
        return [e.to_dict() for e in self.entries]

    def filter(
        self,
        video_name: Optional[str] = None,
        codec_name: Optional[str] = None,
        quality_param: Optional[str] = None
    ) -> List[ResultEntry]:
        """Filtre les resultats."""
        results = self.entries

        if video_name:
            results = [e for e in results if e.video_name == video_name]
        if codec_name:
            results = [e for e in results if e.codec_name == codec_name]
        if quality_param:
            results = [e for e in results if e.quality_param == quality_param]

        return results

    def get_summary_by_codec(self) -> Dict[str, Dict]:
        """Resume par codec."""
        if not self.entries:
            return {}

        # Grouper par codec sans pandas
        by_codec = {}
        for e in self.entries:
            if e.codec_name not in by_codec:
                by_codec[e.codec_name] = []
            by_codec[e.codec_name].append(e)

        summary = {}
        for codec, entries in by_codec.items():
            n = len(entries)
            summary[codec] = {
                'n_videos': n,
                'avg_compression_ratio': sum(e.compression_ratio for e in entries) / n,
                'avg_psnr': sum(e.psnr for e in entries) / n,
                'avg_ssim': sum(e.ssim for e in entries) / n,
                'avg_encode_fps': sum(e.encode_fps for e in entries) / n,
                'avg_bitrate_kbps': sum(e.bitrate_kbps for e in entries) / n,
            }

        return summary

    def get_rate_distortion_points(
        self,
        codec_name: str,
        metric: str = 'psnr'
    ) -> Tuple[List[float], List[float]]:
        """
        Retourne les points rate-distortion pour un codec.

        Args:
            codec_name: Nom du codec
            metric: 'psnr' ou 'ssim'

        Returns:
            (bitrates, qualities)
        """
        entries = self.filter(codec_name=codec_name)

        bitrates = [e.bitrate_kbps for e in entries]
        if metric == 'psnr':
            qualities = [e.psnr for e in entries]
        else:
            qualities = [e.ssim for e in entries]

        # Trier par bitrate
        sorted_pairs = sorted(zip(bitrates, qualities))
        bitrates = [p[0] for p in sorted_pairs]
        qualities = [p[1] for p in sorted_pairs]

        return bitrates, qualities


class ResultsAnalyzer:
    """
    Analyseur de resultats.

    Genere des statistiques et comparaisons.
    """

    def __init__(self, store: ResultsStore):
        self.store = store

    def compare_codecs(
        self,
        codecs: List[str],
        metric: str = 'psnr'
    ) -> Dict:
        """
        Compare plusieurs codecs.

        Args:
            codecs: Liste des codecs a comparer
            metric: Metrique de comparaison

        Returns:
            Comparaison detaillee
        """
        entries = self.store.entries
        if not entries:
            return {}

        comparison = {}

        for codec in codecs:
            codec_entries = [e for e in entries if e.codec_name == codec]
            if not codec_entries:
                continue

            if metric == 'psnr':
                values = [e.psnr for e in codec_entries]
            elif metric == 'ssim':
                values = [e.ssim for e in codec_entries]
            elif metric == 'compression_ratio':
                values = [e.compression_ratio for e in codec_entries]
            else:
                values = [getattr(e, metric, 0) for e in codec_entries]

            if values:
                comparison[codec] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                }

        return comparison

    def get_best_codec_per_video(
        self,
        metric: str = 'psnr'
    ) -> Dict[str, str]:
        """
        Trouve le meilleur codec pour chaque video.

        Args:
            metric: Metrique a optimiser

        Returns:
            Dict video_name -> best_codec
        """
        entries = self.store.entries
        if not entries:
            return {}

        # Grouper par video
        by_video = {}
        for e in entries:
            if e.video_name not in by_video:
                by_video[e.video_name] = []
            by_video[e.video_name].append(e)

        best = {}
        for video, video_entries in by_video.items():
            if metric in ['psnr', 'ssim']:
                best_entry = max(video_entries, key=lambda e: getattr(e, metric, 0))
            else:
                best_entry = min(video_entries, key=lambda e: getattr(e, metric, float('inf')))
            best[video] = best_entry.codec_name

        return best

    def calculate_bd_rate(
        self,
        codec1: str,
        codec2: str
    ) -> float:
        """
        Calcule le BD-Rate entre deux codecs.

        Args:
            codec1: Codec de reference
            codec2: Codec a comparer

        Returns:
            BD-Rate (negatif = codec2 meilleur)
        """
        br1, q1 = self.store.get_rate_distortion_points(codec1)
        br2, q2 = self.store.get_rate_distortion_points(codec2)

        if len(br1) < 4 or len(br2) < 4:
            return 0.0

        from benchmark.metrics.performance import calculate_bd_rate
        return calculate_bd_rate(br1, q1, br2, q2)

    def generate_summary_table(self) -> str:
        """Genere un tableau resume."""
        summary = self.store.get_summary_by_codec()
        if not summary:
            return "No results"

        # Formatter en tableau texte
        lines = []
        header = f"{'Codec':<15} {'Ratio':>8} {'PSNR':>8} {'SSIM':>8} {'FPS':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        for codec, stats in summary.items():
            line = f"{codec:<15} {stats['avg_compression_ratio']:>8.2f} {stats['avg_psnr']:>8.2f} {stats['avg_ssim']:>8.4f} {stats['avg_encode_fps']:>8.1f}"
            lines.append(line)

        return "\n".join(lines)
