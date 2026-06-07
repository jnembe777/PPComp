"""
performance.py - Metriques de performance
==========================================

Metriques implementees:
- Compression Ratio: input_bytes / output_bytes
- Encoding Speed: frames/seconde
- Decoding Speed: frames/seconde
- Bitrate: kbps

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PerformanceMetrics:
    """Metriques de performance d'un encodage."""

    # Compression
    input_size_bytes: int
    output_size_bytes: int
    compression_ratio: float  # input / output

    # Bitrate
    bitrate_kbps: float
    bits_per_pixel: float

    # Vitesse
    encode_time_sec: float
    decode_time_sec: float
    encode_fps: float
    decode_fps: float

    # Video info
    width: int = 0
    height: int = 0
    n_frames: int = 0
    duration_sec: float = 0.0
    original_bitrate_kbps: float = 0.0

    def __post_init__(self):
        """Calculs derives."""
        if self.output_size_bytes > 0 and self.input_size_bytes > 0:
            self.compression_ratio = self.input_size_bytes / self.output_size_bytes

        if self.duration_sec > 0 and self.output_size_bytes > 0:
            self.bitrate_kbps = (self.output_size_bytes * 8 / 1000) / self.duration_sec

        pixels = self.width * self.height * self.n_frames
        if pixels > 0 and self.output_size_bytes > 0:
            self.bits_per_pixel = (self.output_size_bytes * 8) / pixels
        else:
            self.bits_per_pixel = 0.0

    @classmethod
    def from_encode_result(
        cls,
        encode_result,
        decode_time: float = 0.0
    ) -> 'PerformanceMetrics':
        """Cree les metriques depuis un EncodeResult."""
        decode_fps = encode_result.n_frames / decode_time if decode_time > 0 else 0.0

        return cls(
            input_size_bytes=encode_result.input_size_bytes,
            output_size_bytes=encode_result.output_size_bytes,
            compression_ratio=encode_result.compression_ratio,
            bitrate_kbps=encode_result.bitrate_kbps,
            bits_per_pixel=0.0,  # Calcule dans __post_init__
            encode_time_sec=encode_result.encode_time_sec,
            decode_time_sec=decode_time,
            encode_fps=encode_result.encode_fps,
            decode_fps=decode_fps,
            width=encode_result.width,
            height=encode_result.height,
            n_frames=encode_result.n_frames,
            duration_sec=encode_result.duration_sec
        )

    def to_dict(self) -> Dict:
        return {
            'input_size_bytes': self.input_size_bytes,
            'output_size_bytes': self.output_size_bytes,
            'compression_ratio': self.compression_ratio,
            'bitrate_kbps': self.bitrate_kbps,
            'bits_per_pixel': self.bits_per_pixel,
            'encode_time_sec': self.encode_time_sec,
            'decode_time_sec': self.decode_time_sec,
            'encode_fps': self.encode_fps,
            'decode_fps': self.decode_fps,
            'width': self.width,
            'height': self.height,
            'n_frames': self.n_frames,
            'duration_sec': self.duration_sec,
        }

    @property
    def size_reduction_percent(self) -> float:
        """Pourcentage de reduction de taille."""
        if self.input_size_bytes == 0:
            return 0.0
        return (1 - self.output_size_bytes / self.input_size_bytes) * 100

    @property
    def encoding_realtime_factor(self) -> float:
        """Facteur temps reel (1.0 = temps reel)."""
        if self.encode_time_sec == 0:
            return 0.0
        return self.duration_sec / self.encode_time_sec


@dataclass
class AggregatePerformance:
    """Metriques de performance agregees sur plusieurs videos."""

    n_videos: int
    total_frames: int
    total_input_bytes: int
    total_output_bytes: int
    total_encode_time: float
    total_decode_time: float

    # Moyennes
    avg_compression_ratio: float
    avg_bitrate_kbps: float
    avg_encode_fps: float
    avg_decode_fps: float

    # Min/Max
    min_compression_ratio: float
    max_compression_ratio: float
    min_encode_fps: float
    max_encode_fps: float

    # Ecarts-types
    std_compression_ratio: float = 0.0
    std_bitrate_kbps: float = 0.0

    @classmethod
    def from_metrics_list(
        cls,
        metrics_list: List[PerformanceMetrics]
    ) -> 'AggregatePerformance':
        """Calcule les metriques agregees."""
        if not metrics_list:
            return cls(
                n_videos=0,
                total_frames=0,
                total_input_bytes=0,
                total_output_bytes=0,
                total_encode_time=0,
                total_decode_time=0,
                avg_compression_ratio=0,
                avg_bitrate_kbps=0,
                avg_encode_fps=0,
                avg_decode_fps=0,
                min_compression_ratio=0,
                max_compression_ratio=0,
                min_encode_fps=0,
                max_encode_fps=0
            )

        ratios = [m.compression_ratio for m in metrics_list]
        bitrates = [m.bitrate_kbps for m in metrics_list]
        encode_fps = [m.encode_fps for m in metrics_list if m.encode_fps > 0]
        decode_fps = [m.decode_fps for m in metrics_list if m.decode_fps > 0]

        return cls(
            n_videos=len(metrics_list),
            total_frames=sum(m.n_frames for m in metrics_list),
            total_input_bytes=sum(m.input_size_bytes for m in metrics_list),
            total_output_bytes=sum(m.output_size_bytes for m in metrics_list),
            total_encode_time=sum(m.encode_time_sec for m in metrics_list),
            total_decode_time=sum(m.decode_time_sec for m in metrics_list),
            avg_compression_ratio=np.mean(ratios),
            avg_bitrate_kbps=np.mean(bitrates),
            avg_encode_fps=np.mean(encode_fps) if encode_fps else 0,
            avg_decode_fps=np.mean(decode_fps) if decode_fps else 0,
            min_compression_ratio=np.min(ratios),
            max_compression_ratio=np.max(ratios),
            min_encode_fps=np.min(encode_fps) if encode_fps else 0,
            max_encode_fps=np.max(encode_fps) if encode_fps else 0,
            std_compression_ratio=np.std(ratios),
            std_bitrate_kbps=np.std(bitrates)
        )

    def to_dict(self) -> Dict:
        return {
            'n_videos': self.n_videos,
            'total_frames': self.total_frames,
            'total_input_mb': self.total_input_bytes / (1024 * 1024),
            'total_output_mb': self.total_output_bytes / (1024 * 1024),
            'total_encode_time_sec': self.total_encode_time,
            'total_decode_time_sec': self.total_decode_time,
            'avg_compression_ratio': self.avg_compression_ratio,
            'avg_bitrate_kbps': self.avg_bitrate_kbps,
            'avg_encode_fps': self.avg_encode_fps,
            'avg_decode_fps': self.avg_decode_fps,
            'min_compression_ratio': self.min_compression_ratio,
            'max_compression_ratio': self.max_compression_ratio,
            'std_compression_ratio': self.std_compression_ratio,
        }

    @property
    def overall_compression_ratio(self) -> float:
        """Ratio de compression global."""
        if self.total_output_bytes == 0:
            return 0.0
        return self.total_input_bytes / self.total_output_bytes

    @property
    def overall_encode_fps(self) -> float:
        """FPS d'encodage global."""
        if self.total_encode_time == 0:
            return 0.0
        return self.total_frames / self.total_encode_time

    @property
    def overall_decode_fps(self) -> float:
        """FPS de decodage global."""
        if self.total_decode_time == 0:
            return 0.0
        return self.total_frames / self.total_decode_time


def calculate_bd_rate(
    bitrates1: List[float],
    qualities1: List[float],
    bitrates2: List[float],
    qualities2: List[float]
) -> float:
    """
    Calcule le BD-Rate (Bjontegaard Delta Rate).

    Mesure le pourcentage de difference de bitrate entre deux codecs
    a qualite egale.

    Args:
        bitrates1: Bitrates du codec 1 (kbps)
        qualities1: Qualites du codec 1 (PSNR ou SSIM)
        bitrates2: Bitrates du codec 2 (kbps)
        qualities2: Qualites du codec 2

    Returns:
        BD-Rate en pourcentage (negatif = codec 2 meilleur)
    """
    if len(bitrates1) < 4 or len(bitrates2) < 4:
        return 0.0

    # Convertir en log
    log_r1 = np.log10(bitrates1)
    log_r2 = np.log10(bitrates2)

    # Fit polynomial de degre 3
    try:
        p1 = np.polyfit(qualities1, log_r1, 3)
        p2 = np.polyfit(qualities2, log_r2, 3)
    except np.linalg.LinAlgError:
        return 0.0

    # Plage commune de qualite
    min_q = max(min(qualities1), min(qualities2))
    max_q = min(max(qualities1), max(qualities2))

    if min_q >= max_q:
        return 0.0

    # Integrale
    def polyval_integral(p, a, b):
        p_int = np.polyint(p)
        return np.polyval(p_int, b) - np.polyval(p_int, a)

    avg1 = polyval_integral(p1, min_q, max_q) / (max_q - min_q)
    avg2 = polyval_integral(p2, min_q, max_q) / (max_q - min_q)

    # BD-Rate
    bd_rate = (10 ** (avg2 - avg1) - 1) * 100

    return bd_rate
