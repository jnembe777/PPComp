"""
base.py - Interface abstraite pour les codecs
==============================================

Definit l'interface commune a tous les codecs (FFmpeg, LMD-PPV).

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import time


@dataclass
class EncodeResult:
    """Resultat d'un encodage."""
    success: bool
    input_path: Path
    output_path: Path
    input_size_bytes: int
    output_size_bytes: int
    encode_time_sec: float
    codec_name: str
    quality_param: Any  # CRF, QP, etc.

    # Metriques calculees
    compression_ratio: float = 0.0
    bitrate_kbps: float = 0.0
    encode_fps: float = 0.0

    # Metadonnees video
    width: int = 0
    height: int = 0
    n_frames: int = 0
    duration_sec: float = 0.0

    # Erreur eventuelle
    error: Optional[str] = None

    # Parametres additionnels
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calcul des metriques derivees."""
        if self.output_size_bytes > 0 and self.input_size_bytes > 0:
            self.compression_ratio = self.input_size_bytes / self.output_size_bytes

        if self.duration_sec > 0:
            self.bitrate_kbps = (self.output_size_bytes * 8 / 1000) / self.duration_sec

        if self.encode_time_sec > 0 and self.n_frames > 0:
            self.encode_fps = self.n_frames / self.encode_time_sec

    def to_dict(self) -> Dict:
        """Conversion en dictionnaire."""
        return {
            'success': self.success,
            'input_path': str(self.input_path),
            'output_path': str(self.output_path),
            'input_size_bytes': self.input_size_bytes,
            'output_size_bytes': self.output_size_bytes,
            'encode_time_sec': self.encode_time_sec,
            'codec_name': self.codec_name,
            'quality_param': self.quality_param,
            'compression_ratio': self.compression_ratio,
            'bitrate_kbps': self.bitrate_kbps,
            'encode_fps': self.encode_fps,
            'width': self.width,
            'height': self.height,
            'n_frames': self.n_frames,
            'duration_sec': self.duration_sec,
            'error': self.error,
            'extra_params': self.extra_params
        }


@dataclass
class DecodeResult:
    """Resultat d'un decodage."""
    success: bool
    input_path: Path
    output_path: Path
    decode_time_sec: float
    codec_name: str

    # Metriques
    n_frames: int = 0
    decode_fps: float = 0.0

    # Erreur eventuelle
    error: Optional[str] = None

    def __post_init__(self):
        if self.decode_time_sec > 0 and self.n_frames > 0:
            self.decode_fps = self.n_frames / self.decode_time_sec

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'input_path': str(self.input_path),
            'output_path': str(self.output_path),
            'decode_time_sec': self.decode_time_sec,
            'codec_name': self.codec_name,
            'n_frames': self.n_frames,
            'decode_fps': self.decode_fps,
            'error': self.error
        }


class CodecBase(ABC):
    """
    Interface abstraite pour tous les codecs.

    Definit les methodes encode() et decode() que tous les codecs
    doivent implementer.
    """

    def __init__(self, name: str):
        """
        Initialise le codec.

        Args:
            name: Nom du codec (ex: 'H.264', 'LMD-PPV')
        """
        self.name = name

    @abstractmethod
    def encode(
        self,
        input_path: Path,
        output_path: Path,
        quality: Any = None,
        **kwargs
    ) -> EncodeResult:
        """
        Encode une video.

        Args:
            input_path: Chemin de la video source
            output_path: Chemin du fichier de sortie
            quality: Parametre de qualite (CRF, QP, etc.)
            **kwargs: Parametres supplementaires

        Returns:
            EncodeResult avec les metriques
        """
        pass

    @abstractmethod
    def decode(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> DecodeResult:
        """
        Decode une video.

        Args:
            input_path: Chemin du fichier encode
            output_path: Chemin de la video decodee
            **kwargs: Parametres supplementaires

        Returns:
            DecodeResult avec les metriques
        """
        pass

    @abstractmethod
    def get_supported_qualities(self) -> list:
        """Retourne les valeurs de qualite supportees."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Verifie si le codec est disponible sur le systeme."""
        pass

    def get_output_extension(self) -> str:
        """Retourne l'extension de fichier de sortie."""
        return '.mp4'

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class Timer:
    """Context manager pour mesurer le temps d'execution."""

    def __init__(self):
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
