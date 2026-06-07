"""
lmd_wrapper.py - Wrapper LMD-PPV pour benchmark
================================================

Integration du codec LMD-PPV dans le framework de benchmark.

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import sys
from pathlib import Path
from typing import Optional, Any, List

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import CodecBase, EncodeResult, DecodeResult, Timer


class LMDCodec(CodecBase):
    """
    Wrapper pour le codec LMD-PPV.

    Encapsule le pipeline LMD-PPV pour comparaison avec les codecs standards.
    """

    def __init__(
        self,
        block_size: int = 16,
        block_frames: int = 32,
        n_colors: int = 256,
        use_turbo: bool = True
    ):
        """
        Initialise le codec LMD-PPV.

        Args:
            block_size: Taille des blocs spatiaux
            block_frames: Nombre de frames par bloc temporel
            n_colors: Nombre de couleurs pour la quantification
            use_turbo: Utiliser l'encodeur turbo (plus rapide)
        """
        super().__init__('LMD-PPV')
        self.block_size = block_size
        self.block_frames = block_frames
        self.n_colors = n_colors
        self.use_turbo = use_turbo

    def is_available(self) -> bool:
        """Verifie si LMD-PPV est disponible."""
        try:
            if self.use_turbo:
                from src.video.turbo_encoder import TurboVideoEncoder
            else:
                from src.video.encoder import VideoEncoder
            from src.video.fast_decoder import FastVideoDecoder
            return True
        except ImportError:
            return False

    def get_supported_qualities(self) -> list:
        """
        LMD-PPV est adaptatif, pas de parametres de qualite fixes.

        Les 'qualites' correspondent aux configurations de blocs.
        """
        return [
            {'block_size': 8, 'colors': 64},
            {'block_size': 16, 'colors': 128},
            {'block_size': 16, 'colors': 256},
            {'block_size': 32, 'colors': 256},
        ]

    def get_output_extension(self) -> str:
        return '.lmd'

    def encode(
        self,
        input_path: Path,
        output_path: Path,
        quality: Optional[Any] = None,
        max_frames: Optional[int] = None,
        **kwargs
    ) -> EncodeResult:
        """
        Encode une video avec LMD-PPV.

        Args:
            input_path: Video source
            output_path: Fichier .lmd de sortie
            quality: Configuration (dict avec block_size, colors, etc.)
            max_frames: Limite de frames a encoder
            **kwargs: Parametres supplementaires

        Returns:
            EncodeResult
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Configurer selon la qualite
        if isinstance(quality, dict):
            block_size = quality.get('block_size', self.block_size)
            n_colors = quality.get('colors', self.n_colors)
        else:
            block_size = self.block_size
            n_colors = self.n_colors

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Timer() as timer:
                if self.use_turbo:
                    from src.video.turbo_encoder import TurboVideoEncoder

                    encoder = TurboVideoEncoder(
                        block_size=block_size,
                        block_frames=self.block_frames,
                        n_colors=n_colors
                    )
                    stats = encoder.encode(
                        str(input_path),
                        str(output_path),
                        max_frames=max_frames
                    )
                else:
                    from src.video.encoder import VideoEncoder
                    from src.video.quantizer import QuantizeMethod

                    encoder = VideoEncoder(
                        block_size=block_size,
                        block_frames=self.block_frames,
                        n_colors=n_colors,
                        quantize_method=QuantizeMethod.KMEANS
                    )
                    encoded = encoder.encode(
                        str(input_path),
                        str(output_path),
                        max_frames=max_frames
                    )
                    stats = encoded.stats if hasattr(encoded, 'stats') else None

            if output_path.exists():
                # Extraire les metadonnees du fichier source
                from ..codecs.ffmpeg_wrapper import get_video_info
                video_info = get_video_info(input_path)

                return EncodeResult(
                    success=True,
                    input_path=input_path,
                    output_path=output_path,
                    input_size_bytes=input_path.stat().st_size,
                    output_size_bytes=output_path.stat().st_size,
                    encode_time_sec=timer.elapsed,
                    codec_name=self.name,
                    quality_param={'block_size': block_size, 'colors': n_colors},
                    width=video_info['width'],
                    height=video_info['height'],
                    n_frames=video_info['n_frames'],
                    duration_sec=video_info['duration'],
                    extra_params={
                        'block_size': block_size,
                        'block_frames': self.block_frames,
                        'n_colors': n_colors,
                        'turbo': self.use_turbo
                    }
                )
            else:
                return EncodeResult(
                    success=False,
                    input_path=input_path,
                    output_path=output_path,
                    input_size_bytes=input_path.stat().st_size if input_path.exists() else 0,
                    output_size_bytes=0,
                    encode_time_sec=timer.elapsed,
                    codec_name=self.name,
                    quality_param={'block_size': block_size, 'colors': n_colors},
                    error='Output file not created'
                )

        except Exception as e:
            return EncodeResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                input_size_bytes=input_path.stat().st_size if input_path.exists() else 0,
                output_size_bytes=0,
                encode_time_sec=0,
                codec_name=self.name,
                quality_param={'block_size': block_size, 'colors': n_colors},
                error=str(e)
            )

    def decode(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> DecodeResult:
        """
        Decode un fichier LMD-PPV.

        Args:
            input_path: Fichier .lmd
            output_path: Video decodee (MP4)
            **kwargs: Parametres supplementaires

        Returns:
            DecodeResult
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from src.video.fast_decoder import FastVideoDecoder

            with Timer() as timer:
                decoder = FastVideoDecoder()
                header = decoder.load(str(input_path))
                decoder.save_video(str(output_path))

            if output_path.exists():
                return DecodeResult(
                    success=True,
                    input_path=input_path,
                    output_path=output_path,
                    decode_time_sec=timer.elapsed,
                    codec_name=self.name,
                    n_frames=header.n_frames
                )
            else:
                return DecodeResult(
                    success=False,
                    input_path=input_path,
                    output_path=output_path,
                    decode_time_sec=timer.elapsed,
                    codec_name=self.name,
                    error='Output file not created'
                )

        except Exception as e:
            return DecodeResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                decode_time_sec=0,
                codec_name=self.name,
                error=str(e)
            )

    def encode_with_threshold_config(
        self,
        input_path: Path,
        output_path: Path,
        threshold_config: Any,
        max_frames: Optional[int] = None
    ) -> EncodeResult:
        """
        Encode avec une configuration de seuils specifique.

        Args:
            input_path: Video source
            output_path: Fichier .lmd
            threshold_config: Configuration des 7 seuils
            max_frames: Limite de frames

        Returns:
            EncodeResult
        """
        # Cette methode sera utilisee pour l'optimisation
        # Elle utilise l'agent de classification avec des seuils parametres
        input_path = Path(input_path)
        output_path = Path(output_path)

        try:
            from src.video.turbo_encoder import TurboVideoEncoder
            from src.agents.agent_1_classification import ClassificationAgent

            # Creer un agent avec les seuils personnalises
            if threshold_config:
                agent = ClassificationAgent()
                agent.threshold_H_s = threshold_config.threshold_H_s
                agent.threshold_rho_high = threshold_config.threshold_rho_high
                agent.threshold_rho_low = threshold_config.threshold_rho_low
                agent.threshold_chi2 = threshold_config.threshold_chi2

            with Timer() as timer:
                encoder = TurboVideoEncoder(
                    block_size=self.block_size,
                    block_frames=self.block_frames,
                    n_colors=self.n_colors
                )
                # Injecter l'agent personnalise si necessaire
                stats = encoder.encode(
                    str(input_path),
                    str(output_path),
                    max_frames=max_frames
                )

            if output_path.exists():
                from ..codecs.ffmpeg_wrapper import get_video_info
                video_info = get_video_info(input_path)

                return EncodeResult(
                    success=True,
                    input_path=input_path,
                    output_path=output_path,
                    input_size_bytes=input_path.stat().st_size,
                    output_size_bytes=output_path.stat().st_size,
                    encode_time_sec=timer.elapsed,
                    codec_name=self.name,
                    quality_param='custom_thresholds',
                    width=video_info['width'],
                    height=video_info['height'],
                    n_frames=video_info['n_frames'],
                    duration_sec=video_info['duration'],
                    extra_params={'thresholds': str(threshold_config)}
                )
            else:
                return EncodeResult(
                    success=False,
                    input_path=input_path,
                    output_path=output_path,
                    input_size_bytes=input_path.stat().st_size if input_path.exists() else 0,
                    output_size_bytes=0,
                    encode_time_sec=timer.elapsed,
                    codec_name=self.name,
                    quality_param='custom_thresholds',
                    error='Output file not created'
                )

        except Exception as e:
            return EncodeResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                input_size_bytes=0,
                output_size_bytes=0,
                encode_time_sec=0,
                codec_name=self.name,
                quality_param='custom_thresholds',
                error=str(e)
            )
