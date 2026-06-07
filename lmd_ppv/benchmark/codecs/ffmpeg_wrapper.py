"""
ffmpeg_wrapper.py - Wrappers FFmpeg pour H.264/H.265/VP9/AV1
=============================================================

Codecs supportes:
- libx264 (H.264) - preset medium, CRF 18-28
- libx265 (H.265/HEVC) - preset medium, CRF 18-28
- libvpx-vp9 (VP9) - CRF 18-35
- libaom-av1 (AV1) - CRF 20-40

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import subprocess
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from .base import CodecBase, EncodeResult, DecodeResult, Timer


def run_ffmpeg(args: List[str], timeout: int = 3600) -> Tuple[bool, str, str]:
    """
    Execute une commande FFmpeg.

    Args:
        args: Arguments de la commande
        timeout: Timeout en secondes

    Returns:
        (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            ['ffmpeg'] + args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, '', 'Timeout expired'
    except FileNotFoundError:
        return False, '', 'FFmpeg not found'
    except Exception as e:
        return False, '', str(e)


def probe_video(path: Path) -> Dict:
    """
    Extrait les metadonnees video avec FFprobe.

    Args:
        path: Chemin de la video

    Returns:
        Dictionnaire avec les metadonnees
    """
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(path)
            ],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return {}


def get_video_info(path: Path) -> Dict:
    """
    Extrait les informations cles d'une video.

    Args:
        path: Chemin de la video

    Returns:
        Dictionnaire avec width, height, fps, n_frames, duration
    """
    probe = probe_video(path)
    video_stream = next(
        (s for s in probe.get('streams', []) if s.get('codec_type') == 'video'),
        {}
    )

    fps_str = video_stream.get('r_frame_rate', '0/1')
    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den > 0 else 0
    else:
        fps = float(fps_str) if fps_str else 0

    duration = float(probe.get('format', {}).get('duration', 0))
    n_frames = int(video_stream.get('nb_frames', 0))
    if n_frames == 0 and fps > 0 and duration > 0:
        n_frames = int(fps * duration)

    return {
        'width': video_stream.get('width', 0),
        'height': video_stream.get('height', 0),
        'fps': fps,
        'n_frames': n_frames,
        'duration': duration
    }


class FFmpegCodec(CodecBase):
    """
    Codec generique base sur FFmpeg.

    Classe de base pour H.264, H.265, VP9, AV1.
    """

    def __init__(
        self,
        name: str,
        encoder: str,
        decoder: Optional[str] = None,
        default_crf: int = 23,
        crf_range: Tuple[int, int] = (18, 28),
        preset: str = 'medium',
        extra_encode_args: Optional[List[str]] = None,
        extra_decode_args: Optional[List[str]] = None,
        output_extension: str = '.mp4'
    ):
        """
        Initialise le codec FFmpeg.

        Args:
            name: Nom du codec
            encoder: Nom de l'encodeur FFmpeg (libx264, libx265, etc.)
            decoder: Nom du decodeur (None = auto)
            default_crf: CRF par defaut
            crf_range: Plage CRF supportee (min, max)
            preset: Preset d'encodage
            extra_encode_args: Arguments supplementaires pour l'encodage
            extra_decode_args: Arguments supplementaires pour le decodage
            output_extension: Extension du fichier de sortie
        """
        super().__init__(name)
        self.encoder = encoder
        self.decoder = decoder
        self.default_crf = default_crf
        self.crf_range = crf_range
        self.preset = preset
        self.extra_encode_args = extra_encode_args or []
        self.extra_decode_args = extra_decode_args or []
        self._output_extension = output_extension

    def is_available(self) -> bool:
        """Verifie si FFmpeg et l'encodeur sont disponibles."""
        # Verifier FFmpeg
        if not shutil.which('ffmpeg'):
            return False

        # Verifier l'encodeur
        success, _, stderr = run_ffmpeg(['-encoders'], timeout=10)
        if not success:
            return False

        return self.encoder in stderr

    def get_supported_qualities(self) -> list:
        """Retourne les CRF supportes."""
        return list(range(self.crf_range[0], self.crf_range[1] + 1))

    def get_output_extension(self) -> str:
        return self._output_extension

    def encode(
        self,
        input_path: Path,
        output_path: Path,
        quality: Optional[int] = None,
        **kwargs
    ) -> EncodeResult:
        """
        Encode une video avec FFmpeg.

        Args:
            input_path: Chemin source
            output_path: Chemin destination
            quality: Valeur CRF (defaut: self.default_crf)
            **kwargs: Parametres supplementaires

        Returns:
            EncodeResult
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        crf = quality if quality is not None else self.default_crf

        # Obtenir les infos de la video source
        video_info = get_video_info(input_path)

        # Construire la commande
        args = [
            '-y',  # Overwrite
            '-i', str(input_path),
            '-c:v', self.encoder,
        ]

        # Ajouter CRF/qualite
        args.extend(self._get_quality_args(crf))

        # Ajouter preset
        args.extend(self._get_preset_args())

        # Arguments supplementaires
        args.extend(self.extra_encode_args)

        # Pas d'audio pour le benchmark
        args.extend(['-an'])

        # Fichier de sortie
        args.append(str(output_path))

        # Encoder
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Timer() as timer:
            success, stdout, stderr = run_ffmpeg(args)

        if success and output_path.exists():
            return EncodeResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                input_size_bytes=input_path.stat().st_size,
                output_size_bytes=output_path.stat().st_size,
                encode_time_sec=timer.elapsed,
                codec_name=self.name,
                quality_param=crf,
                width=video_info['width'],
                height=video_info['height'],
                n_frames=video_info['n_frames'],
                duration_sec=video_info['duration'],
                extra_params={'preset': self.preset, 'encoder': self.encoder}
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
                quality_param=crf,
                error=stderr
            )

    def decode(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs
    ) -> DecodeResult:
        """
        Decode une video avec FFmpeg.

        Args:
            input_path: Fichier encode
            output_path: Video decodee (Y4M ou autre)
            **kwargs: Parametres supplementaires

        Returns:
            DecodeResult
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Obtenir le nombre de frames
        video_info = get_video_info(input_path)

        # Construire la commande
        args = [
            '-y',
            '-i', str(input_path),
            '-c:v', 'rawvideo',  # Decode vers raw
            '-pix_fmt', 'yuv420p',
        ]
        args.extend(self.extra_decode_args)
        args.append(str(output_path))

        # Decoder
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Timer() as timer:
            success, stdout, stderr = run_ffmpeg(args)

        if success and output_path.exists():
            return DecodeResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                decode_time_sec=timer.elapsed,
                codec_name=self.name,
                n_frames=video_info['n_frames']
            )
        else:
            return DecodeResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                decode_time_sec=timer.elapsed,
                codec_name=self.name,
                error=stderr
            )

    def _get_quality_args(self, crf: int) -> List[str]:
        """Retourne les arguments de qualite."""
        return ['-crf', str(crf)]

    def _get_preset_args(self) -> List[str]:
        """Retourne les arguments de preset."""
        return ['-preset', self.preset]


class H264Codec(FFmpegCodec):
    """Codec H.264/AVC via libx264."""

    def __init__(
        self,
        preset: str = 'medium',
        tune: Optional[str] = None,
        profile: str = 'high'
    ):
        extra_args = ['-profile:v', profile]
        if tune:
            extra_args.extend(['-tune', tune])

        super().__init__(
            name='H.264',
            encoder='libx264',
            default_crf=23,
            crf_range=(18, 28),
            preset=preset,
            extra_encode_args=extra_args,
            output_extension='.mp4'
        )


class H265Codec(FFmpegCodec):
    """Codec H.265/HEVC via libx265."""

    def __init__(
        self,
        preset: str = 'medium',
        tune: Optional[str] = None
    ):
        extra_args = ['-tag:v', 'hvc1']  # Compatibilite
        if tune:
            extra_args.extend(['-tune', tune])
        # Reduire les logs x265
        extra_args.extend(['-x265-params', 'log-level=error'])

        super().__init__(
            name='H.265',
            encoder='libx265',
            default_crf=23,
            crf_range=(18, 28),
            preset=preset,
            extra_encode_args=extra_args,
            output_extension='.mp4'
        )


class VP9Codec(FFmpegCodec):
    """Codec VP9 via libvpx-vp9."""

    def __init__(
        self,
        cpu_used: int = 2,
        tile_columns: int = 2
    ):
        extra_args = [
            '-b:v', '0',  # Mode CRF
            '-cpu-used', str(cpu_used),
            '-tile-columns', str(tile_columns),
            '-row-mt', '1',  # Multi-threading
        ]

        super().__init__(
            name='VP9',
            encoder='libvpx-vp9',
            default_crf=30,
            crf_range=(20, 40),
            preset='good',
            extra_encode_args=extra_args,
            output_extension='.webm'
        )

    def _get_preset_args(self) -> List[str]:
        """VP9 utilise -deadline au lieu de -preset."""
        return ['-deadline', self.preset]


class AV1Codec(FFmpegCodec):
    """Codec AV1 via libaom-av1."""

    def __init__(
        self,
        cpu_used: int = 4,
        tile_columns: int = 2,
        tile_rows: int = 1
    ):
        extra_args = [
            '-strict', 'experimental',
            '-cpu-used', str(cpu_used),
            '-tile-columns', str(tile_columns),
            '-tile-rows', str(tile_rows),
            '-row-mt', '1',
        ]

        super().__init__(
            name='AV1',
            encoder='libaom-av1',
            default_crf=35,
            crf_range=(25, 45),
            preset=str(cpu_used),
            extra_encode_args=extra_args,
            output_extension='.mp4'
        )

    def _get_preset_args(self) -> List[str]:
        """AV1 utilise -cpu-used (deja dans extra_args)."""
        return []


# Dictionnaire des codecs disponibles
FFMPEG_CODECS = {
    'h264': H264Codec,
    'h265': H265Codec,
    'vp9': VP9Codec,
    'av1': AV1Codec,
}


def get_available_codecs() -> Dict[str, FFmpegCodec]:
    """Retourne les codecs FFmpeg disponibles sur le systeme."""
    available = {}
    for name, codec_class in FFMPEG_CODECS.items():
        codec = codec_class()
        if codec.is_available():
            available[name] = codec
    return available
