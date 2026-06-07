"""
encoder.py - Encodeur video LMD-PPV
====================================

Encode une video complete en utilisant le pipeline LMD.

Structure du fichier compresse:
- Header global (version, dimensions, palette, etc.)
- Blocs encodes sequentiellement

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass, field
from pathlib import Path
import struct
import time

from .loader import VideoLoader, VideoInfo
from .quantizer import ColorQuantizer, QuantizeMethod, Palette
from ..pipeline import LMDPipeline, PipelineResult
from ..core.cartouche import Cartouche
from ..utils.io_utils import BitWriter


# Magic number pour identifier le format
LMD_MAGIC = b'LMDV'
LMD_VERSION = 0x0600  # v6.00


@dataclass
class EncodedVideo:
    """Video encodee."""
    header: bytes
    blocks: List[bytes]
    palette: Palette
    info: VideoInfo
    stats: 'EncodingStats'

    def to_bytes(self) -> bytes:
        """Serialise la video complete."""
        writer = BitWriter()

        # Magic + version
        for b in LMD_MAGIC:
            writer.write_bits(b, 8)
        writer.write_bits(LMD_VERSION, 16)

        # Header
        writer.write_bits(len(self.header), 32)
        for b in self.header:
            writer.write_bits(b, 8)

        # Palette
        palette_bytes = self._encode_palette()
        writer.write_bits(len(palette_bytes), 32)
        for b in palette_bytes:
            writer.write_bits(b, 8)

        # Nombre de blocs
        writer.write_bits(len(self.blocks), 32)

        # Blocs
        for block in self.blocks:
            writer.write_bits(len(block), 32)
            for b in block:
                writer.write_bits(b, 8)

        return writer.get_bytes()

    def _encode_palette(self) -> bytes:
        """Encode la palette."""
        data = bytearray()
        data.extend(struct.pack('<H', self.palette.n_colors))
        data.extend(self.palette.colors.tobytes())
        return bytes(data)

    def save(self, path: str):
        """Sauvegarde la video encodee."""
        data = self.to_bytes()
        with open(path, 'wb') as f:
            f.write(data)

    @property
    def total_bits(self) -> int:
        """Nombre total de bits."""
        return len(self.to_bytes()) * 8


@dataclass
class EncodingStats:
    """Statistiques d'encodage."""
    total_frames: int = 0
    total_blocks: int = 0
    total_jumps: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    compression_ratio: float = 0.0
    encoding_time_sec: float = 0.0
    fps_encoding: float = 0.0

    # Par type de processus
    process_counts: Dict[str, int] = field(default_factory=dict)
    # Par mode couleur
    color_mode_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'total_frames': self.total_frames,
            'total_blocks': self.total_blocks,
            'total_jumps': self.total_jumps,
            'input_bytes': self.input_bytes,
            'output_bytes': self.output_bytes,
            'compression_ratio': self.compression_ratio,
            'encoding_time_sec': self.encoding_time_sec,
            'fps_encoding': self.fps_encoding,
            'process_counts': self.process_counts,
            'color_mode_counts': self.color_mode_counts
        }


class VideoEncoder:
    """
    Encodeur video LMD-PPV.

    Encode une video en blocs spatio-temporels compresses.
    """

    def __init__(
        self,
        block_size: int = 16,
        block_frames: int = 32,
        n_colors: int = 256,
        quantize_method: QuantizeMethod = QuantizeMethod.KMEANS
    ):
        """
        Initialise l'encodeur.

        Args:
            block_size: Taille spatiale des blocs
            block_frames: Nombre de frames par bloc
            n_colors: Nombre de couleurs de la palette
            quantize_method: Methode de quantification
        """
        self.block_size = block_size
        self.block_frames = block_frames
        self.n_colors = n_colors
        self.quantize_method = quantize_method

        self.pipeline = LMDPipeline(block_size=block_size)
        self.quantizer = ColorQuantizer(n_colors=n_colors, method=quantize_method)

    def encode(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> EncodedVideo:
        """
        Encode une video complete.

        Args:
            video_path: Chemin de la video source
            output_path: Chemin de sortie (optionnel)
            max_frames: Limite de frames a encoder
            progress_callback: Callback pour progression (0.0 - 1.0)

        Returns:
            Video encodee
        """
        start_time = time.time()

        # Charge la video
        loader = VideoLoader(video_path)
        info = loader.info

        print(f"[LMD] Encodage de: {video_path}")
        print(f"[LMD] Resolution: {info.width}x{info.height}")
        print(f"[LMD] Frames: {info.frame_count} @ {info.fps:.1f} fps")
        print(f"[LMD] Duree: {info.duration_sec:.1f}s")

        # Limite de frames
        total_frames = min(info.frame_count, max_frames) if max_frames else info.frame_count

        # Statistiques
        stats = EncodingStats(
            total_frames=total_frames,
            input_bytes=total_frames * info.width * info.height * 3
        )

        # Phase 1: Calcule la palette sur un echantillon
        print(f"[LMD] Calcul de la palette ({self.n_colors} couleurs)...")
        sample_frames = self._sample_frames(loader, n_samples=min(30, total_frames))
        palette = self.quantizer.fit(sample_frames)
        print(f"[LMD] Palette: {palette.n_colors} couleurs ({palette.method.name})")

        # Phase 2: Encode les blocs
        print(f"[LMD] Encodage des blocs ({self.block_size}x{self.block_size}x{self.block_frames})...")

        encoded_blocks = []
        block_idx = 0

        n_blocks_x = (info.width + self.block_size - 1) // self.block_size
        n_blocks_y = (info.height + self.block_size - 1) // self.block_size
        n_blocks_t = (total_frames + self.block_frames - 1) // self.block_frames
        total_blocks = n_blocks_x * n_blocks_y * n_blocks_t

        for t_start in range(0, total_frames, self.block_frames):
            # Lit le bloc temporel
            loader.seek(t_start)
            t_end = min(t_start + self.block_frames, total_frames)
            frames_rgb = loader.read_frames(t_end - t_start)

            if frames_rgb.size == 0:
                break

            # Quantifie
            frames_idx = self.quantizer.quantize_batch(frames_rgb)

            # Parcourt les blocs spatiaux
            for y in range(0, info.height, self.block_size):
                for x in range(0, info.width, self.block_size):
                    # Extrait le bloc
                    y_end = min(y + self.block_size, info.height)
                    x_end = min(x + self.block_size, info.width)

                    block = frames_idx[:, y:y_end, x:x_end]

                    # Encode via pipeline
                    result = self.pipeline.process_block(block)
                    encoded_blocks.append(result.encoded.bitstream)

                    # Met a jour les stats
                    stats.total_jumps += result.extraction.features.N

                    proc_name = result.cartouche.process_type_name
                    stats.process_counts[proc_name] = stats.process_counts.get(proc_name, 0) + 1

                    mode_name = result.cartouche.color_mode_name
                    stats.color_mode_counts[mode_name] = stats.color_mode_counts.get(mode_name, 0) + 1

                    block_idx += 1

                    # Progression
                    if progress_callback and block_idx % 100 == 0:
                        progress_callback(block_idx / total_blocks)

            # Affiche progression
            progress = (t_start + len(frames_rgb)) / total_frames * 100
            print(f"\r[LMD] Progression: {progress:.1f}%  Blocs: {block_idx}", end='')

        print()  # Nouvelle ligne

        # Finalise les stats
        stats.total_blocks = block_idx
        stats.encoding_time_sec = time.time() - start_time
        stats.fps_encoding = total_frames / stats.encoding_time_sec if stats.encoding_time_sec > 0 else 0

        # Cree le header
        header = self._create_header(info, total_frames)

        # Assemble la video encodee
        encoded = EncodedVideo(
            header=header,
            blocks=encoded_blocks,
            palette=palette,
            info=info,
            stats=stats
        )

        # Calcule la compression
        stats.output_bytes = len(encoded.to_bytes())
        stats.compression_ratio = stats.input_bytes / stats.output_bytes if stats.output_bytes > 0 else 0

        # Sauvegarde si demande
        if output_path:
            encoded.save(output_path)
            print(f"[LMD] Sauvegarde: {output_path}")

        # Affiche le rapport
        self._print_report(stats)

        loader.close()
        return encoded

    def _sample_frames(self, loader: VideoLoader, n_samples: int) -> np.ndarray:
        """Echantillonne des frames pour la palette."""
        total = loader.info.frame_count
        indices = np.linspace(0, total - 1, n_samples, dtype=int)

        frames = []
        for idx in indices:
            loader.seek(idx)
            frame = loader.read_frame()
            if frame is not None:
                frames.append(frame)

        return np.stack(frames) if frames else np.zeros((1, 100, 100, 3), dtype=np.uint8)

    def _create_header(self, info: VideoInfo, n_frames: int) -> bytes:
        """Cree le header global."""
        data = bytearray()

        # Dimensions
        data.extend(struct.pack('<H', info.width))
        data.extend(struct.pack('<H', info.height))
        data.extend(struct.pack('<I', n_frames))

        # FPS (float)
        data.extend(struct.pack('<f', info.fps))

        # Parametres de bloc
        data.extend(struct.pack('<B', self.block_size))
        data.extend(struct.pack('<B', self.block_frames))

        # Nombre de couleurs
        data.extend(struct.pack('<H', self.n_colors))

        return bytes(data)

    def _print_report(self, stats: EncodingStats):
        """Affiche le rapport d'encodage."""
        print("\n" + "=" * 60)
        print("RAPPORT D'ENCODAGE LMD-PPV")
        print("=" * 60)

        print(f"\nFrames:      {stats.total_frames}")
        print(f"Blocs:       {stats.total_blocks}")
        print(f"Sauts:       {stats.total_jumps}")

        print(f"\nEntree:      {stats.input_bytes / 1024 / 1024:.2f} MB")
        print(f"Sortie:      {stats.output_bytes / 1024:.2f} KB")
        print(f"Ratio:       {stats.compression_ratio:.1f}x")

        print(f"\nTemps:       {stats.encoding_time_sec:.1f}s")
        print(f"Vitesse:     {stats.fps_encoding:.1f} fps")

        print("\nTypes de processus:")
        for name, count in stats.process_counts.items():
            pct = count / stats.total_blocks * 100 if stats.total_blocks > 0 else 0
            print(f"  {name}: {count} ({pct:.1f}%)")

        print("\nModes couleur:")
        for name, count in stats.color_mode_counts.items():
            pct = count / stats.total_blocks * 100 if stats.total_blocks > 0 else 0
            print(f"  {name}: {count} ({pct:.1f}%)")

        print("=" * 60)


def encode_video(
    input_path: str,
    output_path: Optional[str] = None,
    block_size: int = 16,
    n_colors: int = 256,
    max_frames: Optional[int] = None
) -> EncodedVideo:
    """
    Fonction utilitaire pour encoder une video.

    Args:
        input_path: Chemin de la video source
        output_path: Chemin de sortie
        block_size: Taille des blocs
        n_colors: Nombre de couleurs
        max_frames: Limite de frames

    Returns:
        Video encodee
    """
    encoder = VideoEncoder(
        block_size=block_size,
        n_colors=n_colors,
        quantize_method=QuantizeMethod.KMEANS
    )

    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.lmd'))

    return encoder.encode(input_path, output_path, max_frames)
