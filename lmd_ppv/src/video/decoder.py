"""
decoder.py - Decodeur video LMD-PPV
====================================

Decode une video compresse LMD vers frames RGB.

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import List, Optional, Tuple, Iterator
from dataclasses import dataclass
from pathlib import Path
import struct

from .quantizer import Palette
from ..core.cartouche import Cartouche
from ..agents.agent_6_encoder import EncoderAgent
from ..utils.io_utils import BitReader


LMD_MAGIC = b'LMDV'
LMD_VERSION = 0x0600


@dataclass
class DecodedFrame:
    """Frame decodee."""
    index: int
    data: np.ndarray  # RGB (H, W, 3)


@dataclass
class VideoHeader:
    """Header de la video."""
    width: int
    height: int
    n_frames: int
    fps: float
    block_size: int
    block_frames: int
    n_colors: int


class VideoDecoder:
    """
    Decodeur video LMD-PPV.

    Reconstruit les frames a partir des blocs compresses.
    """

    def __init__(self):
        """Initialise le decodeur."""
        self.encoder_agent = EncoderAgent()
        self.header: Optional[VideoHeader] = None
        self.palette: Optional[Palette] = None
        self.blocks: List[bytes] = []

    def load(self, path: str) -> VideoHeader:
        """
        Charge une video compresse.

        Args:
            path: Chemin du fichier .lmd

        Returns:
            Header de la video
        """
        with open(path, 'rb') as f:
            data = f.read()

        return self._parse(data)

    def _parse(self, data: bytes) -> VideoHeader:
        """Parse les donnees compresses."""
        reader = BitReader(data)

        # Magic
        magic = bytes([reader.read_bits(8) for _ in range(4)])
        if magic != LMD_MAGIC:
            raise ValueError(f"Format invalide: magic={magic}")

        # Version
        version = reader.read_bits(16)
        if version > LMD_VERSION:
            print(f"[WARNING] Version {version:04x} > supportee {LMD_VERSION:04x}")

        # Header
        header_len = reader.read_bits(32)
        header_bytes = bytes([reader.read_bits(8) for _ in range(header_len)])
        self.header = self._parse_header(header_bytes)

        # Palette
        palette_len = reader.read_bits(32)
        palette_bytes = bytes([reader.read_bits(8) for _ in range(palette_len)])
        self.palette = self._parse_palette(palette_bytes)

        # Blocs
        n_blocks = reader.read_bits(32)
        self.blocks = []

        for _ in range(n_blocks):
            block_len = reader.read_bits(32)
            block_bytes = bytes([reader.read_bits(8) for _ in range(block_len)])
            self.blocks.append(block_bytes)

        return self.header

    def _parse_header(self, data: bytes) -> VideoHeader:
        """Parse le header."""
        width = struct.unpack('<H', data[0:2])[0]
        height = struct.unpack('<H', data[2:4])[0]
        n_frames = struct.unpack('<I', data[4:8])[0]
        fps = struct.unpack('<f', data[8:12])[0]
        block_size = data[12]
        block_frames = data[13]
        n_colors = struct.unpack('<H', data[14:16])[0]

        return VideoHeader(
            width=width,
            height=height,
            n_frames=n_frames,
            fps=fps,
            block_size=block_size,
            block_frames=block_frames,
            n_colors=n_colors
        )

    def _parse_palette(self, data: bytes) -> Palette:
        """Parse la palette."""
        from .quantizer import QuantizeMethod

        n_colors = struct.unpack('<H', data[0:2])[0]
        colors = np.frombuffer(data[2:], dtype=np.uint8).reshape(-1, 3)

        return Palette(
            colors=colors[:n_colors],
            n_colors=n_colors,
            method=QuantizeMethod.KMEANS
        )

    def decode_frame(self, frame_idx: int) -> np.ndarray:
        """
        Decode une frame specifique.

        Args:
            frame_idx: Index de la frame (0-indexed)

        Returns:
            Frame RGB (H, W, 3)
        """
        if self.header is None:
            raise ValueError("Video non chargee")

        h = self.header
        frame = np.zeros((h.height, h.width), dtype=np.uint16)

        # Calcule quels blocs contiennent cette frame
        t_block = frame_idx // h.block_frames
        t_offset = frame_idx % h.block_frames

        n_blocks_x = (h.width + h.block_size - 1) // h.block_size
        n_blocks_y = (h.height + h.block_size - 1) // h.block_size
        blocks_per_temporal = n_blocks_x * n_blocks_y

        block_start = t_block * blocks_per_temporal

        # Decode les blocs spatiaux
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                block_idx = block_start + by * n_blocks_x + bx

                if block_idx >= len(self.blocks):
                    continue

                # Decode le bloc
                try:
                    block_data = self._decode_block(self.blocks[block_idx])

                    # Extrait la frame du bloc
                    if t_offset < len(block_data):
                        block_frame = block_data[t_offset]

                        # Copie dans la frame
                        y_start = by * h.block_size
                        x_start = bx * h.block_size
                        y_end = min(y_start + h.block_size, h.height)
                        x_end = min(x_start + h.block_size, h.width)

                        bh, bw = block_frame.shape[:2]
                        frame[y_start:y_end, x_start:x_end] = block_frame[:y_end-y_start, :x_end-x_start]
                except Exception as e:
                    # Bloc corrompu - laisse noir
                    pass

        # Convertit vers RGB
        rgb = self.palette.colors[frame]
        return rgb.astype(np.uint8)

    def _decode_block(self, block_data: bytes) -> np.ndarray:
        """Decode un bloc compresse."""
        # Utilise l'agent encodeur pour decoder
        cartouche, times, marks, features = self.encoder_agent.decode_block(block_data)

        # Reconstruit le bloc
        # Simplifie: cree un bloc avec les sauts aux positions decodees
        h = self.header
        block = np.zeros((h.block_frames, h.block_size, h.block_size), dtype=np.uint16)

        if len(times) > 0 and len(marks) > 0:
            # Place les couleurs
            for t, m in zip(times.astype(int), marks):
                if 0 <= t < h.block_frames:
                    # Position spatiale (simplifiee - centre)
                    block[t, :, :] = m

        return block

    def iter_frames(self, start: int = 0, end: Optional[int] = None) -> Iterator[DecodedFrame]:
        """
        Iterateur sur les frames.

        Args:
            start: Frame de debut
            end: Frame de fin

        Yields:
            DecodedFrame pour chaque frame
        """
        if self.header is None:
            raise ValueError("Video non chargee")

        end = end or self.header.n_frames

        for idx in range(start, end):
            frame = self.decode_frame(idx)
            yield DecodedFrame(index=idx, data=frame)

    def decode_all(self) -> np.ndarray:
        """
        Decode toute la video.

        Returns:
            Video (T, H, W, 3) RGB
        """
        if self.header is None:
            raise ValueError("Video non chargee")

        h = self.header
        video = np.zeros((h.n_frames, h.height, h.width, 3), dtype=np.uint8)

        for frame in self.iter_frames():
            video[frame.index] = frame.data

        return video

    def save_frames(self, output_dir: str, format: str = 'png'):
        """
        Sauvegarde les frames en images.

        Args:
            output_dir: Repertoire de sortie
            format: Format d'image (png, jpg)
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV requis pour sauvegarder les frames")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for frame in self.iter_frames():
            filename = output_path / f"frame_{frame.index:06d}.{format}"
            # OpenCV attend BGR
            bgr = frame.data[..., ::-1]
            cv2.imwrite(str(filename), bgr)

        print(f"[LMD] Frames sauvegardees: {output_dir}")

    def save_video(self, output_path: str, codec: str = 'mp4v'):
        """
        Sauvegarde en fichier video.

        Args:
            output_path: Chemin de sortie
            codec: Codec video (mp4v, XVID, etc.)
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV requis pour sauvegarder la video")

        h = self.header
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, h.fps, (h.width, h.height))

        for frame in self.iter_frames():
            bgr = frame.data[..., ::-1]
            writer.write(bgr)

        writer.release()
        print(f"[LMD] Video sauvegardee: {output_path}")


def decode_video(input_path: str, output_path: Optional[str] = None) -> np.ndarray:
    """
    Fonction utilitaire pour decoder une video.

    Args:
        input_path: Chemin du fichier .lmd
        output_path: Chemin video de sortie (optionnel)

    Returns:
        Video decodee (T, H, W, 3)
    """
    decoder = VideoDecoder()
    decoder.load(input_path)

    if output_path:
        decoder.save_video(output_path)

    return decoder.decode_all()
