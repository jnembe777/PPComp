"""
loader.py - Chargement de fichiers video
=========================================

Supporte: MP4, AVI, MKV, MOV, WEBM, etc.
Utilise OpenCV pour la lecture.

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Tuple, Optional, Iterator, List
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("[WARNING] OpenCV non installe. Installez avec: pip install opencv-python")


@dataclass
class VideoInfo:
    """Informations sur la video."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    codec: str


class VideoLoader:
    """
    Chargeur de fichiers video.

    Lit les videos frame par frame ou par blocs temporels.
    """

    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', '.y4m']

    def __init__(self, video_path: str):
        """
        Initialise le loader.

        Args:
            video_path: Chemin vers le fichier video
        """
        if not HAS_OPENCV:
            raise ImportError("OpenCV requis. Installez avec: pip install opencv-python")

        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video non trouvee: {video_path}")

        if self.path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Format non supporte: {self.path.suffix}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la video: {video_path}")

        self._info = self._get_info()

    def _get_info(self) -> VideoInfo:
        """Recupere les informations de la video."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Codec (fourcc)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        return VideoInfo(
            path=str(self.path),
            width=width,
            height=height,
            fps=fps if fps > 0 else 30.0,
            frame_count=frame_count,
            duration_sec=frame_count / fps if fps > 0 else 0,
            codec=codec
        )

    @property
    def info(self) -> VideoInfo:
        """Retourne les informations de la video."""
        return self._info

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Lit la prochaine frame.

        Returns:
            Frame BGR (H, W, 3) ou None si fin de video
        """
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def read_frames(self, n: int) -> np.ndarray:
        """
        Lit n frames consecutives.

        Args:
            n: Nombre de frames a lire

        Returns:
            Tableau (n, H, W, 3) BGR
        """
        frames = []
        for _ in range(n):
            frame = self.read_frame()
            if frame is None:
                break
            frames.append(frame)

        if not frames:
            return np.array([])

        return np.stack(frames, axis=0)

    def read_block(
        self,
        n_frames: int,
        x: int, y: int,
        width: int, height: int
    ) -> np.ndarray:
        """
        Lit un bloc spatio-temporel.

        Args:
            n_frames: Nombre de frames
            x, y: Position du bloc
            width, height: Taille du bloc

        Returns:
            Bloc (T, H, W, 3) BGR
        """
        frames = self.read_frames(n_frames)
        if frames.size == 0:
            return np.array([])

        # Extraction du bloc spatial
        y_end = min(y + height, frames.shape[1])
        x_end = min(x + width, frames.shape[2])

        return frames[:, y:y_end, x:x_end, :]

    def seek(self, frame_number: int):
        """
        Se positionne a une frame specifique.

        Args:
            frame_number: Numero de frame (0-indexed)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def iter_frames(self, start: int = 0, end: Optional[int] = None) -> Iterator[np.ndarray]:
        """
        Iterateur sur les frames.

        Args:
            start: Frame de debut
            end: Frame de fin (None = jusqu'a la fin)

        Yields:
            Frames BGR (H, W, 3)
        """
        self.seek(start)
        current = start
        end = end or self._info.frame_count

        while current < end:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame
            current += 1

    def iter_blocks(
        self,
        block_frames: int = 32,
        block_size: int = 16,
        overlap: int = 0
    ) -> Iterator[Tuple[int, int, int, np.ndarray]]:
        """
        Iterateur sur les blocs spatio-temporels.

        Args:
            block_frames: Nombre de frames par bloc
            block_size: Taille spatiale des blocs
            overlap: Chevauchement spatial

        Yields:
            (frame_idx, y, x, block) pour chaque bloc
        """
        stride = block_size - overlap
        frame_idx = 0

        while True:
            # Lit le bloc temporel
            self.seek(frame_idx)
            frames = self.read_frames(block_frames)

            if frames.size == 0:
                break

            T, H, W, C = frames.shape

            # Parcourt les blocs spatiaux
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    y_end = min(y + block_size, H)
                    x_end = min(x + block_size, W)

                    block = frames[:, y:y_end, x:x_end, :]
                    yield (frame_idx, y, x, block)

            frame_idx += block_frames

            if frame_idx >= self._info.frame_count:
                break

    def get_thumbnail(self, frame_idx: int = 0, max_size: int = 256) -> np.ndarray:
        """
        Cree une miniature de la video.

        Args:
            frame_idx: Frame a utiliser
            max_size: Taille maximale

        Returns:
            Image redimensionnee
        """
        self.seek(frame_idx)
        frame = self.read_frame()

        if frame is None:
            return np.zeros((max_size, max_size, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        return cv2.resize(frame, (new_w, new_h))

    def close(self):
        """Ferme la video."""
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


def list_videos(directory: str) -> List[str]:
    """
    Liste les fichiers video dans un repertoire.

    Args:
        directory: Chemin du repertoire

    Returns:
        Liste des chemins video
    """
    path = Path(directory)
    videos = []

    for ext in VideoLoader.SUPPORTED_FORMATS:
        videos.extend(path.glob(f"*{ext}"))
        videos.extend(path.glob(f"*{ext.upper()}"))

    return [str(v) for v in sorted(videos)]
