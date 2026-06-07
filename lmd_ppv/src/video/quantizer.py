"""
quantizer.py - Quantification des couleurs RGB vers palette indexee
====================================================================

Methodes de quantification:
- Uniform: Division uniforme de l'espace RGB
- KMeans: Clustering k-means adaptatif
- MedianCut: Decoupe mediane de l'histogramme
- Octree: Arbre octaire pour reduction

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum, auto


class QuantizeMethod(Enum):
    """Methodes de quantification."""
    UNIFORM = auto()      # Grille uniforme RGB
    KMEANS = auto()       # K-means clustering
    MEDIAN_CUT = auto()   # Decoupe mediane
    OCTREE = auto()       # Arbre octaire


@dataclass
class Palette:
    """Palette de couleurs."""
    colors: np.ndarray       # (m, 3) RGB
    n_colors: int            # Nombre de couleurs
    method: QuantizeMethod   # Methode utilisee

    def to_rgb(self, indices: np.ndarray) -> np.ndarray:
        """Convertit indices vers RGB."""
        return self.colors[indices]

    def save(self, path: str):
        """Sauvegarde la palette."""
        np.savez(path, colors=self.colors, n_colors=self.n_colors,
                 method=self.method.name)

    @classmethod
    def load(cls, path: str) -> 'Palette':
        """Charge une palette."""
        data = np.load(path)
        return cls(
            colors=data['colors'],
            n_colors=int(data['n_colors']),
            method=QuantizeMethod[str(data['method'])]
        )


class ColorQuantizer:
    """
    Quantificateur de couleurs.

    Convertit les images RGB 24-bit en images indexees.
    """

    def __init__(
        self,
        n_colors: int = 256,
        method: QuantizeMethod = QuantizeMethod.UNIFORM
    ):
        """
        Initialise le quantificateur.

        Args:
            n_colors: Nombre de couleurs cibles (2-256)
            method: Methode de quantification
        """
        self.n_colors = min(max(n_colors, 2), 256)
        self.method = method
        self.palette: Optional[Palette] = None

    def fit(self, frames: np.ndarray) -> Palette:
        """
        Calcule la palette optimale pour les frames.

        Args:
            frames: Tableau (T, H, W, 3) ou (H, W, 3) BGR/RGB

        Returns:
            Palette calculee
        """
        # Normalise les dimensions
        if frames.ndim == 3:
            frames = frames[np.newaxis, ...]

        # Extrait tous les pixels
        pixels = frames.reshape(-1, 3).astype(np.float32)

        if self.method == QuantizeMethod.UNIFORM:
            colors = self._uniform_palette()
        elif self.method == QuantizeMethod.KMEANS:
            colors = self._kmeans_palette(pixels)
        elif self.method == QuantizeMethod.MEDIAN_CUT:
            colors = self._median_cut_palette(pixels)
        elif self.method == QuantizeMethod.OCTREE:
            colors = self._octree_palette(pixels)
        else:
            colors = self._uniform_palette()

        self.palette = Palette(
            colors=colors.astype(np.uint8),
            n_colors=len(colors),
            method=self.method
        )

        return self.palette

    def quantize(self, frame: np.ndarray) -> np.ndarray:
        """
        Quantifie une frame vers indices.

        Args:
            frame: Image (H, W, 3) BGR/RGB

        Returns:
            Image indexee (H, W) avec valeurs 0 to m-1
        """
        if self.palette is None:
            self.fit(frame)

        h, w = frame.shape[:2]
        pixels = frame.reshape(-1, 3).astype(np.float32)

        # Trouve la couleur la plus proche pour chaque pixel
        indices = self._find_nearest(pixels, self.palette.colors)

        return indices.reshape(h, w).astype(np.uint16)

    def quantize_batch(self, frames: np.ndarray) -> np.ndarray:
        """
        Quantifie un batch de frames.

        Args:
            frames: Tableau (T, H, W, 3)

        Returns:
            Tableau indexe (T, H, W)
        """
        T, H, W, _ = frames.shape
        result = np.zeros((T, H, W), dtype=np.uint16)

        for t in range(T):
            result[t] = self.quantize(frames[t])

        return result

    def _uniform_palette(self) -> np.ndarray:
        """Cree une palette uniforme."""
        # Determine le nombre de niveaux par canal
        levels = int(np.ceil(self.n_colors ** (1/3)))

        colors = []
        step = 255 / (levels - 1) if levels > 1 else 255

        for r in range(levels):
            for g in range(levels):
                for b in range(levels):
                    if len(colors) >= self.n_colors:
                        break
                    colors.append([r * step, g * step, b * step])
                if len(colors) >= self.n_colors:
                    break
            if len(colors) >= self.n_colors:
                break

        return np.array(colors[:self.n_colors])

    def _kmeans_palette(self, pixels: np.ndarray, max_iter: int = 20) -> np.ndarray:
        """K-means clustering pour palette."""
        n_samples = min(10000, len(pixels))
        sample_idx = np.random.choice(len(pixels), n_samples, replace=False)
        samples = pixels[sample_idx]

        # Initialisation aleatoire
        init_idx = np.random.choice(n_samples, self.n_colors, replace=False)
        centers = samples[init_idx].copy()

        for _ in range(max_iter):
            # Assignation
            labels = self._find_nearest(samples, centers)

            # Mise a jour des centres
            new_centers = np.zeros_like(centers)
            for k in range(self.n_colors):
                mask = labels == k
                if mask.sum() > 0:
                    new_centers[k] = samples[mask].mean(axis=0)
                else:
                    new_centers[k] = centers[k]

            # Convergence
            if np.allclose(centers, new_centers, atol=1):
                break

            centers = new_centers

        return centers

    def _median_cut_palette(self, pixels: np.ndarray) -> np.ndarray:
        """Decoupe mediane pour palette."""
        n_samples = min(50000, len(pixels))
        sample_idx = np.random.choice(len(pixels), n_samples, replace=False)
        samples = pixels[sample_idx]

        # Liste des boites a diviser
        boxes = [samples]

        while len(boxes) < self.n_colors:
            # Trouve la boite avec la plus grande plage
            max_range = -1
            max_idx = 0
            max_channel = 0

            for i, box in enumerate(boxes):
                if len(box) < 2:
                    continue
                for c in range(3):
                    r = box[:, c].max() - box[:, c].min()
                    if r > max_range:
                        max_range = r
                        max_idx = i
                        max_channel = c

            if max_range <= 0:
                break

            # Divise la boite
            box = boxes.pop(max_idx)
            median = np.median(box[:, max_channel])

            box1 = box[box[:, max_channel] <= median]
            box2 = box[box[:, max_channel] > median]

            if len(box1) > 0:
                boxes.append(box1)
            if len(box2) > 0:
                boxes.append(box2)

        # Calcule les centres
        colors = []
        for box in boxes:
            if len(box) > 0:
                colors.append(box.mean(axis=0))

        # Complete si necessaire
        while len(colors) < self.n_colors:
            colors.append(colors[-1] if colors else [128, 128, 128])

        return np.array(colors[:self.n_colors])

    def _octree_palette(self, pixels: np.ndarray) -> np.ndarray:
        """Arbre octaire pour palette (simplifie)."""
        # Utilise median cut comme fallback
        return self._median_cut_palette(pixels)

    def _find_nearest(self, pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Trouve l'indice de couleur le plus proche."""
        # Distance euclidienne au carre
        # pixels: (N, 3), palette: (M, 3)
        # result: (N,)

        # Methode optimisee par batch
        batch_size = 10000
        n_pixels = len(pixels)
        indices = np.zeros(n_pixels, dtype=np.int32)

        for start in range(0, n_pixels, batch_size):
            end = min(start + batch_size, n_pixels)
            batch = pixels[start:end]

            # Calcul des distances
            diff = batch[:, np.newaxis, :] - palette[np.newaxis, :, :]
            dist = (diff ** 2).sum(axis=2)
            indices[start:end] = dist.argmin(axis=1)

        return indices

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """
        Convertit indices vers RGB.

        Args:
            indices: Image indexee (H, W) ou (T, H, W)

        Returns:
            Image RGB
        """
        if self.palette is None:
            raise ValueError("Palette non initialisee")

        return self.palette.colors[indices]

    def get_color_distribution(self, indices: np.ndarray) -> Dict[int, float]:
        """
        Calcule la distribution des couleurs.

        Args:
            indices: Image indexee

        Returns:
            Distribution {color_idx: probability}
        """
        flat = indices.flatten()
        counts = np.bincount(flat, minlength=self.n_colors)
        total = counts.sum()

        if total == 0:
            return {}

        return {i: counts[i] / total for i in range(self.n_colors) if counts[i] > 0}


def rgb_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convertit RGB vers niveaux de gris."""
    if frame.ndim == 2:
        return frame
    # Formule standard: Y = 0.299*R + 0.587*G + 0.114*B
    return (0.299 * frame[..., 2] + 0.587 * frame[..., 1] + 0.114 * frame[..., 0]).astype(np.uint8)


def compute_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calcule la difference entre deux frames."""
    diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
    return diff.mean()
