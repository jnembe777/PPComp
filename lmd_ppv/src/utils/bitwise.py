"""
bitwise.py - Opérations bitwise optimisées pour l'extraction
=============================================================

Structures et fonctions SIMD-like pour:
- BinaryMask: masques binaires alignés
- XOR inter-frames
- POPCOUNT pour comptage
- TZCNT pour localisation

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Numba optionnel - fallback sur pure Python
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range


@dataclass
class BinaryMask:
    """
    Masque binaire pour représentation spatiale.

    Pour chaque couleur c: M_c(t,p) = I{C(p,t) = c} — 1 bit par pixel.

    Structure alignée 32 octets pour optimisation SIMD (AVX2).

    Attributes:
        data: Tableau uint64 (4 × 64 = 256 bits = 256 pixels max)
        width: Largeur du bloc en pixels
        height: Hauteur du bloc en pixels
    """
    data: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.uint64))
    width: int = 16
    height: int = 16

    def __post_init__(self):
        """Assure l'alignement et la taille correcte."""
        n_bits = self.width * self.height
        n_uint64 = (n_bits + 63) // 64
        if len(self.data) < n_uint64:
            self.data = np.zeros(n_uint64, dtype=np.uint64)

    @property
    def n_pixels(self) -> int:
        return self.width * self.height

    def set_bit(self, x: int, y: int, value: bool = True):
        """Définit le bit à la position (x, y)."""
        idx = y * self.width + x
        word_idx = idx // 64
        bit_idx = idx % 64
        if value:
            self.data[word_idx] |= np.uint64(1) << bit_idx
        else:
            self.data[word_idx] &= ~(np.uint64(1) << bit_idx)

    def get_bit(self, x: int, y: int) -> bool:
        """Lit le bit à la position (x, y)."""
        idx = y * self.width + x
        word_idx = idx // 64
        bit_idx = idx % 64
        return bool((self.data[word_idx] >> bit_idx) & 1)

    def popcount(self) -> int:
        """Compte le nombre de bits à 1 (POPCOUNT)."""
        return popcount_array(self.data)

    def xor(self, other: 'BinaryMask') -> 'BinaryMask':
        """XOR avec un autre masque."""
        result = BinaryMask(width=self.width, height=self.height)
        result.data = np.bitwise_xor(self.data, other.data)
        return result

    def and_(self, other: 'BinaryMask') -> 'BinaryMask':
        """AND avec un autre masque."""
        result = BinaryMask(width=self.width, height=self.height)
        result.data = np.bitwise_and(self.data, other.data)
        return result

    def or_(self, other: 'BinaryMask') -> 'BinaryMask':
        """OR avec un autre masque."""
        result = BinaryMask(width=self.width, height=self.height)
        result.data = np.bitwise_or(self.data, other.data)
        return result

    def get_set_positions(self) -> List[Tuple[int, int]]:
        """Retourne les positions (x, y) des bits à 1."""
        positions = []
        for i in range(self.n_pixels):
            word_idx = i // 64
            bit_idx = i % 64
            if (self.data[word_idx] >> bit_idx) & 1:
                x = i % self.width
                y = i // self.width
                positions.append((x, y))
        return positions

    @classmethod
    def from_frame(cls, frame: np.ndarray, color: int) -> 'BinaryMask':
        """
        Crée un masque depuis une frame pour une couleur donnée.

        Args:
            frame: Image 2D (height, width) avec indices de couleur
            color: Couleur à masquer

        Returns:
            BinaryMask avec 1 où frame == color
        """
        height, width = frame.shape
        mask = cls(width=width, height=height)

        for y in range(height):
            for x in range(width):
                if frame[y, x] == color:
                    mask.set_bit(x, y, True)

        return mask


# === Fonctions optimisées Numba ===

@njit(cache=True)
def popcount_u64(x: np.uint64) -> int:
    """POPCOUNT pour un uint64."""
    x = x - ((x >> 1) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
    x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return int(x & np.uint64(0x7F))


@njit(cache=True, parallel=True)
def popcount_array(data: np.ndarray) -> int:
    """POPCOUNT pour un tableau uint64."""
    total = 0
    for i in prange(len(data)):
        total += popcount_u64(data[i])
    return total


def popcount(x) -> int:
    """POPCOUNT générique."""
    if isinstance(x, np.ndarray):
        return popcount_array(x.astype(np.uint64))
    elif isinstance(x, BinaryMask):
        return x.popcount()
    else:
        return popcount_u64(np.uint64(x))


@njit(cache=True)
def tzcnt_u64(x: np.uint64) -> int:
    """
    Trailing Zero Count - position du premier bit à 1.

    Retourne 64 si x == 0.
    """
    if x == 0:
        return 64
    count = 0
    while (x & 1) == 0:
        x >>= 1
        count += 1
    return count


def tzcnt(x: np.uint64) -> int:
    """Trailing Zero Count."""
    return tzcnt_u64(np.uint64(x))


@njit(cache=True, parallel=True)
def xor_frames(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """
    XOR optimisé entre deux frames.

    Args:
        frame1, frame2: Frames 2D (uint8 ou uint16)

    Returns:
        Masque booléen des différences
    """
    height, width = frame1.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for y in prange(height):
        for x in range(width):
            if frame1[y, x] != frame2[y, x]:
                result[y, x] = 1

    return result


@njit(cache=True)
def count_transitions(sequence: np.ndarray) -> int:
    """
    Compte les transitions dans une séquence.

    N_trans = #{k: m_k ≠ m_{k-1}}
    """
    if len(sequence) <= 1:
        return 0
    count = 0
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i-1]:
            count += 1
    return count


@njit(cache=True, parallel=True)
def extract_jumps_positions(xor_result: np.ndarray) -> List:
    """
    Extrait les positions des sauts depuis le résultat XOR.

    Args:
        xor_result: Masque booléen des changements

    Returns:
        Liste de positions (y, x)
    """
    positions = []
    height, width = xor_result.shape

    for y in range(height):
        for x in range(width):
            if xor_result[y, x]:
                positions.append((y, x))

    return positions


def compute_spatial_homogeneity(masks: List[BinaryMask]) -> float:
    """
    DEPRECATED: Utiliser compute_spatial_homogeneity_from_frame() à la place.

    Cette version basée sur les masques binaires retourne toujours 0
    car les masques de couleurs sont mutuellement exclusifs.
    """
    return 0.0


def compute_color_correlation(mask1: BinaryMask, mask2: BinaryMask) -> float:
    """
    DEPRECATED: Utiliser compute_spatial_autocorrelation() à la place.

    Cette version basée sur XOR de masques ne mesure pas la corrélation spatiale.
    """
    return 0.0


def compute_spatial_homogeneity_from_frame(frame: np.ndarray) -> float:
    """
    Calcule l'homogénéité spatiale H_s d'une frame.

    H_s = fraction des paires de pixels voisins ayant la même couleur.

    - H_s proche de 1.0 = bloc très homogène (grandes régions uniformes)
    - H_s proche de 0.0 = bloc fragmenté (beaucoup de transitions)

    Args:
        frame: Image 2D (height, width) avec indices de couleur

    Returns:
        H_s ∈ [0, 1]
    """
    if frame.size == 0:
        return 1.0

    height, width = frame.shape

    if height < 2 and width < 2:
        return 1.0

    # Compter les paires de pixels voisins identiques
    same_horizontal = 0
    same_vertical = 0
    total_horizontal = 0
    total_vertical = 0

    # Voisins horizontaux
    if width >= 2:
        for y in range(height):
            for x in range(width - 1):
                total_horizontal += 1
                if frame[y, x] == frame[y, x + 1]:
                    same_horizontal += 1

    # Voisins verticaux
    if height >= 2:
        for y in range(height - 1):
            for x in range(width):
                total_vertical += 1
                if frame[y, x] == frame[y + 1, x]:
                    same_vertical += 1

    total_pairs = total_horizontal + total_vertical
    same_pairs = same_horizontal + same_vertical

    if total_pairs == 0:
        return 1.0

    return same_pairs / total_pairs


def compute_spatial_autocorrelation(frame: np.ndarray, lag: int = 1) -> float:
    """
    Calcule l'autocorrélation spatiale des couleurs (rho_corr).

    Mesure la corrélation entre un pixel et ses voisins décalés.

    - rho proche de 1.0 = forte corrélation spatiale (gradients lisses)
    - rho proche de 0.0 = faible corrélation (bruit, textures complexes)

    Args:
        frame: Image 2D (height, width) avec indices de couleur
        lag: Décalage spatial (1 = voisins immédiats)

    Returns:
        rho ∈ [0, 1]
    """
    if frame.size == 0:
        return 0.5

    height, width = frame.shape

    if height <= lag or width <= lag:
        return 0.5

    # Convertir en float pour les calculs
    f = frame.astype(np.float64)

    # Moyenne et variance globales
    mean_val = np.mean(f)
    var_val = np.var(f)

    if var_val < 1e-10:
        # Variance nulle = bloc uniforme = corrélation maximale
        return 1.0

    # Autocorrélation horizontale
    f_left = f[:, :-lag]
    f_right = f[:, lag:]
    cov_h = np.mean((f_left - mean_val) * (f_right - mean_val))

    # Autocorrélation verticale
    f_top = f[:-lag, :]
    f_bottom = f[lag:, :]
    cov_v = np.mean((f_top - mean_val) * (f_bottom - mean_val))

    # Moyenne des corrélations
    rho = (cov_h + cov_v) / (2 * var_val)

    # Normaliser dans [0, 1]
    # rho théorique ∈ [-1, 1], on le ramène à [0, 1]
    rho_normalized = (rho + 1) / 2

    return float(np.clip(rho_normalized, 0.0, 1.0))


def compute_texture_complexity(frame: np.ndarray) -> float:
    """
    Mesure la complexité de texture du bloc.

    Basé sur le nombre de couleurs distinctes et leur distribution spatiale.

    - Complexité proche de 0 = bloc simple (peu de couleurs, régions uniformes)
    - Complexité proche de 1 = bloc complexe (beaucoup de couleurs, fragmenté)

    Args:
        frame: Image 2D (height, width) avec indices de couleur

    Returns:
        Complexité ∈ [0, 1]
    """
    if frame.size == 0:
        return 0.0

    height, width = frame.shape
    n_pixels = height * width

    # Nombre de couleurs distinctes
    unique_colors = np.unique(frame)
    n_colors = len(unique_colors)

    # Normaliser par le nombre max théorique
    max_colors = min(n_pixels, 256)  # Supposer 256 couleurs max
    color_diversity = n_colors / max_colors

    # Compter les transitions (changements de couleur entre voisins)
    transitions = 0
    total_edges = 0

    # Transitions horizontales
    if width >= 2:
        for y in range(height):
            for x in range(width - 1):
                total_edges += 1
                if frame[y, x] != frame[y, x + 1]:
                    transitions += 1

    # Transitions verticales
    if height >= 2:
        for y in range(height - 1):
            for x in range(width):
                total_edges += 1
                if frame[y, x] != frame[y + 1, x]:
                    transitions += 1

    if total_edges == 0:
        edge_density = 0.0
    else:
        edge_density = transitions / total_edges

    # Combiner les métriques
    complexity = 0.3 * color_diversity + 0.7 * edge_density

    return float(np.clip(complexity, 0.0, 1.0))
