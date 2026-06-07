"""
agent_7_hierarchy.py - Agent Hiérarchie + Arbre + Zoom
=======================================================

Phase 4 - Dimensions E, F, G

Fonctions:
- HierarchicalColor avec color_cost_hier()
- SpatialProcessTree (quadtree)
- ZoomController avec adaptation bande passante
- Estimation d'intensité (Histogrammes, Splines, Ondelettes, Trig)

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..core.process_types import ColorMode, IntensityFamily, ChromaticLevel
from ..core.features import BlockFeatures
from ..core.cartouche import Cartouche
from ..codecs.huffman import HuffmanCodec


@dataclass
class HierarchicalColor:
    """
    Couleur hiérarchique multi-niveaux.

    Pour F niveaux actifs: C_color_hier(Bc, L) = N·Σ H(P̂_i) + Σ D_huf_i
    """
    levels: int = 2  # Nombre de niveaux (8b, 16b, 24b)
    level_bits: List[int] = field(default_factory=lambda: [8, 8])
    level_distributions: List[Dict[int, float]] = field(default_factory=list)

    def color_cost_hier(
        self,
        mode: ColorMode,
        N: int,
        m_per_level: Optional[List[int]] = None
    ) -> float:
        """
        Calcule le coût couleur hiérarchique.

        Args:
            mode: Mode de codage
            N: Nombre de sauts
            m_per_level: Nombre de couleurs par niveau

        Returns:
            Coût total en bits
        """
        if N == 0:
            return 0.0

        if mode == ColorMode.UNIFORM:
            # Bb: N · 8L bits
            return N * sum(self.level_bits)

        elif mode == ColorMode.HUFFMAN:
            # Bc: N·Σ H(P̂_i) + Σ D_huf_i
            total = 0.0
            for i, dist in enumerate(self.level_distributions):
                if not dist:
                    continue
                # Entropie du niveau
                H = sum(-p * np.log2(p) for p in dist.values() if p > 0)
                # Overhead dictionnaire
                m = max(dist.keys()) + 1 if dist else 256
                D_huf = m * (int(np.floor(np.log2(m))) + 1)
                total += N * H + D_huf
            return total

        elif mode == ColorMode.SEQUENTIAL:
            # Complexe pour hiérarchique - simplifié
            return N * sum(self.level_bits) * 0.3  # Estimation

        return N * sum(self.level_bits)

    def downgrade(self, target_level: int) -> 'HierarchicalColor':
        """Réduit au niveau spécifié."""
        new_levels = min(target_level, self.levels)
        return HierarchicalColor(
            levels=new_levels,
            level_bits=self.level_bits[:new_levels],
            level_distributions=self.level_distributions[:new_levels]
        )


@dataclass
class QuadTreeNode:
    """Nœud de l'arbre quadtree."""
    x: int
    y: int
    size: int
    level: int
    is_leaf: bool = True
    children: List['QuadTreeNode'] = field(default_factory=list)
    features: Optional[BlockFeatures] = None
    cartouche: Optional[Cartouche] = None
    encoded_bits: int = 0


class SpatialProcessTree:
    """
    Arbre quadtree pour compression spatiale adaptative.

    Critère de subdivision MDL:
    L4(parent, B) > Σ L4(enfants, B) + log₂(4)
    """

    def __init__(self, width: int, height: int, min_size: int = 2):
        self.width = width
        self.height = height
        self.min_size = min_size
        self.root: Optional[QuadTreeNode] = None

    def build(self, video_block: np.ndarray, features_func) -> QuadTreeNode:
        """
        Construit l'arbre quadtree.

        Args:
            video_block: Bloc vidéo (T, H, W)
            features_func: Fonction pour calculer les features d'un sous-bloc

        Returns:
            Racine de l'arbre
        """
        T, H, W = video_block.shape
        self.root = self._build_node(video_block, 0, 0, min(H, W), 0, features_func)
        return self.root

    def _build_node(
        self,
        video_block: np.ndarray,
        x: int, y: int,
        size: int,
        level: int,
        features_func
    ) -> QuadTreeNode:
        """Construit récursivement un nœud."""
        T, H, W = video_block.shape

        # Extrait le sous-bloc
        sub_block = video_block[:, y:y+size, x:x+size]
        features = features_func(sub_block)

        node = QuadTreeNode(
            x=x, y=y, size=size, level=level,
            features=features
        )

        # Coût du parent
        L_parent = self._compute_block_cost(features)

        # Test de subdivision
        if size > self.min_size:
            half = size // 2
            children_cost = np.log2(4)  # Overhead structure

            # Calcul des enfants potentiels
            children = []
            for dy in [0, half]:
                for dx in [0, half]:
                    if y + dy + half <= H and x + dx + half <= W:
                        child = self._build_node(
                            video_block, x + dx, y + dy,
                            half, level + 1, features_func
                        )
                        children.append(child)
                        children_cost += child.encoded_bits

            # Décision: subdiviser si les enfants coûtent moins
            if children_cost < L_parent:
                node.is_leaf = False
                node.children = children
                node.encoded_bits = int(children_cost)
            else:
                node.is_leaf = True
                node.encoded_bits = int(L_parent)
        else:
            node.encoded_bits = int(L_parent)

        return node

    def _compute_block_cost(self, features: BlockFeatures) -> float:
        """Calcule le coût L4 d'un bloc."""
        N = features.N
        r = features.r
        m = features.m

        if N == 0:
            return 0.0

        from ..utils.math_utils import logC
        L_temp = np.log2(N + 1) + logC(r, N)

        # Meilleur coût couleur
        H = features.H_color
        log2_m = np.log2(m) if m > 1 else 0
        D_huf = features.get_huffman_overhead()

        C_colors = [
            log2_m + features.N_trans * log2_m,  # Ba
            N * log2_m,                           # Bb
            N * H + D_huf                         # Bc
        ]

        return L_temp + min(C_colors)

    def get_leaves(self) -> List[QuadTreeNode]:
        """Retourne toutes les feuilles."""
        leaves = []
        self._collect_leaves(self.root, leaves)
        return leaves

    def _collect_leaves(self, node: Optional[QuadTreeNode], leaves: List):
        if node is None:
            return
        if node.is_leaf:
            leaves.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, leaves)

    def get_total_bits(self) -> int:
        """Retourne le nombre total de bits."""
        if self.root is None:
            return 0
        return self.root.encoded_bits


class ZoomController:
    """
    Contrôleur de zoom adaptatif.

    Adapte le niveau de détail (dim. F, G) selon la bande passante
    et les demandes de zoom.
    """

    def __init__(self, max_latency_ms: float = 100.0):
        self.max_latency_ms = max_latency_ms
        self.current_level = 0
        self.cached_levels: Dict[int, bytes] = {}

    def get_level_for_bandwidth(
        self,
        bandwidth_kbps: float,
        block_bits: int,
        fps: float = 30.0
    ) -> int:
        """
        Détermine le niveau optimal pour la bande passante.

        Args:
            bandwidth_kbps: Bande passante en kbit/s
            block_bits: Bits par bloc au niveau max
            fps: Images par seconde

        Returns:
            Niveau optimal (0 = max qualité, 3 = min)
        """
        bits_per_frame = block_bits
        bits_per_second = bits_per_frame * fps

        if bits_per_second <= bandwidth_kbps * 1000:
            return 0  # Qualité max
        elif bits_per_second <= bandwidth_kbps * 1000 * 2:
            return 1  # Légère réduction
        elif bits_per_second <= bandwidth_kbps * 1000 * 4:
            return 2  # Réduction moyenne
        else:
            return 3  # Réduction forte

    def adapt_cartouche(
        self,
        cartouche: Cartouche,
        target_level: int
    ) -> Cartouche:
        """
        Adapte le cartouche au niveau de qualité.

        Args:
            cartouche: Cartouche original
            target_level: Niveau cible (0-3)

        Returns:
            Cartouche adapté
        """
        # Copie du cartouche
        adapted = Cartouche(
            A=cartouche.A,
            B=cartouche.B,
            C=cartouche.C,
            D=cartouche.D,
            E=cartouche.E,
            F=cartouche.F,
            G=cartouche.G,
            H=cartouche.H
        )

        if target_level >= 1:
            # Réduire la résolution chromatique
            adapted.F = max(1, cartouche.F - target_level)

        if target_level >= 2:
            # Réduire la résolution spatiale
            adapted.G = min(3, cartouche.G + 1)

        if target_level >= 3:
            # Passer en mode séquentiel (moins de bits couleur)
            adapted.B = ColorMode.SEQUENTIAL

        return adapted


# === Estimateurs d'intensité ===

class IntensityEstimator(ABC):
    """Estimateur d'intensité de base."""

    @abstractmethod
    def fit(self, jump_times: np.ndarray, r: int) -> np.ndarray:
        """Estime α̂(t) sur les bins."""
        pass

    @abstractmethod
    def get_cdf(self) -> np.ndarray:
        """Retourne la CDF Λ̂(t)/Λ̂(T)."""
        pass


class HistogramEstimator(IntensityEstimator):
    """Estimateur par histogramme."""

    def __init__(self, n_bins: int = 16):
        self.n_bins = n_bins
        self.alpha: Optional[np.ndarray] = None
        self.cdf: Optional[np.ndarray] = None

    def fit(self, jump_times: np.ndarray, r: int) -> np.ndarray:
        bin_width = r / self.n_bins
        counts = np.zeros(self.n_bins)

        for t in jump_times:
            bin_idx = min(int(t / bin_width), self.n_bins - 1)
            counts[bin_idx] += 1

        # Intensité = comptage / largeur
        self.alpha = counts / bin_width

        # CDF
        cumsum = np.cumsum(self.alpha * bin_width)
        self.cdf = cumsum / cumsum[-1] if cumsum[-1] > 0 else np.linspace(0, 1, self.n_bins)

        return self.alpha

    def get_cdf(self) -> np.ndarray:
        return self.cdf if self.cdf is not None else np.array([])


class SplineEstimator(IntensityEstimator):
    """Estimateur par splines cubiques."""

    def __init__(self, n_knots: int = 10):
        self.n_knots = n_knots
        self.alpha: Optional[np.ndarray] = None
        self.cdf: Optional[np.ndarray] = None

    def fit(self, jump_times: np.ndarray, r: int) -> np.ndarray:
        # Simplifié: utilise un lissage par moyenne mobile
        n_points = r
        self.alpha = np.zeros(n_points)

        for t in jump_times:
            idx = int(t) % n_points
            self.alpha[idx] += 1

        # Lissage
        from scipy.ndimage import uniform_filter1d
        try:
            self.alpha = uniform_filter1d(self.alpha, size=max(1, n_points // self.n_knots))
        except:
            pass

        # CDF
        cumsum = np.cumsum(self.alpha)
        self.cdf = cumsum / cumsum[-1] if cumsum[-1] > 0 else np.linspace(0, 1, n_points)

        return self.alpha

    def get_cdf(self) -> np.ndarray:
        return self.cdf if self.cdf is not None else np.array([])


class HierarchyAgent:
    """
    Agent 7: Hiérarchie + Arbre + Zoom

    Gère les fonctionnalités avancées F1-F6.
    """

    def __init__(self):
        self.estimators = {
            IntensityFamily.HISTOGRAM: HistogramEstimator(),
            IntensityFamily.SPLINES: SplineEstimator(),
        }
        self.zoom_controller = ZoomController()

    def build_quadtree(
        self,
        video_block: np.ndarray,
        features_func
    ) -> SpatialProcessTree:
        """Construit l'arbre quadtree."""
        T, H, W = video_block.shape
        tree = SpatialProcessTree(W, H)
        tree.build(video_block, features_func)
        return tree

    def estimate_intensity(
        self,
        jump_times: np.ndarray,
        r: int,
        family: IntensityFamily
    ) -> np.ndarray:
        """Estime l'intensité."""
        if family in self.estimators:
            return self.estimators[family].fit(jump_times, r)
        return np.ones(r) / r

    def compute_hierarchical_cost(
        self,
        features: BlockFeatures,
        n_levels: int,
        mode: ColorMode
    ) -> float:
        """Calcule le coût hiérarchique."""
        hier = HierarchicalColor(levels=n_levels)
        return hier.color_cost_hier(mode, features.N)
