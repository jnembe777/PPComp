"""
agent_4_precoding.py - Agent Précodage R1-R4b (Dimension C)
============================================================

Phase 2 - Dimension C

Représentations temporelles:
- R1: Timestamps
- R2: Count (histogramme)
- R3: Intervalles
- R4a: Boolean
- R4b: Combinatoire (N, Index) - OPTIMAL

L_i(B) = L_temporel_i + C_color(B)

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..core.process_types import ColorMode, Representation
from ..core.features import BlockFeatures
from ..utils.math_utils import logC, binomial, combinatorial_index, decode_combinatorial_index


@dataclass
class PrecodingResult:
    """Résultat du précodage."""
    best_repr: Representation
    best_color_mode: ColorMode
    best_total_cost: float
    temporal_costs: Dict[Representation, float]
    total_costs: Dict[Tuple[Representation, ColorMode], float]
    encoded_data: Optional[bytes] = None


class PrecodingAgent:
    """
    Agent 4: Précodage R1-R4b (Dimension C)

    Sélectionne la représentation temporelle optimale
    et calcule L_i(B) pour les 20 combinaisons C×B.
    """

    def __init__(self):
        pass

    def analyze(
        self,
        jump_times: np.ndarray,
        N: int,
        r: int,
        m: int,
        features: BlockFeatures
    ) -> PrecodingResult:
        """
        Analyse et sélectionne la meilleure combinaison (C, B).

        Args:
            jump_times: Temps des sauts
            N: Nombre de sauts
            r: Nombre de bins
            m: Nombre de couleurs
            features: Caractéristiques du bloc

        Returns:
            PrecodingResult avec combinaison optimale
        """
        # 1. Calcul des longueurs temporelles (indépendantes de B)
        temporal_costs = {
            Representation.TIMESTAMPS: self.L_temporal_R1(N, r),
            Representation.COUNT: self.L_temporal_R2(N, r),
            Representation.INTERVALS: self.L_temporal_R3(N, r),
            Representation.BOOLEAN: self.L_temporal_R4a(r),
            Representation.COMBINATORIAL: self.L_temporal_R4b(N, r)
        }

        # 2. Calcul des coûts couleur
        color_costs = self._compute_color_costs(features)

        # 3. Calcul des 20 combinaisons
        total_costs = {}
        for repr_type, L_temp in temporal_costs.items():
            for color_mode, C_color in color_costs.items():
                total_costs[(repr_type, color_mode)] = L_temp + C_color

        # 4. Sélection du minimum
        best_combo = min(total_costs, key=total_costs.get)
        best_repr, best_color_mode = best_combo

        return PrecodingResult(
            best_repr=best_repr,
            best_color_mode=best_color_mode,
            best_total_cost=total_costs[best_combo],
            temporal_costs=temporal_costs,
            total_costs=total_costs
        )

    def L_temporal_R1(self, N: int, r: int) -> float:
        """
        Longueur temporelle R1 (Timestamps).

        L^temporel_R1 = log₂N + N·log₂r

        Args:
            N: Nombre de sauts
            r: Nombre de bins

        Returns:
            Longueur en bits
        """
        if N == 0:
            return 0.0
        return np.log2(N + 1) + N * np.log2(r)

    def L_temporal_R2(self, N: int, r: int) -> float:
        """
        Longueur temporelle R2 (Count/Histogramme).

        L^temporel_R2 = r · log₂(N/r + 1)

        Args:
            N: Nombre de sauts
            r: Nombre de bins

        Returns:
            Longueur en bits
        """
        if r == 0:
            return 0.0
        avg = N / r
        return r * np.log2(avg + 1)

    def L_temporal_R3(self, N: int, r: int) -> float:
        """
        Longueur temporelle R3 (Intervalles).

        L^temporel_R3 = N · log₂r

        Args:
            N: Nombre de sauts
            r: Nombre de bins

        Returns:
            Longueur en bits
        """
        if N == 0:
            return 0.0
        return N * np.log2(r)

    def L_temporal_R4a(self, r: int) -> float:
        """
        Longueur temporelle R4a (Boolean).

        L^temporel_R4a = r bits

        Args:
            r: Nombre de bins

        Returns:
            Longueur en bits
        """
        return float(r)

    def L_temporal_R4b(self, N: int, r: int) -> float:
        """
        Longueur temporelle R4b (Combinatoire).

        L^temporel_R4b = log₂N + log₂C(r, N)

        C'est la BORNE INFÉRIEURE pour le codage uniforme.

        Args:
            N: Nombre de sauts
            r: Nombre de bins

        Returns:
            Longueur en bits
        """
        if N == 0:
            return 0.0
        return np.log2(N + 1) + logC(r, N)

    def _compute_color_costs(self, features: BlockFeatures) -> Dict[ColorMode, float]:
        """Calcule les coûts couleur pour chaque mode."""
        N = features.N
        m = features.m
        H = features.H_color
        N_trans = features.N_trans

        if N == 0 or m <= 1:
            return {mode: 0.0 for mode in ColorMode}

        log2_m = np.log2(m)
        D_huf = features.get_huffman_overhead()

        return {
            ColorMode.SEQUENTIAL: log2_m + N_trans * log2_m,
            ColorMode.UNIFORM: N * log2_m,
            ColorMode.HUFFMAN: N * H + D_huf,
            ColorMode.ELIAS: N * (log2_m + 2 * np.log2(max(1, log2_m)))
        }

    def encode_R1(self, jump_times: np.ndarray, r: int) -> Tuple[int, np.ndarray]:
        """
        Encode en représentation R1 (Timestamps).

        Args:
            jump_times: Temps des sauts
            r: Nombre de bins

        Returns:
            (N, times_quantifiés)
        """
        N = len(jump_times)
        # Quantification sur r bins
        quantized = np.clip(jump_times.astype(int), 0, r - 1)
        return N, np.sort(quantized)

    def encode_R2(self, jump_times: np.ndarray, r: int) -> np.ndarray:
        """
        Encode en représentation R2 (Count).

        Args:
            jump_times: Temps des sauts
            r: Nombre de bins

        Returns:
            Histogramme des comptages
        """
        return np.bincount(jump_times.astype(int).clip(0, r-1), minlength=r)

    def encode_R3(self, jump_times: np.ndarray) -> np.ndarray:
        """
        Encode en représentation R3 (Intervalles).

        Args:
            jump_times: Temps des sauts triés

        Returns:
            Intervalles Δ_k
        """
        if len(jump_times) <= 1:
            return np.array([])
        sorted_times = np.sort(jump_times)
        return np.diff(sorted_times)

    def encode_R4a(self, jump_times: np.ndarray, r: int) -> np.ndarray:
        """
        Encode en représentation R4a (Boolean).

        Args:
            jump_times: Temps des sauts
            r: Nombre de bins

        Returns:
            Vecteur booléen de longueur r
        """
        boolean = np.zeros(r, dtype=bool)
        for t in jump_times:
            idx = int(t)
            if 0 <= idx < r:
                boolean[idx] = True
        return boolean

    def encode_R4b(self, jump_times: np.ndarray, r: int) -> Tuple[int, int]:
        """
        Encode en représentation R4b (Combinatoire).

        Args:
            jump_times: Temps des sauts
            r: Nombre de bins

        Returns:
            (N, Index combinatoire)
        """
        N = len(jump_times)
        if N == 0:
            return 0, 0

        # Positions triées
        positions = np.sort(jump_times.astype(int).clip(0, r-1))
        positions = np.unique(positions)  # Supprime les doublons

        # Calcul de l'index combinatoire
        index = combinatorial_index(positions, r)

        return len(positions), index

    def decode_R4b(self, N: int, index: int, r: int) -> np.ndarray:
        """
        Décode la représentation R4b.

        Args:
            N: Nombre de sauts
            index: Index combinatoire
            r: Nombre de bins

        Returns:
            Positions des sauts
        """
        return decode_combinatorial_index(index, N, r)

    def select_repr_and_color(
        self,
        N: int,
        r: int,
        m: int,
        color_dist: Dict[int, float],
        N_trans: int
    ) -> Tuple[Representation, ColorMode, float]:
        """
        Sélectionne la combinaison (C, B) optimale.

        Explore les 5×4 = 20 combinaisons.

        Args:
            N: Nombre de sauts
            r: Nombre de bins
            m: Nombre de couleurs
            color_dist: Distribution des couleurs
            N_trans: Nombre de transitions

        Returns:
            (repr_optimale, mode_couleur_optimal, coût_total)
        """
        # Créer des features temporaires
        H = sum(-p * np.log2(p) for p in color_dist.values() if p > 0) if color_dist else 0.0

        features = BlockFeatures(
            N=N, r=r, m=m,
            H_color=H, N_trans=N_trans,
            color_dist=color_dist
        )

        result = self.analyze(
            jump_times=np.array([]),  # Non utilisé pour le calcul des coûts
            N=N, r=r, m=m,
            features=features
        )

        return result.best_repr, result.best_color_mode, result.best_total_cost

    def get_cost_matrix(
        self,
        N: int,
        r: int,
        m: int,
        H: float,
        N_trans: int
    ) -> np.ndarray:
        """
        Retourne la matrice 5×4 des coûts totaux.

        Args:
            N, r, m: Paramètres du bloc
            H: Entropie couleur
            N_trans: Nombre de transitions

        Returns:
            Matrice (5 repr × 4 modes)
        """
        # Longueurs temporelles
        L_temp = [
            self.L_temporal_R1(N, r),
            self.L_temporal_R2(N, r),
            self.L_temporal_R3(N, r),
            self.L_temporal_R4a(r),
            self.L_temporal_R4b(N, r)
        ]

        # Coûts couleur
        log2_m = np.log2(m) if m > 1 else 0
        D_huf = m * (int(np.floor(log2_m)) + 1) if m > 1 else 0

        C_color = [
            log2_m + N_trans * log2_m,  # Ba
            N * log2_m,                  # Bb
            N * H + D_huf,               # Bc
            N * (log2_m + 2 * np.log2(max(1, log2_m)))  # Bd
        ]

        # Matrice des coûts totaux
        matrix = np.zeros((5, 4))
        for i in range(5):
            for j in range(4):
                matrix[i, j] = L_temp[i] + C_color[j]

        return matrix
