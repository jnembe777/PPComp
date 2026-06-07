"""
agent_0_extraction.py - Agent d'Extraction Bitwise
===================================================

Phase 1 - Dimensions A, B

Fonctions:
- Extraction des masques binaires M_c(t,p)
- Calcul des 7 features bitwise
- Distribution empirique P̂(m_k) pour dim. B
- Opérations XOR / POPCOUNT / TZCNT

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..core.features import BlockFeatures
from ..core.process_types import ColorMode
from ..utils.bitwise import (
    BinaryMask, popcount, xor_frames, count_transitions,
    compute_spatial_homogeneity_from_frame, compute_spatial_autocorrelation
)


@dataclass
class ExtractionResult:
    """Résultat de l'extraction bitwise."""
    features: BlockFeatures
    masks: Dict[int, List[BinaryMask]]  # color -> masks par frame
    jump_positions: List[Tuple[int, int, int, int]]  # (frame, y, x, color)
    color_costs: Dict[ColorMode, float]


class ExtractionAgent:
    """
    Agent 0: Extraction Bitwise

    Extrait les caractéristiques d'un bloc vidéo pour déterminer
    le cartouche ABCDEFGH optimal.
    """

    def __init__(self, block_width: int = 16, block_height: int = 16):
        """
        Initialise l'agent.

        Args:
            block_width: Largeur du bloc en pixels
            block_height: Hauteur du bloc en pixels
        """
        self.block_width = block_width
        self.block_height = block_height

    def extract(self, video_block: np.ndarray) -> ExtractionResult:
        """
        Extrait les features d'un bloc vidéo.

        Args:
            video_block: Bloc vidéo (T, H, W) avec indices de couleur

        Returns:
            ExtractionResult avec features et données extraites
        """
        T, H, W = video_block.shape
        r = T  # Nombre de bins temporels

        # 1. Détection des couleurs présentes
        unique_colors = np.unique(video_block)
        m = len(unique_colors)
        color_map = {c: i for i, c in enumerate(unique_colors)}

        # 2. Extraction des masques et sauts
        masks = {c: [] for c in range(m)}
        jumps = []  # (frame, y, x, new_color)
        all_marks = []  # Séquence des marques couleur

        for t in range(T):
            frame = video_block[t]

            # Masques pour chaque couleur
            for orig_c in unique_colors:
                c = color_map[orig_c]
                mask = BinaryMask.from_frame(frame, orig_c)
                masks[c].append(mask)

            # Détection des sauts (comparaison avec frame précédente)
            if t > 0:
                prev_frame = video_block[t - 1]
                xor = xor_frames(prev_frame, frame)

                for y in range(H):
                    for x in range(W):
                        if xor[y, x]:
                            new_color = color_map[frame[y, x]]
                            jumps.append((t, y, x, new_color))
                            all_marks.append(new_color)

        # 3. Calcul des features
        N = len(jumps)
        marks_array = np.array(all_marks, dtype=int) if all_marks else np.array([], dtype=int)

        features = self._compute_features(
            N=N,
            r=r,
            m=m,
            n_pixels=H * W,
            marks=marks_array,
            masks=masks,
            jumps=jumps,
            video_block=video_block
        )

        # 4. Calcul des coûts couleur pour chaque mode B
        color_costs = self._compute_color_costs(features)

        return ExtractionResult(
            features=features,
            masks=masks,
            jump_positions=jumps,
            color_costs=color_costs
        )

    def _compute_features(
        self,
        N: int,
        r: int,
        m: int,
        n_pixels: int,
        marks: np.ndarray,
        masks: Dict[int, List[BinaryMask]],
        jumps: List[Tuple],
        video_block: np.ndarray
    ) -> BlockFeatures:
        """Calcule les 7 features bitwise."""

        # lambda_avg: Intensité moyenne
        lambda_avg = N / r if r > 0 else 0.0

        # H_color: Entropie des couleurs
        H_color = 0.0
        color_dist = {}
        if len(marks) > 0:
            counts = np.bincount(marks, minlength=m)
            total = counts.sum()
            for c in range(m):
                if counts[c] > 0:
                    p = counts[c] / total
                    color_dist[c] = p
                    H_color -= p * np.log2(p)

        # N_trans: Nombre de transitions couleur
        N_trans = count_transitions(marks) if len(marks) > 1 else 0

        # H_s: Homogénéité spatiale (sur la dernière frame)
        # Mesure la fraction de paires de pixels voisins ayant la même couleur
        H_s = 0.0
        if video_block is not None and len(video_block) > 0:
            last_frame = video_block[-1]
            H_s = compute_spatial_homogeneity_from_frame(last_frame)

        # rho_corr: Autocorrélation spatiale des couleurs
        # Mesure la corrélation entre un pixel et ses voisins
        rho_corr = 0.5  # Valeur neutre par défaut
        if video_block is not None and len(video_block) > 0:
            last_frame = video_block[-1]
            rho_corr = compute_spatial_autocorrelation(last_frame)

        # R_temp: Régularité temporelle
        R_temp = 1.0
        if len(jumps) >= 2:
            # Calcul des intervalles entre sauts
            times = np.array([j[0] for j in sorted(jumps, key=lambda x: x[0])])
            if len(times) >= 2:
                intervals = np.diff(times)
                if len(intervals) > 0:
                    mean_int = np.mean(intervals)
                    var_int = np.var(intervals)
                    if var_int > 0:
                        R_temp = (mean_int ** 2) / var_int

        # m_eff: Couleurs effectives (avec activité significative)
        m_eff = 0
        if len(marks) > 0:
            counts = np.bincount(marks, minlength=m)
            threshold = max(2, N * 0.01)  # Au moins 1% des sauts
            m_eff = np.sum(counts >= threshold)

        return BlockFeatures(
            N=N,
            r=r,
            m=m,
            n_pixels=n_pixels,
            lambda_avg=lambda_avg,
            H_s=H_s,
            rho_corr=rho_corr,
            R_temp=R_temp,
            m_eff=m_eff,
            H_color=H_color,
            N_trans=N_trans,
            color_dist=color_dist
        )

    def _compute_color_costs(self, features: BlockFeatures) -> Dict[ColorMode, float]:
        """Calcule C_color(B) pour chaque mode."""
        N = features.N
        m = features.m
        H = features.H_color
        N_trans = features.N_trans

        if N == 0 or m <= 1:
            return {mode: 0.0 for mode in ColorMode}

        log2_m = np.log2(m)
        D_huf = features.get_huffman_overhead()

        costs = {
            ColorMode.SEQUENTIAL: log2_m + N_trans * log2_m,  # Ba
            ColorMode.UNIFORM: N * log2_m,                     # Bb
            ColorMode.HUFFMAN: N * H + D_huf,                  # Bc
            ColorMode.ELIAS: N * (log2_m + 2 * np.log2(max(1, np.log2(m))))  # Bd
        }

        return costs

    def extract_jumps_bitwise(
        self,
        frames: np.ndarray,
        color: int
    ) -> List[int]:
        """
        Extrait les temps de sauts pour une couleur spécifique.

        Utilise XOR + POPCOUNT + TZCNT pour extraction rapide.

        Args:
            frames: Séquence de frames (T, H, W)
            color: Indice de la couleur

        Returns:
            Liste des temps de sauts
        """
        T = frames.shape[0]
        jump_times = []

        prev_mask = BinaryMask.from_frame(frames[0], color)

        for t in range(1, T):
            curr_mask = BinaryMask.from_frame(frames[t], color)
            xor_mask = prev_mask.xor(curr_mask)

            # Si des bits diffèrent, il y a des sauts
            if xor_mask.popcount() > 0:
                jump_times.append(t)

            prev_mask = curr_mask

        return jump_times

    def compute_distribution(self, marks: np.ndarray, m: int) -> Dict[int, float]:
        """
        Calcule la distribution empirique P̂(m_k).

        Args:
            marks: Séquence des marques couleur
            m: Nombre de couleurs

        Returns:
            Distribution {couleur: probabilité}
        """
        if len(marks) == 0:
            return {}

        counts = np.bincount(marks, minlength=m)
        total = counts.sum()

        return {c: counts[c] / total for c in range(m) if counts[c] > 0}

    def select_best_color_mode(
        self,
        features: BlockFeatures
    ) -> Tuple[ColorMode, Dict[ColorMode, float]]:
        """
        Sélectionne le mode B optimal.

        Args:
            features: Caractéristiques du bloc

        Returns:
            (meilleur_mode, dictionnaire_des_coûts)
        """
        costs = self._compute_color_costs(features)
        best = min(costs, key=costs.get)
        return best, costs


def create_test_video_block(
    T: int = 64,
    H: int = 16,
    W: int = 16,
    m: int = 4,
    jump_rate: float = 0.1
) -> np.ndarray:
    """
    Crée un bloc vidéo de test.

    Args:
        T: Nombre de frames
        H, W: Dimensions spatiales
        m: Nombre de couleurs
        jump_rate: Taux de sauts par pixel par frame

    Returns:
        Bloc vidéo (T, H, W)
    """
    video = np.zeros((T, H, W), dtype=int)

    # État initial aléatoire
    video[0] = np.random.randint(0, m, size=(H, W))

    # Génération des sauts
    for t in range(1, T):
        video[t] = video[t - 1].copy()

        # Sauts aléatoires
        jump_mask = np.random.random((H, W)) < jump_rate
        new_colors = np.random.randint(0, m, size=(H, W))
        video[t][jump_mask] = new_colors[jump_mask]

    return video
