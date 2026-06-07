"""
agent_3_structures.py - Agent Structures + Features
====================================================

Phase 2 - Dimensions A, B

Gestion des dataclasses avec intégration du mode B:
- color_cost(mode_B) pour chaque structure
- select_color_mode() pour choix optimal
- Primitives d'accès universelles

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..core.process_types import ProcessType, ColorMode, Representation
from ..core.point_process import (
    PointProcess, MarkedProcess, MonochromaticProcess,
    VectorialProcess, MarkovianProcess
)
from ..core.features import BlockFeatures


@dataclass
class ProcessData:
    """
    Conteneur universel pour les données de processus.

    Encapsule n'importe quel type de processus avec les méthodes
    d'accès standardisées.
    """
    process: PointProcess
    process_type: ProcessType
    features: BlockFeatures
    optimal_color_mode: Optional[ColorMode] = None
    color_costs: Dict[ColorMode, float] = field(default_factory=dict)

    def get_jump_times(self) -> np.ndarray:
        """Retourne les temps de sauts."""
        return self.process.get_jump_times()

    def get_N(self) -> int:
        """Retourne le nombre de sauts."""
        return self.process.get_N()

    def get_color_distribution(self) -> Dict[int, float]:
        """Retourne la distribution des couleurs."""
        return self.process.get_color_distribution()

    def color_cost(self, mode: ColorMode) -> float:
        """Retourne le coût couleur pour un mode donné."""
        if mode in self.color_costs:
            return self.color_costs[mode]
        return self.process.color_cost(mode)


class StructuresAgent:
    """
    Agent 3: Structures + Features

    Crée et gère les dataclasses pour les différents types
    de processus avec intégration du mode couleur.
    """

    def __init__(self):
        pass

    def create_process_data(
        self,
        jumps: List[Tuple[float, int]],
        r: int,
        m: int,
        features: BlockFeatures
    ) -> ProcessData:
        """
        Crée une structure ProcessData à partir des sauts.

        Choisit automatiquement le type de processus optimal.

        Args:
            jumps: Liste (temps, couleur)
            r: Nombre de bins temporels
            m: Nombre de couleurs
            features: Caractéristiques pré-calculées

        Returns:
            ProcessData avec type optimal
        """
        # Détermination du type optimal
        process_type = features.suggest_process_type()

        # Création du processus selon le type
        if process_type == ProcessType.MONOCHROMATIC:
            process = self._create_monochromatic(jumps, r, m)
        elif process_type in (ProcessType.VECTORIAL_MARG, ProcessType.VECTORIAL_JOINT):
            process = self._create_vectorial(jumps, r, m, process_type)
        elif process_type == ProcessType.MARKOVIAN:
            process = self._create_markovian(jumps, r, m)
        else:
            process = MarkedProcess(jumps=jumps, r=r, m=m)

        # Calcul des coûts couleur
        color_costs = self._compute_all_color_costs(process, features)

        # Mode optimal
        if process_type == ProcessType.MONOCHROMATIC:
            optimal_mode = ColorMode.UNIFORM  # C_color = 0 de toute façon
        else:
            optimal_mode = min(color_costs, key=color_costs.get)

        return ProcessData(
            process=process,
            process_type=process_type,
            features=features,
            optimal_color_mode=optimal_mode,
            color_costs=color_costs
        )

    def _create_monochromatic(
        self,
        jumps: List[Tuple[float, int]],
        r: int,
        m: int
    ) -> MonochromaticProcess:
        """Crée un processus monochromatique."""
        processes = {c: [] for c in range(m)}
        for time, color in jumps:
            if 0 <= color < m:
                processes[color].append(time)

        # Retire les couleurs vides
        processes = {c: times for c, times in processes.items() if times}

        return MonochromaticProcess(processes=processes, r=r)

    def _create_vectorial(
        self,
        jumps: List[Tuple[float, int]],
        r: int,
        m: int,
        vec_type: ProcessType
    ) -> VectorialProcess:
        """Crée un processus vectoriel."""
        # Divise les sauts par couleur
        components = []
        for c in range(m):
            c_jumps = [(t, c) for t, col in jumps if col == c]
            if c_jumps:
                components.append(MarkedProcess(jumps=c_jumps, r=r, m=m))

        is_joint = (vec_type == ProcessType.VECTORIAL_JOINT)
        return VectorialProcess(components=components, is_joint=is_joint, r=r, m=m)

    def _create_markovian(
        self,
        jumps: List[Tuple[float, int]],
        r: int,
        m: int
    ) -> MarkovianProcess:
        """Crée un processus markovien."""
        states = list(range(m))
        transitions = []

        # Reconstruit les transitions
        sorted_jumps = sorted(jumps, key=lambda x: x[0])
        for i in range(1, len(sorted_jumps)):
            time = sorted_jumps[i][0]
            from_state = sorted_jumps[i-1][1]
            to_state = sorted_jumps[i][1]
            if from_state != to_state:
                transitions.append((time, from_state, to_state))

        process = MarkovianProcess(
            states=states,
            transitions=transitions,
            r=r
        )
        process.build_transition_matrix()

        return process

    def _compute_all_color_costs(
        self,
        process: PointProcess,
        features: BlockFeatures
    ) -> Dict[ColorMode, float]:
        """Calcule les coûts couleur pour tous les modes."""
        costs = {}
        for mode in ColorMode:
            costs[mode] = process.color_cost(mode)
        return costs

    def select_color_mode(
        self,
        N: int,
        m: int,
        color_dist: Dict[int, float],
        N_trans: int
    ) -> Tuple[ColorMode, Dict[ColorMode, float]]:
        """
        Sélectionne le mode B optimal.

        Args:
            N: Nombre de sauts
            m: Nombre de couleurs
            color_dist: Distribution des couleurs
            N_trans: Nombre de transitions

        Returns:
            (meilleur_mode, dictionnaire_coûts)
        """
        if N == 0 or m <= 1:
            return ColorMode.UNIFORM, {mode: 0.0 for mode in ColorMode}

        # Entropie
        H = 0.0
        for p in color_dist.values():
            if p > 0:
                H -= p * np.log2(p)

        log2_m = np.log2(m)
        D_huf = m * (int(np.floor(log2_m)) + 1)

        costs = {
            ColorMode.SEQUENTIAL: log2_m + N_trans * log2_m,
            ColorMode.UNIFORM: N * log2_m,
            ColorMode.HUFFMAN: N * H + D_huf,
            ColorMode.ELIAS: N * (log2_m + 2 * np.log2(max(1, log2_m)))
        }

        best = min(costs, key=costs.get)
        return best, costs

    def get_primitives(self, data: ProcessData) -> Dict[str, Any]:
        """
        Retourne les primitives d'accès universelles.

        Args:
            data: ProcessData

        Returns:
            Dict avec les fonctions primitives
        """
        return {
            "get_jump_times": data.get_jump_times,
            "get_N": data.get_N,
            "get_color_distribution": data.get_color_distribution,
            "color_cost": data.color_cost,
            "process_type": data.process_type,
            "optimal_color_mode": data.optimal_color_mode
        }


class CompressedVideoProcessor:
    """
    Processeur de vidéo compressée.

    Gère la transformation vidéo -> ProcessData et vice-versa.
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.structures_agent = StructuresAgent()

    def video_to_process_data(
        self,
        video_block: np.ndarray,
        features: BlockFeatures
    ) -> ProcessData:
        """
        Convertit un bloc vidéo en ProcessData.

        Args:
            video_block: Bloc (T, H, W)
            features: Caractéristiques pré-calculées

        Returns:
            ProcessData
        """
        T, H, W = video_block.shape

        # Extraction des sauts
        jumps = []
        for t in range(1, T):
            for y in range(H):
                for x in range(W):
                    if video_block[t, y, x] != video_block[t-1, y, x]:
                        jumps.append((float(t), int(video_block[t, y, x])))

        # Détermination de m
        m = int(video_block.max()) + 1

        return self.structures_agent.create_process_data(
            jumps=jumps,
            r=T,
            m=m,
            features=features
        )

    def process_data_to_video(
        self,
        data: ProcessData,
        initial_frame: np.ndarray,
        T: int
    ) -> np.ndarray:
        """
        Reconstruit un bloc vidéo depuis ProcessData.

        Args:
            data: ProcessData
            initial_frame: Frame initiale (H, W)
            T: Nombre de frames

        Returns:
            Bloc vidéo (T, H, W)
        """
        H, W = initial_frame.shape
        video = np.zeros((T, H, W), dtype=initial_frame.dtype)
        video[0] = initial_frame.copy()

        # Application des sauts
        jump_times = data.get_jump_times()
        # ... reconstruction selon le type de processus

        return video
