"""
point_process.py - Structures de données pour les processus ponctuels
======================================================================

Classes pour les 5 types de processus (dim. A):
- MarkedProcess (Aa)
- MonochromaticProcess (Ab)
- VectorialProcess (Ac/Ad)
- MarkovianProcess (Ae)

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
import numpy as np

from .process_types import ColorMode, ProcessType


class PointProcess(ABC):
    """Classe abstraite pour tous les processus ponctuels."""

    @abstractmethod
    def get_jump_times(self) -> np.ndarray:
        """Retourne les temps de sauts."""
        pass

    @abstractmethod
    def get_N(self) -> int:
        """Retourne le nombre de sauts."""
        pass

    @abstractmethod
    def color_cost(self, mode: ColorMode, **kwargs) -> float:
        """Calcule le coût de codage couleur selon le mode B."""
        pass

    @abstractmethod
    def get_color_distribution(self) -> Dict[int, float]:
        """Retourne la distribution empirique P̂(m_k) des couleurs."""
        pass


@dataclass
class MarkedProcess(PointProcess):
    """
    Processus Ponctuel Réel Marqué (Aa)

    Structure: [(τ_k, m_k)] où τ_k est le temps et m_k la marque (couleur).

    Attributes:
        jumps: Liste de tuples (temps, marque)
        r: Nombre de bins temporels
        m: Nombre de couleurs possibles
    """
    jumps: List[Tuple[float, int]] = field(default_factory=list)
    r: int = 256
    m: int = 16

    def get_jump_times(self) -> np.ndarray:
        """Retourne les temps de sauts triés."""
        if not self.jumps:
            return np.array([])
        return np.array([j[0] for j in sorted(self.jumps, key=lambda x: x[0])])

    def get_marks(self) -> np.ndarray:
        """Retourne les marques (couleurs) dans l'ordre temporel."""
        if not self.jumps:
            return np.array([], dtype=int)
        sorted_jumps = sorted(self.jumps, key=lambda x: x[0])
        return np.array([j[1] for j in sorted_jumps], dtype=int)

    def get_N(self) -> int:
        """Nombre de sauts."""
        return len(self.jumps)

    def get_N_transitions(self) -> int:
        """Nombre de transitions couleur #{k: m_k ≠ m_{k-1}}."""
        marks = self.get_marks()
        if len(marks) <= 1:
            return 0
        return int(np.sum(marks[1:] != marks[:-1]))

    def get_color_distribution(self) -> Dict[int, float]:
        """Distribution empirique P̂(m_k) des couleurs."""
        marks = self.get_marks()
        if len(marks) == 0:
            return {}
        counts = np.bincount(marks, minlength=self.m)
        total = counts.sum()
        if total == 0:
            return {}
        return {c: counts[c] / total for c in range(self.m) if counts[c] > 0}

    def get_entropy(self) -> float:
        """Entropie H(P̂) de la distribution couleur."""
        dist = self.get_color_distribution()
        if not dist:
            return 0.0
        return -sum(p * np.log2(p) for p in dist.values() if p > 0)

    def color_cost(self, mode: ColorMode, **kwargs) -> float:
        """
        Calcule C_color(B) selon le mode de codage.

        Args:
            mode: Mode de codage couleur (Ba, Bb, Bc, Bd)

        Returns:
            Coût en bits du codage couleur
        """
        N = self.get_N()
        if N == 0:
            return 0.0

        log2_m = np.log2(self.m) if self.m > 1 else 0.0

        if mode == ColorMode.SEQUENTIAL:  # Ba
            N_trans = self.get_N_transitions()
            return log2_m + N_trans * log2_m

        elif mode == ColorMode.UNIFORM:  # Bb - terme des formules L1-L4
            return N * log2_m

        elif mode == ColorMode.HUFFMAN:  # Bc
            H = self.get_entropy()
            D_huf = self.m * (int(np.floor(log2_m)) + 1) if self.m > 1 else 0
            return N * H + D_huf

        elif mode == ColorMode.ELIAS:  # Bd
            # Code δ d'Elias: log₂m + 2·log₂log₂m par marque
            marks = self.get_marks()
            total = 0.0
            for mark in marks:
                n = mark + 1
                if n >= 1:
                    k = int(np.floor(np.log2(n))) if n > 0 else 0
                    total += 1 + k + 2 * int(np.floor(np.log2(1 + k + 1e-9)))
            return total

        return 0.0

    def select_best_color_mode(self) -> Tuple[ColorMode, Dict[ColorMode, float]]:
        """
        Sélectionne le mode B optimal.

        Returns:
            (meilleur_mode, dictionnaire_des_coûts)
        """
        costs = {mode: self.color_cost(mode) for mode in ColorMode}
        best = min(costs, key=costs.get)
        return best, costs


@dataclass
class MonochromaticProcess(PointProcess):
    """
    Processus Ponctuel Monochromatique (Ab)

    La couleur est portée par l'indice du processus.
    C_color = 0 quel que soit le mode B.

    Structure: {couleur: [τ_k]} - dictionnaire couleur → liste de temps

    Attributes:
        processes: Dict couleur → liste de temps de sauts
        r: Nombre de bins temporels
    """
    processes: Dict[int, List[float]] = field(default_factory=dict)
    r: int = 256

    @property
    def m(self) -> int:
        """Nombre de couleurs (processus)."""
        return len(self.processes)

    def get_jump_times(self) -> np.ndarray:
        """Tous les temps de sauts fusionnés et triés."""
        all_times = []
        for times in self.processes.values():
            all_times.extend(times)
        return np.array(sorted(all_times))

    def get_jump_times_by_color(self, color: int) -> np.ndarray:
        """Temps de sauts pour une couleur spécifique."""
        return np.array(sorted(self.processes.get(color, [])))

    def get_N(self) -> int:
        """Nombre total de sauts."""
        return sum(len(times) for times in self.processes.values())

    def get_N_by_color(self, color: int) -> int:
        """Nombre de sauts pour une couleur."""
        return len(self.processes.get(color, []))

    def get_color_distribution(self) -> Dict[int, float]:
        """Distribution des sauts par couleur."""
        N = self.get_N()
        if N == 0:
            return {}
        return {c: len(times) / N for c, times in self.processes.items() if times}

    def color_cost(self, mode: ColorMode, **kwargs) -> float:
        """
        C_color = 0 pour le processus monochromatique.

        La couleur est l'indice du processus, pas une marque explicite.
        """
        return 0.0

    def get_combinatorial_length(self) -> float:
        """
        Longueur combinatoire optimale pour processus mono.

        L_comb = Σ_c (log₂N_c + log₂C(r, N_c))
        """
        from .math_utils import logC

        total = 0.0
        for color, times in self.processes.items():
            N_c = len(times)
            if N_c > 0:
                total += np.log2(N_c + 1) + logC(self.r, N_c)
        return total


@dataclass
class VectorialProcess(PointProcess):
    """
    Processus Ponctuel Vectoriel (Ac Marginal / Ad Joint)

    Plusieurs composantes corrélées ou indépendantes.

    Attributes:
        components: Liste de processus composantes
        is_joint: True si Ad (joint), False si Ac (marginal)
        r: Nombre de bins temporels
        m: Nombre de couleurs
    """
    components: List[MarkedProcess] = field(default_factory=list)
    is_joint: bool = False
    r: int = 256
    m: int = 16

    @property
    def d(self) -> int:
        """Dimension du vecteur (nombre de composantes)."""
        return len(self.components)

    def get_jump_times(self) -> np.ndarray:
        """Tous les temps de sauts fusionnés."""
        all_times = []
        for comp in self.components:
            all_times.extend(comp.get_jump_times())
        return np.array(sorted(all_times))

    def get_N(self) -> int:
        """Nombre total de sauts."""
        return sum(comp.get_N() for comp in self.components)

    def get_color_distribution(self) -> Dict[int, float]:
        """Distribution couleur globale."""
        all_marks = []
        for comp in self.components:
            all_marks.extend(comp.get_marks())
        if not all_marks:
            return {}
        marks = np.array(all_marks)
        counts = np.bincount(marks, minlength=self.m)
        total = counts.sum()
        return {c: counts[c] / total for c in range(self.m) if counts[c] > 0}

    def color_cost(self, mode: ColorMode, **kwargs) -> float:
        """
        Coût couleur selon marginal ou joint.

        - Marginal (Ac): somme des coûts par composante
        - Joint (Ad): coût de la loi jointe
        """
        if self.is_joint:
            # Loi jointe - traiter comme un seul processus
            all_marks = []
            for comp in self.components:
                all_marks.extend(comp.get_marks())
            if not all_marks:
                return 0.0

            combined = MarkedProcess(
                jumps=[(i, m) for i, m in enumerate(all_marks)],
                r=self.r, m=self.m
            )
            return combined.color_cost(mode)
        else:
            # Marginal - somme des composantes
            return sum(comp.color_cost(mode) for comp in self.components)


@dataclass
class MarkovianProcess(PointProcess):
    """
    Processus Markovien des Transitions (Ae)

    Modèle de type Aalen-Johansen avec intensités de transition α^{hj}(t).

    Attributes:
        states: Liste des états possibles
        transitions: Liste de (temps, état_from, état_to)
        transition_counts: Matrice N_hj des comptages
        r: Nombre de bins temporels
    """
    states: List[int] = field(default_factory=list)
    transitions: List[Tuple[float, int, int]] = field(default_factory=list)
    transition_counts: Optional[np.ndarray] = None
    r: int = 256

    @property
    def m(self) -> int:
        """Nombre d'états."""
        return len(self.states) if self.states else 0

    def get_jump_times(self) -> np.ndarray:
        """Temps des transitions."""
        if not self.transitions:
            return np.array([])
        return np.array(sorted([t[0] for t in self.transitions]))

    def get_N(self) -> int:
        """Nombre de transitions."""
        return len(self.transitions)

    def get_color_distribution(self) -> Dict[int, float]:
        """Distribution des états de destination."""
        if not self.transitions:
            return {}
        destinations = [t[2] for t in self.transitions]
        counts = {}
        for d in destinations:
            counts[d] = counts.get(d, 0) + 1
        total = len(destinations)
        return {k: v / total for k, v in counts.items()}

    def build_transition_matrix(self) -> np.ndarray:
        """Construit la matrice de comptage N_hj."""
        m = self.m
        if m == 0:
            return np.array([[]])
        matrix = np.zeros((m, m), dtype=int)
        for _, h, j in self.transitions:
            if 0 <= h < m and 0 <= j < m:
                matrix[h, j] += 1
        self.transition_counts = matrix
        return matrix

    def color_cost(self, mode: ColorMode, **kwargs) -> float:
        """
        Coût couleur basé sur les transitions.

        Les "couleurs" sont les états de destination des transitions.
        """
        marks = np.array([t[2] for t in self.transitions])
        if len(marks) == 0:
            return 0.0

        temp_process = MarkedProcess(
            jumps=[(i, m) for i, m in enumerate(marks)],
            r=self.r, m=self.m
        )
        return temp_process.color_cost(mode)
