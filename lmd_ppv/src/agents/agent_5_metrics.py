"""
agent_5_metrics.py - Agent Métriques LMD
=========================================

Phase 3 - Dimensions B, C

Métriques et formules:
- L1-L4 généralisées par mode B
- Gain Huffman (Bc vs Bb)
- Seuil N*
- Borne de Hellinger H²

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from ..core.process_types import ColorMode, Representation
from ..core.features import BlockFeatures
from ..utils.math_utils import logC, entropy, hellinger_bound, model_complexity


@dataclass
class MetricsResult:
    """Résultat des métriques LMD."""
    # Longueurs de code
    L1: Dict[ColorMode, float]
    L2: Dict[ColorMode, float]
    L3: Dict[ColorMode, float]
    L4: Dict[ColorMode, float]
    L4_mono: float  # Monochromatique (C_color = 0)

    # Métriques dérivées
    best_L: float
    best_repr: Representation
    best_color_mode: ColorMode

    # Seuils et gains
    N_star: float
    gain_Bc_vs_Bb: float
    gain_percent: float

    # Borne de Hellinger
    hellinger_bound: float


class MetricsAgent:
    """
    Agent 5: Métriques LMD

    Calcule les formules L1-L4 généralisées et les métriques de performance.
    """

    def __init__(self):
        pass

    def compute_all_metrics(self, features: BlockFeatures) -> MetricsResult:
        """
        Calcule toutes les métriques LMD.

        Args:
            features: Caractéristiques du bloc

        Returns:
            MetricsResult complet
        """
        N = features.N
        r = features.r
        m = features.m
        H = features.H_color
        N_trans = features.N_trans

        # Calcul des L1-L4 pour chaque mode B
        L1 = self._compute_L1(N, r, m, H, N_trans)
        L2 = self._compute_L2(N, r, m, H, N_trans)
        L3 = self._compute_L3(N, r, m, H, N_trans)
        L4 = self._compute_L4(N, r, m, H, N_trans)
        L4_mono = self._compute_L4_mono(N, r)

        # Meilleure combinaison
        all_lengths = []
        for repr_name, lengths in [("L1", L1), ("L2", L2), ("L3", L3), ("L4", L4)]:
            for mode, length in lengths.items():
                all_lengths.append((length, repr_name, mode))

        # Ajoute L4_mono
        all_lengths.append((L4_mono, "L4_mono", ColorMode.UNIFORM))

        best = min(all_lengths, key=lambda x: x[0])
        best_L, best_repr_name, best_mode = best

        repr_map = {"L1": Representation.TIMESTAMPS, "L2": Representation.COUNT,
                    "L3": Representation.INTERVALS, "L4": Representation.COMBINATORIAL,
                    "L4_mono": Representation.COMBINATORIAL}

        # Seuil N*
        N_star = self._compute_N_star(m, H)

        # Gain Bc vs Bb
        gain = L4[ColorMode.UNIFORM] - L4[ColorMode.HUFFMAN]
        gain_pct = (gain / L4[ColorMode.UNIFORM] * 100) if L4[ColorMode.UNIFORM] > 0 else 0

        # Borne de Hellinger
        complexity = model_complexity(k=10, n=N, method="mdl")  # k estimé
        h_bound = hellinger_bound(N, complexity, bias_term=0.0)

        return MetricsResult(
            L1=L1, L2=L2, L3=L3, L4=L4, L4_mono=L4_mono,
            best_L=best_L,
            best_repr=repr_map[best_repr_name],
            best_color_mode=best_mode,
            N_star=N_star,
            gain_Bc_vs_Bb=gain,
            gain_percent=gain_pct,
            hellinger_bound=h_bound
        )

    def _compute_L1(
        self, N: int, r: int, m: int, H: float, N_trans: int
    ) -> Dict[ColorMode, float]:
        """
        L1 = r (partie temporelle) + C_color(B)

        Encodage état complet.
        """
        L_temp = float(r)
        return self._add_color_costs(L_temp, N, m, H, N_trans)

    def _compute_L2(
        self, N: int, r: int, m: int, H: float, N_trans: int
    ) -> Dict[ColorMode, float]:
        """
        L2 = log₂N + N·log₂r (partie temporelle) + C_color(B)

        Liste des sauts.
        """
        L_temp = np.log2(N + 1) + N * np.log2(r) if N > 0 else 0.0
        return self._add_color_costs(L_temp, N, m, H, N_trans)

    def _compute_L3(
        self, N: int, r: int, m: int, H: float, N_trans: int
    ) -> Dict[ColorMode, float]:
        """
        L3 = r (vecteur booléen) + C_color(B)

        Note: dans certaines versions, L3 = r + N·log₂m pour Bb.
        """
        L_temp = float(r)
        return self._add_color_costs(L_temp, N, m, H, N_trans)

    def _compute_L4(
        self, N: int, r: int, m: int, H: float, N_trans: int
    ) -> Dict[ColorMode, float]:
        """
        L4 = log₂N + log₂C(r,N) (partie temporelle) + C_color(B)

        Adresse combinatoire - BORNE INFÉRIEURE uniforme.
        """
        L_temp = np.log2(N + 1) + logC(r, N) if N > 0 else 0.0
        return self._add_color_costs(L_temp, N, m, H, N_trans)

    def _compute_L4_mono(self, N: int, r: int) -> float:
        """
        L4_mono = log₂N + log₂C(r,N)

        Mode monochromatique: C_color = 0.
        """
        if N == 0:
            return 0.0
        return np.log2(N + 1) + logC(r, N)

    def _add_color_costs(
        self,
        L_temp: float,
        N: int,
        m: int,
        H: float,
        N_trans: int
    ) -> Dict[ColorMode, float]:
        """Ajoute les coûts couleur à la partie temporelle."""
        if N == 0 or m <= 1:
            return {mode: L_temp for mode in ColorMode}

        log2_m = np.log2(m)
        D_huf = m * (int(np.floor(log2_m)) + 1)

        return {
            ColorMode.SEQUENTIAL: L_temp + log2_m + N_trans * log2_m,
            ColorMode.UNIFORM: L_temp + N * log2_m,
            ColorMode.HUFFMAN: L_temp + N * H + D_huf,
            ColorMode.ELIAS: L_temp + N * (log2_m + 2 * np.log2(max(1, log2_m)))
        }

    def _compute_N_star(self, m: int, H: float) -> float:
        """
        Calcule N* = D_huf / (log₂m - H)

        Seuil au-delà duquel Bc bat Bb.
        """
        if m <= 1:
            return float('inf')

        log2_m = np.log2(m)
        D_huf = m * (int(np.floor(log2_m)) + 1)
        delta = log2_m - H

        if delta <= 0:
            return float('inf')

        return D_huf / delta

    def should_use_mdl(
        self,
        L_mdl: float,
        features: BlockFeatures
    ) -> Tuple[bool, float]:
        """
        Détermine si le codage MDL est meilleur que l'uniforme.

        Args:
            L_mdl: Longueur MDL calculée
            features: Caractéristiques du bloc

        Returns:
            (utiliser_mdl, meilleure_longueur_uniforme)
        """
        metrics = self.compute_all_metrics(features)

        # Meilleur uniforme parmi L1-L4
        best_uniform = min(
            min(metrics.L1.values()),
            min(metrics.L2.values()),
            min(metrics.L3.values()),
            min(metrics.L4.values()),
            metrics.L4_mono
        )

        return L_mdl < best_uniform, best_uniform

    def compute_compression_ratio(
        self,
        raw_bits: int,
        encoded_bits: int
    ) -> float:
        """Calcule le taux de compression."""
        if encoded_bits == 0:
            return float('inf')
        return raw_bits / encoded_bits

    def compute_redundancy(
        self,
        encoded_bits: int,
        theoretical_entropy: float,
        n_symbols: int
    ) -> float:
        """
        Calcule la redondance.

        Redundancy = L_encoded - n·H
        """
        return encoded_bits - n_symbols * theoretical_entropy


def generate_metrics_report(features: BlockFeatures) -> str:
    """
    Génère un rapport textuel des métriques.

    Args:
        features: Caractéristiques du bloc

    Returns:
        Rapport formaté
    """
    agent = MetricsAgent()
    m = agent.compute_all_metrics(features)

    report = []
    report.append("=" * 60)
    report.append("MÉTRIQUES LMD - Codage Versatile v6.0")
    report.append("=" * 60)
    report.append(f"\nParamètres: N={features.N}, r={features.r}, m={features.m}")
    report.append(f"H_color={features.H_color:.3f} bits, N_trans={features.N_trans}")
    report.append(f"\n{'Mode':<12} {'L1':>10} {'L2':>10} {'L3':>10} {'L4':>10}")
    report.append("-" * 54)

    for mode in ColorMode:
        mode_name = ["Ba", "Bb", "Bc", "Bd"][mode.value]
        report.append(
            f"{mode_name:<12} {m.L1[mode]:>10.1f} {m.L2[mode]:>10.1f} "
            f"{m.L3[mode]:>10.1f} {m.L4[mode]:>10.1f}"
        )

    report.append(f"\nL4_mono (Ab): {m.L4_mono:.1f} bits (C_color = 0)")
    report.append(f"\nMeilleure combinaison: {m.best_repr.name} + {m.best_color_mode.name}")
    report.append(f"Longueur optimale: {m.best_L:.1f} bits")
    report.append(f"\nSeuil Huffman N* = {m.N_star:.1f}")
    report.append(f"Gain Bc vs Bb: {m.gain_Bc_vs_Bb:.1f} bits ({m.gain_percent:.1f}%)")
    report.append(f"Borne Hellinger H²: {m.hellinger_bound:.4f}")
    report.append("=" * 60)

    return "\n".join(report)
