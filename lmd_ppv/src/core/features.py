"""
features.py - Caractéristiques extraites d'un bloc vidéo
=========================================================

7 features bitwise pour le choix du cartouche ABCDEFGH.

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class BlockFeatures:
    """
    Caractéristiques statistiques extraites d'un bloc vidéo.

    Ces features déterminent le cartouche ABCDEFGH optimal.

    Attributes:
        N: Nombre total de sauts (changements d'état)
        r: Nombre de bins temporels
        m: Nombre de couleurs distinctes
        n_pixels: Nombre de pixels dans le bloc

        lambda_avg: Intensité moyenne des sauts (N/r)
        H_s: Homogénéité spatiale (variance inter-pixels)
        rho_corr: Corrélation spatiale entre pixels voisins
        R_temp: Régularité temporelle E[Δτ]²/Var[Δτ]
        m_eff: Nombre de couleurs effectives (avec activité > seuil)
        H_color: Entropie couleur -Σ p_c·log₂p_c
        N_trans: Nombre de transitions couleur #{k: m_k ≠ m_{k-1}}

        color_dist: Distribution empirique P̂(m_k) des couleurs
    """
    # Paramètres de base
    N: int = 0                    # Nombre de sauts
    r: int = 256                  # Bins temporels
    m: int = 16                   # Nombre de couleurs
    n_pixels: int = 256           # Pixels dans le bloc (16x16)

    # 7 features bitwise
    lambda_avg: float = 0.0       # Intensité moyenne
    H_s: float = 0.0              # Homogénéité spatiale [0,1]
    rho_corr: float = 0.0         # Corrélation couleurs [0,1]
    R_temp: float = 1.0           # Régularité temporelle
    m_eff: int = 0                # Couleurs effectives
    H_color: float = 0.0          # Entropie couleur (bits)
    N_trans: int = 0              # Transitions couleur

    # Distribution couleur
    color_dist: Dict[int, float] = field(default_factory=dict)

    # Statistiques additionnelles
    var_delta_tau: float = 0.0    # Variance des intervalles
    mean_delta_tau: float = 0.0   # Moyenne des intervalles

    def __post_init__(self):
        """Calculs dérivés après initialisation."""
        if self.r > 0:
            self.lambda_avg = self.N / self.r

    @property
    def density(self) -> float:
        """Densité de sauts N/r."""
        return self.N / self.r if self.r > 0 else 0.0

    @property
    def log2_m(self) -> float:
        """log₂(m) pour les calculs de coût."""
        return np.log2(self.m) if self.m > 1 else 0.0

    def get_huffman_overhead(self) -> int:
        """D_huf ≈ m·(⌊log₂m⌋+1) bits pour le dictionnaire Huffman."""
        if self.m <= 1:
            return 0
        return self.m * (int(np.floor(np.log2(self.m))) + 1)

    def get_huffman_threshold(self) -> float:
        """
        N* - Seuil au-delà duquel Huffman (Bc) bat Uniforme (Bb).

        N* = D_huf / (log₂m - H_color)
        """
        delta = self.log2_m - self.H_color
        if delta <= 0:
            return float('inf')
        return self.get_huffman_overhead() / delta

    def suggest_process_type(self) -> int:
        """Suggère le type de processus (dim. A) selon les features."""
        from .process_types import ProcessType

        # Processus joint si forte corrélation et homogénéité
        if self.H_s > 0.7 and self.rho_corr > 0.8:
            return ProcessType.VECTORIAL_JOINT

        # Vectoriel marginal si faible corrélation
        if self.rho_corr < 0.15:
            return ProcessType.VECTORIAL_MARG

        # Monochromatique si une seule couleur dominante
        if self.m_eff == 1 or (self.color_dist and max(self.color_dist.values(), default=0) > 0.95):
            return ProcessType.MONOCHROMATIC

        # Par défaut: marqué
        return ProcessType.MARKED

    def suggest_color_mode(self) -> int:
        """Suggère le mode couleur (dim. B) selon les features."""
        from .process_types import ColorMode

        # Monochromatique -> pas de coût couleur
        if self.suggest_process_type() == 1:  # MONOCHROMATIC
            return ColorMode.UNIFORM  # B ignoré si A=Ab

        D_huf = self.get_huffman_overhead()
        N_star = self.get_huffman_threshold()

        # Coûts des différents modes
        cost_Ba = self.log2_m + self.N_trans * self.log2_m
        cost_Bb = self.N * self.log2_m
        cost_Bc = self.N * self.H_color + D_huf

        costs = {
            ColorMode.SEQUENTIAL: cost_Ba,
            ColorMode.UNIFORM: cost_Bb,
            ColorMode.HUFFMAN: cost_Bc,
        }

        return min(costs, key=costs.get)

    def suggest_representation(self, threshold_N: int = 50) -> int:
        """
        Suggère la représentation (dim. C) selon le nombre de sauts N.

        Règles optimales (dérivées de l'analyse exhaustive):
        - N = 0: TIMESTAMPS (R1) - bloc vide, rien à encoder
        - N < threshold: COUNT (R2) - peu d'événements, comptage efficace
        - N >= threshold: COMBINATORIAL (R4b) - beaucoup d'événements

        Args:
            threshold_N: Seuil pour distinguer COUNT vs COMBINATORIAL (défaut: 50)

        Returns:
            Representation optimale
        """
        from .process_types import Representation

        if self.N == 0:
            return Representation.TIMESTAMPS  # R1 - bloc vide
        elif self.N < threshold_N:
            return Representation.COUNT  # R2 - peu d'événements
        else:
            return Representation.COMBINATORIAL  # R4b - beaucoup d'événements

    def to_dict(self) -> Dict:
        """Conversion en dictionnaire."""
        return {
            'N': self.N, 'r': self.r, 'm': self.m, 'n_pixels': self.n_pixels,
            'lambda_avg': self.lambda_avg, 'H_s': self.H_s,
            'rho_corr': self.rho_corr, 'R_temp': self.R_temp,
            'm_eff': self.m_eff, 'H_color': self.H_color,
            'N_trans': self.N_trans, 'density': self.density,
            'N_star': self.get_huffman_threshold()
        }
