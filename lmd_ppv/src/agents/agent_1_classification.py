"""
agent_1_classification.py - Agent de Classification (Dimension A)
==================================================================

Phase 1 - Dimension A

Classification du type de processus ponctuel:
- Aa: Réel Marqué (défaut)
- Ab: Monochromatique (C_color = 0)
- Ac: Vectoriel Marginal
- Ad: Vectoriel Joint
- Ae: Markovien

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from scipy import stats

from ..core.process_types import ProcessType, ColorMode
from ..core.features import BlockFeatures
from ..utils.math_utils import logC

# Import conditionnel pour éviter les imports circulaires
if TYPE_CHECKING:
    from optimization.threshold_config import ThresholdConfig


@dataclass
class ClassificationResult:
    """Résultat de la classification."""
    process_type: ProcessType
    confidence: float
    scores: Dict[ProcessType, float]
    mono_gain: float  # Gain si monochromatique
    is_mono_better: bool


@dataclass
class ThresholdConfigSimple:
    """Configuration simplifiée des seuils pour éviter les imports circulaires."""
    threshold_H_s: float = 0.7       # Homogénéité spatiale
    threshold_rho_high: float = 0.8  # Corrélation forte
    threshold_rho_low: float = 0.15  # Corrélation faible
    threshold_chi2: float = 0.05     # p-value chi² pour Markov

    # Seuils pour la représentation (dimension C)
    density_R1_max: float = 0.10     # Sparse -> R1
    density_R4b_R2: float = 0.80     # Dense -> R2
    density_R4a_min: float = 0.50    # Boolean -> R4a


class ClassificationAgent:
    """
    Agent 1: Classification du Type de Processus

    Détermine la dimension A du cartouche ABCDEFGH.

    Supporte les seuils paramétrables via ThresholdConfig.
    """

    def __init__(self, threshold_config: Optional['ThresholdConfig'] = None):
        """
        Initialise l'agent de classification.

        Args:
            threshold_config: Configuration des seuils (optionnel).
                             Si None, utilise les valeurs par défaut.
        """
        # Configuration par défaut ou fournie
        if threshold_config is not None:
            self.threshold_H_s = threshold_config.threshold_H_s
            self.threshold_rho_high = threshold_config.threshold_rho_high
            self.threshold_rho_low = threshold_config.threshold_rho_low
            self.threshold_chi2 = threshold_config.threshold_chi2

            # Seuils pour dimension C (si disponibles)
            if hasattr(threshold_config, 'density_R1_max'):
                self.density_R1_max = threshold_config.density_R1_max
                self.density_R4b_R2 = threshold_config.density_R4b_R2
                self.density_R4a_min = threshold_config.density_R4a_min
            else:
                self.density_R1_max = 0.10
                self.density_R4b_R2 = 0.80
                self.density_R4a_min = 0.50
        else:
            # Seuils de classification (calibrés par défaut)
            self.threshold_H_s = 0.7       # Homogénéité spatiale
            self.threshold_rho_high = 0.8  # Corrélation forte
            self.threshold_rho_low = 0.15  # Corrélation faible
            self.threshold_chi2 = 0.05     # p-value chi² pour Markov

            # Seuils pour dimension C
            self.density_R1_max = 0.10
            self.density_R4b_R2 = 0.80
            self.density_R4a_min = 0.50

    @classmethod
    def with_thresholds(
        cls,
        threshold_H_s: float = 0.7,
        threshold_rho_high: float = 0.8,
        threshold_rho_low: float = 0.15,
        threshold_chi2: float = 0.05,
        density_R1_max: float = 0.10,
        density_R4b_R2: float = 0.80,
        density_R4a_min: float = 0.50
    ) -> 'ClassificationAgent':
        """
        Factory method pour créer un agent avec des seuils spécifiques.

        Args:
            threshold_H_s: Seuil d'homogénéité spatiale
            threshold_rho_high: Seuil de corrélation haute
            threshold_rho_low: Seuil de corrélation basse
            threshold_chi2: Seuil p-value chi²
            density_R1_max: Densité max pour R1
            density_R4b_R2: Densité pour R2
            density_R4a_min: Densité min pour R4a

        Returns:
            ClassificationAgent configuré
        """
        config = ThresholdConfigSimple(
            threshold_H_s=threshold_H_s,
            threshold_rho_high=threshold_rho_high,
            threshold_rho_low=threshold_rho_low,
            threshold_chi2=threshold_chi2,
            density_R1_max=density_R1_max,
            density_R4b_R2=density_R4b_R2,
            density_R4a_min=density_R4a_min
        )
        return cls(config)

    def get_thresholds(self) -> Dict[str, float]:
        """Retourne les seuils actuels."""
        return {
            'threshold_H_s': self.threshold_H_s,
            'threshold_rho_high': self.threshold_rho_high,
            'threshold_rho_low': self.threshold_rho_low,
            'threshold_chi2': self.threshold_chi2,
            'density_R1_max': self.density_R1_max,
            'density_R4b_R2': self.density_R4b_R2,
            'density_R4a_min': self.density_R4a_min,
        }

    def set_thresholds(self, **kwargs) -> None:
        """Met à jour les seuils."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def classify(self, features: BlockFeatures) -> ClassificationResult:
        """
        Classifie le type de processus.

        Logique de classification (basée sur l'analyse empirique):
        1. Blocs vides (N=0) → MARKED (coût minimal)
        2. Blocs avec m petits et actifs → MONOCHROMATIC (C_color=0)
        3. Sinon comparer MONO vs MARKED via coût d'encodage

        Args:
            features: Caractéristiques du bloc

        Returns:
            ClassificationResult avec type et scores
        """
        # Cas 1: Bloc vide (aucun saut) → MARKED par défaut
        if features.N == 0:
            return ClassificationResult(
                process_type=ProcessType.MARKED,
                confidence=1.0,
                scores={ProcessType.MARKED: 1.0},
                mono_gain=0.0,
                is_mono_better=False
            )

        # Cas 2: Blocs avec peu de couleurs actives → MONOCHROMATIC
        # Quand m <= 4 et toutes les couleurs sont actives (m_eff == m),
        # MONOCHROMATIC est toujours optimal car C_color = 0.
        # L'analyse exhaustive montre que cost_A[MONO] = 0 dans ces cas.
        if features.N > 0 and features.m <= 4 and features.m_eff >= features.m - 1:
            return ClassificationResult(
                process_type=ProcessType.MONOCHROMATIC,
                confidence=0.95,
                scores={ProcessType.MONOCHROMATIC: 0.95, ProcessType.MARKED: 0.05},
                mono_gain=features.N,  # Gain approximatif (évite le coût couleur)
                is_mono_better=True
            )

        # Cas 3: Comparer MONO vs MARKED pour autres cas
        mono_gain, is_mono_better = self._compare_mono_vs_marked(features)

        if is_mono_better:
            return ClassificationResult(
                process_type=ProcessType.MONOCHROMATIC,
                confidence=min(1.0, mono_gain / 100),
                scores={ProcessType.MONOCHROMATIC: 0.9, ProcessType.MARKED: 0.1},
                mono_gain=mono_gain,
                is_mono_better=True
            )

        # Cas 4: Par défaut MARKED (pas de VECTORIAL car rarement optimal)
        return ClassificationResult(
            process_type=ProcessType.MARKED,
            confidence=0.8,
            scores={ProcessType.MARKED: 0.8, ProcessType.MONOCHROMATIC: 0.1},
            mono_gain=mono_gain,
            is_mono_better=False
        )

    def _compute_scores(self, features: BlockFeatures) -> Dict[ProcessType, float]:
        """Calcule les scores pour chaque type de processus."""
        scores = {
            ProcessType.MARKED: 0.5,  # Score par défaut
            ProcessType.MONOCHROMATIC: 0.0,
            ProcessType.VECTORIAL_MARG: 0.0,
            ProcessType.VECTORIAL_JOINT: 0.0,
            ProcessType.MARKOVIAN: 0.0
        }

        # Vectoriel Joint (Ad): forte corrélation ET homogénéité
        if features.H_s > self.threshold_H_s and features.rho_corr > self.threshold_rho_high:
            scores[ProcessType.VECTORIAL_JOINT] = 0.8 + 0.2 * features.rho_corr

        # Vectoriel Marginal (Ac): faible corrélation
        if features.rho_corr < self.threshold_rho_low:
            scores[ProcessType.VECTORIAL_MARG] = 0.7 + 0.3 * (1 - features.rho_corr)

        # Monochromatique (Ab): une couleur dominante
        if features.color_dist:
            max_prob = max(features.color_dist.values())
            if max_prob > 0.9:
                scores[ProcessType.MONOCHROMATIC] = max_prob
            elif features.m_eff == 1:
                scores[ProcessType.MONOCHROMATIC] = 0.95

        # Markovien (Ae): structure de transitions
        # Simplifié: utilise la régularité temporelle comme proxy
        if features.R_temp > 1.5:
            scores[ProcessType.MARKOVIAN] = 0.3 + 0.4 * min(1.0, features.R_temp / 3.0)

        # Normalisation pour que le max soit au moins 0.5
        max_score = max(scores.values())
        if max_score < 0.5:
            scores[ProcessType.MARKED] = 0.6

        return scores

    def _compare_mono_vs_marked(
        self,
        features: BlockFeatures
    ) -> Tuple[float, bool]:
        """
        Compare le coût monochromatique vs marqué.

        Règle: Choisir Ab si Σ_c log₂C(r,N_c) < L_temporel(R4b) + C_color(B_optimal)

        Utilise color_dist si disponible, sinon estime à partir de H_color et m_eff.

        Args:
            features: Caractéristiques du bloc

        Returns:
            (gain_en_bits, monochromatique_est_meilleur)
        """
        N = features.N
        r = features.r
        m = features.m
        m_eff = features.m_eff

        if N == 0:
            return 0.0, False

        # Coût marqué avec R4b et meilleur mode B
        L_temp_R4b = np.log2(N + 1) + logC(r, N)

        # Meilleur C_color
        H = features.H_color
        N_trans = features.N_trans
        log2_m = np.log2(m) if m > 1 else 0
        D_huf = features.get_huffman_overhead()

        C_color_Bb = N * log2_m
        C_color_Bc = N * H + D_huf
        C_color_Ba = log2_m + N_trans * log2_m

        C_color_best = min(C_color_Bb, C_color_Bc, C_color_Ba)

        L_marked = L_temp_R4b + C_color_best

        # Coût monochromatique: Σ_c (log₂N_c + log₂C(r,N_c))
        L_mono = 0.0

        if features.color_dist:
            # Utiliser la distribution réelle si disponible
            for c, prob in features.color_dist.items():
                N_c = int(prob * N)
                if N_c > 0:
                    L_mono += np.log2(N_c + 1) + logC(r, N_c)
        elif m_eff > 0:
            # Estimer une distribution uniforme parmi les couleurs effectives
            # Quand H_color ≈ log2(m_eff), la distribution est quasi-uniforme
            N_per_color = N // m_eff if m_eff > 0 else N
            remainder = N % m_eff if m_eff > 0 else 0

            for c in range(m_eff):
                N_c = N_per_color + (1 if c < remainder else 0)
                if N_c > 0:
                    L_mono += np.log2(N_c + 1) + logC(r, N_c)
        else:
            # Pas d'info sur les couleurs, supposer mono avec tout N
            L_mono = np.log2(N + 1) + logC(r, N)

        gain = L_marked - L_mono
        is_better = gain > 0

        return gain, is_better

    def classify_5types(
        self,
        features: BlockFeatures,
        has_markov_structure: bool = False
    ) -> Tuple[ProcessType, Dict]:
        """
        Classification complète avec 5 types.

        Args:
            features: Caractéristiques du bloc
            has_markov_structure: Résultat du test chi² externe

        Returns:
            (type, détails)
        """
        details = {
            "H_s": features.H_s,
            "rho_corr": features.rho_corr,
            "R_temp": features.R_temp,
            "m_eff": features.m_eff,
            "N_trans": features.N_trans
        }

        # 1. Test Joint (Ad)
        if features.H_s > self.threshold_H_s and features.rho_corr > self.threshold_rho_high:
            return ProcessType.VECTORIAL_JOINT, details

        # 2. Test Marginal (Ac)
        if features.rho_corr < self.threshold_rho_low:
            return ProcessType.VECTORIAL_MARG, details

        # 3. Test Markovien (Ae)
        if has_markov_structure:
            return ProcessType.MARKOVIAN, details

        # 4. Test Monochromatique (Ab)
        _, is_mono = self._compare_mono_vs_marked(features)
        if is_mono:
            return ProcessType.MONOCHROMATIC, details

        # 5. Défaut: Marqué (Aa)
        return ProcessType.MARKED, details

    def test_markov_structure(
        self,
        transition_counts: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Test chi² pour structure markovienne.

        Teste si les transitions sont significativement différentes
        d'un modèle indépendant.

        Args:
            transition_counts: Matrice N_hj des comptages

        Returns:
            (est_markovien, p_value)
        """
        if transition_counts.sum() == 0:
            return False, 1.0

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(
                transition_counts + 1  # +1 pour éviter les zéros
            )
            is_markov = p_value < self.threshold_chi2
            return is_markov, p_value
        except:
            return False, 1.0

    def get_color_cost_by_type(
        self,
        process_type: ProcessType,
        features: BlockFeatures,
        color_mode: ColorMode
    ) -> float:
        """
        Retourne le coût couleur selon le type de processus.

        Args:
            process_type: Type de processus (dim A)
            features: Caractéristiques du bloc
            color_mode: Mode couleur (dim B)

        Returns:
            Coût couleur en bits
        """
        N = features.N
        m = features.m
        H = features.H_color
        N_trans = features.N_trans

        # Monochromatique: C_color = 0 quel que soit B
        if process_type == ProcessType.MONOCHROMATIC:
            return 0.0

        log2_m = np.log2(m) if m > 1 else 0
        D_huf = features.get_huffman_overhead()

        if color_mode == ColorMode.SEQUENTIAL:
            return log2_m + N_trans * log2_m
        elif color_mode == ColorMode.UNIFORM:
            return N * log2_m
        elif color_mode == ColorMode.HUFFMAN:
            return N * H + D_huf
        elif color_mode == ColorMode.ELIAS:
            return N * (log2_m + 2 * np.log2(max(1, log2_m)))

        return 0.0


def get_optimal_cartouche_A(features: BlockFeatures) -> Tuple[ProcessType, str]:
    """
    Détermine la valeur optimale de la dimension A.

    Args:
        features: Caractéristiques du bloc

    Returns:
        (ProcessType, explication)
    """
    agent = ClassificationAgent()
    result = agent.classify(features)

    explanations = {
        ProcessType.MARKED: "Processus marqué standard - C_color(B) selon le mode",
        ProcessType.MONOCHROMATIC: f"Monochromatique - C_color = 0, gain = {result.mono_gain:.1f} bits",
        ProcessType.VECTORIAL_MARG: f"Vectoriel marginal - rho_corr = {features.rho_corr:.2f} < 0.15",
        ProcessType.VECTORIAL_JOINT: f"Vectoriel joint - H_s = {features.H_s:.2f}, rho = {features.rho_corr:.2f}",
        ProcessType.MARKOVIAN: f"Markovien - R_temp = {features.R_temp:.2f}"
    }

    return result.process_type, explanations.get(result.process_type, "")
