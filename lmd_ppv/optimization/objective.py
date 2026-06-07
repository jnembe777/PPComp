"""
objective.py - Fonction objectif pour l'optimisation
=====================================================

Fonction objectif:
J = alpha * Sum(L(predit)) + beta * Sum(max(0, L(predit) - L(optimal)))

- alpha = 1.0 (cout d'encodage)
- beta = 0.5 (penalite de mauvaise prediction)

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cartouche import Cartouche
from src.core.features import BlockFeatures
from src.agents.agent_1_classification import ClassificationAgent

from .threshold_config import ThresholdConfig
from .exhaustive_search import ExhaustiveSearch, BlockOptimalResult


@dataclass
class OptimizationResult:
    """Resultat de l'evaluation d'une configuration de seuils."""
    config: ThresholdConfig
    total_cost: float  # J = alpha * encoding + beta * penalty
    encoding_cost: float  # Sum(L(predit))
    penalty_cost: float  # Sum(max(0, L(predit) - L(optimal)))

    # Metriques detaillees
    n_blocks: int = 0
    exact_matches: int = 0
    accuracy_A: float = 0.0
    accuracy_B: float = 0.0
    accuracy_C: float = 0.0

    # Distribution des erreurs
    cost_penalties: List[float] = field(default_factory=list)

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / self.n_blocks if self.n_blocks > 0 else 0.0

    @property
    def mean_penalty(self) -> float:
        return np.mean(self.cost_penalties) if self.cost_penalties else 0.0

    @property
    def max_penalty(self) -> float:
        return np.max(self.cost_penalties) if self.cost_penalties else 0.0

    def to_dict(self) -> Dict:
        return {
            'config': self.config.to_dict(),
            'total_cost': self.total_cost,
            'encoding_cost': self.encoding_cost,
            'penalty_cost': self.penalty_cost,
            'n_blocks': self.n_blocks,
            'exact_matches': self.exact_matches,
            'exact_accuracy': self.exact_accuracy,
            'accuracy_A': self.accuracy_A,
            'accuracy_B': self.accuracy_B,
            'accuracy_C': self.accuracy_C,
            'mean_penalty': self.mean_penalty,
            'max_penalty': self.max_penalty,
        }


class ObjectiveFunction:
    """
    Fonction objectif pour l'optimisation des seuils.

    Evalure une configuration de seuils sur un ensemble de blocs
    en comparant les predictions aux cartouches optimaux.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        exhaustive_search: Optional[ExhaustiveSearch] = None
    ):
        """
        Initialise la fonction objectif.

        Args:
            alpha: Poids du cout d'encodage
            beta: Poids de la penalite de mauvaise prediction
            exhaustive_search: Instance de recherche exhaustive
        """
        self.alpha = alpha
        self.beta = beta
        self.exhaustive_search = exhaustive_search or ExhaustiveSearch()

        # Cache des resultats optimaux
        self._optimal_cache: Dict[int, BlockOptimalResult] = {}

    def precompute_optima(
        self,
        features_list: List[BlockFeatures],
        n_workers: int = 4
    ) -> None:
        """
        Pre-calcule les cartouches optimaux pour tous les blocs.

        Args:
            features_list: Liste des caracteristiques de blocs
            n_workers: Nombre de workers
        """
        results = self.exhaustive_search.find_optimal_batch(features_list, n_workers)
        for result in results:
            self._optimal_cache[result.block_id] = result

    def evaluate(
        self,
        config: ThresholdConfig,
        features_list: List[BlockFeatures],
        precomputed: bool = True
    ) -> OptimizationResult:
        """
        Evalue une configuration de seuils.

        Args:
            config: Configuration de seuils a evaluer
            features_list: Liste des caracteristiques de blocs
            precomputed: Utiliser le cache des optimaux

        Returns:
            OptimizationResult avec les metriques
        """
        # Creer l'agent avec les seuils
        agent = self._create_agent(config)

        # Accumulateurs
        encoding_cost = 0.0
        penalty_cost = 0.0
        cost_penalties = []
        exact_matches = 0
        match_A = 0
        match_B = 0
        match_C = 0

        for i, features in enumerate(features_list):
            # Obtenir le cartouche optimal
            if precomputed and i in self._optimal_cache:
                optimal = self._optimal_cache[i]
            else:
                optimal = self.exhaustive_search.find_optimal(features, i)
                self._optimal_cache[i] = optimal

            # Predire avec l'agent
            predicted_cartouche = self._predict_cartouche(agent, features, config)

            # Calculer le cout predit
            predicted_cost = self._compute_cost(predicted_cartouche, optimal)
            optimal_cost = optimal.optimal_cost

            # Accumuler les couts
            encoding_cost += predicted_cost
            penalty = max(0, predicted_cost - optimal_cost)
            penalty_cost += penalty
            cost_penalties.append(penalty)

            # Verifier les correspondances
            opt = optimal.optimal_cartouche
            if predicted_cartouche.A == opt.A:
                match_A += 1
            if predicted_cartouche.B == opt.B:
                match_B += 1
            if predicted_cartouche.C == opt.C:
                match_C += 1
            if (predicted_cartouche.A == opt.A and
                predicted_cartouche.B == opt.B and
                predicted_cartouche.C == opt.C):
                exact_matches += 1

        n_blocks = len(features_list)
        total_cost = self.alpha * encoding_cost + self.beta * penalty_cost

        return OptimizationResult(
            config=config,
            total_cost=total_cost,
            encoding_cost=encoding_cost,
            penalty_cost=penalty_cost,
            n_blocks=n_blocks,
            exact_matches=exact_matches,
            accuracy_A=match_A / n_blocks if n_blocks > 0 else 0.0,
            accuracy_B=match_B / n_blocks if n_blocks > 0 else 0.0,
            accuracy_C=match_C / n_blocks if n_blocks > 0 else 0.0,
            cost_penalties=cost_penalties
        )

    def _create_agent(self, config: ThresholdConfig) -> ClassificationAgent:
        """Cree un agent avec les seuils specifies."""
        agent = ClassificationAgent()
        agent.threshold_H_s = config.threshold_H_s
        agent.threshold_rho_high = config.threshold_rho_high
        agent.threshold_rho_low = config.threshold_rho_low
        agent.threshold_chi2 = config.threshold_chi2
        return agent

    def _predict_cartouche(
        self,
        agent: ClassificationAgent,
        features: BlockFeatures,
        config: ThresholdConfig
    ) -> Cartouche:
        """
        Predit le cartouche pour un bloc.

        Utilise l'agent pour la dimension A et les regles optimales
        pour les dimensions B et C.
        """
        # Dimension A
        result = agent.classify(features)
        A = result.process_type

        # Dimension B (selon les features)
        B = features.suggest_color_mode()

        # Dimension C (basé sur N, pas sur density)
        # Règles optimales dérivées de l'analyse exhaustive:
        # - N = 0: TIMESTAMPS (bloc vide)
        # - N < 50: COUNT (peu d'événements)
        # - N >= 50: COMBINATORIAL (beaucoup d'événements)
        threshold_N = 50  # Seuil optimal

        if features.N == 0:
            C = 0  # R1 - Timestamps
        elif features.N < threshold_N:
            C = 1  # R2 - Count
        else:
            C = 4  # R4b - Combinatorial

        return Cartouche(A=A, B=B, C=C)

    def _compute_cost(
        self,
        cartouche: Cartouche,
        optimal: BlockOptimalResult
    ) -> float:
        """Calcule le cout d'un cartouche."""
        return (
            optimal.cost_A.get(cartouche.A, float('inf')) +
            optimal.cost_B.get(cartouche.B, float('inf')) +
            optimal.cost_C.get(cartouche.C, float('inf'))
        )


class ObjectiveEvaluator:
    """
    Evaluateur haute performance pour le grid search.

    Optimise pour evaluer rapidement de nombreuses configurations.
    """

    def __init__(
        self,
        features_list: List[BlockFeatures],
        alpha: float = 1.0,
        beta: float = 0.5,
        n_workers: int = 4
    ):
        """
        Initialise l'evaluateur.

        Args:
            features_list: Liste des caracteristiques de blocs
            alpha: Poids du cout d'encodage
            beta: Poids de la penalite
            n_workers: Nombre de workers
        """
        self.features_list = features_list
        self.alpha = alpha
        self.beta = beta
        self.n_workers = n_workers

        # Pre-calculer les optimaux
        self.objective = ObjectiveFunction(alpha, beta)
        self.objective.precompute_optima(features_list, n_workers)

    def evaluate(self, config: ThresholdConfig) -> OptimizationResult:
        """Evalue une configuration."""
        return self.objective.evaluate(config, self.features_list, precomputed=True)

    def evaluate_batch(
        self,
        configs: List[ThresholdConfig]
    ) -> List[OptimizationResult]:
        """Evalue un lot de configurations."""
        return [self.evaluate(config) for config in configs]

    def find_best(
        self,
        configs: List[ThresholdConfig]
    ) -> Tuple[ThresholdConfig, OptimizationResult]:
        """
        Trouve la meilleure configuration parmi une liste.

        Args:
            configs: Liste de configurations

        Returns:
            (meilleure_config, resultat)
        """
        results = self.evaluate_batch(configs)
        best_idx = np.argmin([r.total_cost for r in results])
        return configs[best_idx], results[best_idx]
