"""
predictor_vs_optimal.py - Comparaison predit vs optimal
=========================================================

Compare les predictions du classifieur aux cartouches optimaux
trouves par recherche exhaustive.

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.features import BlockFeatures
from src.core.cartouche import Cartouche
from src.core.lookup_tables import predict_dimension_C, select_representation_algo1
from src.agents.agent_1_classification import ClassificationAgent

from optimization.threshold_config import ThresholdConfig
from optimization.exhaustive_search import ExhaustiveSearch, BlockOptimalResult


@dataclass
class BlockValidation:
    """Resultat de validation pour un bloc."""
    block_id: int
    predicted: Cartouche
    optimal: Cartouche

    # Correspondances
    match_A: bool
    match_B: bool
    match_C: bool
    exact_match: bool

    # Couts
    predicted_cost: float
    optimal_cost: float
    cost_penalty: float

    def to_dict(self) -> Dict:
        return {
            'block_id': self.block_id,
            'predicted': self.predicted.to_string(),
            'optimal': self.optimal.to_string(),
            'match_A': self.match_A,
            'match_B': self.match_B,
            'match_C': self.match_C,
            'exact_match': self.exact_match,
            'predicted_cost': self.predicted_cost,
            'optimal_cost': self.optimal_cost,
            'cost_penalty': self.cost_penalty,
        }


@dataclass
class ValidationResult:
    """Resultats complets de la validation."""
    n_blocks: int
    validations: List[BlockValidation]

    # Metriques globales
    exact_accuracy: float
    accuracy_A: float
    accuracy_B: float
    accuracy_C: float

    # Couts
    total_predicted_cost: float
    total_optimal_cost: float
    total_penalty: float
    mean_penalty: float
    max_penalty: float

    # Configuration utilisee
    threshold_config: Optional[ThresholdConfig] = None

    def to_dict(self) -> Dict:
        return {
            'n_blocks': self.n_blocks,
            'exact_accuracy': self.exact_accuracy,
            'accuracy_A': self.accuracy_A,
            'accuracy_B': self.accuracy_B,
            'accuracy_C': self.accuracy_C,
            'total_predicted_cost': self.total_predicted_cost,
            'total_optimal_cost': self.total_optimal_cost,
            'total_penalty': self.total_penalty,
            'mean_penalty': self.mean_penalty,
            'max_penalty': self.max_penalty,
            'cost_overhead_percent': (self.total_penalty / self.total_optimal_cost * 100)
                                     if self.total_optimal_cost > 0 else 0,
            'threshold_config': self.threshold_config.to_dict() if self.threshold_config else None,
        }

    def save(self, path: Path) -> None:
        """Sauvegarde les resultats."""
        data = self.to_dict()
        data['validations'] = [v.to_dict() for v in self.validations]
        path.write_text(json.dumps(data, indent=2))

    def print_summary(self) -> None:
        """Affiche un resume des resultats."""
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        print(f"\nBlocks analyzed: {self.n_blocks}")
        print(f"\nAccuracy:")
        print(f"  Exact match:  {self.exact_accuracy:.2%}")
        print(f"  Dimension A:  {self.accuracy_A:.2%}")
        print(f"  Dimension B:  {self.accuracy_B:.2%}")
        print(f"  Dimension C:  {self.accuracy_C:.2%}")
        print(f"\nCost Analysis:")
        print(f"  Total predicted: {self.total_predicted_cost:.2f} bits")
        print(f"  Total optimal:   {self.total_optimal_cost:.2f} bits")
        print(f"  Total penalty:   {self.total_penalty:.2f} bits")
        print(f"  Mean penalty:    {self.mean_penalty:.2f} bits/block")
        print(f"  Max penalty:     {self.max_penalty:.2f} bits")
        overhead = (self.total_penalty / self.total_optimal_cost * 100) if self.total_optimal_cost > 0 else 0
        print(f"  Overhead:        {overhead:.2f}%")
        print("=" * 60)


class PredictorValidator:
    """
    Validateur du classifieur.

    Compare les predictions aux cartouches optimaux sur
    l'ensemble de test.
    """

    def __init__(
        self,
        threshold_config: Optional[ThresholdConfig] = None,
        exhaustive_search: Optional[ExhaustiveSearch] = None
    ):
        """
        Initialise le validateur.

        Args:
            threshold_config: Configuration des seuils (defaut = defaut)
            exhaustive_search: Instance de recherche exhaustive
        """
        self.threshold_config = threshold_config or ThresholdConfig.default()
        self.exhaustive_search = exhaustive_search or ExhaustiveSearch()

        # Creer l'agent avec les seuils
        self.agent = self._create_agent()

    def _create_agent(self) -> ClassificationAgent:
        """Cree l'agent de classification avec les seuils configures."""
        agent = ClassificationAgent()
        agent.threshold_H_s = self.threshold_config.threshold_H_s
        agent.threshold_rho_high = self.threshold_config.threshold_rho_high
        agent.threshold_rho_low = self.threshold_config.threshold_rho_low
        agent.threshold_chi2 = self.threshold_config.threshold_chi2
        return agent

    def update_thresholds(self, config: ThresholdConfig) -> None:
        """Met a jour la configuration des seuils."""
        self.threshold_config = config
        self.agent = self._create_agent()

    def validate_block(
        self,
        features: BlockFeatures,
        block_id: int = 0
    ) -> BlockValidation:
        """
        Valide la prediction pour un bloc.

        Args:
            features: Caracteristiques du bloc
            block_id: Identifiant du bloc

        Returns:
            BlockValidation
        """
        # Recherche exhaustive
        optimal_result = self.exhaustive_search.find_optimal(features, block_id)
        optimal = optimal_result.optimal_cartouche

        # Prediction
        predicted = self._predict(features)

        # Calculer le cout predit
        predicted_cost = (
            optimal_result.cost_A.get(predicted.A, float('inf')) +
            optimal_result.cost_B.get(predicted.B, float('inf')) +
            optimal_result.cost_C.get(predicted.C, float('inf'))
        )

        return BlockValidation(
            block_id=block_id,
            predicted=predicted,
            optimal=optimal,
            match_A=(predicted.A == optimal.A),
            match_B=(predicted.B == optimal.B),
            match_C=(predicted.C == optimal.C),
            exact_match=(predicted.A == optimal.A and
                         predicted.B == optimal.B and
                         predicted.C == optimal.C),
            predicted_cost=predicted_cost,
            optimal_cost=optimal_result.optimal_cost,
            cost_penalty=max(0, predicted_cost - optimal_result.optimal_cost)
        )

    def _predict(self, features: BlockFeatures) -> Cartouche:
        """
        Predit le cartouche pour un bloc.

        Utilise l'Algorithme 1 corrigé (Nembé, 2015) avec les tables
        de lookup T2-T4 pour la dimension C.
        """
        # Dimension A
        result = self.agent.classify(features)
        A = result.process_type

        # Dimension B
        B = features.suggest_color_mode()

        # Dimension C - Algorithme 1 corrigé avec tables de lookup
        # Remplace le seuil fixe par les bornes théoriques γ_l(r), γ_r(r), δ(r,m), λ(r,m)
        C = predict_dimension_C(features.N, features.r, features.m)

        return Cartouche(A=A, B=B, C=C)

    def validate_all(
        self,
        features_list: List[BlockFeatures],
        progress_callback: Optional[callable] = None
    ) -> ValidationResult:
        """
        Valide sur une liste de blocs.

        Args:
            features_list: Liste des caracteristiques
            progress_callback: Callback de progression

        Returns:
            ValidationResult complet
        """
        validations = []

        for i, features in enumerate(features_list):
            validation = self.validate_block(features, i)
            validations.append(validation)

            if progress_callback:
                progress_callback(i + 1, len(features_list))

        # Calculer les metriques
        n = len(validations)
        exact_matches = sum(1 for v in validations if v.exact_match)
        match_A = sum(1 for v in validations if v.match_A)
        match_B = sum(1 for v in validations if v.match_B)
        match_C = sum(1 for v in validations if v.match_C)

        penalties = [v.cost_penalty for v in validations]
        total_predicted = sum(v.predicted_cost for v in validations)
        total_optimal = sum(v.optimal_cost for v in validations)

        return ValidationResult(
            n_blocks=n,
            validations=validations,
            exact_accuracy=exact_matches / n if n > 0 else 0,
            accuracy_A=match_A / n if n > 0 else 0,
            accuracy_B=match_B / n if n > 0 else 0,
            accuracy_C=match_C / n if n > 0 else 0,
            total_predicted_cost=total_predicted,
            total_optimal_cost=total_optimal,
            total_penalty=sum(penalties),
            mean_penalty=np.mean(penalties) if penalties else 0,
            max_penalty=np.max(penalties) if penalties else 0,
            threshold_config=self.threshold_config
        )

    def compare_configs(
        self,
        features_list: List[BlockFeatures],
        configs: List[ThresholdConfig]
    ) -> List[ValidationResult]:
        """
        Compare plusieurs configurations de seuils.

        Args:
            features_list: Donnees de test
            configs: Liste de configurations a comparer

        Returns:
            Liste de ValidationResult
        """
        results = []

        for config in configs:
            self.update_thresholds(config)
            result = self.validate_all(features_list)
            results.append(result)

        return results


def run_validation(
    test_features: List[BlockFeatures],
    threshold_config: ThresholdConfig,
    output_path: Optional[Path] = None
) -> ValidationResult:
    """
    Execute la validation complete.

    Args:
        test_features: Caracteristiques de l'ensemble de test
        threshold_config: Configuration des seuils optimises
        output_path: Chemin de sortie (optionnel)

    Returns:
        ValidationResult
    """
    validator = PredictorValidator(threshold_config)
    result = validator.validate_all(test_features)

    result.print_summary()

    if output_path:
        result.save(output_path)
        print(f"\nResults saved to: {output_path}")

    return result
