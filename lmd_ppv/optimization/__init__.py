"""
optimization - Optimisation des seuils du classifieur
======================================================

- threshold_config: Configuration des 7 seuils
- exhaustive_search: Recherche du cartouche optimal par bloc
- objective: Fonction objectif pour l'optimisation
- grid_search: Grid search sur les 32,000 configurations

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from .threshold_config import ThresholdConfig, ThresholdRange, THRESHOLD_RANGES
from .exhaustive_search import ExhaustiveSearch, BlockOptimalResult
from .objective import ObjectiveFunction, OptimizationResult
from .grid_search import GridSearch, GridSearchResult

__all__ = [
    'ThresholdConfig', 'ThresholdRange', 'THRESHOLD_RANGES',
    'ExhaustiveSearch', 'BlockOptimalResult',
    'ObjectiveFunction', 'OptimizationResult',
    'GridSearch', 'GridSearchResult'
]
