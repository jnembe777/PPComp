"""
validation - Validation du classifieur
=======================================

- predictor_vs_optimal: Comparaison predit vs optimal
- accuracy_metrics: Metriques de precision
- confusion_matrix: Matrices de confusion

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from .predictor_vs_optimal import PredictorValidator, ValidationResult
from .accuracy_metrics import AccuracyMetrics, compute_accuracy_metrics
from .confusion_matrix import ConfusionMatrixAnalyzer, DimensionConfusionMatrix

__all__ = [
    'PredictorValidator', 'ValidationResult',
    'AccuracyMetrics', 'compute_accuracy_metrics',
    'ConfusionMatrixAnalyzer', 'DimensionConfusionMatrix'
]
