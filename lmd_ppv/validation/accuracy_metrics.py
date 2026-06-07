"""
accuracy_metrics.py - Metriques de precision
=============================================

Metriques de validation:
- Exact Accuracy: % cartouches identiques
- A/B/C Accuracy: % par dimension
- Mean/Max Cost Penalty

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AccuracyMetrics:
    """Metriques de precision du classifieur."""

    # Accuracy globale et par dimension
    exact_accuracy: float  # % cartouches identiques
    accuracy_A: float      # % dimension A correcte
    accuracy_B: float      # % dimension B correcte
    accuracy_C: float      # % dimension C correcte

    # Penalites de cout
    mean_cost_penalty: float   # Surcout moyen (bits)
    max_cost_penalty: float    # Pire cas
    std_cost_penalty: float    # Ecart-type
    median_cost_penalty: float # Mediane

    # Distribution des penalites
    penalty_percentiles: Dict[int, float] = field(default_factory=dict)

    # Nombre d'echantillons
    n_samples: int = 0

    # Par type de processus (A)
    accuracy_by_A: Dict[int, float] = field(default_factory=dict)

    # Par mode couleur (B)
    accuracy_by_B: Dict[int, float] = field(default_factory=dict)

    # Par representation (C)
    accuracy_by_C: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'exact_accuracy': self.exact_accuracy,
            'accuracy_A': self.accuracy_A,
            'accuracy_B': self.accuracy_B,
            'accuracy_C': self.accuracy_C,
            'mean_cost_penalty': self.mean_cost_penalty,
            'max_cost_penalty': self.max_cost_penalty,
            'std_cost_penalty': self.std_cost_penalty,
            'median_cost_penalty': self.median_cost_penalty,
            'penalty_percentiles': self.penalty_percentiles,
            'n_samples': self.n_samples,
            'accuracy_by_A': self.accuracy_by_A,
            'accuracy_by_B': self.accuracy_by_B,
            'accuracy_by_C': self.accuracy_by_C,
        }

    def __str__(self) -> str:
        lines = [
            "Accuracy Metrics:",
            f"  Exact:  {self.exact_accuracy:.2%}",
            f"  Dim A:  {self.accuracy_A:.2%}",
            f"  Dim B:  {self.accuracy_B:.2%}",
            f"  Dim C:  {self.accuracy_C:.2%}",
            f"  Mean Penalty: {self.mean_cost_penalty:.2f} bits",
            f"  Max Penalty:  {self.max_cost_penalty:.2f} bits",
        ]
        return "\n".join(lines)


def compute_accuracy_metrics(
    predictions: List[Tuple[int, int, int]],  # (A, B, C) predit
    actuals: List[Tuple[int, int, int]],       # (A, B, C) optimal
    cost_penalties: Optional[List[float]] = None
) -> AccuracyMetrics:
    """
    Calcule les metriques de precision.

    Args:
        predictions: Liste de (A, B, C) predits
        actuals: Liste de (A, B, C) optimaux
        cost_penalties: Penalites de cout (optionnel)

    Returns:
        AccuracyMetrics
    """
    n = len(predictions)
    if n == 0:
        return AccuracyMetrics(
            exact_accuracy=0, accuracy_A=0, accuracy_B=0, accuracy_C=0,
            mean_cost_penalty=0, max_cost_penalty=0, std_cost_penalty=0,
            median_cost_penalty=0, n_samples=0
        )

    # Calculer les correspondances
    exact_matches = 0
    match_A = 0
    match_B = 0
    match_C = 0

    # Par valeur de dimension
    correct_by_A: Dict[int, int] = {}
    total_by_A: Dict[int, int] = {}
    correct_by_B: Dict[int, int] = {}
    total_by_B: Dict[int, int] = {}
    correct_by_C: Dict[int, int] = {}
    total_by_C: Dict[int, int] = {}

    for (pred_A, pred_B, pred_C), (act_A, act_B, act_C) in zip(predictions, actuals):
        # Global
        if pred_A == act_A:
            match_A += 1
        if pred_B == act_B:
            match_B += 1
        if pred_C == act_C:
            match_C += 1
        if pred_A == act_A and pred_B == act_B and pred_C == act_C:
            exact_matches += 1

        # Par valeur A
        total_by_A[act_A] = total_by_A.get(act_A, 0) + 1
        if pred_A == act_A:
            correct_by_A[act_A] = correct_by_A.get(act_A, 0) + 1

        # Par valeur B
        total_by_B[act_B] = total_by_B.get(act_B, 0) + 1
        if pred_B == act_B:
            correct_by_B[act_B] = correct_by_B.get(act_B, 0) + 1

        # Par valeur C
        total_by_C[act_C] = total_by_C.get(act_C, 0) + 1
        if pred_C == act_C:
            correct_by_C[act_C] = correct_by_C.get(act_C, 0) + 1

    # Accuracy par valeur
    accuracy_by_A = {
        k: correct_by_A.get(k, 0) / v
        for k, v in total_by_A.items()
    }
    accuracy_by_B = {
        k: correct_by_B.get(k, 0) / v
        for k, v in total_by_B.items()
    }
    accuracy_by_C = {
        k: correct_by_C.get(k, 0) / v
        for k, v in total_by_C.items()
    }

    # Penalites
    if cost_penalties:
        penalties = np.array(cost_penalties)
        mean_penalty = np.mean(penalties)
        max_penalty = np.max(penalties)
        std_penalty = np.std(penalties)
        median_penalty = np.median(penalties)
        percentiles = {
            50: np.percentile(penalties, 50),
            75: np.percentile(penalties, 75),
            90: np.percentile(penalties, 90),
            95: np.percentile(penalties, 95),
            99: np.percentile(penalties, 99),
        }
    else:
        mean_penalty = max_penalty = std_penalty = median_penalty = 0.0
        percentiles = {}

    return AccuracyMetrics(
        exact_accuracy=exact_matches / n,
        accuracy_A=match_A / n,
        accuracy_B=match_B / n,
        accuracy_C=match_C / n,
        mean_cost_penalty=mean_penalty,
        max_cost_penalty=max_penalty,
        std_cost_penalty=std_penalty,
        median_cost_penalty=median_penalty,
        penalty_percentiles=percentiles,
        n_samples=n,
        accuracy_by_A=accuracy_by_A,
        accuracy_by_B=accuracy_by_B,
        accuracy_by_C=accuracy_by_C,
    )


def compute_classification_report(
    predictions: List[Tuple[int, int, int]],
    actuals: List[Tuple[int, int, int]],
    dimension_names: Dict[str, Dict[int, str]] = None
) -> str:
    """
    Genere un rapport de classification detaille.

    Args:
        predictions: Predictions
        actuals: Valeurs reelles
        dimension_names: Noms des valeurs par dimension

    Returns:
        Rapport formate
    """
    if dimension_names is None:
        dimension_names = {
            'A': {0: 'Aa', 1: 'Ab', 2: 'Ac', 3: 'Ad', 4: 'Ae'},
            'B': {0: 'Ba', 1: 'Bb', 2: 'Bc', 3: 'Bd'},
            'C': {0: 'R1', 1: 'R2', 2: 'R3', 3: 'R4a', 4: 'R4b'},
        }

    metrics = compute_accuracy_metrics(predictions, actuals)

    lines = [
        "=" * 60,
        "CLASSIFICATION REPORT",
        "=" * 60,
        "",
        f"Total samples: {metrics.n_samples}",
        "",
        "GLOBAL ACCURACY",
        "-" * 40,
        f"  Exact match:     {metrics.exact_accuracy:.2%}",
        f"  Dimension A:     {metrics.accuracy_A:.2%}",
        f"  Dimension B:     {metrics.accuracy_B:.2%}",
        f"  Dimension C:     {metrics.accuracy_C:.2%}",
        "",
    ]

    # Par dimension A
    lines.append("ACCURACY BY A (Process Type)")
    lines.append("-" * 40)
    for k, v in sorted(metrics.accuracy_by_A.items()):
        name = dimension_names['A'].get(k, str(k))
        lines.append(f"  {name}: {v:.2%}")
    lines.append("")

    # Par dimension B
    lines.append("ACCURACY BY B (Color Mode)")
    lines.append("-" * 40)
    for k, v in sorted(metrics.accuracy_by_B.items()):
        name = dimension_names['B'].get(k, str(k))
        lines.append(f"  {name}: {v:.2%}")
    lines.append("")

    # Par dimension C
    lines.append("ACCURACY BY C (Representation)")
    lines.append("-" * 40)
    for k, v in sorted(metrics.accuracy_by_C.items()):
        name = dimension_names['C'].get(k, str(k))
        lines.append(f"  {name}: {v:.2%}")
    lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def compute_error_analysis(
    predictions: List[Tuple[int, int, int]],
    actuals: List[Tuple[int, int, int]],
    features_list: Optional[List] = None
) -> Dict:
    """
    Analyse les erreurs de prediction.

    Args:
        predictions: Predictions
        actuals: Valeurs reelles
        features_list: Caracteristiques des blocs (optionnel)

    Returns:
        Analyse des erreurs
    """
    errors = {
        'total': 0,
        'A_only': 0,  # Erreur uniquement sur A
        'B_only': 0,  # Erreur uniquement sur B
        'C_only': 0,  # Erreur uniquement sur C
        'AB': 0,      # Erreur sur A et B
        'AC': 0,      # Erreur sur A et C
        'BC': 0,      # Erreur sur B et C
        'ABC': 0,     # Erreur sur tout
    }

    error_indices = []

    for i, ((pred_A, pred_B, pred_C), (act_A, act_B, act_C)) in enumerate(zip(predictions, actuals)):
        err_A = pred_A != act_A
        err_B = pred_B != act_B
        err_C = pred_C != act_C

        if not (err_A or err_B or err_C):
            continue

        errors['total'] += 1
        error_indices.append(i)

        if err_A and err_B and err_C:
            errors['ABC'] += 1
        elif err_A and err_B:
            errors['AB'] += 1
        elif err_A and err_C:
            errors['AC'] += 1
        elif err_B and err_C:
            errors['BC'] += 1
        elif err_A:
            errors['A_only'] += 1
        elif err_B:
            errors['B_only'] += 1
        elif err_C:
            errors['C_only'] += 1

    return {
        'error_counts': errors,
        'error_rate': errors['total'] / len(predictions) if predictions else 0,
        'error_indices': error_indices,
    }
