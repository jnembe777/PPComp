"""
confusion_matrix.py - Matrices de confusion
=============================================

Matrices de confusion par dimension:
- Matrice 5x5 pour dimension A (ProcessType)
- Matrice 4x4 pour dimension B (ColorMode)
- Matrice 5x5 pour dimension C (Representation)

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class DimensionConfusionMatrix:
    """Matrice de confusion pour une dimension."""
    dimension: str  # 'A', 'B', 'C'
    labels: List[str]
    matrix: np.ndarray

    # Metriques derivees
    accuracy: float = 0.0
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1_score: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Calcule les metriques derivees."""
        self._compute_metrics()

    def _compute_metrics(self):
        """Calcule precision, recall, F1 pour chaque classe."""
        n = len(self.labels)

        # Accuracy globale
        total = self.matrix.sum()
        correct = np.trace(self.matrix)
        self.accuracy = correct / total if total > 0 else 0

        # Par classe
        for i, label in enumerate(self.labels):
            tp = self.matrix[i, i]
            fp = self.matrix[:, i].sum() - tp  # Colonne i - diagonal
            fn = self.matrix[i, :].sum() - tp  # Ligne i - diagonal

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            self.precision[label] = precision
            self.recall[label] = recall
            self.f1_score[label] = f1

    def to_dict(self) -> Dict:
        return {
            'dimension': self.dimension,
            'labels': self.labels,
            'matrix': self.matrix.tolist(),
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
        }

    def __str__(self) -> str:
        """Representation textuelle de la matrice."""
        n = len(self.labels)
        max_label_len = max(len(l) for l in self.labels)

        lines = [f"Confusion Matrix - Dimension {self.dimension}"]
        lines.append("=" * (max_label_len + n * 8 + 10))

        # Header
        header = " " * (max_label_len + 2) + "Predicted"
        lines.append(header)
        header2 = " " * (max_label_len + 2) + "".join(f"{l:>7}" for l in self.labels)
        lines.append(header2)
        lines.append("-" * len(header2))

        # Rows
        for i, label in enumerate(self.labels):
            prefix = "Actual " if i == len(self.labels) // 2 else "       "
            row = f"{prefix}{label:>{max_label_len}} |"
            row += "".join(f"{int(self.matrix[i, j]):>7}" for j in range(n))
            lines.append(row)

        # Footer
        lines.append("-" * len(header2))
        lines.append(f"Accuracy: {self.accuracy:.2%}")

        return "\n".join(lines)

    def plot(self, save_path: Optional[Path] = None):
        """
        Affiche la matrice de confusion (matplotlib).

        Args:
            save_path: Chemin pour sauvegarder (optionnel)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib/seaborn required for plotting")
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            self.matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.labels,
            yticklabels=self.labels,
            ax=ax
        )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - Dimension {self.dimension}\nAccuracy: {self.accuracy:.2%}')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()


class ConfusionMatrixAnalyzer:
    """
    Analyseur de matrices de confusion.

    Genere les matrices pour les 3 dimensions A, B, C.
    """

    # Labels par dimension
    LABELS = {
        'A': ['Aa', 'Ab', 'Ac', 'Ad', 'Ae'],  # ProcessType
        'B': ['Ba', 'Bb', 'Bc', 'Bd'],         # ColorMode
        'C': ['R1', 'R2', 'R3', 'R4a', 'R4b'], # Representation
    }

    def __init__(self):
        self.matrices: Dict[str, DimensionConfusionMatrix] = {}

    def compute(
        self,
        predictions: List[Tuple[int, int, int]],
        actuals: List[Tuple[int, int, int]]
    ) -> None:
        """
        Calcule les matrices de confusion.

        Args:
            predictions: Liste de (A, B, C) predits
            actuals: Liste de (A, B, C) optimaux
        """
        # Extraire par dimension
        pred_A = [p[0] for p in predictions]
        pred_B = [p[1] for p in predictions]
        pred_C = [p[2] for p in predictions]

        act_A = [a[0] for a in actuals]
        act_B = [a[1] for a in actuals]
        act_C = [a[2] for a in actuals]

        # Calculer les matrices
        self.matrices['A'] = self._compute_matrix('A', pred_A, act_A)
        self.matrices['B'] = self._compute_matrix('B', pred_B, act_B)
        self.matrices['C'] = self._compute_matrix('C', pred_C, act_C)

    def _compute_matrix(
        self,
        dimension: str,
        predictions: List[int],
        actuals: List[int]
    ) -> DimensionConfusionMatrix:
        """Calcule la matrice pour une dimension."""
        labels = self.LABELS[dimension]
        n = len(labels)

        matrix = np.zeros((n, n), dtype=int)

        for pred, act in zip(predictions, actuals):
            if 0 <= pred < n and 0 <= act < n:
                matrix[act, pred] += 1

        return DimensionConfusionMatrix(
            dimension=dimension,
            labels=labels,
            matrix=matrix
        )

    def get_matrix(self, dimension: str) -> Optional[DimensionConfusionMatrix]:
        """Retourne la matrice pour une dimension."""
        return self.matrices.get(dimension)

    def get_all(self) -> Dict[str, DimensionConfusionMatrix]:
        """Retourne toutes les matrices."""
        return self.matrices

    def print_all(self) -> None:
        """Affiche toutes les matrices."""
        for dim in ['A', 'B', 'C']:
            if dim in self.matrices:
                print(self.matrices[dim])
                print()

    def plot_all(self, output_dir: Path) -> None:
        """
        Genere les graphiques pour toutes les matrices.

        Args:
            output_dir: Repertoire de sortie
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for dim in ['A', 'B', 'C']:
            if dim in self.matrices:
                save_path = output_dir / f"confusion_matrix_{dim}.png"
                self.matrices[dim].plot(save_path)

    def to_dict(self) -> Dict:
        return {
            dim: mat.to_dict()
            for dim, mat in self.matrices.items()
        }

    def save(self, path: Path) -> None:
        """Sauvegarde les matrices."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> 'ConfusionMatrixAnalyzer':
        """Charge les matrices."""
        data = json.loads(path.read_text())

        analyzer = cls()
        for dim, mat_data in data.items():
            analyzer.matrices[dim] = DimensionConfusionMatrix(
                dimension=mat_data['dimension'],
                labels=mat_data['labels'],
                matrix=np.array(mat_data['matrix'])
            )

        return analyzer

    def get_summary(self) -> Dict:
        """Resume des performances par dimension."""
        return {
            dim: {
                'accuracy': mat.accuracy,
                'macro_precision': np.mean(list(mat.precision.values())),
                'macro_recall': np.mean(list(mat.recall.values())),
                'macro_f1': np.mean(list(mat.f1_score.values())),
            }
            for dim, mat in self.matrices.items()
        }

    def get_worst_confusions(self, top_k: int = 5) -> Dict[str, List[Tuple[str, str, int]]]:
        """
        Identifie les pires confusions.

        Args:
            top_k: Nombre de confusions a retourner par dimension

        Returns:
            Dict dimension -> [(actual, predicted, count), ...]
        """
        worst = {}

        for dim, mat in self.matrices.items():
            confusions = []
            n = len(mat.labels)

            for i in range(n):
                for j in range(n):
                    if i != j and mat.matrix[i, j] > 0:
                        confusions.append((
                            mat.labels[i],  # actual
                            mat.labels[j],  # predicted
                            int(mat.matrix[i, j])  # count
                        ))

            # Trier par count
            confusions.sort(key=lambda x: x[2], reverse=True)
            worst[dim] = confusions[:top_k]

        return worst
