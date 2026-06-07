"""
charts.py - Generateur de graphiques
=====================================

Graphiques matplotlib pour le rapport:
- Courbes Rate-Distortion
- Barres de comparaison
- Matrices de confusion

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class ChartGenerator:
    """
    Generateur de graphiques pour les rapports.
    """

    # Couleurs par codec
    CODEC_COLORS = {
        'H.264': '#e41a1c',
        'H.265': '#377eb8',
        'VP9': '#4daf4a',
        'AV1': '#984ea3',
        'LMD-PPV': '#ff7f00',
    }

    # Style par defaut
    DEFAULT_STYLE = {
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
    }

    def __init__(self, output_dir: Path, dpi: int = 150):
        """
        Initialise le generateur.

        Args:
            output_dir: Repertoire de sortie pour les images
            dpi: Resolution des images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        if MATPLOTLIB_AVAILABLE:
            plt.rcParams.update(self.DEFAULT_STYLE)

    def plot_rate_distortion(
        self,
        data: Dict[str, Tuple[List[float], List[float]]],
        metric: str = 'PSNR',
        title: Optional[str] = None,
        save_name: str = 'rate_distortion.png'
    ) -> Optional[Path]:
        """
        Trace les courbes Rate-Distortion.

        Args:
            data: Dict codec -> (bitrates, qualities)
            metric: 'PSNR' ou 'SSIM'
            title: Titre du graphique
            save_name: Nom du fichier de sortie

        Returns:
            Chemin du fichier ou None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        for codec, (bitrates, qualities) in data.items():
            color = self.CODEC_COLORS.get(codec, '#333333')
            ax.plot(bitrates, qualities, 'o-',
                    color=color, label=codec, linewidth=2, markersize=6)

        ax.set_xlabel('Bitrate (kbps)')
        ax.set_ylabel(f'{metric} (dB)' if metric == 'PSNR' else metric)
        ax.set_title(title or f'Rate-Distortion Curves ({metric})')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Echelle log pour bitrate
        ax.set_xscale('log')

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

        return save_path

    def plot_codec_comparison(
        self,
        data: Dict[str, Dict[str, float]],
        metric: str = 'psnr',
        title: Optional[str] = None,
        save_name: str = 'codec_comparison.png'
    ) -> Optional[Path]:
        """
        Trace un graphique de comparaison des codecs.

        Args:
            data: Dict codec -> {metric: value, ...}
            metric: Metrique a afficher
            title: Titre
            save_name: Nom du fichier

        Returns:
            Chemin du fichier
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        codecs = list(data.keys())
        values = [data[c].get(metric, 0) for c in codecs]
        colors = [self.CODEC_COLORS.get(c, '#333333') for c in codecs]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(codecs, values, color=colors, edgecolor='black', linewidth=1)

        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Codec')
        ax.set_ylabel(metric.upper())
        ax.set_title(title or f'Comparaison des codecs - {metric.upper()}')
        ax.grid(True, alpha=0.3, axis='y')

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

        return save_path

    def plot_confusion_matrix(
        self,
        matrix: np.ndarray,
        labels: List[str],
        title: str = 'Confusion Matrix',
        save_name: str = 'confusion_matrix.png'
    ) -> Optional[Path]:
        """
        Trace une matrice de confusion.

        Args:
            matrix: Matrice NxN
            labels: Labels des classes
            title: Titre
            save_name: Nom du fichier

        Returns:
            Chemin du fichier
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        if SEABORN_AVAILABLE:
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=ax)
        else:
            im = ax.imshow(matrix, cmap='Blues')
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            # Ajouter les valeurs
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, str(int(matrix[i, j])),
                            ha='center', va='center')

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

        return save_path

    def plot_encoding_speed(
        self,
        data: Dict[str, float],
        title: str = 'Encoding Speed Comparison',
        save_name: str = 'encoding_speed.png'
    ) -> Optional[Path]:
        """
        Trace la comparaison de vitesse d'encodage.

        Args:
            data: Dict codec -> fps
            title: Titre
            save_name: Nom du fichier

        Returns:
            Chemin du fichier
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        codecs = list(data.keys())
        fps = list(data.values())
        colors = [self.CODEC_COLORS.get(c, '#333333') for c in codecs]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(codecs, fps, color=colors, edgecolor='black')

        for bar, val in zip(bars, fps):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f} fps',
                    ha='left', va='center', fontsize=10)

        ax.set_xlabel('Encoding Speed (fps)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

        return save_path

    def plot_bd_rates(
        self,
        bd_rates: Dict[str, float],
        reference: str = 'H.265',
        title: Optional[str] = None,
        save_name: str = 'bd_rates.png'
    ) -> Optional[Path]:
        """
        Trace les BD-Rates par rapport a une reference.

        Args:
            bd_rates: Dict codec -> BD-Rate (%)
            reference: Codec de reference
            title: Titre
            save_name: Nom du fichier

        Returns:
            Chemin du fichier
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        codecs = [c for c in bd_rates.keys() if c != reference]
        rates = [bd_rates[c] for c in codecs]
        colors = ['green' if r < 0 else 'red' for r in rates]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(codecs, rates, color=colors, edgecolor='black')

        for bar, val in zip(bars, rates):
            width = bar.get_width()
            label = f'{val:+.1f}%'
            x_pos = width if width >= 0 else width - 5
            ax.text(x_pos, bar.get_y() + bar.get_height()/2.,
                    label, ha='left' if width >= 0 else 'right',
                    va='center', fontsize=10, fontweight='bold')

        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel(f'BD-Rate vs {reference} (%)')
        ax.set_title(title or f'BD-Rate Comparison (Reference: {reference})')
        ax.grid(True, alpha=0.3, axis='x')

        # Legende
        legend_elements = [
            mpatches.Patch(color='green', label='Better than reference'),
            mpatches.Patch(color='red', label='Worse than reference'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

        return save_path

    def plot_accuracy_by_dimension(
        self,
        accuracies: Dict[str, float],
        title: str = 'Classification Accuracy by Dimension',
        save_name: str = 'accuracy_dimensions.png'
    ) -> Optional[Path]:
        """
        Trace l'accuracy par dimension.

        Args:
            accuracies: Dict dimension -> accuracy
            title: Titre
            save_name: Nom du fichier

        Returns:
            Chemin du fichier
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        dims = list(accuracies.keys())
        accs = [accuracies[d] * 100 for d in dims]

        fig, ax = plt.subplots(figsize=(8, 5))

        colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f']
        bars = ax.bar(dims, accs, color=colors[:len(dims)], edgecolor='black')

        for bar, val in zip(bars, accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylim(0, 105)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        # Ligne a 100%
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

        return save_path

    def plot_cost_penalty_distribution(
        self,
        penalties: List[float],
        title: str = 'Cost Penalty Distribution',
        save_name: str = 'cost_penalties.png'
    ) -> Optional[Path]:
        """
        Trace la distribution des penalites de cout.

        Args:
            penalties: Liste des penalites
            title: Titre
            save_name: Nom du fichier

        Returns:
            Chemin du fichier
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(penalties, bins=50, color='steelblue', edgecolor='black', alpha=0.7)

        # Stats
        mean_p = np.mean(penalties)
        median_p = np.median(penalties)

        ax.axvline(x=mean_p, color='red', linestyle='--', label=f'Mean: {mean_p:.2f}')
        ax.axvline(x=median_p, color='green', linestyle='--', label=f'Median: {median_p:.2f}')

        ax.set_xlabel('Cost Penalty (bits)')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

        return save_path
