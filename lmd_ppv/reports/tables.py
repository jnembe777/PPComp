"""
tables.py - Generateur de tableaux
===================================

Tableaux formates pour le rapport.

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


class TableGenerator:
    """
    Generateur de tableaux formates.
    """

    def __init__(self):
        self.tables = []

    def create_codec_comparison_table(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Codec Comparison"
    ) -> str:
        """
        Cree un tableau de comparaison des codecs.

        Args:
            data: Dict codec -> {psnr, ssim, ratio, speed, ...}
            title: Titre du tableau

        Returns:
            Tableau formate
        """
        headers = ['Codec', 'PSNR (dB)', 'SSIM', 'Ratio', 'Speed (fps)']
        rows = []

        for codec, metrics in data.items():
            rows.append([
                codec,
                f"{metrics.get('psnr', 0):.2f}",
                f"{metrics.get('ssim', 0):.4f}",
                f"{metrics.get('ratio', 0):.1f}x",
                f"{metrics.get('speed', 0):.1f}"
            ])

        if TABULATE_AVAILABLE:
            table = tabulate(rows, headers=headers, tablefmt='grid')
        else:
            table = self._simple_table(headers, rows)

        return f"\n{title}\n{'=' * len(title)}\n{table}"

    def create_threshold_config_table(
        self,
        config: Dict[str, float],
        title: str = "Optimal Threshold Configuration"
    ) -> str:
        """
        Cree un tableau de configuration des seuils.

        Args:
            config: Configuration des seuils
            title: Titre

        Returns:
            Tableau formate
        """
        descriptions = {
            'threshold_H_s': "Spatial homogeneity threshold",
            'threshold_rho_high': "High correlation threshold",
            'threshold_rho_low': "Low correlation threshold",
            'threshold_chi2': "Chi-square p-value threshold",
            'density_R1_max': "Max density for R1 (sparse)",
            'density_R4b_R2': "Density for R2 (dense)",
            'density_R4a_min': "Min density for R4a (boolean)",
        }

        headers = ['Parameter', 'Value', 'Description']
        rows = []

        for param, value in config.items():
            rows.append([
                param,
                f"{value:.4f}",
                descriptions.get(param, "")
            ])

        if TABULATE_AVAILABLE:
            table = tabulate(rows, headers=headers, tablefmt='grid')
        else:
            table = self._simple_table(headers, rows)

        return f"\n{title}\n{'=' * len(title)}\n{table}"

    def create_validation_summary_table(
        self,
        results: Dict[str, Any],
        title: str = "Validation Summary"
    ) -> str:
        """
        Cree un tableau resume de validation.

        Args:
            results: Resultats de validation
            title: Titre

        Returns:
            Tableau formate
        """
        headers = ['Metric', 'Value']
        rows = [
            ['Exact Accuracy', f"{results.get('exact_accuracy', 0):.2%}"],
            ['Dimension A Accuracy', f"{results.get('accuracy_A', 0):.2%}"],
            ['Dimension B Accuracy', f"{results.get('accuracy_B', 0):.2%}"],
            ['Dimension C Accuracy', f"{results.get('accuracy_C', 0):.2%}"],
            ['Mean Cost Penalty', f"{results.get('mean_penalty', 0):.2f} bits"],
            ['Max Cost Penalty', f"{results.get('max_penalty', 0):.2f} bits"],
            ['Total Blocks', f"{results.get('n_blocks', 0)}"],
        ]

        if TABULATE_AVAILABLE:
            table = tabulate(rows, headers=headers, tablefmt='grid')
        else:
            table = self._simple_table(headers, rows)

        return f"\n{title}\n{'=' * len(title)}\n{table}"

    def create_bd_rate_table(
        self,
        bd_rates: Dict[str, float],
        reference: str = 'H.265',
        title: str = "BD-Rate Comparison"
    ) -> str:
        """
        Cree un tableau de BD-Rates.

        Args:
            bd_rates: Dict codec -> BD-Rate
            reference: Codec de reference
            title: Titre

        Returns:
            Tableau formate
        """
        headers = ['Codec', f'BD-Rate vs {reference}', 'Interpretation']
        rows = []

        for codec, rate in bd_rates.items():
            if rate < -10:
                interp = "Much better"
            elif rate < 0:
                interp = "Better"
            elif rate == 0:
                interp = "Reference"
            elif rate < 10:
                interp = "Worse"
            else:
                interp = "Much worse"

            rows.append([
                codec,
                f"{rate:+.1f}%",
                interp
            ])

        if TABULATE_AVAILABLE:
            table = tabulate(rows, headers=headers, tablefmt='grid')
        else:
            table = self._simple_table(headers, rows)

        return f"\n{title}\n{'=' * len(title)}\n{table}"

    def create_per_video_table(
        self,
        results: List[Dict],
        title: str = "Per-Video Results"
    ) -> str:
        """
        Cree un tableau de resultats par video.

        Args:
            results: Liste de resultats
            title: Titre

        Returns:
            Tableau formate
        """
        headers = ['Video', 'Codec', 'CRF', 'PSNR', 'SSIM', 'Ratio', 'Time']
        rows = []

        for r in results:
            rows.append([
                r.get('video_name', '')[:20],
                r.get('codec_name', ''),
                str(r.get('quality_param', '')),
                f"{r.get('psnr', 0):.2f}",
                f"{r.get('ssim', 0):.4f}",
                f"{r.get('compression_ratio', 0):.1f}x",
                f"{r.get('encode_time_sec', 0):.2f}s"
            ])

        if TABULATE_AVAILABLE:
            table = tabulate(rows, headers=headers, tablefmt='grid')
        else:
            table = self._simple_table(headers, rows)

        return f"\n{title}\n{'=' * len(title)}\n{table}"

    def create_dataset_summary_table(
        self,
        stats: Dict,
        title: str = "Dataset Summary"
    ) -> str:
        """
        Cree un tableau resume du dataset.

        Args:
            stats: Statistiques du dataset
            title: Titre

        Returns:
            Tableau formate
        """
        headers = ['Attribute', 'Value']
        rows = [
            ['Total Videos', str(stats.get('total_videos', 0))],
            ['Total Frames', str(stats.get('total_frames', 0))],
            ['Total Duration', f"{stats.get('total_duration_hours', 0):.2f} hours"],
            ['Total Size', f"{stats.get('total_size_gb', 0):.2f} GB"],
            ['Avg FPS', f"{stats.get('avg_fps', 0):.1f}"],
            ['Avg Duration', f"{stats.get('avg_duration', 0):.1f} sec"],
        ]

        # Par source
        for source, count in stats.get('by_source', {}).items():
            rows.append([f'Source: {source}', str(count)])

        # Par resolution
        for res, count in stats.get('by_resolution', {}).items():
            rows.append([f'Resolution: {res}', str(count)])

        if TABULATE_AVAILABLE:
            table = tabulate(rows, headers=headers, tablefmt='grid')
        else:
            table = self._simple_table(headers, rows)

        return f"\n{title}\n{'=' * len(title)}\n{table}"

    def _simple_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Cree un tableau simple sans tabulate."""
        # Calculer les largeurs
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        # Header
        header_line = ' | '.join(h.ljust(widths[i]) for i, h in enumerate(headers))
        separator = '-+-'.join('-' * w for w in widths)

        # Rows
        row_lines = []
        for row in rows:
            row_line = ' | '.join(
                str(cell).ljust(widths[i]) if i < len(widths) else str(cell)
                for i, cell in enumerate(row)
            )
            row_lines.append(row_line)

        return f"{header_line}\n{separator}\n" + '\n'.join(row_lines)

    def to_html(
        self,
        headers: List[str],
        rows: List[List[str]],
        title: Optional[str] = None
    ) -> str:
        """
        Genere un tableau HTML.

        Args:
            headers: En-tetes
            rows: Lignes
            title: Titre optionnel

        Returns:
            HTML
        """
        html = ""
        if title:
            html += f"<h3>{title}</h3>\n"

        html += "<table class='data-table'>\n<thead>\n<tr>\n"
        for h in headers:
            html += f"<th>{h}</th>\n"
        html += "</tr>\n</thead>\n<tbody>\n"

        for row in rows:
            html += "<tr>\n"
            for cell in row:
                html += f"<td>{cell}</td>\n"
            html += "</tr>\n"

        html += "</tbody>\n</table>\n"
        return html

    def to_latex(
        self,
        headers: List[str],
        rows: List[List[str]],
        title: Optional[str] = None
    ) -> str:
        """
        Genere un tableau LaTeX.

        Args:
            headers: En-tetes
            rows: Lignes
            title: Titre optionnel

        Returns:
            LaTeX
        """
        n_cols = len(headers)
        col_spec = '|' + '|'.join(['c'] * n_cols) + '|'

        latex = "\\begin{table}[htbp]\n\\centering\n"
        if title:
            latex += f"\\caption{{{title}}}\n"

        latex += f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n"

        # Header
        latex += ' & '.join(f"\\textbf{{{h}}}" for h in headers) + " \\\\\n\\hline\n"

        # Rows
        for row in rows:
            latex += ' & '.join(str(cell) for cell in row) + " \\\\\n"

        latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
        return latex
