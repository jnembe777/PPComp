"""
generator.py - Generateur principal de rapports
================================================

Genere des rapports en HTML et PDF avec:
- Resume executif
- Resultats du benchmark
- Optimisation des seuils
- Validation du classifieur
- Annexes

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ReportConfig:
    """Configuration du rapport."""
    title: str = "LMD-PPV Benchmark Report"
    author: str = "LMD-PPV Team"
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    # Sections a inclure
    include_executive_summary: bool = True
    include_dataset_description: bool = True
    include_benchmark_results: bool = True
    include_optimization: bool = True
    include_validation: bool = True
    include_per_video_analysis: bool = True
    include_appendix: bool = True

    # Formats de sortie
    output_html: bool = True
    output_pdf: bool = False
    output_json: bool = True

    # Style
    theme: str = "default"


class ReportGenerator:
    """
    Generateur de rapports de benchmark.

    Produit des rapports HTML interactifs et PDF.
    """

    def __init__(
        self,
        config: ReportConfig,
        output_dir: Path
    ):
        """
        Initialise le generateur.

        Args:
            config: Configuration du rapport
            output_dir: Repertoire de sortie
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Donnees du rapport
        self.benchmark_data: Dict = {}
        self.optimization_data: Dict = {}
        self.validation_data: Dict = {}
        self.dataset_data: Dict = {}

    def load_benchmark_results(self, path: Path) -> None:
        """Charge les resultats de benchmark."""
        if path.exists():
            self.benchmark_data = json.loads(path.read_text())

    def load_optimization_results(self, path: Path) -> None:
        """Charge les resultats d'optimisation."""
        if path.exists():
            self.optimization_data = json.loads(path.read_text())

    def load_validation_results(self, path: Path) -> None:
        """Charge les resultats de validation."""
        if path.exists():
            self.validation_data = json.loads(path.read_text())

    def load_dataset_info(self, path: Path) -> None:
        """Charge les informations du dataset."""
        if path.exists():
            self.dataset_data = json.loads(path.read_text())

    def generate(self) -> Dict[str, Path]:
        """
        Genere le rapport complet.

        Returns:
            Dictionnaire format -> chemin du fichier
        """
        outputs = {}

        # Generer les sections
        sections = self._build_sections()

        # HTML
        if self.config.output_html:
            html_path = self.output_dir / "report.html"
            self._generate_html(sections, html_path)
            outputs['html'] = html_path

        # PDF
        if self.config.output_pdf:
            pdf_path = self.output_dir / "report.pdf"
            self._generate_pdf(sections, pdf_path)
            outputs['pdf'] = pdf_path

        # JSON (donnees brutes)
        if self.config.output_json:
            json_path = self.output_dir / "report_data.json"
            self._generate_json(json_path)
            outputs['json'] = json_path

        return outputs

    def _build_sections(self) -> List[Dict]:
        """Construit les sections du rapport."""
        sections = []

        if self.config.include_executive_summary:
            sections.append(self._build_executive_summary())

        if self.config.include_dataset_description:
            sections.append(self._build_dataset_section())

        if self.config.include_benchmark_results:
            sections.append(self._build_benchmark_section())

        if self.config.include_optimization:
            sections.append(self._build_optimization_section())

        if self.config.include_validation:
            sections.append(self._build_validation_section())

        if self.config.include_per_video_analysis:
            sections.append(self._build_per_video_section())

        if self.config.include_appendix:
            sections.append(self._build_appendix())

        return sections

    def _build_executive_summary(self) -> Dict:
        """Construit le resume executif."""
        summary = {
            'id': 'executive-summary',
            'title': '1. Resume Executif',
            'content': []
        }

        # Performances cles
        if self.benchmark_data:
            n_videos = self.benchmark_data.get('n_videos', 0)
            n_codecs = self.benchmark_data.get('n_codecs', 0)
            success_rate = self.benchmark_data.get('success_rate', 0) * 100

            summary['content'].append({
                'type': 'text',
                'value': f"Ce rapport presente les resultats du benchmark LMD-PPV "
                         f"sur {n_videos} videos avec {n_codecs} codecs."
            })

            summary['content'].append({
                'type': 'metrics',
                'value': {
                    'Videos testees': n_videos,
                    'Codecs compares': n_codecs,
                    'Taux de succes': f"{success_rate:.1f}%"
                }
            })

        # Recommandations
        summary['content'].append({
            'type': 'subsection',
            'title': 'Recommandations',
            'content': [
                "LMD-PPV offre un bon compromis compression/qualite pour les contenus varies",
                "Les seuils optimises ameliorent la precision de classification de X%",
                "Pour les videos a fort mouvement, preferer H.265 ou AV1"
            ]
        })

        return summary

    def _build_dataset_section(self) -> Dict:
        """Construit la section dataset."""
        section = {
            'id': 'dataset',
            'title': '2. Description du Dataset',
            'content': []
        }

        if self.dataset_data:
            section['content'].append({
                'type': 'table',
                'title': 'Distribution par source',
                'data': self.dataset_data.get('by_source', {})
            })

            section['content'].append({
                'type': 'table',
                'title': 'Distribution par resolution',
                'data': self.dataset_data.get('by_resolution', {})
            })

        return section

    def _build_benchmark_section(self) -> Dict:
        """Construit la section benchmark."""
        section = {
            'id': 'benchmark',
            'title': '3. Resultats du Benchmark',
            'content': []
        }

        # Courbes Rate-Distortion
        section['content'].append({
            'type': 'chart',
            'chart_type': 'rate_distortion',
            'title': 'Courbes Rate-Distortion (PSNR)',
            'data_key': 'rate_distortion_psnr'
        })

        section['content'].append({
            'type': 'chart',
            'chart_type': 'rate_distortion',
            'title': 'Courbes Rate-Distortion (SSIM)',
            'data_key': 'rate_distortion_ssim'
        })

        # Tableau comparatif
        section['content'].append({
            'type': 'table',
            'title': 'Comparaison des codecs',
            'headers': ['Codec', 'PSNR moyen', 'SSIM moyen', 'Ratio', 'Vitesse'],
            'data_key': 'codec_comparison'
        })

        # BD-Rates
        section['content'].append({
            'type': 'table',
            'title': 'BD-Rates vs H.265',
            'data_key': 'bd_rates'
        })

        return section

    def _build_optimization_section(self) -> Dict:
        """Construit la section optimisation."""
        section = {
            'id': 'optimization',
            'title': '4. Optimisation des Seuils',
            'content': []
        }

        if self.optimization_data:
            best_config = self.optimization_data.get('best_config', {})

            section['content'].append({
                'type': 'table',
                'title': 'Configuration optimale',
                'data': best_config
            })

            section['content'].append({
                'type': 'text',
                'value': f"Amelioration vs configuration par defaut: "
                         f"{self.optimization_data.get('improvement', 0):.2f}%"
            })

        return section

    def _build_validation_section(self) -> Dict:
        """Construit la section validation."""
        section = {
            'id': 'validation',
            'title': '5. Validation du Classifieur',
            'content': []
        }

        if self.validation_data:
            # Precision globale
            section['content'].append({
                'type': 'metrics',
                'value': {
                    'Precision exacte': f"{self.validation_data.get('exact_accuracy', 0):.2%}",
                    'Precision A': f"{self.validation_data.get('accuracy_A', 0):.2%}",
                    'Precision B': f"{self.validation_data.get('accuracy_B', 0):.2%}",
                    'Precision C': f"{self.validation_data.get('accuracy_C', 0):.2%}",
                }
            })

            # Matrices de confusion
            section['content'].append({
                'type': 'chart',
                'chart_type': 'confusion_matrix',
                'title': 'Matrice de confusion - Dimension A',
                'data_key': 'confusion_A'
            })

        return section

    def _build_per_video_section(self) -> Dict:
        """Construit la section par video."""
        section = {
            'id': 'per-video',
            'title': '6. Analyse Par Video',
            'content': []
        }

        if self.benchmark_data and 'results' in self.benchmark_data:
            results = self.benchmark_data['results'][:20]  # Top 20

            section['content'].append({
                'type': 'table',
                'title': 'Resultats detailles',
                'headers': ['Video', 'Codec', 'PSNR', 'Ratio', 'Temps'],
                'rows': [
                    [r['video_name'], r['codec_name'],
                     f"{r['psnr']:.2f}", f"{r['compression_ratio']:.1f}x",
                     f"{r['encode_time_sec']:.2f}s"]
                    for r in results
                ]
            })

        return section

    def _build_appendix(self) -> Dict:
        """Construit les annexes."""
        return {
            'id': 'appendix',
            'title': '7. Annexes',
            'content': [
                {
                    'type': 'text',
                    'value': "Les donnees brutes sont disponibles dans le fichier "
                             "report_data.json"
                }
            ]
        }

    def _generate_html(self, sections: List[Dict], output_path: Path) -> None:
        """Genere le rapport HTML."""
        html_content = self._render_html(sections)
        output_path.write_text(html_content, encoding='utf-8')

    def _render_html(self, sections: List[Dict]) -> str:
        """Rend les sections en HTML."""
        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #444;
            margin-top: 40px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #007bff;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card h3 {{
            margin: 0;
            font-size: 2em;
        }}
        .metric-card p {{
            margin: 5px 0 0;
            opacity: 0.9;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
        }}
        .toc {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .toc ul {{
            list-style: none;
            padding-left: 20px;
        }}
        .toc a {{
            color: #007bff;
            text-decoration: none;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        footer {{
            text-align: center;
            margin-top: 40px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.config.title}</h1>
        <p><strong>Auteur:</strong> {self.config.author} | <strong>Date:</strong> {self.config.date}</p>

        <div class="toc">
            <h3>Table des matieres</h3>
            <ul>
"""
        # Table des matieres
        for section in sections:
            html += f'                <li><a href="#{section["id"]}">{section["title"]}</a></li>\n'

        html += """            </ul>
        </div>
"""

        # Sections
        for section in sections:
            html += self._render_section_html(section)

        html += f"""
        <footer>
            <p>Genere par LMD-PPV Benchmark Suite | {self.config.date}</p>
        </footer>
    </div>
</body>
</html>"""

        return html

    def _render_section_html(self, section: Dict) -> str:
        """Rend une section en HTML."""
        html = f'\n        <section id="{section["id"]}">\n'
        html += f'            <h2>{section["title"]}</h2>\n'

        for content in section.get('content', []):
            content_type = content.get('type')

            if content_type == 'text':
                html += f'            <p>{content["value"]}</p>\n'

            elif content_type == 'metrics':
                html += '            <div class="metrics">\n'
                for key, value in content['value'].items():
                    html += f'''                <div class="metric-card">
                    <h3>{value}</h3>
                    <p>{key}</p>
                </div>\n'''
                html += '            </div>\n'

            elif content_type == 'table':
                html += f'            <h3>{content.get("title", "")}</h3>\n'
                html += '            <table>\n'

                if 'headers' in content:
                    html += '                <tr>\n'
                    for h in content['headers']:
                        html += f'                    <th>{h}</th>\n'
                    html += '                </tr>\n'

                if 'rows' in content:
                    for row in content['rows']:
                        html += '                <tr>\n'
                        for cell in row:
                            html += f'                    <td>{cell}</td>\n'
                        html += '                </tr>\n'

                elif 'data' in content and isinstance(content['data'], dict):
                    for key, value in content['data'].items():
                        html += f'                <tr><td>{key}</td><td>{value}</td></tr>\n'

                html += '            </table>\n'

            elif content_type == 'chart':
                html += f'''            <div class="chart-container">
                <h3>{content.get("title", "")}</h3>
                <p>[Graphique: {content.get("chart_type", "")}]</p>
            </div>\n'''

            elif content_type == 'subsection':
                html += f'            <h3>{content.get("title", "")}</h3>\n'
                html += '            <ul>\n'
                for item in content.get('content', []):
                    html += f'                <li>{item}</li>\n'
                html += '            </ul>\n'

        html += '        </section>\n'
        return html

    def _generate_pdf(self, sections: List[Dict], output_path: Path) -> None:
        """Genere le rapport PDF."""
        try:
            from weasyprint import HTML

            # Generer HTML d'abord
            html_content = self._render_html(sections)
            HTML(string=html_content).write_pdf(str(output_path))

        except ImportError:
            print("WeasyPrint non installe, PDF non genere")
            print("Installez avec: pip install weasyprint")

    def _generate_json(self, output_path: Path) -> None:
        """Genere les donnees JSON."""
        data = {
            'config': {
                'title': self.config.title,
                'author': self.config.author,
                'date': self.config.date,
            },
            'benchmark': self.benchmark_data,
            'optimization': self.optimization_data,
            'validation': self.validation_data,
            'dataset': self.dataset_data,
        }
        output_path.write_text(json.dumps(data, indent=2))


def generate_report(
    output_dir: Path,
    benchmark_path: Optional[Path] = None,
    optimization_path: Optional[Path] = None,
    validation_path: Optional[Path] = None,
    formats: List[str] = ['html', 'json']
) -> Dict[str, Path]:
    """
    Genere un rapport complet.

    Args:
        output_dir: Repertoire de sortie
        benchmark_path: Chemin des resultats benchmark
        optimization_path: Chemin des resultats optimisation
        validation_path: Chemin des resultats validation
        formats: Formats de sortie ('html', 'pdf', 'json')

    Returns:
        Dictionnaire format -> chemin
    """
    config = ReportConfig(
        output_html='html' in formats,
        output_pdf='pdf' in formats,
        output_json='json' in formats
    )

    generator = ReportGenerator(config, output_dir)

    if benchmark_path:
        generator.load_benchmark_results(benchmark_path)
    if optimization_path:
        generator.load_optimization_results(optimization_path)
    if validation_path:
        generator.load_validation_results(validation_path)

    return generator.generate()
