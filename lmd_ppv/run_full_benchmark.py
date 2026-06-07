#!/usr/bin/env python3
"""
run_full_benchmark.py - Pipeline Complet de Benchmark LMD-PPV
==============================================================

Execute automatiquement toutes les etapes du benchmark:
1. Telechargement des datasets (si necessaire)
2. Extraction des features des videos
3. Optimisation des seuils du classifieur
4. Execution du benchmark sur tous les codecs
5. Validation du classifieur
6. Generation du rapport final

Usage:
    python run_full_benchmark.py
    python run_full_benchmark.py --quick           # Mode rapide (moins de videos)
    python run_full_benchmark.py --skip-download   # Utiliser les videos existantes
    python run_full_benchmark.py --codecs h264,h265,lmd  # Codecs specifiques
    python run_full_benchmark.py --resume          # Reprendre un benchmark interrompu

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class PipelineConfig:
    """Configuration du pipeline de benchmark."""
    # Repertoires
    base_dir: Path = field(default_factory=lambda: Path('./benchmark_output'))
    datasets_dir: Path = field(default_factory=lambda: Path('./datasets'))
    cache_dir: Path = field(default_factory=lambda: Path('./cache'))

    # Options de telechargement
    sources: List[str] = field(default_factory=lambda: ['xiph'])
    skip_download: bool = False

    # Options d'optimisation
    skip_optimization: bool = False
    optimization_alpha: float = 1.0
    optimization_beta: float = 0.5
    quick_optimization: bool = False

    # Options de benchmark
    codecs: List[str] = field(default_factory=lambda: ['h264', 'h265', 'vp9', 'av1', 'lmd'])
    max_videos: Optional[int] = None
    n_workers: int = 4

    # Options de validation
    skip_validation: bool = False

    # Options de rapport
    report_formats: List[str] = field(default_factory=lambda: ['html', 'json'])

    # Mode
    quick_mode: bool = False  # Mode rapide avec moins de videos
    resume: bool = False  # Reprendre un benchmark interrompu

    def __post_init__(self):
        """Appliquer les parametres du mode rapide."""
        if self.quick_mode:
            self.max_videos = self.max_videos or 5
            self.quick_optimization = True
            self.sources = ['xiph']
            logger.info("Mode rapide active: max 5 videos, optimisation rapide")

        # Creer les repertoires
        for dir_path in [self.base_dir, self.datasets_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineState:
    """Etat du pipeline pour la reprise."""
    started_at: str = ""
    current_step: str = ""
    completed_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        data = {
            'started_at': self.started_at,
            'current_step': self.current_step,
            'completed_steps': self.completed_steps,
            'step_results': self.step_results,
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> 'PipelineState':
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            started_at=data.get('started_at', ''),
            current_step=data.get('current_step', ''),
            completed_steps=data.get('completed_steps', []),
            step_results=data.get('step_results', {}),
        )


class BenchmarkPipeline:
    """
    Pipeline complet de benchmark LMD-PPV.

    Execute toutes les etapes du benchmark de maniere automatisee.
    """

    STEPS = [
        'download',
        'extract_features',
        'optimize',
        'benchmark',
        'validate',
        'report'
    ]

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state_path = config.base_dir / 'pipeline_state.json'
        self.state = PipelineState.load(self.state_path) if config.resume else PipelineState()

        # Timestamps
        self.start_time = time.time()

    def run(self) -> Dict:
        """
        Execute le pipeline complet.

        Returns:
            Dictionnaire avec les resultats de chaque etape
        """
        self._print_header()

        # Initialiser l'etat
        if not self.config.resume or not self.state.started_at:
            self.state.started_at = datetime.now().isoformat()

        results = {}

        try:
            # Etape 1: Telechargement
            if self._should_run_step('download'):
                results['download'] = self._step_download()
                self._complete_step('download', results['download'])

            # Etape 2: Extraction des features
            if self._should_run_step('extract_features'):
                results['extract_features'] = self._step_extract_features()
                self._complete_step('extract_features', results['extract_features'])

            # Etape 3: Optimisation
            if self._should_run_step('optimize'):
                results['optimize'] = self._step_optimize()
                self._complete_step('optimize', results['optimize'])

            # Etape 4: Benchmark
            if self._should_run_step('benchmark'):
                results['benchmark'] = self._step_benchmark()
                self._complete_step('benchmark', results['benchmark'])

            # Etape 5: Validation
            if self._should_run_step('validate'):
                results['validate'] = self._step_validate()
                self._complete_step('validate', results['validate'])

            # Etape 6: Rapport
            if self._should_run_step('report'):
                results['report'] = self._step_report()
                self._complete_step('report', results['report'])

            self._print_summary(results)

        except KeyboardInterrupt:
            logger.warning("\nInterruption utilisateur - etat sauvegarde")
            self.state.save(self.state_path)
            raise

        except Exception as e:
            logger.error(f"Erreur: {e}")
            self.state.save(self.state_path)
            raise

        return results

    def _should_run_step(self, step: str) -> bool:
        """Verifie si une etape doit etre executee."""
        # Verifier si deja completee
        if self.config.resume and step in self.state.completed_steps:
            logger.info(f"[SKIP] Etape '{step}' deja completee")
            return False

        # Verifier les options de skip
        if step == 'download' and self.config.skip_download:
            logger.info("[SKIP] Telechargement ignore (--skip-download)")
            return False

        if step == 'optimize' and self.config.skip_optimization:
            logger.info("[SKIP] Optimisation ignoree (--skip-optimization)")
            return False

        if step == 'validate' and self.config.skip_validation:
            logger.info("[SKIP] Validation ignoree (--skip-validation)")
            return False

        return True

    def _complete_step(self, step: str, result: Any) -> None:
        """Marque une etape comme completee."""
        self.state.completed_steps.append(step)
        self.state.step_results[step] = result
        self.state.save(self.state_path)

    def _print_header(self) -> None:
        """Affiche l'en-tete du pipeline."""
        print("\n" + "=" * 70)
        print("     LMD-PPV BENCHMARK PIPELINE")
        print("     Reference: J. Nembe, Codage LMD Versatile v6.0")
        print("=" * 70)
        print(f"  Repertoire de sortie: {self.config.base_dir}")
        print(f"  Codecs:               {', '.join(self.config.codecs)}")
        print(f"  Sources datasets:     {', '.join(self.config.sources)}")
        print(f"  Mode rapide:          {'Oui' if self.config.quick_mode else 'Non'}")
        print(f"  Workers:              {self.config.n_workers}")
        print("=" * 70 + "\n")

    def _print_step(self, step: str, number: int, total: int) -> None:
        """Affiche le debut d'une etape."""
        print(f"\n[{number}/{total}] === {step.upper()} " + "=" * (50 - len(step)))

    def _print_summary(self, results: Dict) -> None:
        """Affiche le resume final."""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 70)
        print("     RESUME DU BENCHMARK")
        print("=" * 70)

        # Statistiques par etape
        for step, result in results.items():
            status = "OK" if result.get('success', True) else "ERREUR"
            duration = result.get('duration_sec', 0)
            print(f"  {step:20} [{status}] ({duration:.1f}s)")

        # Statistiques globales
        if 'benchmark' in results:
            bench = results['benchmark']
            print(f"\n  Videos testees:    {bench.get('n_videos', 0)}")
            print(f"  Runs reussis:      {bench.get('successful_runs', 0)}/{bench.get('total_runs', 0)}")

        if 'validate' in results:
            val = results['validate']
            print(f"  Precision exacte:  {val.get('exact_accuracy', 0):.1%}")

        print(f"\n  Temps total:       {elapsed/60:.1f} minutes")
        print(f"  Resultats dans:    {self.config.base_dir}")
        print("=" * 70 + "\n")

    # =========================================================================
    # ETAPES DU PIPELINE
    # =========================================================================

    def _step_download(self) -> Dict:
        """Etape 1: Telecharger les datasets."""
        self._print_step('Telechargement des datasets', 1, 6)
        start = time.time()

        from benchmark.datasets.downloader import DatasetDownloader

        downloader = DatasetDownloader(self.config.datasets_dir)

        total_videos = 0
        by_source = {}

        for source in self.config.sources:
            logger.info(f"Telechargement depuis {source}...")

            try:
                videos = downloader.download_from_source(source)
                by_source[source] = len(videos)
                total_videos += len(videos)
                logger.info(f"  -> {len(videos)} videos")
            except Exception as e:
                logger.warning(f"  -> Erreur: {e}")
                by_source[source] = 0

        # Sauvegarder le manifest
        manifest_path = self.config.datasets_dir / 'manifest.json'
        downloader.save_manifest(manifest_path)

        duration = time.time() - start
        logger.info(f"Total: {total_videos} videos en {duration:.1f}s")

        return {
            'success': total_videos > 0,
            'total_videos': total_videos,
            'by_source': by_source,
            'duration_sec': duration,
        }

    def _step_extract_features(self) -> Dict:
        """Etape 2: Extraire les features des videos."""
        self._print_step('Extraction des features', 2, 6)
        start = time.time()

        from benchmark.datasets.downloader import DatasetDownloader
        from src.video.fast_encoder import FastVideoEncoder
        from src.core.features import BlockFeatures
        import numpy as np

        # Charger les videos
        downloader = DatasetDownloader(self.config.datasets_dir)
        videos = downloader.get_all_videos()

        if self.config.max_videos:
            videos = videos[:self.config.max_videos]

        logger.info(f"Extraction des features pour {len(videos)} videos...")

        all_features = []
        encoder = FastVideoEncoder(block_size=16, block_frames=32, n_colors=256)

        for i, video in enumerate(videos):
            try:
                logger.info(f"  [{i+1}/{len(videos)}] {video.name}")

                # Encoder avec stats
                stats = encoder.encode(video.path, max_frames=100)

                # Extraire les features de chaque bloc
                for block_stats in stats.get('block_stats', []):
                    features = BlockFeatures(
                        N=block_stats.get('n_jumps', 0),
                        r=block_stats.get('n_bins', 256),
                        m=block_stats.get('n_colors_used', 16),
                        H_s=block_stats.get('spatial_homogeneity', 0.5),
                        rho_corr=block_stats.get('color_correlation', 0.5),
                        H_color=block_stats.get('color_entropy', 3.0),
                        N_trans=block_stats.get('n_transitions', 0),
                    )
                    all_features.append(features)

            except Exception as e:
                logger.warning(f"    Erreur: {e}")

        # Si pas de features, generer synthetiques
        if not all_features:
            logger.warning("Pas de features extraites, generation synthetique...")
            np.random.seed(42)
            for _ in range(500):
                all_features.append(BlockFeatures(
                    N=np.random.randint(10, 200),
                    r=256,
                    m=np.random.randint(4, 64),
                    H_s=np.random.random(),
                    rho_corr=np.random.random(),
                    H_color=np.random.random() * 6,
                    N_trans=np.random.randint(5, 100)
                ))

        # Sauvegarder les features
        features_path = self.config.base_dir / 'features.json'
        features_data = [f.to_dict() if hasattr(f, 'to_dict') else f.__dict__ for f in all_features]
        features_path.write_text(json.dumps(features_data, indent=2))

        duration = time.time() - start
        logger.info(f"Extrait {len(all_features)} blocs en {duration:.1f}s")

        return {
            'success': True,
            'n_features': len(all_features),
            'features_path': str(features_path),
            'duration_sec': duration,
        }

    def _step_optimize(self) -> Dict:
        """Etape 3: Optimiser les seuils du classifieur."""
        self._print_step('Optimisation des seuils', 3, 6)
        start = time.time()

        from optimization.grid_search import run_optimization_pipeline
        from src.core.features import BlockFeatures

        # Charger les features
        features_path = self.config.base_dir / 'features.json'
        if features_path.exists():
            features_data = json.loads(features_path.read_text())
            # Filtrer les cles valides pour BlockFeatures
            valid_keys = {'N', 'r', 'm', 'n_pixels', 'H_s', 'rho_corr', 'H_color', 'N_trans',
                          'lambda_avg', 'R_temp', 'm_eff', 'var_delta_tau', 'mean_delta_tau'}
            features_list = [
                BlockFeatures(**{k: v for k, v in f.items() if k in valid_keys})
                for f in features_data
            ]
        else:
            logger.warning("Pas de features, utilisation de donnees synthetiques")
            import numpy as np
            np.random.seed(42)
            features_list = [
                BlockFeatures(
                    N=np.random.randint(10, 200),
                    r=256,
                    m=np.random.randint(4, 64),
                    H_s=np.random.random(),
                    rho_corr=np.random.random(),
                    H_color=np.random.random() * 6,
                    N_trans=np.random.randint(5, 100)
                )
                for _ in range(500)
            ]

        logger.info(f"Optimisation sur {len(features_list)} blocs...")

        output_path = self.config.base_dir / 'optimization.json'

        result = run_optimization_pipeline(
            features_list,
            output_path,
            alpha=self.config.optimization_alpha,
            beta=self.config.optimization_beta,
            n_workers=self.config.n_workers,
            quick_search=self.config.quick_optimization
        )

        duration = time.time() - start
        logger.info(f"Optimisation terminee en {duration:.1f}s")

        # Extraire les resultats (result peut etre un GridSearchResult ou un dict)
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        else:
            result_dict = result if isinstance(result, dict) else {}

        return {
            'success': True,
            'best_config': result_dict.get('best_config', {}),
            'improvement': result_dict.get('improvement', 0),
            'output_path': str(output_path),
            'duration_sec': duration,
        }

    def _step_benchmark(self) -> Dict:
        """Etape 4: Executer le benchmark sur tous les codecs."""
        self._print_step('Benchmark des codecs', 4, 6)
        start = time.time()

        from benchmark.runner import run_benchmark_pipeline
        from benchmark.datasets.downloader import DatasetDownloader
        from benchmark.config import BenchmarkConfig

        # Charger les videos
        downloader = DatasetDownloader(self.config.datasets_dir)
        videos = downloader.get_all_videos()

        if not videos:
            logger.warning("Aucune video trouvee, creation de videos de test...")
            # Utiliser les videos existantes dans le projet
            for ext in ['*.mp4', '*.y4m', '*.avi']:
                for vpath in self.config.datasets_dir.parent.glob(ext):
                    from benchmark.datasets.downloader import VideoInfo
                    videos.append(VideoInfo(
                        name=vpath.stem,
                        path=str(vpath),
                        source='local',
                        resolution='unknown'
                    ))

        if self.config.max_videos:
            videos = videos[:self.config.max_videos]

        logger.info(f"Benchmark sur {len(videos)} videos avec {len(self.config.codecs)} codecs...")

        output_path = self.config.base_dir / 'benchmark_results'
        output_path.mkdir(parents=True, exist_ok=True)

        config = BenchmarkConfig(
            output_dir=output_path,
            datasets_dir=self.config.datasets_dir,
            cache_dir=self.config.cache_dir,
            max_videos=self.config.max_videos,
            n_workers=self.config.n_workers
        )

        summary = run_benchmark_pipeline(
            videos,
            output_path,
            config,
            codecs=self.config.codecs
        )

        duration = time.time() - start

        return {
            'success': summary.successful_runs > 0,
            'n_videos': summary.n_videos,
            'n_codecs': summary.n_codecs,
            'total_runs': summary.total_runs,
            'successful_runs': summary.successful_runs,
            'output_path': str(output_path),
            'duration_sec': duration,
        }

    def _step_validate(self) -> Dict:
        """Etape 5: Valider le classifieur avec les seuils optimises."""
        self._print_step('Validation du classifieur', 5, 6)
        start = time.time()

        from validation.predictor_vs_optimal import run_validation
        from optimization.threshold_config import ThresholdConfig
        from src.core.features import BlockFeatures

        # Charger les seuils optimises
        opt_path = self.config.base_dir / 'optimization.json'
        if opt_path.exists():
            try:
                config = ThresholdConfig.load(opt_path)
                logger.info("Seuils optimises charges")
            except Exception as e:
                logger.warning(f"Erreur chargement seuils: {e}")
                config = ThresholdConfig.default()
        else:
            config = ThresholdConfig.default()
            logger.info("Utilisation des seuils par defaut")

        # Charger les features de test
        features_path = self.config.base_dir / 'features.json'
        if features_path.exists():
            features_data = json.loads(features_path.read_text())
            # Filtrer les cles valides pour BlockFeatures
            valid_keys = {'N', 'r', 'm', 'n_pixels', 'H_s', 'rho_corr', 'H_color', 'N_trans',
                          'lambda_avg', 'R_temp', 'm_eff', 'var_delta_tau', 'mean_delta_tau'}
            # Utiliser les derniers 20% pour le test
            test_size = max(50, len(features_data) // 5)
            test_features = [
                BlockFeatures(**{k: v for k, v in f.items() if k in valid_keys})
                for f in features_data[-test_size:]
            ]
        else:
            logger.warning("Pas de features, generation synthetique pour test")
            import numpy as np
            np.random.seed(123)
            test_features = [
                BlockFeatures(
                    N=np.random.randint(10, 200),
                    r=256,
                    m=np.random.randint(4, 64),
                    H_s=np.random.random(),
                    rho_corr=np.random.random(),
                    H_color=np.random.random() * 6,
                    N_trans=np.random.randint(5, 100)
                )
                for _ in range(100)
            ]

        logger.info(f"Validation sur {len(test_features)} blocs...")

        output_path = self.config.base_dir / 'validation_results.json'
        result = run_validation(test_features, config, output_path)

        duration = time.time() - start
        logger.info(f"Validation terminee en {duration:.1f}s")

        # Extraire les resultats (result peut etre un ValidationResult ou un dict)
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        else:
            result_dict = result if isinstance(result, dict) else {}

        return {
            'success': True,
            'exact_accuracy': result_dict.get('exact_accuracy', 0),
            'accuracy_A': result_dict.get('accuracy_A', 0),
            'accuracy_B': result_dict.get('accuracy_B', 0),
            'accuracy_C': result_dict.get('accuracy_C', 0),
            'output_path': str(output_path),
            'duration_sec': duration,
        }

    def _step_report(self) -> Dict:
        """Etape 6: Generer le rapport final."""
        self._print_step('Generation du rapport', 6, 6)
        start = time.time()

        from reports.generator import generate_report

        report_dir = self.config.base_dir / 'report'

        outputs = generate_report(
            report_dir,
            benchmark_path=self.config.base_dir / 'benchmark_results' / 'benchmark_results.json',
            optimization_path=self.config.base_dir / 'optimization.json',
            validation_path=self.config.base_dir / 'validation_results.json',
            formats=self.config.report_formats
        )

        duration = time.time() - start

        # Afficher les chemins des rapports
        logger.info("Rapports generes:")
        for fmt, path in outputs.items():
            logger.info(f"  {fmt.upper()}: {path}")

        return {
            'success': True,
            'outputs': {k: str(v) for k, v in outputs.items()},
            'duration_sec': duration,
        }


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline complet de benchmark LMD-PPV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python run_full_benchmark.py                          # Benchmark complet
  python run_full_benchmark.py --quick                  # Mode rapide (5 videos)
  python run_full_benchmark.py --skip-download          # Utiliser videos existantes
  python run_full_benchmark.py --codecs h264,h265,lmd   # Codecs specifiques
  python run_full_benchmark.py --resume                 # Reprendre un benchmark
  python run_full_benchmark.py --output ./results       # Repertoire de sortie
        """
    )

    # Options generales
    parser.add_argument('-o', '--output', type=Path, default=Path('./benchmark_output'),
                        help='Repertoire de sortie (defaut: ./benchmark_output)')
    parser.add_argument('--datasets', type=Path, default=Path('./datasets'),
                        help='Repertoire des datasets (defaut: ./datasets)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Nombre de workers paralleles (defaut: 4)')

    # Mode rapide
    parser.add_argument('--quick', action='store_true',
                        help='Mode rapide (5 videos, optimisation rapide)')
    parser.add_argument('--max-videos', type=int,
                        help='Nombre maximum de videos')

    # Options de telechargement
    parser.add_argument('--skip-download', action='store_true',
                        help='Utiliser les videos existantes')
    parser.add_argument('--sources', default='xiph',
                        help='Sources de datasets: xiph,cdvl,vimeo (defaut: xiph)')

    # Options de benchmark
    parser.add_argument('--codecs', default='h264,h265,vp9,av1,lmd',
                        help='Codecs a tester (defaut: h264,h265,vp9,av1,lmd)')

    # Options d'optimisation
    parser.add_argument('--skip-optimization', action='store_true',
                        help='Ignorer l\'optimisation des seuils')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Poids du cout d\'encodage (defaut: 1.0)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Poids de la penalite (defaut: 0.5)')

    # Options de validation
    parser.add_argument('--skip-validation', action='store_true',
                        help='Ignorer la validation')

    # Options de rapport
    parser.add_argument('--format', default='html,json',
                        help='Formats de rapport: html,pdf,json (defaut: html,json)')

    # Reprise
    parser.add_argument('--resume', action='store_true',
                        help='Reprendre un benchmark interrompu')

    args = parser.parse_args()

    # Configuration
    config = PipelineConfig(
        base_dir=args.output,
        datasets_dir=args.datasets,
        sources=args.sources.split(','),
        skip_download=args.skip_download,
        codecs=args.codecs.split(','),
        max_videos=args.max_videos,
        n_workers=args.workers,
        skip_optimization=args.skip_optimization,
        optimization_alpha=args.alpha,
        optimization_beta=args.beta,
        skip_validation=args.skip_validation,
        report_formats=args.format.split(','),
        quick_mode=args.quick,
        resume=args.resume,
    )

    # Executer le pipeline
    pipeline = BenchmarkPipeline(config)

    try:
        results = pipeline.run()
        return 0
    except KeyboardInterrupt:
        print("\n[INTERROMPU] Utilisez --resume pour reprendre")
        return 1
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
