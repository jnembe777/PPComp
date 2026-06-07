"""
grid_search.py - Grid search sur les 32,000 configurations
============================================================

Parcourt les 32,000 combinaisons de seuils:
5 x 5 x 5 x 4 x 4 x 4 x 4 = 32,000

Parallelize avec multiprocessing pour performance.

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.features import BlockFeatures

from .threshold_config import ThresholdConfig, ThresholdConfigGenerator
from .objective import ObjectiveFunction, OptimizationResult, ObjectiveEvaluator


@dataclass
class GridSearchResult:
    """Resultat du grid search."""
    best_config: ThresholdConfig
    best_result: OptimizationResult
    all_results: List[OptimizationResult] = field(default_factory=list)

    # Statistiques
    n_evaluated: int = 0
    total_time_sec: float = 0.0
    evaluations_per_sec: float = 0.0

    # Top configurations
    top_k: List[Tuple[ThresholdConfig, OptimizationResult]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'best_config': self.best_config.to_dict(),
            'best_result': self.best_result.to_dict(),
            'n_evaluated': self.n_evaluated,
            'total_time_sec': self.total_time_sec,
            'evaluations_per_sec': self.evaluations_per_sec,
            'top_k': [
                {'config': c.to_dict(), 'cost': r.total_cost}
                for c, r in self.top_k
            ]
        }

    def save(self, path: Path) -> None:
        """Sauvegarde les resultats."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> 'GridSearchResult':
        """Charge les resultats."""
        data = json.loads(path.read_text())

        best_config = ThresholdConfig.from_dict(data['best_config'])
        best_result = OptimizationResult(
            config=best_config,
            total_cost=data['best_result']['total_cost'],
            encoding_cost=data['best_result']['encoding_cost'],
            penalty_cost=data['best_result']['penalty_cost'],
            n_blocks=data['best_result']['n_blocks'],
            exact_matches=data['best_result']['exact_matches'],
            accuracy_A=data['best_result']['accuracy_A'],
            accuracy_B=data['best_result']['accuracy_B'],
            accuracy_C=data['best_result']['accuracy_C'],
        )

        return cls(
            best_config=best_config,
            best_result=best_result,
            n_evaluated=data['n_evaluated'],
            total_time_sec=data['total_time_sec'],
            evaluations_per_sec=data['evaluations_per_sec']
        )


class GridSearch:
    """
    Grid search pour optimiser les seuils du classifieur.

    Explore systematiquement les 32,000 configurations possibles
    et identifie celle minimisant la fonction objectif.
    """

    def __init__(
        self,
        features_list: List[BlockFeatures],
        alpha: float = 1.0,
        beta: float = 0.5,
        n_workers: int = 4,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ):
        """
        Initialise le grid search.

        Args:
            features_list: Liste des caracteristiques de blocs
            alpha: Poids du cout d'encodage
            beta: Poids de la penalite
            n_workers: Nombre de workers paralleles
            progress_callback: Callback (current, total, best_cost)
        """
        self.features_list = features_list
        self.alpha = alpha
        self.beta = beta
        self.n_workers = n_workers
        self.progress_callback = progress_callback

        # Evaluateur
        self.evaluator = ObjectiveEvaluator(
            features_list, alpha, beta, n_workers
        )

        # Generateur de configurations
        self.generator = ThresholdConfigGenerator()

    def run(
        self,
        max_configs: Optional[int] = None,
        save_all: bool = False,
        checkpoint_path: Optional[Path] = None
    ) -> GridSearchResult:
        """
        Execute le grid search complet.

        Args:
            max_configs: Limite du nombre de configurations (None = toutes)
            save_all: Sauvegarder tous les resultats
            checkpoint_path: Chemin pour les checkpoints

        Returns:
            GridSearchResult
        """
        start_time = time.time()

        # Generer les configurations
        total = self.generator.total_combinations
        if max_configs:
            total = min(total, max_configs)

        configs = list(self.generator.generate_all())
        if max_configs:
            configs = configs[:max_configs]

        # Evaluer
        all_results = []
        best_config = None
        best_result = None
        best_cost = float('inf')

        for i, config in enumerate(tqdm(configs, desc="Grid Search")):
            result = self.evaluator.evaluate(config)
            all_results.append(result)

            if result.total_cost < best_cost:
                best_cost = result.total_cost
                best_config = config
                best_result = result

            if self.progress_callback:
                self.progress_callback(i + 1, total, best_cost)

            # Checkpoint
            if checkpoint_path and (i + 1) % 1000 == 0:
                self._save_checkpoint(checkpoint_path, best_config, best_result, i + 1)

        total_time = time.time() - start_time

        # Top-k configurations
        top_k = self._get_top_k(all_results, k=10)

        return GridSearchResult(
            best_config=best_config,
            best_result=best_result,
            all_results=all_results if save_all else [],
            n_evaluated=len(configs),
            total_time_sec=total_time,
            evaluations_per_sec=len(configs) / total_time if total_time > 0 else 0,
            top_k=top_k
        )

    def run_parallel(
        self,
        max_configs: Optional[int] = None,
        batch_size: int = 100
    ) -> GridSearchResult:
        """
        Execute le grid search en parallele.

        Args:
            max_configs: Limite du nombre de configurations
            batch_size: Taille des lots

        Returns:
            GridSearchResult
        """
        start_time = time.time()

        configs = list(self.generator.generate_all())
        if max_configs:
            configs = configs[:max_configs]

        # Diviser en lots
        batches = [
            configs[i:i + batch_size]
            for i in range(0, len(configs), batch_size)
        ]

        all_results = []
        best_config = None
        best_result = None
        best_cost = float('inf')

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []

            for batch in batches:
                future = executor.submit(
                    self._evaluate_batch_worker,
                    batch,
                    self.features_list,
                    self.alpha,
                    self.beta
                )
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Grid Search"):
                batch_results = future.result()
                all_results.extend(batch_results)

                for result in batch_results:
                    if result.total_cost < best_cost:
                        best_cost = result.total_cost
                        best_config = result.config
                        best_result = result

        total_time = time.time() - start_time
        top_k = self._get_top_k(all_results, k=10)

        return GridSearchResult(
            best_config=best_config,
            best_result=best_result,
            all_results=[],
            n_evaluated=len(configs),
            total_time_sec=total_time,
            evaluations_per_sec=len(configs) / total_time if total_time > 0 else 0,
            top_k=top_k
        )

    @staticmethod
    def _evaluate_batch_worker(
        configs: List[ThresholdConfig],
        features_list: List[BlockFeatures],
        alpha: float,
        beta: float
    ) -> List[OptimizationResult]:
        """Worker pour evaluation parallele."""
        evaluator = ObjectiveEvaluator(features_list, alpha, beta, n_workers=1)
        return evaluator.evaluate_batch(configs)

    def run_random_sample(
        self,
        n_samples: int = 1000,
        seed: int = 42
    ) -> GridSearchResult:
        """
        Execute un grid search sur un echantillon aleatoire.

        Utile pour une exploration rapide avant le search complet.

        Args:
            n_samples: Nombre d'echantillons
            seed: Graine aleatoire

        Returns:
            GridSearchResult
        """
        start_time = time.time()

        configs = self.generator.generate_sample(n_samples, seed)

        all_results = []
        best_config = None
        best_result = None
        best_cost = float('inf')

        for config in tqdm(configs, desc="Random Search"):
            result = self.evaluator.evaluate(config)
            all_results.append(result)

            if result.total_cost < best_cost:
                best_cost = result.total_cost
                best_config = config
                best_result = result

        total_time = time.time() - start_time
        top_k = self._get_top_k(all_results, k=10)

        return GridSearchResult(
            best_config=best_config,
            best_result=best_result,
            all_results=[],
            n_evaluated=len(configs),
            total_time_sec=total_time,
            evaluations_per_sec=len(configs) / total_time if total_time > 0 else 0,
            top_k=top_k
        )

    def run_local_search(
        self,
        initial_config: Optional[ThresholdConfig] = None,
        max_iterations: int = 100
    ) -> GridSearchResult:
        """
        Execute une recherche locale a partir d'une configuration initiale.

        Args:
            initial_config: Configuration de depart (defaut = defaut)
            max_iterations: Nombre max d'iterations

        Returns:
            GridSearchResult
        """
        start_time = time.time()

        current = initial_config or ThresholdConfig.default()
        current_result = self.evaluator.evaluate(current)
        best = current
        best_result = current_result

        n_evaluated = 1
        all_results = [current_result]

        for _ in range(max_iterations):
            # Generer les voisins
            neighbors = self.generator.generate_neighbors(current)

            improved = False
            for neighbor in neighbors:
                result = self.evaluator.evaluate(neighbor)
                all_results.append(result)
                n_evaluated += 1

                if result.total_cost < best_result.total_cost:
                    best = neighbor
                    best_result = result
                    current = neighbor
                    current_result = result
                    improved = True
                    break

            if not improved:
                break

        total_time = time.time() - start_time
        top_k = self._get_top_k(all_results, k=10)

        return GridSearchResult(
            best_config=best,
            best_result=best_result,
            all_results=[],
            n_evaluated=n_evaluated,
            total_time_sec=total_time,
            evaluations_per_sec=n_evaluated / total_time if total_time > 0 else 0,
            top_k=top_k
        )

    def _get_top_k(
        self,
        results: List[OptimizationResult],
        k: int = 10
    ) -> List[Tuple[ThresholdConfig, OptimizationResult]]:
        """Retourne les k meilleures configurations."""
        sorted_results = sorted(results, key=lambda r: r.total_cost)
        return [(r.config, r) for r in sorted_results[:k]]

    def _save_checkpoint(
        self,
        path: Path,
        best_config: ThresholdConfig,
        best_result: OptimizationResult,
        n_evaluated: int
    ) -> None:
        """Sauvegarde un checkpoint."""
        checkpoint = {
            'n_evaluated': n_evaluated,
            'best_config': best_config.to_dict(),
            'best_cost': best_result.total_cost
        }
        path.write_text(json.dumps(checkpoint, indent=2))


def run_optimization_pipeline(
    features_list: List[BlockFeatures],
    output_path: Path,
    alpha: float = 1.0,
    beta: float = 0.5,
    n_workers: int = 4,
    quick_search: bool = True
) -> GridSearchResult:
    """
    Pipeline complet d'optimisation des seuils.

    Args:
        features_list: Caracteristiques des blocs
        output_path: Chemin de sortie pour les resultats
        alpha: Poids du cout d'encodage
        beta: Poids de la penalite
        n_workers: Nombre de workers
        quick_search: Utiliser la recherche rapide

    Returns:
        GridSearchResult
    """
    grid_search = GridSearch(features_list, alpha, beta, n_workers)

    if quick_search:
        # 1. Random search rapide
        print("Phase 1: Random search (1000 samples)...")
        random_result = grid_search.run_random_sample(n_samples=1000)

        # 2. Local search depuis les meilleures
        print("Phase 2: Local search from top candidates...")
        best_result = random_result

        for config, _ in random_result.top_k[:5]:
            local_result = grid_search.run_local_search(
                initial_config=config,
                max_iterations=50
            )
            if local_result.best_result.total_cost < best_result.best_result.total_cost:
                best_result = local_result

        result = best_result
    else:
        # Grid search complet
        print("Running full grid search (32,000 configurations)...")
        result = grid_search.run()

    # Sauvegarder
    result.save(output_path)
    print(f"\nResults saved to: {output_path}")
    print(f"Best configuration:")
    print(f"  Cost: {result.best_result.total_cost:.2f}")
    print(f"  Accuracy: {result.best_result.exact_accuracy:.2%}")
    print(f"  Config: {result.best_config.to_dict()}")

    return result
