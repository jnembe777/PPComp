#!/usr/bin/env python3
"""
run_optimization.py - Pipeline complet d'optimisation des seuils
=================================================================

Ce script:
1. Charge les vidéos du dataset
2. Extrait les 7 features bitwise de chaque bloc
3. Calcule le cartouche optimal pour chaque bloc (100 combinaisons A×B×C)
4. Optimise les 7 seuils de classification via grid search
5. Génère un rapport avec les métriques de précision

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
from tqdm import tqdm

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

# Imports du projet
from src.core.features import BlockFeatures
from src.core.cartouche import Cartouche
from src.core.process_types import ProcessType, ColorMode, Representation
from src.agents.agent_0_extraction import ExtractionAgent, create_test_video_block
from src.agents.agent_1_classification import ClassificationAgent

from optimization.threshold_config import ThresholdConfig, ThresholdConfigGenerator, THRESHOLD_RANGES
from optimization.exhaustive_search import ExhaustiveSearch, BlockOptimalResult
from optimization.objective import ObjectiveFunction, OptimizationResult
from optimization.grid_search import GridSearch, GridSearchResult


class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON pour les types numpy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def json_dumps(obj, **kwargs):
    """json.dumps avec support numpy."""
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)


@dataclass
class VideoFeatures:
    """Features extraites d'une vidéo."""
    video_name: str
    video_path: str
    n_blocks: int
    block_features: List[BlockFeatures] = field(default_factory=list)

    # Statistiques globales
    mean_N: float = 0.0
    mean_density: float = 0.0
    mean_H_color: float = 0.0
    mean_H_s: float = 0.0
    mean_rho_corr: float = 0.0

    def compute_stats(self):
        """Calcule les statistiques globales."""
        if not self.block_features:
            return

        self.mean_N = np.mean([f.N for f in self.block_features])
        self.mean_density = np.mean([f.density for f in self.block_features])
        self.mean_H_color = np.mean([f.H_color for f in self.block_features])
        self.mean_H_s = np.mean([f.H_s for f in self.block_features])
        self.mean_rho_corr = np.mean([f.rho_corr for f in self.block_features])

    def to_dict(self) -> Dict:
        return {
            'video_name': self.video_name,
            'video_path': self.video_path,
            'n_blocks': self.n_blocks,
            'mean_N': self.mean_N,
            'mean_density': self.mean_density,
            'mean_H_color': self.mean_H_color,
            'mean_H_s': self.mean_H_s,
            'mean_rho_corr': self.mean_rho_corr,
        }


def load_video_as_blocks(
    video_path: Path,
    block_size: int = 16,
    n_frames: int = 64,
    n_colors: int = 64,
    max_blocks: Optional[int] = None
) -> List[np.ndarray]:
    """
    Charge une vidéo et la découpe en blocs.

    Args:
        video_path: Chemin vers la vidéo
        block_size: Taille des blocs (16x16 par défaut)
        n_frames: Nombre de frames à charger
        n_colors: Nombre de couleurs pour la quantification
        max_blocks: Nombre max de blocs à extraire

    Returns:
        Liste de blocs vidéo (T, H, W) avec indices de couleur
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV non installé. Installation: pip install opencv-python")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Erreur: impossible d'ouvrir {video_path}")
        return []

    # Lire les frames
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Convertir en niveaux de gris et quantifier
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quantized = (gray * (n_colors - 1) / 255).astype(np.int32)
        frames.append(quantized)

    cap.release()

    if len(frames) < 2:
        print(f"Pas assez de frames dans {video_path}")
        return []

    # Stack en array (T, H, W)
    video = np.stack(frames, axis=0)
    T, H, W = video.shape

    # Découper en blocs
    blocks = []
    n_blocks_h = H // block_size
    n_blocks_w = W // block_size

    for i in range(n_blocks_h):
        for j in range(n_blocks_w):
            y_start = i * block_size
            x_start = j * block_size
            block = video[:, y_start:y_start+block_size, x_start:x_start+block_size]
            blocks.append(block)

            if max_blocks and len(blocks) >= max_blocks:
                return blocks

    return blocks


def extract_features_from_video(
    video_path: Path,
    block_size: int = 16,
    n_frames: int = 64,
    n_colors: int = 64,
    max_blocks: int = 100
) -> VideoFeatures:
    """
    Extrait les features de tous les blocs d'une vidéo.

    Args:
        video_path: Chemin vers la vidéo
        block_size: Taille des blocs
        n_frames: Nombre de frames
        n_colors: Nombre de couleurs
        max_blocks: Nombre max de blocs

    Returns:
        VideoFeatures avec les caractéristiques de chaque bloc
    """
    blocks = load_video_as_blocks(
        video_path, block_size, n_frames, n_colors, max_blocks
    )

    if not blocks:
        return VideoFeatures(
            video_name=video_path.name,
            video_path=str(video_path),
            n_blocks=0
        )

    # Extraire les features de chaque bloc
    agent = ExtractionAgent(block_size, block_size)
    features_list = []

    for block in blocks:
        result = agent.extract(block)
        features_list.append(result.features)

    video_features = VideoFeatures(
        video_name=video_path.name,
        video_path=str(video_path),
        n_blocks=len(features_list),
        block_features=features_list
    )
    video_features.compute_stats()

    return video_features


def extract_features_from_dataset(
    dataset_dir: Path,
    block_size: int = 16,
    n_frames: int = 64,
    n_colors: int = 64,
    max_blocks_per_video: int = 50,
    max_videos: Optional[int] = None
) -> Tuple[List[BlockFeatures], List[VideoFeatures]]:
    """
    Extrait les features de toutes les vidéos d'un dataset.

    Args:
        dataset_dir: Répertoire du dataset
        block_size: Taille des blocs
        n_frames: Nombre de frames par vidéo
        n_colors: Nombre de couleurs
        max_blocks_per_video: Max blocs par vidéo
        max_videos: Max vidéos à traiter

    Returns:
        (liste_features_blocs, liste_features_videos)
    """
    # Trouver les vidéos
    video_extensions = ['*.mp4', '*.y4m', '*.avi', '*.mkv', '*.mov']
    video_paths = []

    for ext in video_extensions:
        video_paths.extend(dataset_dir.glob(f'**/{ext}'))

    if max_videos:
        video_paths = video_paths[:max_videos]

    print(f"Trouvé {len(video_paths)} vidéos dans {dataset_dir}")

    all_features = []
    video_features_list = []

    for video_path in tqdm(video_paths, desc="Extraction des features"):
        video_features = extract_features_from_video(
            video_path,
            block_size=block_size,
            n_frames=n_frames,
            n_colors=n_colors,
            max_blocks=max_blocks_per_video
        )

        if video_features.n_blocks > 0:
            all_features.extend(video_features.block_features)
            video_features_list.append(video_features)

    print(f"Total: {len(all_features)} blocs extraits de {len(video_features_list)} vidéos")

    return all_features, video_features_list


def run_optimization(
    features_list: List[BlockFeatures],
    output_dir: Path,
    quick_mode: bool = True,
    n_workers: int = 4
) -> GridSearchResult:
    """
    Exécute l'optimisation des seuils.

    Args:
        features_list: Liste des features de blocs
        output_dir: Répertoire de sortie
        quick_mode: Utiliser la recherche rapide
        n_workers: Nombre de workers

    Returns:
        GridSearchResult
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Optimisation des seuils ===")
    print(f"Nombre de blocs: {len(features_list)}")
    print(f"Mode: {'Rapide (Random + Local)' if quick_mode else 'Complet (32,000 configs)'}")

    # Créer le grid search
    grid_search = GridSearch(
        features_list=features_list,
        alpha=1.0,
        beta=0.5,
        n_workers=n_workers
    )

    start_time = time.time()

    if quick_mode:
        # Phase 1: Random search
        print("\nPhase 1: Random search (1000 échantillons)...")
        random_result = grid_search.run_random_sample(n_samples=1000)
        print(f"  Meilleur coût: {random_result.best_result.total_cost:.2f}")
        print(f"  Précision exacte: {random_result.best_result.exact_accuracy:.2%}")

        # Phase 2: Local search
        print("\nPhase 2: Recherche locale depuis les 5 meilleurs...")
        best_result = random_result

        for i, (config, _) in enumerate(random_result.top_k[:5]):
            local_result = grid_search.run_local_search(
                initial_config=config,
                max_iterations=50
            )
            print(f"  Config {i+1}: coût = {local_result.best_result.total_cost:.2f}")

            if local_result.best_result.total_cost < best_result.best_result.total_cost:
                best_result = local_result

        result = best_result
    else:
        # Grid search complet
        print("\nRecherche exhaustive (32,000 configurations)...")
        result = grid_search.run(save_all=False)

    total_time = time.time() - start_time

    # Sauvegarder les résultats
    result.save(output_dir / 'optimization_result.json')

    # Sauvegarder la configuration optimale séparément
    result.best_config.save(output_dir / 'optimal_thresholds.json')

    print(f"\n=== Résultats de l'optimisation ===")
    print(f"Temps total: {total_time:.1f}s")
    print(f"Configurations évaluées: {result.n_evaluated}")
    print(f"Évaluations/sec: {result.evaluations_per_sec:.1f}")
    print(f"\nMeilleure configuration:")
    print(f"  Coût total: {result.best_result.total_cost:.2f}")
    print(f"  Coût encodage: {result.best_result.encoding_cost:.2f}")
    print(f"  Pénalité: {result.best_result.penalty_cost:.2f}")
    print(f"\nPrécision de classification:")
    print(f"  Dimension A (type): {result.best_result.accuracy_A:.2%}")
    print(f"  Dimension B (couleur): {result.best_result.accuracy_B:.2%}")
    print(f"  Dimension C (représentation): {result.best_result.accuracy_C:.2%}")
    print(f"  Exacte (A+B+C): {result.best_result.exact_accuracy:.2%}")
    print(f"\nSeuils optimaux:")
    for key, value in result.best_config.to_dict().items():
        print(f"  {key}: {value:.4f}")

    return result


def validate_thresholds(
    features_list: List[BlockFeatures],
    config: ThresholdConfig,
    output_dir: Path
) -> Dict:
    """
    Valide les seuils sur un ensemble de blocs.

    Args:
        features_list: Liste des features
        config: Configuration des seuils
        output_dir: Répertoire de sortie

    Returns:
        Dictionnaire avec les métriques de validation
    """
    print(f"\n=== Validation des seuils ===")

    # Recherche exhaustive pour les optimaux
    exhaustive = ExhaustiveSearch()

    # Agent de classification avec les seuils
    agent = ClassificationAgent.with_thresholds(**config.to_dict())

    # Comparer prédit vs optimal
    results = {
        'match_A': 0,
        'match_B': 0,
        'match_C': 0,
        'exact_match': 0,
        'total_predicted_cost': 0.0,
        'total_optimal_cost': 0.0,
        'cost_penalties': [],
        'confusion_A': {},
        'confusion_B': {},
        'confusion_C': {},
    }

    for i, features in enumerate(tqdm(features_list, desc="Validation")):
        # Optimal
        optimal = exhaustive.find_optimal(features, i)

        # Prédit
        classification = agent.classify(features)
        predicted_A = classification.process_type
        predicted_B = features.suggest_color_mode()

        # Dimension C (basé sur N, pas sur density)
        # Règles optimales dérivées de l'analyse exhaustive:
        # - N = 0: TIMESTAMPS (bloc vide)
        # - N < 50: COUNT (peu d'événements)
        # - N >= 50: COMBINATORIAL (beaucoup d'événements)
        threshold_N = 50
        if features.N == 0:
            predicted_C = 0  # R1 - Timestamps
        elif features.N < threshold_N:
            predicted_C = 1  # R2 - Count
        else:
            predicted_C = 4  # R4b - Combinatorial

        predicted = Cartouche(A=predicted_A, B=predicted_B, C=predicted_C)

        # Comparaison
        opt = optimal.optimal_cartouche
        if predicted.A == opt.A:
            results['match_A'] += 1
        if predicted.B == opt.B:
            results['match_B'] += 1
        if predicted.C == opt.C:
            results['match_C'] += 1
        if predicted.A == opt.A and predicted.B == opt.B and predicted.C == opt.C:
            results['exact_match'] += 1

        # Coûts
        predicted_cost = (
            optimal.cost_A.get(predicted.A, float('inf')) +
            optimal.cost_B.get(predicted.B, float('inf')) +
            optimal.cost_C.get(predicted.C, float('inf'))
        )
        results['total_predicted_cost'] += predicted_cost
        results['total_optimal_cost'] += optimal.optimal_cost
        results['cost_penalties'].append(max(0, predicted_cost - optimal.optimal_cost))

    n = len(features_list)

    validation_report = {
        'n_blocks': n,
        'accuracy_A': results['match_A'] / n,
        'accuracy_B': results['match_B'] / n,
        'accuracy_C': results['match_C'] / n,
        'exact_accuracy': results['exact_match'] / n,
        'total_predicted_cost': results['total_predicted_cost'],
        'total_optimal_cost': results['total_optimal_cost'],
        'overhead_ratio': results['total_predicted_cost'] / results['total_optimal_cost'] if results['total_optimal_cost'] > 0 else 1.0,
        'mean_penalty': np.mean(results['cost_penalties']),
        'max_penalty': np.max(results['cost_penalties']),
        'thresholds': config.to_dict()
    }

    # Sauvegarder
    output_path = output_dir / 'validation_report.json'
    output_path.write_text(json_dumps(validation_report, indent=2))

    print(f"\nRapport de validation:")
    print(f"  Précision A: {validation_report['accuracy_A']:.2%}")
    print(f"  Précision B: {validation_report['accuracy_B']:.2%}")
    print(f"  Précision C: {validation_report['accuracy_C']:.2%}")
    print(f"  Précision exacte: {validation_report['exact_accuracy']:.2%}")
    print(f"  Overhead ratio: {validation_report['overhead_ratio']:.4f}")
    print(f"  Pénalité moyenne: {validation_report['mean_penalty']:.2f} bits")
    print(f"\nRapport sauvegardé: {output_path}")

    return validation_report


def main():
    parser = argparse.ArgumentParser(
        description="Optimisation des seuils de classification LMD-PPV"
    )
    parser.add_argument(
        '--dataset', '-d',
        type=Path,
        default=Path('datasets'),
        help="Répertoire du dataset vidéo"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('optimization_output'),
        help="Répertoire de sortie"
    )
    parser.add_argument(
        '--block-size', '-b',
        type=int,
        default=16,
        help="Taille des blocs (défaut: 16)"
    )
    parser.add_argument(
        '--n-frames', '-f',
        type=int,
        default=64,
        help="Nombre de frames par vidéo (défaut: 64)"
    )
    parser.add_argument(
        '--n-colors', '-c',
        type=int,
        default=64,
        help="Nombre de couleurs (défaut: 64)"
    )
    parser.add_argument(
        '--max-blocks',
        type=int,
        default=50,
        help="Max blocs par vidéo (défaut: 50)"
    )
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help="Max vidéos à traiter"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        default=True,
        help="Mode rapide (défaut: True)"
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help="Mode complet (32,000 configs)"
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help="Nombre de workers (défaut: 4)"
    )
    parser.add_argument(
        '--validate-only',
        type=Path,
        default=None,
        help="Valider avec des seuils existants"
    )

    args = parser.parse_args()

    # Mode complet override quick
    quick_mode = not args.full

    print("=" * 60)
    print("LMD-PPV - Optimisation des Seuils de Classification")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Block size: {args.block_size}x{args.block_size}")
    print(f"Frames: {args.n_frames}")
    print(f"Colors: {args.n_colors}")
    print(f"Mode: {'Rapide' if quick_mode else 'Complet'}")
    print("=" * 60)

    args.output.mkdir(parents=True, exist_ok=True)

    # Étape 1: Extraction des features
    print("\n[1/3] Extraction des features...")
    features_list, video_features = extract_features_from_dataset(
        args.dataset,
        block_size=args.block_size,
        n_frames=args.n_frames,
        n_colors=args.n_colors,
        max_blocks_per_video=args.max_blocks,
        max_videos=args.max_videos
    )

    if not features_list:
        print("Aucun bloc extrait. Vérifiez le dataset.")
        return 1

    # Sauvegarder les stats des vidéos
    video_stats = [vf.to_dict() for vf in video_features]
    (args.output / 'video_stats.json').write_text(json_dumps(video_stats, indent=2))

    # Sauvegarder les features (pour réutilisation)
    features_data = [f.to_dict() for f in features_list]
    (args.output / 'features.json').write_text(json_dumps(features_data, indent=2))
    print(f"Features sauvegardées: {args.output / 'features.json'}")

    # Validation uniquement?
    if args.validate_only:
        print(f"\n[Validation avec seuils: {args.validate_only}]")
        config = ThresholdConfig.load(args.validate_only)
        validate_thresholds(features_list, config, args.output)
        return 0

    # Étape 2: Optimisation
    print("\n[2/3] Optimisation des seuils...")
    result = run_optimization(
        features_list,
        args.output,
        quick_mode=quick_mode,
        n_workers=args.workers
    )

    # Étape 3: Validation
    print("\n[3/3] Validation finale...")
    validate_thresholds(features_list, result.best_config, args.output)

    print("\n" + "=" * 60)
    print("OPTIMISATION TERMINÉE")
    print("=" * 60)
    print(f"Fichiers générés dans: {args.output}")
    print(f"  - optimization_result.json  (résultats complets)")
    print(f"  - optimal_thresholds.json   (seuils optimaux)")
    print(f"  - validation_report.json    (rapport de validation)")
    print(f"  - features.json             (features extraites)")
    print(f"  - video_stats.json          (statistiques vidéos)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
