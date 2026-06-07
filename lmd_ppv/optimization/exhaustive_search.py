"""
exhaustive_search.py - Recherche exhaustive du cartouche optimal
=================================================================

Pour chaque bloc video, teste toutes les combinaisons A x B x C
(5 x 4 x 5 = 100 combinaisons) et retourne le cartouche optimal.

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ajouter le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.process_types import ProcessType, ColorMode, Representation
from src.core.features import BlockFeatures
from src.core.cartouche import Cartouche
from src.utils.math_utils import logC


@dataclass
class BlockOptimalResult:
    """Resultat de la recherche exhaustive pour un bloc."""
    block_id: int
    optimal_cartouche: Cartouche
    optimal_cost: float

    # Couts par dimension
    cost_A: Dict[int, float] = field(default_factory=dict)  # ProcessType -> cost
    cost_B: Dict[int, float] = field(default_factory=dict)  # ColorMode -> cost
    cost_C: Dict[int, float] = field(default_factory=dict)  # Representation -> cost

    # Meilleur pour chaque dimension
    best_A: int = 0
    best_B: int = 0
    best_C: int = 0

    # Features du bloc
    features: Optional[BlockFeatures] = None

    def to_dict(self) -> Dict:
        return {
            'block_id': self.block_id,
            'optimal_cartouche': self.optimal_cartouche.to_string(),
            'optimal_cost': self.optimal_cost,
            'best_A': self.best_A,
            'best_B': self.best_B,
            'best_C': self.best_C,
            'cost_A': {str(k): v for k, v in self.cost_A.items()},
            'cost_B': {str(k): v for k, v in self.cost_B.items()},
            'cost_C': {str(k): v for k, v in self.cost_C.items()},
        }


class ExhaustiveSearch:
    """
    Recherche exhaustive du cartouche optimal.

    Teste les 100 combinaisons A x B x C pour chaque bloc
    et retourne le cartouche minimisant le cout d'encodage.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialise la recherche exhaustive.

        Args:
            cache_dir: Repertoire de cache pour les resultats
        """
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache en memoire
        self._cache: Dict[int, BlockOptimalResult] = {}

    def find_optimal(self, features: BlockFeatures, block_id: int = 0) -> BlockOptimalResult:
        """
        Trouve le cartouche optimal pour un bloc.

        Args:
            features: Caracteristiques du bloc
            block_id: Identifiant du bloc

        Returns:
            BlockOptimalResult avec le cartouche optimal
        """
        # Verifier le cache
        cache_key = hash((block_id, features.N, features.r, features.m))
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Calculer les couts pour chaque dimension
        cost_A = self._compute_costs_A(features)
        cost_B = self._compute_costs_B(features)
        cost_C = self._compute_costs_C(features)

        # Trouver les minimums
        best_A = min(cost_A, key=cost_A.get)
        best_B = min(cost_B, key=cost_B.get)
        best_C = min(cost_C, key=cost_C.get)

        # Cout total optimal
        optimal_cost = cost_A[best_A] + cost_B[best_B] + cost_C[best_C]

        # Creer le cartouche optimal
        optimal_cartouche = Cartouche(A=best_A, B=best_B, C=best_C)

        result = BlockOptimalResult(
            block_id=block_id,
            optimal_cartouche=optimal_cartouche,
            optimal_cost=optimal_cost,
            cost_A=cost_A,
            cost_B=cost_B,
            cost_C=cost_C,
            best_A=best_A,
            best_B=best_B,
            best_C=best_C,
            features=features
        )

        self._cache[cache_key] = result
        return result

    def _compute_costs_A(self, features: BlockFeatures) -> Dict[int, float]:
        """
        Calcule le cout pour chaque type de processus (dim A).

        Le cout de A est principalement determine par C_color.
        """
        N = features.N
        m = features.m
        costs = {}

        log2_m = np.log2(m) if m > 1 else 0
        H = features.H_color

        # Aa - Marked: cout couleur standard
        costs[ProcessType.MARKED] = N * log2_m

        # Ab - Monochromatic: C_color = 0
        costs[ProcessType.MONOCHROMATIC] = 0.0

        # Ac - Vectorial Marginal: traitement independant
        costs[ProcessType.VECTORIAL_MARG] = N * H  # Huffman adapte

        # Ad - Vectorial Joint: correlation exploitee
        costs[ProcessType.VECTORIAL_JOINT] = N * H * 0.8  # Gain correlation

        # Ae - Markovian: transitions
        costs[ProcessType.MARKOVIAN] = features.N_trans * log2_m

        return costs

    def _compute_costs_B(self, features: BlockFeatures) -> Dict[int, float]:
        """
        Calcule le cout pour chaque mode couleur (dim B).
        """
        N = features.N
        m = features.m
        H = features.H_color
        N_trans = features.N_trans

        log2_m = np.log2(m) if m > 1 else 0
        D_huf = features.get_huffman_overhead()

        costs = {
            # Ba - Sequential: log2(m) + N_trans * log2(m)
            ColorMode.SEQUENTIAL: log2_m + N_trans * log2_m,

            # Bb - Uniform: N * log2(m)
            ColorMode.UNIFORM: N * log2_m,

            # Bc - Huffman: N * H + D_huf
            ColorMode.HUFFMAN: N * H + D_huf,

            # Bd - Elias: N * L*(m) ~ N * (log2(m) + 2*log2(log2(m)))
            ColorMode.ELIAS: N * (log2_m + 2 * np.log2(max(1, log2_m))) if log2_m > 0 else 0,
        }

        return costs

    def _compute_costs_C(self, features: BlockFeatures) -> Dict[int, float]:
        """
        Calcule le cout pour chaque representation temporelle (dim C).
        """
        N = features.N
        r = features.r

        costs = {}

        # R1 - Timestamps: N * log2(r)
        costs[Representation.TIMESTAMPS] = N * np.log2(r) if r > 0 else 0

        # R2 - Count/Histogram: r * log2(N/r + 1)
        avg_count = N / r if r > 0 else 0
        costs[Representation.COUNT] = r * np.log2(avg_count + 1) if avg_count > 0 else r

        # R3 - Intervals: N * log2(r/N) si r > N
        if N > 0 and r > 0:
            avg_interval = r / N
            costs[Representation.INTERVALS] = N * np.log2(avg_interval + 1)
        else:
            costs[Representation.INTERVALS] = float('inf')

        # R4a - Boolean: r bits
        costs[Representation.BOOLEAN] = r

        # R4b - Combinatorial: log2(N+1) + log2(C(r,N))
        costs[Representation.COMBINATORIAL] = np.log2(N + 1) + logC(r, N)

        return costs

    def find_optimal_batch(
        self,
        features_list: List[BlockFeatures],
        n_workers: int = 4
    ) -> List[BlockOptimalResult]:
        """
        Trouve les cartouches optimaux pour une liste de blocs.

        Args:
            features_list: Liste des caracteristiques de blocs
            n_workers: Nombre de workers paralleles

        Returns:
            Liste de BlockOptimalResult
        """
        results = []

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.find_optimal, f, i): i
                    for i, f in enumerate(features_list)
                }

                for future in as_completed(futures):
                    results.append(future.result())

            # Trier par block_id
            results.sort(key=lambda x: x.block_id)
        else:
            for i, features in enumerate(features_list):
                results.append(self.find_optimal(features, i))

        return results

    def save_cache(self, path: Path) -> None:
        """Sauvegarde le cache sur disque."""
        data = {
            str(k): v.to_dict()
            for k, v in self._cache.items()
        }
        path.write_text(json.dumps(data, indent=2))

    def load_cache(self, path: Path) -> None:
        """Charge le cache depuis le disque."""
        if not path.exists():
            return

        data = json.loads(path.read_text())
        # Note: La reconstruction complete necessite les features
        # Pour l'instant, on stocke juste les resultats cles


def compare_predicted_vs_optimal(
    predicted: Cartouche,
    optimal: BlockOptimalResult
) -> Dict:
    """
    Compare un cartouche predit au cartouche optimal.

    Args:
        predicted: Cartouche predit par le classifieur
        optimal: Resultat de la recherche exhaustive

    Returns:
        Dictionnaire avec les metriques de comparaison
    """
    opt = optimal.optimal_cartouche

    # Verifier l'exactitude par dimension
    match_A = predicted.A == opt.A
    match_B = predicted.B == opt.B
    match_C = predicted.C == opt.C
    exact_match = match_A and match_B and match_C

    # Calculer le cout du cartouche predit
    predicted_cost = (
        optimal.cost_A.get(predicted.A, float('inf')) +
        optimal.cost_B.get(predicted.B, float('inf')) +
        optimal.cost_C.get(predicted.C, float('inf'))
    )

    # Penalite de cout
    cost_penalty = predicted_cost - optimal.optimal_cost

    return {
        'exact_match': exact_match,
        'match_A': match_A,
        'match_B': match_B,
        'match_C': match_C,
        'predicted': predicted.to_string(),
        'optimal': opt.to_string(),
        'predicted_cost': predicted_cost,
        'optimal_cost': optimal.optimal_cost,
        'cost_penalty': cost_penalty,
        'cost_ratio': predicted_cost / optimal.optimal_cost if optimal.optimal_cost > 0 else 1.0
    }
