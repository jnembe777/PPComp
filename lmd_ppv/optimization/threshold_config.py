"""
threshold_config.py - Configuration des 7 seuils de classification
===================================================================

Les 7 seuils a optimiser:
1. threshold_H_s: Homogeneite spatiale -> Vectorial Joint
2. threshold_rho_high: Correlation haute -> Joint
3. threshold_rho_low: Correlation basse -> Marginal
4. threshold_chi2: p-value Markov
5. density_R1_max: Seuil R1 (sparse)
6. density_R4b_R2: Seuil R2 (dense)
7. density_R4a_min: Zone R4a (boolean)

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterator, Optional
import json
from pathlib import Path
import numpy as np


@dataclass
class ThresholdRange:
    """Plage de valeurs pour un seuil."""
    name: str
    min_val: float
    max_val: float
    steps: int
    default: float
    description: str = ""

    @property
    def values(self) -> List[float]:
        """Genere les valeurs possibles."""
        return list(np.linspace(self.min_val, self.max_val, self.steps))

    def __iter__(self):
        return iter(self.values)


# Definition des 7 seuils avec leurs plages
THRESHOLD_RANGES = {
    'threshold_H_s': ThresholdRange(
        name='threshold_H_s',
        min_val=0.5,
        max_val=0.9,
        steps=5,
        default=0.7,
        description="Homogeneite spatiale -> Vectorial Joint"
    ),
    'threshold_rho_high': ThresholdRange(
        name='threshold_rho_high',
        min_val=0.6,
        max_val=0.95,
        steps=5,
        default=0.8,
        description="Correlation haute -> Joint"
    ),
    'threshold_rho_low': ThresholdRange(
        name='threshold_rho_low',
        min_val=0.05,
        max_val=0.30,
        steps=5,
        default=0.15,
        description="Correlation basse -> Marginal"
    ),
    'threshold_chi2': ThresholdRange(
        name='threshold_chi2',
        min_val=0.01,
        max_val=0.10,
        steps=4,
        default=0.05,
        description="p-value chi-carre pour test Markov"
    ),
    'density_R1_max': ThresholdRange(
        name='density_R1_max',
        min_val=0.05,
        max_val=0.20,
        steps=4,
        default=0.10,
        description="Seuil de densite max pour R1 (sparse)"
    ),
    'density_R4b_R2': ThresholdRange(
        name='density_R4b_R2',
        min_val=0.70,
        max_val=0.90,
        steps=4,
        default=0.80,
        description="Seuil de densite pour R2 (dense)"
    ),
    'density_R4a_min': ThresholdRange(
        name='density_R4a_min',
        min_val=0.40,
        max_val=0.65,
        steps=4,
        default=0.50,
        description="Zone de densite min pour R4a (boolean)"
    ),
}


@dataclass
class ThresholdConfig:
    """
    Configuration complete des 7 seuils de classification.

    Utilisee par ClassificationAgent et pour le grid search.
    """

    # Seuils pour la dimension A (type de processus)
    threshold_H_s: float = 0.7       # Homogeneite spatiale
    threshold_rho_high: float = 0.8  # Correlation haute
    threshold_rho_low: float = 0.15  # Correlation basse
    threshold_chi2: float = 0.05     # p-value Markov

    # Seuils pour la dimension C (representation)
    density_R1_max: float = 0.10     # Sparse -> R1
    density_R4b_R2: float = 0.80     # Dense -> R2
    density_R4a_min: float = 0.50    # Boolean -> R4a

    def to_dict(self) -> Dict[str, float]:
        """Conversion en dictionnaire."""
        return {
            'threshold_H_s': self.threshold_H_s,
            'threshold_rho_high': self.threshold_rho_high,
            'threshold_rho_low': self.threshold_rho_low,
            'threshold_chi2': self.threshold_chi2,
            'density_R1_max': self.density_R1_max,
            'density_R4b_R2': self.density_R4b_R2,
            'density_R4a_min': self.density_R4a_min,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'ThresholdConfig':
        """Creation depuis un dictionnaire."""
        return cls(
            threshold_H_s=d.get('threshold_H_s', 0.7),
            threshold_rho_high=d.get('threshold_rho_high', 0.8),
            threshold_rho_low=d.get('threshold_rho_low', 0.15),
            threshold_chi2=d.get('threshold_chi2', 0.05),
            density_R1_max=d.get('density_R1_max', 0.10),
            density_R4b_R2=d.get('density_R4b_R2', 0.80),
            density_R4a_min=d.get('density_R4a_min', 0.50),
        )

    @classmethod
    def default(cls) -> 'ThresholdConfig':
        """Configuration par defaut."""
        return cls()

    def to_tuple(self) -> Tuple[float, ...]:
        """Conversion en tuple pour hashage."""
        return (
            self.threshold_H_s,
            self.threshold_rho_high,
            self.threshold_rho_low,
            self.threshold_chi2,
            self.density_R1_max,
            self.density_R4b_R2,
            self.density_R4a_min,
        )

    @classmethod
    def from_tuple(cls, t: Tuple[float, ...]) -> 'ThresholdConfig':
        """Creation depuis un tuple."""
        return cls(
            threshold_H_s=t[0],
            threshold_rho_high=t[1],
            threshold_rho_low=t[2],
            threshold_chi2=t[3],
            density_R1_max=t[4],
            density_R4b_R2=t[5],
            density_R4a_min=t[6],
        )

    def __hash__(self):
        return hash(self.to_tuple())

    def save(self, path: Path) -> None:
        """Sauvegarde en JSON."""
        data = self.to_dict()
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> 'ThresholdConfig':
        """Charge depuis un JSON."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    def distance(self, other: 'ThresholdConfig') -> float:
        """Distance euclidienne vers une autre configuration."""
        t1 = np.array(self.to_tuple())
        t2 = np.array(other.to_tuple())
        return np.linalg.norm(t1 - t2)


class ThresholdConfigGenerator:
    """
    Generateur de configurations de seuils pour grid search.

    Genere les 32,000 combinaisons possibles:
    5 x 5 x 5 x 4 x 4 x 4 x 4 = 32,000
    """

    def __init__(self, ranges: Optional[Dict[str, ThresholdRange]] = None):
        """
        Initialise le generateur.

        Args:
            ranges: Plages de seuils (defaut: THRESHOLD_RANGES)
        """
        self.ranges = ranges or THRESHOLD_RANGES

    @property
    def total_combinations(self) -> int:
        """Nombre total de combinaisons."""
        total = 1
        for r in self.ranges.values():
            total *= r.steps
        return total

    def generate_all(self) -> Iterator[ThresholdConfig]:
        """
        Genere toutes les configurations possibles.

        Yields:
            ThresholdConfig pour chaque combinaison
        """
        # Generer les valeurs pour chaque seuil
        values = {
            name: r.values
            for name, r in self.ranges.items()
        }

        # Produit cartesien
        import itertools
        keys = list(values.keys())
        for combo in itertools.product(*[values[k] for k in keys]):
            yield ThresholdConfig(**dict(zip(keys, combo)))

    def generate_sample(self, n: int, seed: int = 42) -> List[ThresholdConfig]:
        """
        Genere un echantillon aleatoire de configurations.

        Args:
            n: Nombre de configurations
            seed: Graine aleatoire

        Returns:
            Liste de ThresholdConfig
        """
        np.random.seed(seed)

        configs = []
        for _ in range(n):
            config_dict = {}
            for name, r in self.ranges.items():
                config_dict[name] = np.random.choice(r.values)
            configs.append(ThresholdConfig.from_dict(config_dict))

        return configs

    def generate_neighbors(
        self,
        config: ThresholdConfig,
        step: int = 1
    ) -> List[ThresholdConfig]:
        """
        Genere les configurations voisines.

        Args:
            config: Configuration centrale
            step: Nombre de pas dans chaque direction

        Returns:
            Liste des configurations voisines
        """
        neighbors = []
        base_dict = config.to_dict()

        for name, r in self.ranges.items():
            values = r.values
            current = base_dict[name]

            # Trouver l'index actuel
            try:
                idx = values.index(current)
            except ValueError:
                idx = np.argmin([abs(v - current) for v in values])

            # Generer les voisins
            for delta in [-step, step]:
                new_idx = idx + delta
                if 0 <= new_idx < len(values):
                    new_dict = base_dict.copy()
                    new_dict[name] = values[new_idx]
                    neighbors.append(ThresholdConfig.from_dict(new_dict))

        return neighbors
