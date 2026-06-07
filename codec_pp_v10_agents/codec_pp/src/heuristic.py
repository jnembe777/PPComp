"""
Heuristic M6 — Classifieur rapide pour prédiction du Code Méthode.

Au lieu de calculer les 4 longueurs de description MDL pour chaque bloc
(ce qui nécessite de construire les matrices et d'estimer chaque type),
le classifieur prédit directement le meilleur (proc_type, best_rep,
pred_mode) depuis des features statistiques légères.

Features par bloc (calculées en O(block_size² × r)) :
  f1 : n_jumps_mean       — nombre moyen de sauts par pixel
  f2 : n_jumps_std        — écart-type du nombre de sauts
  f3 : n_distinct_colors  — nombre de couleurs distinctes dans le bloc
  f4 : spatial_entropy    — entropie spatiale (frame moyenne)
  f5 : temporal_entropy   — entropie temporelle (séquence moyenne)
  f6 : jump_pattern_dominance — fraction de pixels partageant le pattern dominant
  f7 : n_distinct_transitions — nombre de transitions (from, to) distinctes
  f8 : change_rate        — taux moyen de changement par frame
  f9 : color_concentration — ratio max_freq / total pour les couleurs
  f10: block_variance     — variance des valeurs dans le bloc

Label : code_method = (proc_type * 4 + best_rep) encodé en entier
        + pred_mode (0-3)

Pipeline :
  1. generate_training_data() — exécute le MDL exhaustif sur des vidéos
  2. train_decision_tree()    — entraîne un arbre de décision scikit-free
  3. DecisionHeuristic.predict_method() — prédiction en production
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from math import ceil, log2, log
from collections import Counter

from .matrices import build_all_matrices, compute_representation_lengths
from .process_types import _analyze_block_pixels, classify_block
from .prediction import choose_best_mode, PRED_BYPASS, NUM_PRED_MODES
from .video_io import generate_synthetic_video
from .benchmark import (
    gen_static_bg, gen_gradient, gen_animation,
    gen_noise, gen_periodic,
)

# ═══════════════════════════════════════════════════════════════
#  EXTRACTION DE FEATURES
# ═══════════════════════════════════════════════════════════════

def extract_block_features(
    plane: np.ndarray,
    i0: int, i1: int, j0: int, j1: int,
    gop_r: int,
) -> np.ndarray:
    """
    Extrait 10 features statistiques d'un bloc (version vectorisée).

    Returns:
        features: ndarray shape (10,)
    """
    block = plane[:, i0:i1, j0:j1]  # (gop_r, bh, bw)
    bh = i1 - i0
    bw = j1 - j0
    n_pixels = bh * bw

    # Matrice de changements : diff[t, i, j] = 1 si block[t] != block[t-1]
    changes = (block[1:] != block[:-1]).astype(np.int32)  # (gop_r-1, bh, bw)

    # f1, f2 : nombre de sauts par pixel (1 + nb de changements)
    n_jumps_per_pixel = 1 + np.sum(changes, axis=0).flatten().astype(float)
    f1 = float(np.mean(n_jumps_per_pixel))
    f2 = float(np.std(n_jumps_per_pixel))

    # f3 : couleurs distinctes
    f3 = float(len(np.unique(block)))

    # f4 : variance spatiale de la frame moyenne (proxy d'entropie spatiale)
    frame_mean = np.mean(block.astype(float), axis=0)
    f4 = float(np.std(frame_mean))

    # f5 : variance temporelle globale (proxy d'entropie temporelle)
    f5 = float(np.std(block.astype(float)))

    # f6 : dominance du pattern de sauts
    # Convertir chaque séquence de changements en tuple hashable
    changes_flat = changes.reshape(gop_r - 1, -1).T  # (n_pixels, gop_r-1)
    # Hash rapide : convertir en bytes
    pattern_hashes = [row.tobytes() for row in changes_flat]
    pattern_counts = Counter(pattern_hashes)
    dom_count = pattern_counts.most_common(1)[0][1]
    f6 = float(dom_count) / n_pixels

    # f7 : transitions distinctes
    # Paires (from, to) uniques sur tout le bloc
    froms = block[:-1][changes == 1].flatten()
    tos = block[1:][changes == 1].flatten()
    if len(froms) > 0:
        pairs = set(zip(froms.tolist(), tos.tolist()))
        f7 = float(len(pairs))
    else:
        f7 = 0.0

    # f8 : taux de changement
    total_changes = int(np.sum(changes))
    total_possible = n_pixels * (gop_r - 1)
    f8 = float(total_changes) / max(total_possible, 1)

    # f9 : concentration des couleurs
    flat = block.flatten()
    vals, counts = np.unique(flat, return_counts=True)
    f9 = float(np.max(counts)) / len(flat)

    # f10 : variance du bloc
    f10 = float(np.var(block.astype(float)))

    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])


FEATURE_NAMES = [
    "n_jumps_mean", "n_jumps_std", "n_distinct_colors",
    "spatial_entropy", "temporal_entropy", "jump_pattern_dominance",
    "n_distinct_transitions", "change_rate", "color_concentration",
    "block_variance",
]


def _entropy(values) -> float:
    """Entropie de Shannon."""
    n = len(values)
    if n == 0:
        return 0.0
    counts = Counter(values)
    ent = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            ent -= p * log(p, 2)
    return ent


# ═══════════════════════════════════════════════════════════════
#  GÉNÉRATION DU DATASET D'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════

def generate_training_data(
    block_size: int = 8,
    color_bits: int = 8,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère un dataset d'entraînement en exécutant le MDL exhaustif
    sur des vidéos synthétiques variées.

    Returns:
        X: features, shape (n_samples, 10)
        y_proc: labels proc_type, shape (n_samples,)
        y_pred: labels pred_mode, shape (n_samples,)
    """
    generators = [
        # (generator, kwargs, repetitions)
        (gen_static_bg,  dict(nl=64, nc=64, r=32), 3),
        (gen_gradient,   dict(nl=64, nc=64, r=32), 3),
        (gen_animation,  dict(nl=64, nc=64, r=32), 3),
        (gen_noise,      dict(nl=64, nc=64, r=32), 3),
        (gen_periodic,   dict(nl=64, nc=64, r=32), 3),
        # Variations de paramètres
        (gen_static_bg,  dict(nl=32, nc=32, r=16), 2),
        (gen_gradient,   dict(nl=32, nc=32, r=64), 2),
        (gen_animation,  dict(nl=48, nc=48, r=48), 2),
    ]

    # Aussi des vidéos aléatoires avec différentes probabilités
    random_configs = [
        dict(nl=64, nc=64, r=32, n_colors=2,  change_prob=0.05),
        dict(nl=64, nc=64, r=32, n_colors=4,  change_prob=0.10),
        dict(nl=64, nc=64, r=32, n_colors=8,  change_prob=0.15),
        dict(nl=64, nc=64, r=32, n_colors=16, change_prob=0.20),
        dict(nl=64, nc=64, r=32, n_colors=4,  change_prob=0.50),
        dict(nl=64, nc=64, r=32, n_colors=2,  change_prob=0.01),
    ]

    X_list = []
    y_proc_list = []
    y_pred_list = []

    sample_count = 0

    if verbose:
        print("  Génération du dataset d'entraînement...")

    for gen_fn, kwargs, reps in generators:
        for rep in range(reps):
            kw = dict(kwargs, seed=42 + rep * 7)
            M, name, _ = gen_fn(**kw)
            gop_r = M.shape[0]
            nl, nc = M.shape[1], M.shape[2]

            _collect_labels(
                M, gop_r, nl, nc, block_size, color_bits,
                X_list, y_proc_list, y_pred_list,
            )
            sample_count += ceil(nl / block_size) * ceil(nc / block_size)

    for cfg in random_configs:
        M = generate_synthetic_video(seed=42 + sample_count, **cfg)
        gop_r = M.shape[0]
        nl, nc = M.shape[1], M.shape[2]

        _collect_labels(
            M, gop_r, nl, nc, block_size, color_bits,
            X_list, y_proc_list, y_pred_list,
        )
        sample_count += ceil(nl / block_size) * ceil(nc / block_size)

    X = np.array(X_list)
    y_proc = np.array(y_proc_list)
    y_pred = np.array(y_pred_list)

    if verbose:
        print(f"  Dataset : {len(X)} échantillons, "
              f"{len(FEATURE_NAMES)} features")
        for proc in range(4):
            cnt = np.sum(y_proc == proc)
            pct = cnt / len(y_proc) * 100
            print(f"    PROC={proc}: {cnt} ({pct:.1f}%)")
        for pred in range(NUM_PRED_MODES):
            cnt = np.sum(y_pred == pred)
            pct = cnt / len(y_pred) * 100
            print(f"    PRED={pred}: {cnt} ({pct:.1f}%)")

    return X, y_proc, y_pred


def _collect_labels(M, gop_r, nl, nc, block_size, color_bits,
                    X_list, y_proc_list, y_pred_list):
    """Collecte features + labels MDL pour une vidéo."""
    gop_duration_bits = max(1, ceil(log2(max(gop_r, 2))))
    data = build_all_matrices(M)
    nb_bi = ceil(nl / block_size)
    nb_bj = ceil(nc / block_size)

    for bi in range(nb_bi):
        for bj in range(nb_bj):
            i0 = bi * block_size
            i1 = min(i0 + block_size, nl)
            j0 = bj * block_size
            j1 = min(j0 + block_size, nc)

            # Features
            feats = extract_block_features(M, i0, i1, j0, j1, gop_r)

            # Label proc_type (MDL exhaustif)
            pixels = _analyze_block_pixels(M, data, gop_r, i0, i1, j0, j1)
            proc_type, _, _ = classify_block(
                pixels, gop_r, color_bits, gop_duration_bits
            )

            # Label pred_mode
            pred_mode, _ = choose_best_mode(M, i0, i1, j0, j1)

            X_list.append(feats)
            y_proc_list.append(proc_type)
            y_pred_list.append(pred_mode)


# ═══════════════════════════════════════════════════════════════
#  ARBRE DE DÉCISION (SANS SCIKIT-LEARN)
# ═══════════════════════════════════════════════════════════════

class DecisionNode:
    """Nœud d'un arbre de décision binaire."""
    __slots__ = ('feature_idx', 'threshold', 'left', 'right', 'label')

    def __init__(self, feature_idx=None, threshold=None,
                 left=None, right=None, label=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def predict(self, x: np.ndarray) -> int:
        if self.label is not None:
            return self.label
        if x[self.feature_idx] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def to_dict(self) -> dict:
        if self.label is not None:
            return {'label': int(self.label)}
        return {
            'f': int(self.feature_idx),
            't': float(self.threshold),
            'l': self.left.to_dict(),
            'r': self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'DecisionNode':
        if 'label' in d:
            return cls(label=d['label'])
        return cls(
            feature_idx=d['f'],
            threshold=d['t'],
            left=cls.from_dict(d['l']),
            right=cls.from_dict(d['r']),
        )


def _gini(labels: np.ndarray) -> float:
    """Indice de Gini."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels.tolist())
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _best_split(X: np.ndarray, y: np.ndarray, n_features_sample: int = None):
    """Trouve le meilleur split (feature, threshold)."""
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None, None, None

    best_gini = float('inf')
    best_feat = None
    best_thresh = None
    best_mask = None

    # Sous-échantillonner les features (√n)
    if n_features_sample is None:
        n_features_sample = max(1, int(np.sqrt(n_features)))
    feat_indices = np.random.choice(
        n_features, size=min(n_features_sample, n_features), replace=False
    )

    for fi in feat_indices:
        vals = X[:, fi]
        # Essayer quelques seuils (quantiles)
        thresholds = np.unique(np.percentile(vals, [10, 25, 40, 50, 60, 75, 90]))

        for thresh in thresholds:
            mask = vals <= thresh
            n_left = np.sum(mask)
            n_right = n_samples - n_left
            if n_left == 0 or n_right == 0:
                continue

            gini = (n_left * _gini(y[mask]) +
                    n_right * _gini(y[~mask])) / n_samples

            if gini < best_gini:
                best_gini = gini
                best_feat = fi
                best_thresh = thresh
                best_mask = mask

    return best_feat, best_thresh, best_mask


def build_decision_tree(
    X: np.ndarray, y: np.ndarray,
    max_depth: int = 10,
    min_samples_leaf: int = 5,
    depth: int = 0,
) -> DecisionNode:
    """Construit un arbre de décision par partitionnement récursif."""
    n = len(y)

    # Critère d'arrêt
    if n <= min_samples_leaf or depth >= max_depth or len(np.unique(y)) == 1:
        label = Counter(y.tolist()).most_common(1)[0][0]
        return DecisionNode(label=label)

    feat, thresh, mask = _best_split(X, y)

    if feat is None:
        label = Counter(y.tolist()).most_common(1)[0][0]
        return DecisionNode(label=label)

    left = build_decision_tree(
        X[mask], y[mask], max_depth, min_samples_leaf, depth + 1
    )
    right = build_decision_tree(
        X[~mask], y[~mask], max_depth, min_samples_leaf, depth + 1
    )

    return DecisionNode(feature_idx=feat, threshold=thresh,
                        left=left, right=right)


# ═══════════════════════════════════════════════════════════════
#  CLASSIFIEUR HEURISTIQUE
# ═══════════════════════════════════════════════════════════════

class DecisionHeuristic:
    """
    Heuristique M6 — prédit le Code Méthode depuis les features.

    Usage :
        heuristic = DecisionHeuristic.train(block_size=8)
        proc, pred = heuristic.predict_method(features)
    """

    def __init__(self):
        self.tree_proc: Optional[DecisionNode] = None
        self.tree_pred: Optional[DecisionNode] = None
        self.accuracy_proc: float = 0.0
        self.accuracy_pred: float = 0.0

    @classmethod
    def train(
        cls,
        block_size: int = 8,
        color_bits: int = 8,
        max_depth: int = 8,
        verbose: bool = True,
    ) -> 'DecisionHeuristic':
        """Entraîne le classifieur sur des vidéos synthétiques."""
        h = cls()

        X, y_proc, y_pred = generate_training_data(
            block_size=block_size,
            color_bits=color_bits,
            verbose=verbose,
        )

        # Split train/test (80/20)
        n = len(X)
        indices = np.random.RandomState(42).permutation(n)
        split = int(n * 0.8)
        train_idx = indices[:split]
        test_idx = indices[split:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_proc_train, y_proc_test = y_proc[train_idx], y_proc[test_idx]
        y_pred_train, y_pred_test = y_pred[train_idx], y_pred[test_idx]

        if verbose:
            print(f"\n  Entraînement arbre proc_type "
                  f"(train={len(X_train)}, test={len(X_test)})...")

        np.random.seed(42)
        h.tree_proc = build_decision_tree(
            X_train, y_proc_train, max_depth=max_depth
        )

        # Évaluation
        preds = np.array([h.tree_proc.predict(x) for x in X_test])
        h.accuracy_proc = np.mean(preds == y_proc_test)

        if verbose:
            print(f"  Précision proc_type : {h.accuracy_proc:.1%}")

        if verbose:
            print(f"  Entraînement arbre pred_mode...")

        np.random.seed(43)
        h.tree_pred = build_decision_tree(
            X_train, y_pred_train, max_depth=max_depth
        )

        preds_pred = np.array([h.tree_pred.predict(x) for x in X_test])
        h.accuracy_pred = np.mean(preds_pred == y_pred_test)

        if verbose:
            print(f"  Précision pred_mode : {h.accuracy_pred:.1%}")

        return h

    def predict_method(
        self, features: np.ndarray
    ) -> Tuple[int, int]:
        """
        Prédit le proc_type et pred_mode depuis les features.

        Returns:
            (proc_type, pred_mode)
        """
        proc = self.tree_proc.predict(features) if self.tree_proc else 1
        pred = self.tree_pred.predict(features) if self.tree_pred else 0
        return int(proc), int(pred)

    def to_dict(self) -> dict:
        return {
            'tree_proc': self.tree_proc.to_dict() if self.tree_proc else None,
            'tree_pred': self.tree_pred.to_dict() if self.tree_pred else None,
            'accuracy_proc': self.accuracy_proc,
            'accuracy_pred': self.accuracy_pred,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'DecisionHeuristic':
        h = cls()
        if d['tree_proc']:
            h.tree_proc = DecisionNode.from_dict(d['tree_proc'])
        if d['tree_pred']:
            h.tree_pred = DecisionNode.from_dict(d['tree_pred'])
        h.accuracy_proc = d.get('accuracy_proc', 0)
        h.accuracy_pred = d.get('accuracy_pred', 0)
        return h

    def save(self, path: str) -> None:
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> 'DecisionHeuristic':
        import json
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
