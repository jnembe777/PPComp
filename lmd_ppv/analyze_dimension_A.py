"""
Analyse des erreurs de la dimension A
"""
import sys
from pathlib import Path
import json
import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from src.core.features import BlockFeatures
from src.core.process_types import ProcessType
from src.agents.agent_1_classification import ClassificationAgent
from optimization.exhaustive_search import ExhaustiveSearch

def load_features(path: Path) -> list:
    """Charge les features depuis le JSON."""
    data = json.loads(path.read_text())
    features_list = []
    for item in data:
        # Remplacer Infinity par une grande valeur
        if item.get('N_star') == 'Infinity' or item.get('N_star') == float('inf'):
            item['N_star'] = float('inf')

        features = BlockFeatures(
            N=item['N'],
            r=item['r'],
            m=item['m'],
            n_pixels=item['n_pixels'],
            lambda_avg=item.get('lambda_avg', 0),
            H_s=item.get('H_s', 0),
            rho_corr=item.get('rho_corr', 0.5),
            R_temp=item.get('R_temp', 1.0),
            m_eff=item.get('m_eff', 0),
            H_color=item.get('H_color', 0),
            N_trans=item.get('N_trans', 0),
        )
        features_list.append(features)
    return features_list


def main():
    features_path = Path("optimization_final/features.json")
    features_list = load_features(features_path)

    print(f"Nombre de blocs: {len(features_list)}")

    # Initialiser
    agent = ClassificationAgent()
    exhaustive = ExhaustiveSearch()

    # Analyse
    results = {
        'match': [],
        'mismatch': []
    }

    optimal_dist = Counter()
    predicted_dist = Counter()
    confusion = {}  # (predicted, optimal) -> count

    mismatch_details = []

    for i, features in enumerate(features_list):
        # Optimal
        optimal_result = exhaustive.find_optimal(features, i)
        optimal_A = optimal_result.optimal_cartouche.A

        # Prédit
        classification = agent.classify(features)
        predicted_A = classification.process_type

        optimal_dist[optimal_A] += 1
        predicted_dist[predicted_A] += 1

        key = (predicted_A, optimal_A)
        confusion[key] = confusion.get(key, 0) + 1

        if predicted_A == optimal_A:
            results['match'].append(i)
        else:
            results['mismatch'].append(i)
            mismatch_details.append({
                'block_id': i,
                'N': features.N,
                'm': features.m,
                'm_eff': features.m_eff,
                'H_color': features.H_color,
                'N_trans': features.N_trans,
                'predicted': predicted_A,
                'optimal': optimal_A,
                'mono_gain': classification.mono_gain,
                'is_mono_better': classification.is_mono_better,
                'cost_A': optimal_result.cost_A,
            })

    # Afficher résultats
    print(f"\n=== Distribution des types optimaux ===")
    for t, count in sorted(optimal_dist.items()):
        name = ProcessType(t).name if t < 5 else f"TYPE_{t}"
        print(f"  {name}: {count} ({count/len(features_list)*100:.1f}%)")

    print(f"\n=== Distribution des types prédits ===")
    for t, count in sorted(predicted_dist.items()):
        name = ProcessType(t).name if t < 5 else f"TYPE_{t}"
        print(f"  {name}: {count} ({count/len(features_list)*100:.1f}%)")

    print(f"\n=== Matrice de confusion ===")
    print("(prédit, optimal) -> count")
    for key, count in sorted(confusion.items()):
        pred_name = ProcessType(key[0]).name if key[0] < 5 else f"TYPE_{key[0]}"
        opt_name = ProcessType(key[1]).name if key[1] < 5 else f"TYPE_{key[1]}"
        print(f"  ({pred_name}, {opt_name}): {count}")

    print(f"\n=== Statistiques des erreurs ===")
    print(f"Matchs: {len(results['match'])} ({len(results['match'])/len(features_list)*100:.1f}%)")
    print(f"Erreurs: {len(results['mismatch'])} ({len(results['mismatch'])/len(features_list)*100:.1f}%)")

    # Analyser les erreurs par type
    print(f"\n=== Analyse des erreurs ===")

    # Grouper par type d'erreur
    error_types = Counter()
    for d in mismatch_details:
        pred_name = ProcessType(d['predicted']).name
        opt_name = ProcessType(d['optimal']).name
        error_types[f"{pred_name} -> {opt_name}"] += 1

    for error, count in error_types.most_common():
        print(f"  {error}: {count}")

    # Détails des premiers mismatches
    print(f"\n=== Détails des 20 premières erreurs ===")
    for d in mismatch_details[:20]:
        pred_name = ProcessType(d['predicted']).name
        opt_name = ProcessType(d['optimal']).name
        print(f"\nBloc {d['block_id']}:")
        print(f"  N={d['N']}, m={d['m']}, m_eff={d['m_eff']}")
        print(f"  H_color={d['H_color']:.3f}, N_trans={d['N_trans']}")
        print(f"  Prédit: {pred_name}, Optimal: {opt_name}")
        print(f"  mono_gain={d['mono_gain']:.2f}, is_mono_better={d['is_mono_better']}")
        print(f"  Coûts A: {d['cost_A']}")

    # Statistiques sur les erreurs
    print(f"\n=== Statistiques des blocs en erreur ===")
    N_values = [d['N'] for d in mismatch_details]
    m_values = [d['m'] for d in mismatch_details]
    m_eff_values = [d['m_eff'] for d in mismatch_details]
    mono_gains = [d['mono_gain'] for d in mismatch_details]

    print(f"  N: min={min(N_values)}, max={max(N_values)}, mean={np.mean(N_values):.1f}")
    print(f"  m: min={min(m_values)}, max={max(m_values)}, mean={np.mean(m_values):.1f}")
    print(f"  m_eff: min={min(m_eff_values)}, max={max(m_eff_values)}, mean={np.mean(m_eff_values):.1f}")
    print(f"  mono_gain: min={min(mono_gains):.2f}, max={max(mono_gains):.2f}, mean={np.mean(mono_gains):.2f}")


if __name__ == "__main__":
    main()
