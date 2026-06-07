"""
Calcul des longueurs de code: stratégies uniformes L1-L4 et MDL
"""
import numpy as np
from scipy.special import comb, loggamma
from typing import List, Dict


def log_comb(n: int, k: int) -> float:
    """Calcule log(C(n,k)) de manière numériquement stable."""
    if k < 0 or k > n:
        return -np.inf
    if k == 0 or k == n:
        return 0.0
    return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)


def codelength_L1(r: int, m: int) -> float:
    """L1: État complet spatial - code l'état à chaque bin temporel."""
    return r * np.log2(m)


def codelength_L2(r: int, m: int, N: int) -> float:
    """L2: Liste de sauts - code N, puis N fois (temps + couleur)."""
    if N == 0:
        return np.log2(r)  # Juste coder N=0
    return np.log2(N) + N * (np.log2(r) + np.log2(m))


def codelength_L3(r: int, m: int, N: int) -> float:
    """L3: Vecteur booléen + couleurs - code r bits pour les temps + N·log(m) pour couleurs."""
    return r + N * np.log2(m)


def codelength_L4(r: int, m: int, N: int) -> float:
    """L4: Adresse combinatoire + couleurs - code N + index parmi C(r,N) + N·log(m)."""
    if N == 0 or N == r:
        return np.log2(r) + N * np.log2(m)
    return np.log2(N) + log_comb(r, N) / np.log(2) + N * np.log2(m)


def optimal_uniform_codelength(r: int, m: int, N: int) -> Tuple[str, float]:
    """Sélectionne la meilleure stratégie uniforme parmi L1-L4."""
    candidates = [
        ('L1', codelength_L1(r, m)),
        ('L2', codelength_L2(r, m, N)),
        ('L3', codelength_L3(r, m, N)),
        ('L4', codelength_L4(r, m, N))
    ]
    best_name, best_value = min(candidates, key=lambda x: x[1])
    return best_name, best_value


def codelength_mdl(
    log_likelihood: float,
    dimension: int,
    n_samples: int,
    prec_bits: int,
    n_jumps: int,
    m_colors: int,
    mode: str = 'marked'
) -> float:
    """
    Calcule la longueur de code MDL complète.
    
    LC_MDL = -log L(α̂) + C_n(α̂) + C_couleurs
    """
    # Terme de vraisemblance
    neg_log_L = -log_likelihood
    
    # Pénalité de complexité pour l'intensité
    complexity_intensity = (dimension / 2) * np.log(n_samples) + dimension * prec_bits
    
    # Terme de codage des couleurs
    if mode == 'marked':
        # Mode marqué: N·log(m) bits
        color_cost = n_jumps * np.log2(m_colors)
    else:
        # Mode monochromatique: carte de couleurs codée une fois
        color_cost = m_colors * prec_bits  # Simplification
    
    return neg_log_L + complexity_intensity + color_cost


def compute_block_codelengths(
    block_data: Dict,
    mdl_results: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Calcule toutes les longueurs de code pour un bloc.
    
    Retourne un dictionnaire avec L1, L2, L3, L4, et LC_MDL si disponible.
    """
    params = block_data['params']
    r = params['r_frames']
    m = params['m_colors']
    
    # Nombre total de sauts dans le bloc
    N = sum(len(j) for j in block_data['jumps'])
    
    # Longueurs de code uniformes
    results = {
        'L1': codelength_L1(r, m),
        'L2': codelength_L2(r, m, N),
        'L3': codelength_L3(r, m, N),
        'L4': codelength_L4(r, m, N),
        'N_jumps': N
    }
    
    # Meilleure stratégie uniforme
    best_uniform, best_uniform_value = optimal_uniform_codelength(r, m, N)
    results['best_uniform'] = best_uniform
    results['best_uniform_value'] = best_uniform_value
    
    # Longueur de code MDL si estimée
    if mdl_results is not None:
        results['LC_MDL'] = codelength_mdl(
            log_likelihood=mdl_results['log_likelihood'],
            dimension=mdl_results['dimension'],
            n_samples=params['n_pixels'],
            prec_bits=mdl_results.get('prec_bits', 8),
            n_jumps=N,
            m_colors=m,
            mode=mdl_results.get('mode', 'marked')
        )
        results['mdl_gain_vs_uniform'] = best_uniform_value / results['LC_MDL']
    
    return results