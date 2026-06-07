"""
Cadre 2 : Processus Ponctuel Marqué
Implémentation de la vraisemblance et de l'estimation MDL
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize


def marked_likelihood(
    jumps: List[np.ndarray],
    marks: List[np.ndarray],
    alpha_params: np.ndarray,
    basis_funcs: List[callable],
    Y_func: callable,
    T: float,
    m_colors: int
) -> float:
    """
    Calcule la log-vraisemblance pour le cadre marqué.
    
    L(α) = Σ_k log(α(τ_k)) - ∫₀ᵀ α(s)Y(s)ds + Σ_k log(p_M(m_k))
    """
    # Terme 1: Σ log(α(τ_k)) pour tous les sauts
    log_alpha_sum = 0.0
    for pixel_jumps in jumps:
        for tau in pixel_jumps:
            alpha_t = sum(alpha_params[j] * basis_funcs[j](tau) 
                         for j in range(len(alpha_params)))
            log_alpha_sum += np.log(max(alpha_t, 1e-10))
    
    # Terme 2: -∫ α(s)Y(s)ds (approximation par quadrature)
    n_points = 100
    t_grid = np.linspace(0, T, n_points)
    integral = 0.0
    for t in t_grid:
        alpha_t = sum(alpha_params[j] * basis_funcs[j](t) 
                     for j in range(len(alpha_params)))
        Y_t = Y_func(t)
        integral += alpha_t * Y_t
    integral *= T / n_points
    
    # Terme 3: Σ log(p_M(m_k)) - supposons uniforme pour l'instant
    total_marks = sum(len(m) for m in marks)
    log_p_sum = total_marks * np.log(1.0 / m_colors) if m_colors > 0 else 0.0
    
    return log_alpha_sum - integral + log_p_sum


def estimate_marked_mdl(
    block_ Dict,
    basis_family: str,
    dimension: int,
    prec_bits: int = 8
) -> Tuple[np.ndarray, float, float]:
    """
    Estime l'intensité par MDL dans le cadre marqué.
    
    Retourne
    --------
    alpha_hat : coefficients estimés
    log_likelihood : log-vraisemblance au optimum
    complexity_penalty : pénalité MDL
    """
    jumps = block_data['jumps']
    marks = block_data['marks']
    T = block_data['params']['r_frames']
    m_colors = block_data['params']['m_colors']
    n_pixels = block_data['params']['n_pixels']
    
    # Définir les fonctions de base
    basis_funcs = get_basis_functions(basis_family, dimension, T)
    
    # Fonction objectif: -log L + C_n(α)
    def objective(alpha_params):
        # Contrainte: α(t) >= 0
        if np.any(alpha_params < 0):
            return 1e10
        
        log_L = marked_likelihood(
            jumps, marks, alpha_params, basis_funcs,
            Y_func=lambda t: n_pixels, T=T, m_colors=m_colors)
        
        # Pénalité MDL: (K/2)log(n) + K·prec
        K = len(alpha_params)
        n = n_pixels
        complexity = (K / 2) * np.log(n) + K * prec_bits
        
        return -log_L + complexity
    
    # Initialisation: estimateur histogramme simple
    total_jumps = sum(len(j) for j in jumps)
    alpha_init = np.ones(dimension) * (total_jumps / (n_pixels * T * dimension))
    
    # Optimisation
    bounds = [(1e-10, None) for _ in range(dimension)]
    result = minimize(objective, alpha_init, method='L-BFGS-B', bounds=bounds)
    
    alpha_hat = result.x
    log_likelihood = -marked_likelihood(
        jumps, marks, alpha_hat, basis_funcs,
        Y_func=lambda t: n_pixels, T=T, m_colors=m_colors)
    
    K = len(alpha_hat)
    complexity_penalty = (K / 2) * np.log(n_pixels) + K * prec_bits
    
    return alpha_hat, log_likelihood, complexity_penalty


def get_basis_functions(family: str, dimension: int, T: float) -> List[callable]:
    """Retourne une liste de fonctions de base selon la famille."""
    if family == 'histogram':
        # Partition uniforme de [0, T] en dimension intervalles
        edges = np.linspace(0, T, dimension + 1)
        return [lambda t, a=edges[i], b=edges[i+1]: 1.0 if a <= t < b else 0.0 
                for i in range(dimension)]
    
    elif family == 'polynomial':
        # Polynômes normalisés sur [0, T]
        return [lambda t, k=k, T=T: (t / T)**k for k in range(dimension)]
    
    elif family == 'trigonometric':
        # Fonctions trigonométriques
        funcs = [lambda t: np.ones_like(t) if hasattr(t, '__len__') else 1.0]
        for k in range(1, (dimension + 1) // 2 + 1):
            funcs.append(lambda t, k=k, T=T: np.cos(2 * np.pi * k * t / T))
            funcs.append(lambda t, k=k, T=T: np.sin(2 * np.pi * k * t / T))
        return funcs[:dimension]
    
    else:
        # Fallback: histogramme
        return get_basis_functions('histogram', dimension, T)