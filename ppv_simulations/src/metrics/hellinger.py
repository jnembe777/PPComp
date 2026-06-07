"""
Calcul de la distance de Hellinger et validation des bornes de convergence
"""
import numpy as np
from typing import Callable, List


def hellinger_distance_point_process(
    alpha_true: Callable[[float], float],
    alpha_est: Callable[[float], float],
    Y_func: Callable[[float], float],
    T: float,
    n_points: int = 1000
) -> float:
    """
    Calcule la distance de Hellinger entre deux intensités de processus ponctuel.
    
    H²(P_α, P_β) = ½ ∫₀ᵀ (√α(s) - √β(s))² Y(s) ds
    """
    t_grid = np.linspace(0, T, n_points)
    dt = T / n_points
    
    H2 = 0.0
    for t in t_grid:
        sqrt_alpha = np.sqrt(max(alpha_true(t), 1e-10))
        sqrt_beta = np.sqrt(max(alpha_est(t), 1e-10))
        Y_t = Y_func(t)
        H2 += (sqrt_alpha - sqrt_beta)**2 * Y_t * dt
    
    return np.sqrt(0.5 * H2)


def compute_hellinger_from_basis(
    alpha_true: Callable[[float], float],
    alpha_coeffs: np.ndarray,
    basis_funcs: List[callable],
    Y_func: Callable[[float], float],
    T: float
) -> float:
    """Calcule H à partir de coefficients de base."""
    def alpha_est(t):
        return sum(alpha_coeffs[j] * basis_funcs[j](t) for j in range(len(alpha_coeffs)))
    
    return hellinger_distance_point_process(alpha_true, alpha_est, Y_func, T)


def validate_convergence_bound(
    H2_empirical: float,
    approximation_error: float,
    dimension: int,
    n_samples: int,
    tolerance: float = 1e-6
) -> bool:
    """
    Valide la borne de convergence théorique.
    
    H² ≤ εₙ² + (K·log n)/(2n)
    """
    theoretical_bound = approximation_error**2 + (dimension * np.log(n_samples)) / (2 * n_samples)
    return H2_empirical**2 <= theoretical_bound + tolerance


def estimate_approximation_error(
    alpha_true: Callable[[float], float],
    alpha_est: Callable[[float], float],
    T: float,
    n_points: int = 1000
) -> float:
    """Estime l'erreur d'approximation L² entre α vraie et estimée."""
    t_grid = np.linspace(0, T, n_points)
    errors = []
    for t in t_grid:
        err = (alpha_true(t) - alpha_est(t))**2
        errors.append(err)
    return np.sqrt(np.mean(errors))