"""
Générateurs de processus de Poisson non homogène
Implémentation de la méthode de transformation inverse : A(T_i) = -log(U_i)
"""
import numpy as np
from scipy.optimize import bisect
from typing import Callable, Tuple, Optional


def generate_poisson_nh(
    alpha_func: Callable[[float], float],
    A_func: Callable[[float], float],
    A_inv_func: Callable[[float], float],
    T: float,
    max_jumps: int = 10000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Génère les temps de sauts d'un processus de Poisson non homogène
    par transformation inverse de l'intensité cumulée.
    
    Paramètres
    ----------
    alpha_func : callable
        Fonction intensité α(t)
    A_func : callable
        Intensité cumulée A(t) = ∫₀ᵗ α(s)ds
    A_inv_func : callable
        Inverse de A(t)
    T : float
        Horizon temporel
    max_jumps : int
        Sécurité pour éviter boucles infinies
    seed : int, optional
        Graine pour reproductibilité
    
    Retourne
    --------
    jumps : np.ndarray
        Temps de sauts discrets (frames) dans [0, T)
    """
    if seed is not None:
        np.random.seed(seed)
    
    jumps = []
    t_current = 0.0
    A_current = A_func(t_current)
    
    while t_current < T and len(jumps) < max_jumps:
        # Étape 1: Générer U ~ Uniform[0,1]
        U = np.random.uniform(1e-10, 1.0 - 1e-10)  # Éviter log(0)
        
        # Étape 2: Calculer la cible pour l'inversion
        target = A_current - np.log(U)
        
        # Étape 3: Inverser A pour obtenir t_next
        if target >= A_func(T):
            break
            
        t_next = A_inv_func(target)
        
        # Discrétisation: saut dans [k, k+1) → frame k
        frame = int(np.floor(t_next))
        if frame < T and (len(jumps) == 0 or frame > jumps[-1]):
            jumps.append(frame)
            t_current = t_next
            A_current = A_func(t_current)
    
    return np.array(jumps, dtype=int)


def get_intensity_functions(
    intensity_type: str,
    params: dict,
    T: float
) -> Tuple[Callable, Callable, Callable]:
    """
    Retourne (alpha(t), A(t), A_inv(y)) pour le type d'intensité spécifié.
    """
    if intensity_type == 'homogeneous':
        lambda0 = params.get('lambda0', 0.1)
        alpha = lambda t: lambda0
        A = lambda t: lambda0 * t
        A_inv = lambda y: y / lambda0
        
    elif intensity_type == 'linear':
        a = params.get('a', 0.01)
        b = params.get('b', 0.01)
        alpha = lambda t: a * t + b
        A = lambda t: 0.5 * a * t**2 + b * t
        if abs(a) < 1e-10:
            A_inv = lambda y: y / b
        else:
            A_inv = lambda y: (-b + np.sqrt(max(0, b**2 + 2 * a * y))) / a
            
    elif intensity_type == 'sinusoidal':
        lambda0 = params.get('lambda0', 0.1)
        amp = params.get('amplitude', 0.5)
        omega = params.get('omega', np.pi / 16)
        phi = params.get('phi', 0)
        
        def alpha(t):
            val = lambda0 * (1 + amp * np.sin(omega * t + phi))
            return max(val, 1e-10)  # Éviter intensité négative
            
        def A(t):
            return lambda0 * (t - amp / omega * np.cos(omega * t + phi))
            
        def A_inv(y):
            # Inversion numérique par dichotomie
            return bisect(lambda t: A(t) - y, 0, T, xtol=1e-8)
            
    elif intensity_type == 'exponential':
        lambda0 = params.get('lambda0', 0.1)
        beta = params.get('beta', 0.01)
        if abs(beta) < 1e-8:
            return get_intensity_functions('homogeneous', {'lambda0': lambda0}, T)
        alpha = lambda t: lambda0 * np.exp(beta * t)
        A = lambda t: lambda0 / beta * (np.exp(beta * t) - 1)
        A_inv = lambda y: np.log(1 + beta * y / lambda0) / beta
        
    elif intensity_type == 'logarithmic':
        lambda0 = params.get('lambda0', 0.1)
        gamma = params.get('gamma', 0.1)
        alpha = lambda t: lambda0 * np.log(1 + gamma * t + 1e-10)
        A = lambda t: lambda0 / gamma * ((1 + gamma * t) * np.log(1 + gamma * t + 1e-10) - gamma * t)
        A_inv = lambda y: bisect(lambda t: A(t) - y, 0, T, xtol=1e-8)
        
    elif intensity_type == 'power':
        lambda0 = params.get('lambda0', 0.1)
        p = params.get('p', 1)
        if abs(p + 1) < 1e-10:
            alpha = lambda t: lambda0 / (t + 1e-10)
            A = lambda t: lambda0 * np.log(t + 1)
            A_inv = lambda y: np.exp(y / lambda0) - 1
        else:
            alpha = lambda t: lambda0 * np.maximum(t, 1e-10)**p
            A = lambda t: lambda0 / (p + 1) * np.maximum(t, 1e-10)**(p + 1)
            A_inv = lambda y: np.maximum(0, (p + 1) * y / lambda0)**(1 / (p + 1))
    else:
        raise ValueError(f"Unknown intensity type: {intensity_type}")
    
    return alpha, A, A_inv