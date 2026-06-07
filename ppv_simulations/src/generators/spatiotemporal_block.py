"""
Génération de blocs spatio-temporels vidéo simulés
"""
import numpy as np
from typing import List, Dict, Optional
from .poisson_inhomogeneous import generate_poisson_nh, get_intensity_functions


def generate_marks_uniform(n_jumps: int, m_colors: int, seed: Optional[int] = None) -> np.ndarray:
    """Génère des marques uniformes parmi m_colors."""
    if seed is not None:
        np.random.seed(seed)
    if n_jumps == 0:
        return np.array([], dtype=int)
    return np.random.randint(0, m_colors, size=n_jumps)


def generate_marks_markov(
    n_jumps: int, 
    m_states: int, 
    transition_matrix: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """Génère des marques selon une chaîne de Markov à m états."""
    if seed is not None:
        np.random.seed(seed)
    
    if n_jumps == 0:
        return np.array([], dtype=int)
    
    if transition_matrix is None:
        # Matrice uniforme par défaut
        transition_matrix = np.ones((m_states, m_states)) / m_states
    
    marks = np.zeros(n_jumps, dtype=int)
    marks[0] = np.random.randint(0, m_states)
    
    for k in range(1, n_jumps):
        prev_state = marks[k - 1]
        marks[k] = np.random.choice(m_states, p=transition_matrix[prev_state])
    
    return marks


def generate_spatiotemporal_block(
    n_pixels: int,
    r_frames: int,
    m_colors: int,
    intensity_type: str,
    intensity_params: dict,
    framework: str = 'marked',
    markov_params: Optional[dict] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Génère un bloc vidéo simulé: n_pixels processus ponctuels marqués sur r_frames.
    
    Paramètres
    ----------
    n_pixels : int
        Nombre de pixels (bloc carré: sqrt(n_pixels) × sqrt(n_pixels))
    r_frames : int
        Nombre de frames temporelles
    m_colors : int
        Nombre de couleurs/marques possibles
    intensity_type : str
        Type d'intensité ('homogeneous', 'linear', 'sinusoidal', ...)
    intensity_params : dict
        Paramètres spécifiques au type d'intensité
    framework : str
        Cadre statistique sous-jacent ('vector', 'marked', 'spatial', 'markov')
    markov_params : dict, optional
        Paramètres pour le cadre Markovien (matrice de transition)
    seed : int, optional
        Graine pour reproductibilité
    
    Retourne
    --------
    block_data : dict
        {
            'jumps': List[np.ndarray],      # Temps de sauts par pixel
            'marks': List[np.ndarray],      # Couleurs à chaque saut
            'intensity_true': callable,     # α(t) vraie (pour validation)
            'params': dict                  # Paramètres de génération
        }
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Définir les fonctions d'intensité
    alpha_func, A_func, A_inv_func = get_intensity_functions(
        intensity_type, intensity_params, T=r_frames)
    
    jumps_list = []
    marks_list = []
    
    # 2. Générer chaque pixel indépendamment (hypothèse d'indépendance)
    for pixel_id in range(n_pixels):
        # a) Générer les temps de sauts continus
        jumps_cont = generate_poisson_nh(
            alpha_func, A_func, A_inv_func, T=r_frames, seed=None)
        
        # b) Discrétiser en frames
        jumps_disc = np.unique(np.floor(jumps_cont).astype(int))
        jumps_disc = jumps_disc[jumps_disc < r_frames]
        
        # c) Générer les marques selon le cadre
        n_jumps = len(jumps_disc)
        
        if framework == 'markov' and markov_params is not None:
            marks = generate_marks_markov(
                n_jumps, m_colors, 
                transition_matrix=markov_params.get('transition_matrix'),
                seed=None)
        else:
            marks = generate_marks_uniform(n_jumps, m_colors, seed=None)
        
        jumps_list.append(jumps_disc)
        marks_list.append(marks)
    
    return {
        'jumps': jumps_list,
        'marks': marks_list,
        'intensity_true': alpha_func,
        'params': {
            'n_pixels': n_pixels,
            'r_frames': r_frames,
            'm_colors': m_colors,
            'intensity_type': intensity_type,
            'true_params': intensity_params.copy(),
            'framework': framework,
            'seed': seed
        }
    }