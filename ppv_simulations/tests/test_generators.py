"""Tests pour les générateurs de processus."""
import numpy as np
import pytest
from src.generators.poisson_inhomogeneous import (
    generate_poisson_nh, 
    get_intensity_functions
)


def test_homogeneous_poisson_mean():
    """Test que E[N] ≈ λT pour Poisson homogène."""
    lambda0 = 0.1
    T = 100
    n_rep = 1000
    
    alpha, A, A_inv = get_intensity_functions(
        'homogeneous', {'lambda0': lambda0}, T)
    
    counts = []
    for _ in range(n_rep):
        jumps = generate_poisson_nh(alpha, A, A_inv, T)
        counts.append(len(jumps))
    
    empirical_mean = np.mean(counts)
    theoretical_mean = lambda0 * T
    
    assert abs(empirical_mean - theoretical_mean) < 3 * np.sqrt(theoretical_mean)


def test_inversion_consistency():
    """Test que A(A⁻¹(y)) = y pour intensités avec inversion analytique."""
    T = 100
    test_values = [1, 5, 10, 25, 50]
    
    for intensity_type in ['homogeneous', 'exponential', 'power']:
        params = {'lambda0': 0.1}
        if intensity_type == 'exponential':
            params['beta'] = 0.05
        elif intensity_type == 'power':
            params['p'] = 1
        
        alpha, A, A_inv = get_intensity_functions(intensity_type, params, T)
        
        for y in test_values:
            if y < A(T):
                t_recovered = A_inv(y)
                y_recovered = A(t_recovered)
                assert abs(y - y_recovered) < 1e-6, f"Failed for {intensity_type} at y={y}"


def test_discretization():
    """Test que les sauts sont bien dans [0, T)."""
    T = 64
    alpha, A, A_inv = get_intensity_functions(
        'sinusoidal', 
        {'lambda0': 0.1, 'amplitude': 0.5, 'omega': 0.2}, 
        T)
    
    for _ in range(100):
        jumps = generate_poisson_nh(alpha, A, A_inv, T)
        assert np.all(jumps >= 0), "Negative jumps"
        assert np.all(jumps < T), "Jumps >= T"
        assert len(jumps) == len(np.unique(jumps)), "Duplicate jumps"