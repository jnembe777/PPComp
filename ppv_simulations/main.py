#!/usr/bin/env python3
"""
Script principal pour exécuter les simulations PPV v2.0
"""
import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from src.generators.spatiotemporal_block import generate_spatiotemporal_block
from src.frameworks.marked_framework import estimate_marked_mdl
from src.metrics.codelength import compute_block_codelengths
from src.metrics.hellinger import (
    compute_hellinger_from_basis, 
    validate_convergence_bound,
    estimate_approximation_error
)
from src.estimation.basis_families import get_basis_functions


def run_single_simulation(config: dict, seed: int) -> dict:
    """Exécute une simulation unitaire avec une graine donnée."""
    try:
        # 1. Générer le bloc
        block = generate_spatiotemporal_block(
            n_pixels=config['block_size']**2,
            r_frames=config['r_frames'],
            m_colors=config['m_colors'],
            intensity_type=config['intensity_type'],
            intensity_params=config['intensity_params'],
            framework=config.get('framework', 'marked'),
            seed=seed
        )
        
        # 2. Estimation MDL
        basis_family = config.get('basis_family', 'histogram')
        dimension = config.get('dimension', 4)
        
        mdl_result = estimate_marked_mdl(
            block_data=block,
            basis_family=basis_family,
            dimension=dimension,
            prec_bits=config.get('prec_bits', 8)
        )
        
        alpha_hat, log_likelihood, complexity_penalty = mdl_result
        
        # 3. Calcul des longueurs de code
        codelengths = compute_block_codelengths(
            block_data=block,
            mdl_results={
                'log_likelihood': log_likelihood,
                'dimension': dimension,
                'prec_bits': config.get('prec_bits', 8),
                'mode': 'marked'
            }
        )
        
        # 4. Validation de convergence Hellinger
        basis_funcs = get_basis_functions(basis_family, dimension, config['r_frames'])
        
        def alpha_est_func(t):
            return sum(alpha_hat[j] * basis_funcs[j](t) for j in range(dimension))
        
        H_empirical = compute_hellinger_from_basis(
            alpha_true=block['intensity_true'],
            alpha_coeffs=alpha_hat,
            basis_funcs=basis_funcs,
            Y_func=lambda t: config['block_size']**2,
            T=config['r_frames']
        )
        
        approx_error = estimate_approximation_error(
            block['intensity_true'], alpha_est_func, config['r_frames']
        )
        
        bound_satisfied = validate_convergence_bound(
            H2_empirical=H_empirical**2,
            approximation_error=approx_error,
            dimension=dimension,
            n_samples=config['block_size']**2
        )
        
        # 5. Agréger les résultats
        result = {
            'config': config,
            'seed': seed,
            'n_samples': config['block_size']**2,
            'n_jumps_total': sum(len(j) for j in block['jumps']),
            'codelengths': codelengths,
            'hellinger': {
                'empirical': H_empirical,
                'approximation_error': approx_error,
                'bound_satisfied': bound_satisfied
            },
            'mdl': {
                'log_likelihood': log_likelihood,
                'complexity_penalty': complexity_penalty,
                'alpha_hat': alpha_hat.tolist()
            }
        }
        
        return result
        
    except Exception as e:
        return {
            'config': config,
            'seed': seed,
            'error': str(e)
        }


def generate_config_combinations(grid: dict) -> list:
    """Génère toutes les combinaisons de paramètres depuis la grille YAML."""
    configs = []
    
    for block_size in grid['block_sizes']:
        for r_frames in grid['r_frames']:
            for m_colors in grid['m_colors']:
                for intensity_type in grid['intensity_types']:
                    for intensity_params in grid['intensity_params'].get(intensity_type, [{}]):
                        for framework in grid.get('frameworks', ['marked']):
                            for basis_family in grid.get('basis_families', ['histogram']):
                                for dimension in grid.get('dimensions', [4]):
                                    configs.append({
                                        'block_size': block_size,
                                        'r_frames': r_frames,
                                        'm_colors': m_colors,
                                        'intensity_type': intensity_type,
                                        'intensity_params': intensity_params,
                                        'framework': framework,
                                        'basis_family': basis_family,
                                        'dimension': dimension,
                                        'prec_bits': grid.get('prec_bits', 8)
                                    })
    
    return configs


def main():
    parser = argparse.ArgumentParser(description='PPV Simulation Framework v2.0')
    parser.add_argument('--config', type=str, required=True, 
                       help='Chemin vers fichier YAML de configuration')
    parser.add_argument('--output', type=str, default='results/', 
                       help='Dossier de sortie')
    parser.add_argument('--parallel', action='store_true', 
                       help='Exécution parallèle')
    parser.add_argument('--workers', type=int, default=4, 
                       help='Nombre de workers pour parallélisation')
    parser.add_argument('--n-rep', type=int, default=100,
                       help='Nombre de réplications par configuration')
    
    args = parser.parse_args()
    
    # Charger configuration
    with open(args.config, 'r') as f:
        grid = yaml.safe_load(f)
    
    # Générer toutes les configurations
    configs = generate_config_combinations(grid)
    print(f"✓ {len(configs)} configurations générées")
    
    # Créer dossier de sortie
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    results = []
    total_runs = len(configs) * args.n_rep
    
    if args.parallel:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for cfg in configs:
                for rep in range(args.n_rep):
                    seed = grid['seeds'][rep % len(grid['seeds'])] if 'seeds' in grid else None
                    futures.append(executor.submit(run_single_simulation, cfg, seed))
            
            for future in tqdm(futures, total=total_runs, desc="Simulations"):
                result = future.result()
                results.append(result)
    else:
        for cfg in tqdm(configs, desc="Configurations"):
            for rep in range(args.n_rep):
                seed = grid['seeds'][rep % len(grid['seeds'])] if 'seeds' in grid else None
                result = run_single_simulation(cfg, seed)
                results.append(result)
    
    # Sauvegarder résultats bruts
    with open(f"{args.output}/raw_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Agrégation et statistiques
    summary = aggregate_results(results)
    with open(f"{args.output}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Simulations terminées: {len(results)} exécutions")
    print(f"✓ Résultats sauvegardés dans {args.output}/")
    print(f"✓ Résumé disponible dans {args.output}/summary.json")


def aggregate_results(results: list) -> dict:
    """Agrège les résultats pour produire des statistiques."""
    successful = [r for r in results if 'error' not in r]
    
    if not successful:
        return {'error': 'No successful simulations'}
    
    # Statistiques par configuration
    by_config = {}
    for r in successful:
        key = tuple(sorted(r['config'].items()))
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(r)
    
    summary = {
        'total_runs': len(results),
        'successful': len(successful),
        'failed': len(results) - len(successful),
        'by_config': {}
    }
    
    for cfg_key, runs in by_config.items():
        cfg = dict(cfg_key)
        
        # Statistiques sur les longueurs de code
        lc_values = [r['codelengths'] for r in runs if 'codelengths' in r]
        
        # Statistiques sur Hellinger
        h_values = [r['hellinger']['empirical'] for r in runs 
                   if 'hellinger' in r and 'empirical' in r['hellinger']]
        
        bound_stats = [r['hellinger']['bound_satisfied'] for r in runs 
                      if 'hellinger' in r and 'bound_satisfied' in r['hellinger']]
        
        summary['by_config'][str(cfg)] = {
            'n_runs': len(runs),
            'codelengths': {
                'L4_mean': np.mean([lc['L4'] for lc in lc_values]) if lc_values else None,
                'LC_MDL_mean': np.mean([lc['LC_MDL'] for lc in lc_values 
                                       if 'LC_MDL' in lc]) if lc_values else None,
                'gain_vs_uniform': np.mean([lc['mdl_gain_vs_uniform'] for lc in lc_values 
                                          if 'mdl_gain_vs_uniform' in lc]) if lc_values else None
            },
            'hellinger': {
                'mean': np.mean(h_values) if h_values else None,
                'std': np.std(h_values) if h_values else None,
                'bound_satisfied_ratio': np.mean(bound_stats) if bound_stats else None
            }
        }
    
    return summary


if __name__ == "__main__":
    main()