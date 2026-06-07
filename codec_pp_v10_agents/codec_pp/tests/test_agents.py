#!/usr/bin/env python3
"""
Démonstration du système multi-agent PPV.

Exécute le cycle d'optimisation itérative sur le corpus synthétique :
  PROPOSER → ENCODER → ÉVALUER → DÉCIDER → CATALOGUER → boucle
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codec_pp.src.agents import Orchestrator
from codec_pp.src.benchmark import (
    gen_static_bg, gen_gradient, gen_animation, gen_noise, gen_periodic,
)


def build_corpus():
    """Construit le corpus de test avec les vidéos synthétiques."""
    corpus = []
    for gen_fn, name in [
        (gen_static_bg, "surveillance"),
        (gen_gradient, "gradient"),
        (gen_animation, "animation"),
        (gen_noise, "noise"),
        (gen_periodic, "periodic"),
    ]:
        M, _, desc = gen_fn(nl=32, nc=32, r=32)
        corpus.append({'name': name, 'data': M, 'fps': 30})
    return corpus


def build_h264_reference():
    """Référence H.264 simulée (valeurs du benchmark réel)."""
    return {
        'surveillance': {'bpp': 1.53, 'ratio': 5.2},
        'gradient':     {'bpp': 3.68, 'ratio': 2.2},
        'animation':    {'bpp': 2.08, 'ratio': 3.8},
        'noise':        {'bpp': 7.64, 'ratio': 1.0},
        'periodic':     {'bpp': 3.36, 'ratio': 2.4},
    }


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  SYSTÈME MULTI-AGENT PPV — Cycle d'optimisation      ║")
    print("╚══════════════════════════════════════════════════════╝")

    corpus = build_corpus()
    h264_ref = build_h264_reference()

    print(f"\n  Corpus : {len(corpus)} vidéos")
    for v in corpus:
        print(f"    {v['name']:<15s} {v['data'].shape}")

    orch = Orchestrator(corpus, h264_ref)

    print(f"\n  Démarrage du cycle d'optimisation...")
    print(f"  Convergence si gain < 0.1% sur 5 itérations\n")

    results = orch.run_until_convergence(max_iterations=10)

    # Résumé
    print(f"\n{'═' * 70}")
    print(f"  HISTORIQUE DES ITÉRATIONS")
    print(f"{'─' * 70}")
    print(f"  {'#':>3}  {'Règle':<25s}  {'Statut':<10s}  {'Gain ratio':>10s}  {'Version':>10s}")
    print(f"{'─' * 70}")
    for r in results:
        gain = f"{r['gain']:+.2%}" if r.get('gain') else "—"
        print(f"  {r.get('iteration', '?'):>3}  {r.get('rule', '—'):<25s}  "
              f"{r.get('status', '—'):<10s}  {gain:>10s}  {r.get('version', '—'):>10s}")
    print(f"{'═' * 70}")

    # Catalogue
    print(f"\n  Catalogue sauvegardé : ppv_catalog.json")
    print(f"  Versions : {len(orch.state.catalog)}")

    # Progression
    prog = orch.cataloger.get_progression()
    if prog:
        print(f"\n  Progression des ratios :")
        for p in prog:
            bar = '█' * int(p['ratio_mean'] * 3)
            print(f"    {p['version']:<12s}  ratio={p['ratio_mean']:.2f}x  {bar}")
