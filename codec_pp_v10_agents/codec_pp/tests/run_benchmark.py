#!/usr/bin/env python3
"""Exécute le benchmark complet du codec PPV."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codec_pp.src.benchmark import run_full_benchmark, generate_report

if __name__ == "__main__":
    results = run_full_benchmark(nl=64, nc=64, r=64, verbose=True)

    gray_ok = all(r['lossless'] for r in results if not r['is_color'])
    print(f"\n  Gris lossless : {'✓' if gray_ok else '✗'}")

    report = generate_report(results, "/mnt/user-data/outputs/benchmark_report.json")

    if gray_ok:
        print("  ✓ BENCHMARK COMPLET")
    else:
        sys.exit(1)
