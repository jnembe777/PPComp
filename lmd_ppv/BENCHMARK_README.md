# LMD-PPV Benchmark Suite

> **Référence:** J. Nembé, Codage LMD Versatile v6.0

## Vue d'ensemble

Le benchmark compare LMD-PPV avec les codecs standards (H.264, H.265, VP9, AV1) sur des datasets vidéo de référence.

## Installation rapide

```bash
# Installer les dépendances
pip install -r requirements.txt

# Vérifier l'installation
python check_benchmark_setup.py

# Demo rapide (sans téléchargement)
python run_quick_demo.py
```

## Pipeline complet

### 1. Téléchargement des datasets

```bash
# Télécharger les vidéos Xiph.org
python cli.py benchmark download --source xiph --output ./datasets

# Ou plusieurs sources
python cli.py benchmark download --source xiph,cdvl --output ./datasets
```

### 2. Optimisation des seuils

```bash
# Optimisation rapide (recommandé)
python cli.py benchmark optimize --dataset ./datasets --quick

# Optimisation complète (32,000 configurations)
python cli.py benchmark optimize --dataset ./datasets
```

### 3. Exécution du benchmark

```bash
# Benchmark complet
python cli.py benchmark run --dataset ./datasets --codecs all

# Codecs spécifiques
python cli.py benchmark run --dataset ./datasets --codecs h264,h265,lmd

# Limiter le nombre de vidéos
python cli.py benchmark run --dataset ./datasets --max-videos 10
```

### 4. Validation

```bash
python cli.py benchmark validate --thresholds ./optimization.json
```

### 5. Génération du rapport

```bash
# HTML + JSON
python cli.py benchmark report --results ./benchmark_results

# Avec PDF (nécessite weasyprint)
python cli.py benchmark report --results ./benchmark_results --format html,pdf,json
```

## Script Pipeline Automatisé

Le script `run_full_benchmark.py` exécute toutes les étapes automatiquement :

```bash
# Benchmark complet (toutes les étapes)
python run_full_benchmark.py

# Mode rapide (5 vidéos max)
python run_full_benchmark.py --quick

# Utiliser les vidéos existantes (sans téléchargement)
python run_full_benchmark.py --skip-download

# Reprendre un benchmark interrompu
python run_full_benchmark.py --resume

# Options personnalisées
python run_full_benchmark.py \
    --output ./results \
    --codecs h264,h265,lmd \
    --max-videos 20 \
    --workers 8
```

## Structure des résultats

```
benchmark_output/
├── pipeline_state.json      # État pour la reprise
├── features.json            # Features extraites
├── optimization.json        # Seuils optimisés
├── validation_results.json  # Résultats de validation
├── benchmark_results/
│   ├── benchmark_results.json
│   └── comparison_results.json
└── report/
    ├── report.html
    └── report_data.json
```

## Métriques mesurées

| Métrique | Description |
|----------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) |
| **SSIM** | Structural Similarity Index (0-1) |
| **Ratio** | Taille originale / Taille compressée |
| **Bitrate** | Débit (kbps) |
| **Vitesse** | Frames par seconde (encode/decode) |
| **BD-Rate** | Bjøntegaard Delta Rate (%) |

## Configuration des codecs

Les codecs sont configurés dans `benchmark/config.py` :

```python
'h264': CodecConfig(
    ffmpeg_encoder='libx264',
    crf_range=[18, 23, 28],
    preset='medium'
),
'h265': CodecConfig(
    ffmpeg_encoder='libx265',
    crf_range=[18, 23, 28],
    preset='medium'
),
# etc.
```

## Dépannage

### FFmpeg non trouvé

```bash
# Windows
winget install ffmpeg

# Linux
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Erreur d'import

```bash
# Vérifier l'installation
python check_benchmark_setup.py

# Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall
```

### Mémoire insuffisante

```bash
# Réduire le nombre de vidéos
python run_full_benchmark.py --max-videos 5

# Ou utiliser le mode rapide
python run_full_benchmark.py --quick
```

## Architecture des modules

```
benchmark/
├── config.py          # Configuration globale
├── runner.py          # Orchestrateur principal
├── results.py         # Stockage des résultats
├── datasets/
│   ├── downloader.py  # Téléchargement Xiph/CDVL/Vimeo
│   ├── splitter.py    # Train/test split
│   └── manifest.py    # Gestion des métadonnées
├── codecs/
│   ├── base.py        # Interface abstraite
│   ├── ffmpeg_wrapper.py  # H.264, H.265, VP9, AV1
│   └── lmd_wrapper.py     # LMD-PPV
└── metrics/
    ├── quality.py     # PSNR, SSIM
    ├── performance.py # Vitesse, BD-Rate
    └── comparator.py  # Comparaison multi-codecs

optimization/
├── threshold_config.py   # Configuration des seuils
├── objective.py          # Fonction objectif
├── grid_search.py        # Recherche exhaustive
└── exhaustive_search.py  # Recherche du cartouche optimal

validation/
├── predictor_vs_optimal.py  # Comparaison prédit/optimal
├── accuracy_metrics.py      # Métriques de précision
└── confusion_matrix.py      # Matrices de confusion

reports/
├── generator.py   # Générateur principal
├── charts.py      # Graphiques matplotlib
├── tables.py      # Tableaux formatés
└── statistics.py  # Calculs statistiques
```

## Références

- J. Nembé, "Codage LMD Versatile pour Classes de Processus Ponctuels", v6.0
- Xiph.org Video Test Media: https://media.xiph.org/video/derf/
- BD-Rate calculation: Bjøntegaard, "Calculation of average PSNR differences between RD-curves"
