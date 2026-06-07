# Codec PPV — Codage Vidéo par Processus Ponctuels Marqués

## Architecture (17 modules, 4 453 lignes Python)

```
codec_pp/
├── src/
│   ├── bitstream.py         # Lecture/écriture bit à bit + Elias gamma/delta
│   ├── combinatorics.py     # Indexation combinatoire (ranking co-lex pour R4)
│   ├── matrices.py          # Construction B, S, MT, CB, CS, CMT depuis M
│   ├── representations.py   # Encodage/décodage R1-R4 (fixe ou Huffman)
│   ├── huffman.py           # Huffman canonique adaptatif par bloc
│   ├── format.py            # Spécification format .ppv (header + Code Méthode)
│   ├── colorspace.py        # RGB ↔ YCbCr BT.601 + sous-échantillonnage 4:2:0
│   ├── prediction.py        # Prédiction intra-bloc (4 modes : Bypass/Left/Top/DC)
│   ├── process_types.py     # Classification MDL (Mono/PP/Spatial/Markov)
│   ├── blocks.py            # Macroblocs : pipeline encodage/décodage par bloc
│   ├── heuristic.py         # Heuristique M6 : arbre de décision sans sklearn
│   ├── encoder.py           # Encodeur principal (multi-plan, GOPs)
│   ├── decoder.py           # Décodeur principal
│   ├── video_io.py          # Chargement/sauvegarde vidéo (OpenCV ou numpy)
│   └── benchmark.py         # Framework de benchmark + vidéos synthétiques
└── tests/
    ├── test_full_pipeline.py # Tests unitaires + roundtrip lossless
    ├── test_heuristic.py     # Entraînement + comparaison exhaustif vs fast
    └── run_benchmark.py      # Benchmark complet
```

## Utilisation

```python
from codec_pp.src.encoder import PPVEncoder
from codec_pp.src.decoder import PPVDecoder
from codec_pp.src.heuristic import DecisionHeuristic

# Mode exhaustif (qualité optimale)
enc = PPVEncoder(gop_size=32, block_size=8, use_huffman=True)
stats = enc.encode(M=video_array, output_path="video.ppv")

# Mode rapide (heuristique M6)
h = DecisionHeuristic.train()
enc_fast = PPVEncoder(heuristic=h)
stats = enc_fast.encode(M=video_array, output_path="video.ppv")

# Décodage
dec = PPVDecoder()
M_decoded, meta = dec.decode("video.ppv")
```

## Résultats benchmark (64×64, 64 frames)

| Vidéo              | Ratio  | bpp  | Lossless | Type MDL |
|--------------------|--------|------|----------|----------|
| Animation (cartoon)| 47.3x  | 0.17 | ✓        | SPATIAL  |
| Fond statique+objet| 15.6x  | 0.51 | ✓        | SPATIAL  |
| Couleur RGB 4:2:0  | 17.6x  | 0.68 | Y:✓      | SPATIAL  |
| Dégradé lent       | 7.8x   | 1.02 | ✓        | SPATIAL  |
| Oscillation        | 2.4x   | 3.36 | ✓        | R1       |
| Bruit modéré       | 1.0x   | 7.64 | ✓        | R1       |

## Heuristique M6

- Précision proc_type : 100%
- Précision pred_mode : 98.6%
- Speedup : 1.15x (Python pur, >10x attendu en C)
- Perte de ratio : 0% sur 6/7 vidéos
```
