"""
Format — Spécification du format de fichier .ppv (Point Process Video).

Structure du fichier :
══════════════════════

┌──────────────────────────────────────────────────────────────┐
│  HEADER GLOBAL (fixe, 24 octets)                             │
├──────────────────────────────────────────────────────────────┤
│  Magic number      : 4 bytes  "PPV1"                         │
│  Version           : 1 byte   (0x01)                         │
│  Flags             : 1 byte   [COLOR_SPACE:2][RESERVED:6]    │
│  nl (hauteur)      : 2 bytes  uint16                         │
│  nc (largeur)      : 2 bytes  uint16                         │
│  r  (nb frames)    : 2 bytes  uint16                         │
│  color_bits        : 1 byte   (8, 16 ou 24)                  │
│  fps               : 1 byte   frames par seconde             │
│  GOP size          : 2 bytes  uint16 (frames par GOP)        │
│  nb_gops           : 2 bytes  uint16                         │
│  total_bits_body   : 4 bytes  uint32 (taille du corps)       │
│  reserved          : 2 bytes                                 │
├──────────────────────────────────────────────────────────────┤
│  GOP 0                                                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ GOP Header (4 bytes)                                    │  │
│  │  gop_index      : 2 bytes uint16                        │  │
│  │  gop_r          : 2 bytes uint16 (frames dans ce GOP)   │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │ Pixel Data : nl × nc blocs pixel                        │  │
│  │  Pour chaque pixel (i,j) :                              │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │ Code Méthode : 7 bits                             │   │  │
│  │  │   [PROC:2] [REP:2] [COMP:2] [COL:1]             │   │  │
│  │  │ Payload selon REP :                               │   │  │
│  │  │   R1 → r × color_bits couleurs                    │   │  │
│  │  │   R2 → n + n×(date+couleur)                       │   │  │
│  │  │   R3 → r bits + n×couleur                         │   │  │
│  │  │   R4 → n + s + n×couleur                          │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────┘  │
│  GOP 1 ...                                                    │
│  ...                                                          │
└──────────────────────────────────────────────────────────────┘

Code Méthode (7 bits) :
═══════════════════════
  Bits [6:5] — PROC (type de processus ponctuel) :
      00 = Monochromatique (pixel constant sur tout le GOP)
      01 = Processus ponctuel marqué général
      10 = Spatial  (réservé, futur)
      11 = Markovien (réservé, futur)

  Bits [4:3] — REP (représentation choisie) :
      00 = R1 : liste complète (c_1, ..., c_r)
      01 = R2 : dates + couleurs (n, t_1,...,t_n, c_1,...,c_n)
      10 = R3 : booléen + couleurs (b, c_1,...,c_n)
      11 = R4 : indice combinatoire (n, s, c_1,...,c_n)

  Bits [2:1] — COMP (compression entropique appliquée) :
      00 = Aucune
      01 = Huffman (réservé)
      10 = Arithmétique (réservé)
      11 = Réservé

  Bit [0] — COL (espace couleur) :
      0 = Niveaux de gris
      1 = YCbCr / RGB
"""

import struct
from typing import Tuple

# ── Constantes ──────────────────────────────────────────────────

MAGIC = b"PPV1"
VERSION = 0x01

# PROC
PROC_MONO = 0b00
PROC_PP = 0b01       # processus ponctuel marqué général
PROC_SPATIAL = 0b10  # réservé
PROC_MARKOV = 0b11   # réservé

# REP
REP_R1 = 0b00
REP_R2 = 0b01
REP_R3 = 0b10
REP_R4 = 0b11

# COMP
COMP_NONE = 0b00
COMP_HUFFMAN = 0b01
COMP_ARITHMETIC = 0b10

# COL
COL_GRAY = 0
COL_COLOR = 1

# Noms pour affichage
PROC_NAMES = {0: "Mono", 1: "PP", 2: "Spatial", 3: "Markov"}
REP_NAMES = {0: "R1", 1: "R2", 2: "R3", 3: "R4"}
COMP_NAMES = {0: "None", 1: "Huffman", 2: "Arithmetic", 3: "Reserved"}


def make_code_methode(proc: int, rep: int, comp: int, col: int) -> int:
    """Assemble le Code Méthode 7 bits."""
    return ((proc & 0x3) << 5) | ((rep & 0x3) << 3) | ((comp & 0x3) << 1) | (col & 0x1)


def parse_code_methode(cm: int) -> Tuple[int, int, int, int]:
    """Décompose le Code Méthode en (proc, rep, comp, col)."""
    proc = (cm >> 5) & 0x3
    rep = (cm >> 3) & 0x3
    comp = (cm >> 1) & 0x3
    col = cm & 0x1
    return proc, rep, comp, col


def encode_header(
    nl: int, nc: int, r: int, color_bits: int,
    fps: int, gop_size: int, nb_gops: int, total_bits_body: int,
    color_space: int = 0
) -> bytes:
    """Encode le header global (24 octets)."""
    flags = (color_space & 0x3) << 6
    header = struct.pack(
        ">4sBBHHHBBHHIH",
        MAGIC,                  # 4B magic
        VERSION,                # 1B version
        flags,                  # 1B flags
        nl,                     # 2B hauteur
        nc,                     # 2B largeur
        r,                      # 2B nb frames total
        color_bits,             # 1B bits par couleur
        fps,                    # 1B fps
        gop_size,               # 2B frames par GOP
        nb_gops,                # 2B nombre de GOPs
        total_bits_body,        # 4B taille du corps en bits
        0,                      # 2B réservé
    )
    return header


def decode_header(data: bytes) -> dict:
    """Décode le header global (24 octets)."""
    fields = struct.unpack(">4sBBHHHBBHHIH", data[:24])
    magic, version, flags, nl, nc, r, color_bits, fps, gop_size, nb_gops, total_bits, _ = fields
    if magic != MAGIC:
        raise ValueError(f"Magic number invalide : {magic}")
    return {
        'version': version,
        'color_space': (flags >> 6) & 0x3,
        'nl': nl,
        'nc': nc,
        'r': r,
        'color_bits': color_bits,
        'fps': fps,
        'gop_size': gop_size,
        'nb_gops': nb_gops,
        'total_bits_body': total_bits,
    }


HEADER_SIZE = 24
