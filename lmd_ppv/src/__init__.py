"""
LMD-PPV: Codage Vidéo par Processus Ponctuels avec Longueur Minimale de Description
==================================================================================

Framework de compression vidéo adaptative basé sur le principe MDL appliqué
aux processus ponctuels. Développé selon les travaux de J. Nembé.

Version: 6.0 - Cartouche ABCDEFGH (17 bits)

Modules:
    - agents: 9 agents du pipeline (extraction, classification, codage...)
    - core: Structures de données et algorithmes fondamentaux
    - codecs: Encodeurs/décodeurs (Huffman, Elias, arithmétique)
    - utils: Utilitaires (bitwise, mathématiques, I/O)
"""

__version__ = "6.0.0"
__author__ = "J. Nembé"

from .core.cartouche import Cartouche
from .core.process_types import ProcessType, ColorMode, Representation
