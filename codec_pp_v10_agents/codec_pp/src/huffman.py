"""
Huffman — Codage de Huffman canonique adaptatif par GOP.

Stratégie :
  1. Passe 1 sur le GOP → collecter les fréquences des symboles couleur
  2. Construire l'arbre de Huffman → table canonique
  3. Sérialiser la table dans le bitstream (compacte)
  4. Passe 2 → encoder chaque couleur avec son code variable

La table canonique est compacte à sérialiser : on stocke seulement
les longueurs de code triées, le décodeur reconstruit les codes.

Format de la table dans le bitstream :
  - nb_symbols      : Elias gamma (≥ 1)
  - Pour chaque symbole (trié par longueur puis par valeur) :
      symbol_value   : color_bits bits (largeur fixe)
      code_length    : 5 bits (longueurs 1..31)
"""

import heapq
from typing import Dict, List, Tuple, Optional
from collections import Counter

from .bitstream import BitWriter, BitReader


# ═══════════════════════════════════════════════════════════════
#  CONSTRUCTION DE L'ARBRE
# ═══════════════════════════════════════════════════════════════

class HuffmanNode:
    """Nœud de l'arbre de Huffman."""
    __slots__ = ('symbol', 'freq', 'left', 'right')

    def __init__(self, symbol: Optional[int], freq: int,
                 left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        # Tie-break sur le symbole pour déterminisme
        if self.freq == other.freq:
            s1 = self.symbol if self.symbol is not None else -1
            s2 = other.symbol if other.symbol is not None else -1
            return s1 < s2
        return self.freq < other.freq


def build_huffman_tree(freq: Dict[int, int]) -> Optional[HuffmanNode]:
    """Construit l'arbre de Huffman depuis un dictionnaire de fréquences."""
    if not freq:
        return None
    if len(freq) == 1:
        sym, f = next(iter(freq.items()))
        # Un seul symbole → code de longueur 1
        return HuffmanNode(None, f,
                           left=HuffmanNode(sym, f),
                           right=HuffmanNode(sym, f))

    heap = [HuffmanNode(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq,
                             left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0]


def _extract_lengths(node: HuffmanNode, depth: int,
                     lengths: Dict[int, int]) -> None:
    """Parcours récursif pour extraire les longueurs de code."""
    if node.symbol is not None:
        lengths[node.symbol] = max(depth, 1)
        return
    if node.left:
        _extract_lengths(node.left, depth + 1, lengths)
    if node.right:
        _extract_lengths(node.right, depth + 1, lengths)


# ═══════════════════════════════════════════════════════════════
#  TABLE CANONIQUE
# ═══════════════════════════════════════════════════════════════

class HuffmanTable:
    """
    Table de Huffman canonique.

    Les codes sont assignés de manière canonique :
    - Triés par longueur croissante, puis par valeur de symbole
    - Le premier code de longueur L est obtenu en incrémentant
      le dernier code de longueur L-1 et en décalant à gauche
    """

    def __init__(self):
        self.encode_map: Dict[int, Tuple[int, int]] = {}  # symbol → (code, length)
        self.decode_map: Dict[Tuple[int, int], int] = {}   # (code, length) → symbol
        self.symbols: List[Tuple[int, int]] = []  # [(symbol, length), ...] trié
        self.max_length: int = 0

    @classmethod
    def from_frequencies(cls, freq: Dict[int, int]) -> 'HuffmanTable':
        """Construit la table canonique depuis les fréquences."""
        table = cls()
        if not freq:
            return table

        # Construire l'arbre
        tree = build_huffman_tree(freq)
        if tree is None:
            return table

        # Extraire les longueurs
        lengths: Dict[int, int] = {}
        _extract_lengths(tree, 0, lengths)

        # Limiter la profondeur max à 24 bits
        max_allowed = 24
        for sym in lengths:
            if lengths[sym] > max_allowed:
                lengths[sym] = max_allowed

        # Trier : par longueur, puis par valeur de symbole
        sorted_syms = sorted(lengths.items(), key=lambda x: (x[1], x[0]))
        table.symbols = sorted_syms

        # Assigner les codes canoniques
        code = 0
        prev_length = 0
        for sym, length in sorted_syms:
            if prev_length > 0:
                code += 1
                code <<= (length - prev_length)
            table.encode_map[sym] = (code, length)
            table.decode_map[(code, length)] = sym
            prev_length = length
            table.max_length = max(table.max_length, length)

        return table

    def encode_symbol(self, writer: BitWriter, symbol: int) -> int:
        """Encode un symbole. Retourne le nombre de bits écrits."""
        if symbol not in self.encode_map:
            raise ValueError(f"Symbole {symbol} absent de la table Huffman")
        code, length = self.encode_map[symbol]
        writer.write_bits(code, length)
        return length

    def decode_symbol(self, reader: BitReader) -> int:
        """Décode un symbole depuis le flux."""
        code = 0
        for length in range(1, self.max_length + 1):
            code = (code << 1) | reader.read_bits(1)
            if (code, length) in self.decode_map:
                return self.decode_map[(code, length)]
        raise ValueError("Code Huffman invalide dans le flux")

    # ── Sérialisation ──────────────────────────────────────────

    def write_table(self, writer: BitWriter, color_bits: int) -> int:
        """
        Écrit la table dans le bitstream.
        Format : nb_symbols (Elias gamma) + pour chaque symbole :
                 valeur (color_bits) + longueur de code (5 bits)
        Retourne le nombre de bits écrits.
        """
        n = len(self.symbols)
        if n == 0:
            bits = writer.write_elias_gamma(1)
            # Écrire un symbole factice
            writer.write_bits(0, color_bits)
            writer.write_bits(1, 5)
            return bits + color_bits + 5

        bits = writer.write_elias_gamma(n)
        for sym, length in self.symbols:
            writer.write_bits(sym, color_bits)
            writer.write_bits(length, 5)
            bits += color_bits + 5
        return bits

    @classmethod
    def read_table(cls, reader: BitReader, color_bits: int) -> 'HuffmanTable':
        """Lit la table depuis le bitstream et reconstruit les codes."""
        table = cls()
        n = reader.read_elias_gamma()

        sorted_syms = []
        for _ in range(n):
            sym = reader.read_bits(color_bits)
            length = reader.read_bits(5)
            sorted_syms.append((sym, length))

        table.symbols = sorted_syms

        # Reconstruire les codes canoniques
        code = 0
        prev_length = 0
        for sym, length in sorted_syms:
            if prev_length > 0:
                code += 1
                code <<= (length - prev_length)
            table.encode_map[sym] = (code, length)
            table.decode_map[(code, length)] = sym
            prev_length = length
            table.max_length = max(table.max_length, length)

        return table

    def average_bits(self, freq: Dict[int, int]) -> float:
        """Calcule le nombre moyen de bits par symbole."""
        total_count = sum(freq.values())
        if total_count == 0:
            return 0.0
        total_bits = sum(
            freq.get(sym, 0) * length
            for sym, length in self.symbols
        )
        return total_bits / total_count


# ═══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def collect_color_frequencies(
    gop_M, nl: int, nc: int, gop_r: int,
    data: dict
) -> Dict[int, int]:
    """
    Collecte les fréquences de tous les symboles couleur qui seront
    émis dans le bitstream pour ce GOP.

    On ne collecte que les couleurs des sauts (pas les couleurs
    répétées dans R1), car en mode Huffman on encode uniquement
    les couleurs aux points de saut.
    """
    freq: Dict[int, int] = Counter()

    for i in range(nl):
        for j in range(nc):
            colors = data['CB'][i][j]  # couleurs aux sauts
            for c in colors:
                freq[c] += 1

    return dict(freq)


def huffman_color_bits_estimate(
    table: HuffmanTable, colors: List[int]
) -> int:
    """Estime le nombre de bits Huffman pour une liste de couleurs."""
    total = 0
    for c in colors:
        if c in table.encode_map:
            _, length = table.encode_map[c]
            total += length
        else:
            total += 8  # fallback
    return total
