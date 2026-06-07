"""
huffman.py - Codec Huffman pour le mode Bc
==========================================

Implémentation du codage de Huffman pour la dimension B (mode Bc).

C_color(Bc) = N·H(P̂) + D_huf

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import heapq

from ..utils.io_utils import BitWriter, BitReader


@dataclass(order=True)
class HuffmanNode:
    """Nœud de l'arbre de Huffman."""
    freq: float
    symbol: Optional[int] = field(default=None, compare=False)
    left: Optional['HuffmanNode'] = field(default=None, compare=False)
    right: Optional['HuffmanNode'] = field(default=None, compare=False)

    @property
    def is_leaf(self) -> bool:
        return self.symbol is not None


class HuffmanCodec:
    """
    Codec Huffman pour le codage des couleurs (mode Bc).

    Attributes:
        codes: Dictionnaire symbol -> code binaire (str)
        lengths: Dictionnaire symbol -> longueur du code
        tree: Racine de l'arbre de Huffman
    """

    def __init__(self):
        self.codes: Dict[int, str] = {}
        self.lengths: Dict[int, int] = {}
        self.tree: Optional[HuffmanNode] = None
        self.m: int = 0  # Nombre de symboles

    def build_from_distribution(self, distribution: Dict[int, float]):
        """
        Construit l'arbre et les codes depuis une distribution.

        Args:
            distribution: Dict symbol -> probabilité (ou fréquence)
        """
        if not distribution:
            return

        self.m = max(distribution.keys()) + 1

        # Création des feuilles
        heap = []
        for symbol, freq in distribution.items():
            if freq > 0:
                node = HuffmanNode(freq=freq, symbol=symbol)
                heapq.heappush(heap, node)

        # Cas spécial: un seul symbole
        if len(heap) == 1:
            node = heapq.heappop(heap)
            self.tree = HuffmanNode(freq=node.freq, left=node)
            self.codes[node.symbol] = "0"
            self.lengths[node.symbol] = 1
            return

        # Construction de l'arbre
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(
                freq=left.freq + right.freq,
                left=left,
                right=right
            )
            heapq.heappush(heap, merged)

        self.tree = heap[0] if heap else None

        # Génération des codes
        self._generate_codes(self.tree, "")

    def build_from_counts(self, counts: np.ndarray):
        """
        Construit depuis un vecteur de comptages.

        Args:
            counts: Vecteur de comptages par symbole
        """
        total = counts.sum()
        if total == 0:
            return

        distribution = {}
        for i, count in enumerate(counts):
            if count > 0:
                distribution[i] = count / total

        self.build_from_distribution(distribution)

    def _generate_codes(self, node: Optional[HuffmanNode], prefix: str):
        """Génère récursivement les codes."""
        if node is None:
            return

        if node.is_leaf:
            self.codes[node.symbol] = prefix if prefix else "0"
            self.lengths[node.symbol] = len(prefix) if prefix else 1
        else:
            self._generate_codes(node.left, prefix + "0")
            self._generate_codes(node.right, prefix + "1")

    def encode_symbol(self, symbol: int) -> str:
        """Encode un symbole en code binaire."""
        return self.codes.get(symbol, "")

    def encode_sequence(self, symbols: np.ndarray, writer: BitWriter):
        """
        Encode une séquence de symboles.

        Args:
            symbols: Tableau de symboles
            writer: BitWriter pour écrire les bits
        """
        for symbol in symbols:
            code = self.codes.get(int(symbol), "")
            for bit in code:
                writer.write_bit(int(bit))

    def decode_symbol(self, reader: BitReader) -> int:
        """
        Décode un symbole depuis un flux de bits.

        Args:
            reader: BitReader pour lire les bits

        Returns:
            Symbole décodé
        """
        if self.tree is None:
            raise ValueError("Huffman tree not built")

        node = self.tree

        while not node.is_leaf:
            bit = reader.read_bit()
            if bit == 0:
                node = node.left
            else:
                node = node.right

            if node is None:
                raise ValueError("Invalid Huffman code")

        return node.symbol

    def decode_sequence(self, reader: BitReader, n_symbols: int) -> np.ndarray:
        """
        Décode n symboles depuis un flux de bits.

        Args:
            reader: BitReader pour lire les bits
            n_symbols: Nombre de symboles à décoder

        Returns:
            Tableau de symboles décodés
        """
        symbols = np.zeros(n_symbols, dtype=int)
        for i in range(n_symbols):
            symbols[i] = self.decode_symbol(reader)
        return symbols

    def get_average_length(self, distribution: Dict[int, float]) -> float:
        """
        Calcule la longueur moyenne du code.

        L_avg = Σ p_i · l_i

        Args:
            distribution: Distribution des symboles

        Returns:
            Longueur moyenne en bits
        """
        total = 0.0
        for symbol, prob in distribution.items():
            if symbol in self.lengths:
                total += prob * self.lengths[symbol]
        return total

    def get_dictionary_overhead(self) -> int:
        """
        Calcule D_huf - overhead du dictionnaire.

        D_huf ≈ m · (⌊log₂m⌋ + 1) bits

        Returns:
            Overhead en bits
        """
        if self.m <= 1:
            return 0
        return self.m * (int(np.floor(np.log2(self.m))) + 1)

    def serialize_dictionary(self, writer: BitWriter):
        """
        Sérialise le dictionnaire Huffman.

        Format: m(16b) + pour chaque symbole: longueur(8b) + code
        """
        writer.write_uint16(self.m)

        for symbol in range(self.m):
            if symbol in self.codes:
                code = self.codes[symbol]
                writer.write_byte(len(code))
                for bit in code:
                    writer.write_bit(int(bit))
            else:
                writer.write_byte(0)

    def deserialize_dictionary(self, reader: BitReader):
        """
        Désérialise le dictionnaire Huffman.

        Args:
            reader: BitReader pour lire les données
        """
        self.m = reader.read_uint16()
        self.codes = {}
        self.lengths = {}

        for symbol in range(self.m):
            length = reader.read_byte()
            if length > 0:
                code = ""
                for _ in range(length):
                    code += str(reader.read_bit())
                self.codes[symbol] = code
                self.lengths[symbol] = length

        # Reconstruire l'arbre depuis les codes
        self._rebuild_tree()

    def _rebuild_tree(self):
        """Reconstruit l'arbre depuis les codes."""
        self.tree = HuffmanNode(freq=0)

        for symbol, code in self.codes.items():
            node = self.tree
            for bit in code[:-1]:
                if bit == '0':
                    if node.left is None:
                        node.left = HuffmanNode(freq=0)
                    node = node.left
                else:
                    if node.right is None:
                        node.right = HuffmanNode(freq=0)
                    node = node.right

            # Dernier bit -> feuille
            if code[-1] == '0':
                node.left = HuffmanNode(freq=0, symbol=symbol)
            else:
                node.right = HuffmanNode(freq=0, symbol=symbol)

    def verify_prefix_free(self) -> bool:
        """
        Vérifie que le code est sans préfixe.

        Returns:
            True si le code est valide
        """
        codes = list(self.codes.values())
        for i, code1 in enumerate(codes):
            for j, code2 in enumerate(codes):
                if i != j and code2.startswith(code1):
                    return False
        return True


def huffman_threshold(m: int, H: float, D_huf: int = None) -> float:
    """
    Calcule N* - seuil au-delà duquel Huffman bat Uniforme.

    N* = D_huf / (log₂m - H)

    Args:
        m: Nombre de couleurs
        H: Entropie de la distribution
        D_huf: Overhead du dictionnaire (calculé si None)

    Returns:
        N* seuil
    """
    if D_huf is None:
        D_huf = m * (int(np.floor(np.log2(m))) + 1) if m > 1 else 0

    delta = np.log2(m) - H if m > 1 else 0
    if delta <= 0:
        return float('inf')

    return D_huf / delta


def compare_huffman_vs_uniform(N: int, m: int, H: float) -> Dict[str, float]:
    """
    Compare les coûts Huffman (Bc) vs Uniforme (Bb).

    Args:
        N: Nombre de symboles
        m: Nombre de couleurs
        H: Entropie

    Returns:
        Dict avec coûts et gains
    """
    log2_m = np.log2(m) if m > 1 else 0
    D_huf = m * (int(np.floor(log2_m)) + 1) if m > 1 else 0

    cost_Bb = N * log2_m
    cost_Bc = N * H + D_huf
    gain = cost_Bb - cost_Bc
    N_star = huffman_threshold(m, H, D_huf)

    return {
        "cost_Bb": cost_Bb,
        "cost_Bc": cost_Bc,
        "gain": gain,
        "gain_percent": (gain / cost_Bb * 100) if cost_Bb > 0 else 0,
        "N_star": N_star,
        "huffman_better": N > N_star
    }
