"""
universal.py - Codes universels pour entiers
=============================================

Codes pour encoder les paramètres du modèle et les métadonnées.

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import List
from ..utils.io_utils import BitWriter, BitReader


class UniversalCodec:
    """
    Codec pour différents codes universels d'entiers.

    Supporte:
    - Unaire
    - Elias γ / δ / ω
    - Fibonacci
    - Golomb / Rice
    """

    def __init__(self, method: str = "elias_delta"):
        """
        Initialise le codec.

        Args:
            method: "unary", "elias_gamma", "elias_delta", "fibonacci", "golomb_k"
        """
        self.method = method
        self.golomb_k = 4  # Paramètre pour Golomb

    # === Unaire ===
    @staticmethod
    def unary_encode(n: int, writer: BitWriter):
        """n zéros suivis d'un 1."""
        for _ in range(n):
            writer.write_bit(0)
        writer.write_bit(1)

    @staticmethod
    def unary_decode(reader: BitReader) -> int:
        """Décode un code unaire."""
        n = 0
        while reader.read_bit() == 0:
            n += 1
        return n

    @staticmethod
    def unary_length(n: int) -> int:
        return n + 1

    # === Fibonacci ===
    @staticmethod
    def _fibonacci_numbers(max_val: int) -> List[int]:
        """Génère les nombres de Fibonacci jusqu'à max_val."""
        fibs = [1, 2]
        while fibs[-1] < max_val:
            fibs.append(fibs[-1] + fibs[-2])
        return fibs

    @staticmethod
    def fibonacci_encode(n: int, writer: BitWriter):
        """
        Encode avec le code de Fibonacci (Zeckendorf).

        Représentation de n comme somme de Fibonacci non consécutifs,
        terminée par 11.
        """
        if n < 1:
            n = 1

        fibs = UniversalCodec._fibonacci_numbers(n + 1)

        # Représentation de Zeckendorf
        bits = []
        remaining = n
        for f in reversed(fibs):
            if f <= remaining:
                bits.append(1)
                remaining -= f
            else:
                bits.append(0)

        # Inverse et ajoute le terminateur
        bits = bits[::-1]
        while bits and bits[0] == 0:
            bits.pop(0)

        for bit in bits:
            writer.write_bit(bit)
        writer.write_bit(1)  # Terminateur (crée 11)

    @staticmethod
    def fibonacci_decode(reader: BitReader) -> int:
        """Décode un code de Fibonacci."""
        fibs = [1, 2]
        bits = []
        prev_bit = 0

        while True:
            bit = reader.read_bit()
            if bit == 1 and prev_bit == 1:
                break
            bits.append(bit)
            prev_bit = bit

            # Étend les Fibonacci si nécessaire
            if len(bits) >= len(fibs):
                fibs.append(fibs[-1] + fibs[-2])

        # Calcule la valeur
        n = 0
        for i, bit in enumerate(bits):
            if bit == 1:
                n += fibs[i]

        return n

    # === Golomb ===
    def golomb_encode(self, n: int, writer: BitWriter):
        """
        Encode avec le code de Golomb.

        n = q·k + r où q encodé en unaire, r en binaire.
        """
        k = self.golomb_k
        if n < 0:
            n = 0

        q = n // k
        r = n % k

        # q en unaire
        self.unary_encode(q, writer)

        # r en binaire (ceil(log2(k)) bits)
        bits = max(1, int(np.ceil(np.log2(k + 1))))
        writer.write_bits(r, bits)

    def golomb_decode(self, reader: BitReader) -> int:
        """Décode un code de Golomb."""
        k = self.golomb_k

        q = self.unary_decode(reader)
        bits = max(1, int(np.ceil(np.log2(k + 1))))
        r = reader.read_bits(bits)

        return q * k + r

    # === Rice (Golomb avec k = 2^m) ===
    def rice_encode(self, n: int, m: int, writer: BitWriter):
        """Encode avec le code de Rice (k = 2^m)."""
        if n < 0:
            n = 0

        q = n >> m
        r = n & ((1 << m) - 1)

        self.unary_encode(q, writer)
        writer.write_bits(r, m)

    def rice_decode(self, reader: BitReader, m: int) -> int:
        """Décode un code de Rice."""
        q = self.unary_decode(reader)
        r = reader.read_bits(m)
        return (q << m) | r

    # === Interface générique ===
    def encode(self, n: int, writer: BitWriter):
        """Encode selon la méthode configurée."""
        if self.method == "unary":
            self.unary_encode(n, writer)
        elif self.method == "elias_gamma":
            from .elias import EliasCodec
            EliasCodec.gamma_encode(n + 1, writer)
        elif self.method == "elias_delta":
            from .elias import EliasCodec
            EliasCodec.delta_encode(n + 1, writer)
        elif self.method == "fibonacci":
            self.fibonacci_encode(n + 1, writer)
        elif self.method.startswith("golomb"):
            self.golomb_encode(n, writer)
        else:
            from .elias import EliasCodec
            EliasCodec.delta_encode(n + 1, writer)

    def decode(self, reader: BitReader) -> int:
        """Décode selon la méthode configurée."""
        if self.method == "unary":
            return self.unary_decode(reader)
        elif self.method == "elias_gamma":
            from .elias import EliasCodec
            return EliasCodec.gamma_decode(reader) - 1
        elif self.method == "elias_delta":
            from .elias import EliasCodec
            return EliasCodec.delta_decode(reader) - 1
        elif self.method == "fibonacci":
            return self.fibonacci_decode(reader) - 1
        elif self.method.startswith("golomb"):
            return self.golomb_decode(reader)
        else:
            from .elias import EliasCodec
            return EliasCodec.delta_decode(reader) - 1

    def encode_sequence(self, values: np.ndarray, writer: BitWriter):
        """Encode une séquence."""
        for v in values:
            self.encode(int(v), writer)

    def decode_sequence(self, reader: BitReader, n_values: int) -> np.ndarray:
        """Décode une séquence."""
        values = np.zeros(n_values, dtype=int)
        for i in range(n_values):
            values[i] = self.decode(reader)
        return values
