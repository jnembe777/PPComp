"""
elias.py - Codes universels d'Elias pour le mode Bd
====================================================

Implémentation des codes γ et δ d'Elias pour la dimension B (mode Bd).

C_color(Bd) = N · L*(m) où L*(n) ≈ log₂n + 2·log₂log₂n

Utilisé quand m n'est pas connu au décodeur a priori.

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import List, Tuple
from ..utils.io_utils import BitWriter, BitReader


class EliasCodec:
    """
    Codec pour les codes universels d'Elias (γ et δ).

    Attributes:
        method: "gamma" ou "delta"
    """

    def __init__(self, method: str = "delta"):
        """
        Initialise le codec.

        Args:
            method: "gamma" ou "delta"
        """
        self.method = method

    @staticmethod
    def gamma_encode(n: int, writer: BitWriter):
        """
        Encode un entier avec le code γ d'Elias.

        Format: k zéros + n en binaire (k+1 bits)
        où k = ⌊log₂n⌋

        Args:
            n: Entier ≥ 1
            writer: BitWriter
        """
        if n < 1:
            n = 1

        k = int(np.floor(np.log2(n))) if n >= 1 else 0

        # k zéros
        for _ in range(k):
            writer.write_bit(0)

        # n en binaire (k+1 bits)
        writer.write_bits(n, k + 1)

    @staticmethod
    def gamma_decode(reader: BitReader) -> int:
        """
        Décode un entier encodé avec le code γ.

        Args:
            reader: BitReader

        Returns:
            Entier décodé
        """
        # Compte les zéros
        k = 0
        while reader.read_bit() == 0:
            k += 1

        # Lit n (le premier 1 est déjà lu)
        n = 1
        for _ in range(k):
            n = (n << 1) | reader.read_bit()

        return n

    @staticmethod
    def gamma_length(n: int) -> int:
        """
        Longueur du code γ pour n.

        L(n) = 2·⌊log₂n⌋ + 1
        """
        if n < 1:
            return 1
        k = int(np.floor(np.log2(n)))
        return 2 * k + 1

    @staticmethod
    def delta_encode(n: int, writer: BitWriter):
        """
        Encode un entier avec le code δ d'Elias.

        Format: γ(k+1) + (n sans MSB)
        où k = ⌊log₂n⌋

        Args:
            n: Entier ≥ 1
            writer: BitWriter
        """
        if n < 1:
            n = 1

        k = int(np.floor(np.log2(n))) if n >= 1 else 0

        # Encode k+1 avec γ
        EliasCodec.gamma_encode(k + 1, writer)

        # n sans le MSB (k bits)
        if k > 0:
            # Masque pour extraire les k bits de poids faible
            mask = (1 << k) - 1
            writer.write_bits(n & mask, k)

    @staticmethod
    def delta_decode(reader: BitReader) -> int:
        """
        Décode un entier encodé avec le code δ.

        Args:
            reader: BitReader

        Returns:
            Entier décodé
        """
        # Décode k+1 avec γ
        k_plus_1 = EliasCodec.gamma_decode(reader)
        k = k_plus_1 - 1

        if k == 0:
            return 1

        # Lit les k bits de poids faible
        n = 1  # MSB implicite
        for _ in range(k):
            n = (n << 1) | reader.read_bit()

        return n

    @staticmethod
    def delta_length(n: int) -> int:
        """
        Longueur du code δ pour n.

        L(n) = 1 + ⌊log₂n⌋ + 2·⌊log₂(1 + ⌊log₂n⌋)⌋
        """
        if n < 1:
            return 1
        k = int(np.floor(np.log2(n)))
        len_k = int(np.floor(np.log2(k + 1))) if k >= 0 else 0
        return 1 + k + 2 * len_k

    def encode(self, n: int, writer: BitWriter):
        """Encode selon la méthode choisie."""
        if self.method == "gamma":
            self.gamma_encode(n, writer)
        else:
            self.delta_encode(n, writer)

    def decode(self, reader: BitReader) -> int:
        """Décode selon la méthode choisie."""
        if self.method == "gamma":
            return self.gamma_decode(reader)
        else:
            return self.delta_decode(reader)

    def length(self, n: int) -> int:
        """Longueur du code selon la méthode choisie."""
        if self.method == "gamma":
            return self.gamma_length(n)
        else:
            return self.delta_length(n)

    def encode_sequence(self, values: np.ndarray, writer: BitWriter):
        """
        Encode une séquence de valeurs.

        Args:
            values: Tableau d'entiers ≥ 0 (sera encodé comme value + 1)
            writer: BitWriter
        """
        for v in values:
            self.encode(int(v) + 1, writer)

    def decode_sequence(self, reader: BitReader, n_values: int) -> np.ndarray:
        """
        Décode une séquence de valeurs.

        Args:
            reader: BitReader
            n_values: Nombre de valeurs à décoder

        Returns:
            Tableau d'entiers
        """
        values = np.zeros(n_values, dtype=int)
        for i in range(n_values):
            values[i] = self.decode(reader) - 1
        return values

    def total_length(self, values: np.ndarray) -> int:
        """
        Calcule la longueur totale pour encoder une séquence.

        Args:
            values: Tableau d'entiers ≥ 0

        Returns:
            Longueur totale en bits
        """
        return sum(self.length(int(v) + 1) for v in values)


def compare_elias_methods(values: np.ndarray) -> dict:
    """
    Compare les méthodes γ et δ pour une séquence.

    Args:
        values: Tableau d'entiers

    Returns:
        Dict avec longueurs et comparaison
    """
    gamma_codec = EliasCodec("gamma")
    delta_codec = EliasCodec("delta")

    len_gamma = gamma_codec.total_length(values)
    len_delta = delta_codec.total_length(values)

    # Comparaison avec uniforme si on connaît m
    m = int(values.max()) + 1 if len(values) > 0 else 1
    len_uniform = len(values) * np.log2(m) if m > 1 else 0

    return {
        "gamma_length": len_gamma,
        "delta_length": len_delta,
        "uniform_length": len_uniform,
        "best_method": "gamma" if len_gamma < len_delta else "delta",
        "savings_vs_uniform": len_uniform - min(len_gamma, len_delta)
    }


# Codes Omega d'Elias (optionnel, pour très grands entiers)
class OmegaCodec:
    """
    Codec pour le code ω d'Elias (récursif).

    Optimal pour les très grands entiers.
    """

    @staticmethod
    def encode(n: int, writer: BitWriter):
        """Encode avec le code ω."""
        if n < 1:
            n = 1

        # Collecte les représentations récursives
        reps = []
        while n > 1:
            reps.append(n)
            n = int(np.floor(np.log2(n)))

        # Écrit en ordre inverse
        for rep in reversed(reps):
            bits = int(np.floor(np.log2(rep))) + 1
            writer.write_bits(rep, bits)

        # Bit de fin
        writer.write_bit(0)

    @staticmethod
    def decode(reader: BitReader) -> int:
        """Décode le code ω."""
        n = 1

        while True:
            bit = reader.read_bit()
            if bit == 0:
                break

            # Lit n+1 bits supplémentaires
            value = 1
            for _ in range(n):
                value = (value << 1) | reader.read_bit()
            n = value

        return n
