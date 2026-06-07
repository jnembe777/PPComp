"""
io_utils.py - Utilitaires d'entrée/sortie binaire
==================================================

Classes pour lecture/écriture de bits:
- BitWriter: écriture séquentielle de bits
- BitReader: lecture séquentielle de bits

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import List, Optional, Union
from io import BytesIO


class BitWriter:
    """
    Écrivain de bits séquentiel optimisé.

    Accumule les bits et les flush en octets.
    Version optimisée avec écriture par blocs.

    Usage:
        writer = BitWriter()
        writer.write_bits(0b101, 3)  # écrit 101
        writer.write_bit(1)          # écrit 1
        data = writer.get_bytes()
    """

    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.bit_position = 0  # 0-7, position dans l'octet courant

    def write_bit(self, bit: int):
        """Écrit un seul bit (0 ou 1)."""
        if bit:
            self.current_byte |= (1 << (7 - self.bit_position))
        self.bit_position += 1
        if self.bit_position == 8:
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bit_position = 0

    def write_bits(self, value: int, n_bits: int):
        """
        Écrit n_bits de value (MSB first).
        Optimisé pour écrire des octets complets quand possible.

        Args:
            value: Valeur à écrire
            n_bits: Nombre de bits à écrire
        """
        if n_bits == 0:
            return

        # Optimisation: si aligné et >= 8 bits, écrit des octets
        if self.bit_position == 0 and n_bits >= 8:
            # Écrit les octets complets directement
            while n_bits >= 8:
                n_bits -= 8
                self.buffer.append((value >> n_bits) & 0xFF)
            # Reste à écrire
            if n_bits > 0:
                for i in range(n_bits - 1, -1, -1):
                    self.write_bit((value >> i) & 1)
        else:
            # Fallback bit par bit
            for i in range(n_bits - 1, -1, -1):
                self.write_bit((value >> i) & 1)

    def write_byte(self, byte: int):
        """Écrit un octet complet."""
        self.write_bits(byte & 0xFF, 8)

    def write_uint16(self, value: int):
        """Écrit un uint16 (big-endian)."""
        self.write_bits(value & 0xFFFF, 16)

    def write_uint32(self, value: int):
        """Écrit un uint32 (big-endian)."""
        self.write_bits(value & 0xFFFFFFFF, 32)

    def write_elias_delta(self, n: int):
        """
        Écrit un entier avec le code δ d'Elias.

        Args:
            n: Entier ≥ 1
        """
        if n < 1:
            n = 1

        # Calcul de la longueur
        k = int(np.floor(np.log2(n))) if n >= 1 else 0
        len_k = int(np.floor(np.log2(k + 1))) if k >= 0 else 0

        # Écriture: len_k zéros + (k+1) en binaire + n sans le MSB
        for _ in range(len_k):
            self.write_bit(0)
        self.write_bits(k + 1, len_k + 1)
        if k > 0:
            self.write_bits(n, k)  # n sans le MSB (implicite)

    def write_elias_gamma(self, n: int):
        """
        Écrit un entier avec le code γ d'Elias.

        Args:
            n: Entier ≥ 1
        """
        if n < 1:
            n = 1

        k = int(np.floor(np.log2(n))) if n >= 1 else 0

        # k zéros suivis de n en binaire (k+1 bits)
        for _ in range(k):
            self.write_bit(0)
        self.write_bits(n, k + 1)

    def flush(self):
        """Force l'écriture de l'octet courant (padding avec des 0)."""
        if self.bit_position > 0:
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bit_position = 0

    def get_bytes(self) -> bytes:
        """Retourne les données sous forme de bytes."""
        self.flush()
        return bytes(self.buffer)

    def get_bit_length(self) -> int:
        """Retourne le nombre total de bits écrits."""
        return len(self.buffer) * 8 + self.bit_position

    def reset(self):
        """Réinitialise le writer."""
        self.buffer = bytearray()
        self.current_byte = 0
        self.bit_position = 0


class BitReader:
    """
    Lecteur de bits séquentiel.

    Usage:
        reader = BitReader(data)
        value = reader.read_bits(3)  # lit 3 bits
        bit = reader.read_bit()      # lit 1 bit
    """

    def __init__(self, data: bytes):
        self.data = data
        self.byte_position = 0
        self.bit_position = 0  # 0-7

    def read_bit(self) -> int:
        """Lit un seul bit."""
        if self.byte_position >= len(self.data):
            raise EOFError("End of data reached")

        bit = (self.data[self.byte_position] >> (7 - self.bit_position)) & 1
        self.bit_position += 1

        if self.bit_position == 8:
            self.byte_position += 1
            self.bit_position = 0

        return bit

    def read_bits(self, n_bits: int) -> int:
        """
        Lit n_bits et retourne la valeur (MSB first).

        Args:
            n_bits: Nombre de bits à lire

        Returns:
            Valeur lue
        """
        value = 0
        for _ in range(n_bits):
            value = (value << 1) | self.read_bit()
        return value

    def read_byte(self) -> int:
        """Lit un octet."""
        return self.read_bits(8)

    def read_uint16(self) -> int:
        """Lit un uint16 (big-endian)."""
        return self.read_bits(16)

    def read_uint32(self) -> int:
        """Lit un uint32 (big-endian)."""
        return self.read_bits(32)

    def read_elias_delta(self) -> int:
        """
        Lit un entier encodé avec le code δ d'Elias.

        Returns:
            Entier décodé
        """
        # Compte les zéros initiaux
        len_k = 0
        while self.read_bit() == 0:
            len_k += 1

        # Lit k+1
        k_plus_1 = 1  # Le premier 1 est déjà lu
        for _ in range(len_k):
            k_plus_1 = (k_plus_1 << 1) | self.read_bit()
        k = k_plus_1 - 1

        # Lit n
        if k == 0:
            return 1
        n = 1  # MSB implicite
        for _ in range(k):
            n = (n << 1) | self.read_bit()

        return n

    def read_elias_gamma(self) -> int:
        """
        Lit un entier encodé avec le code γ d'Elias.

        Returns:
            Entier décodé
        """
        # Compte les zéros initiaux
        k = 0
        while self.read_bit() == 0:
            k += 1

        # Lit n (k+1 bits, le premier 1 est déjà lu)
        n = 1
        for _ in range(k):
            n = (n << 1) | self.read_bit()

        return n

    def bits_remaining(self) -> int:
        """Retourne le nombre de bits restants."""
        total_bits = len(self.data) * 8
        read_bits = self.byte_position * 8 + self.bit_position
        return total_bits - read_bits

    def is_eof(self) -> bool:
        """Vérifie si fin des données."""
        return self.byte_position >= len(self.data)

    def reset(self):
        """Réinitialise le reader au début."""
        self.byte_position = 0
        self.bit_position = 0


class BitstreamBuffer:
    """
    Buffer pour construction de bitstream avec sections.

    Permet d'organiser le bitstream en sections (header, data, etc.)
    """

    def __init__(self):
        self.sections = {}
        self.current_section = "default"
        self.writers = {"default": BitWriter()}

    def set_section(self, name: str):
        """Change la section courante."""
        if name not in self.writers:
            self.writers[name] = BitWriter()
        self.current_section = name

    def write_bits(self, value: int, n_bits: int):
        """Écrit dans la section courante."""
        self.writers[self.current_section].write_bits(value, n_bits)

    def write_bit(self, bit: int):
        """Écrit un bit dans la section courante."""
        self.writers[self.current_section].write_bit(bit)

    def get_section_bytes(self, name: str) -> bytes:
        """Retourne les bytes d'une section."""
        if name in self.writers:
            return self.writers[name].get_bytes()
        return b""

    def get_all_bytes(self, order: List[str] = None) -> bytes:
        """
        Retourne tous les bytes dans l'ordre spécifié.

        Args:
            order: Liste des noms de sections dans l'ordre

        Returns:
            Bytes concaténés
        """
        if order is None:
            order = list(self.writers.keys())

        result = bytearray()
        for name in order:
            if name in self.writers:
                result.extend(self.writers[name].get_bytes())

        return bytes(result)

    def get_total_bits(self) -> int:
        """Retourne le nombre total de bits."""
        return sum(w.get_bit_length() for w in self.writers.values())
