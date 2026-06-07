"""
Bitstream I/O — Lecture et écriture bit à bit.

Le BitWriter accumule les bits dans un buffer et les sérialise en bytes.
Le BitReader fait l'inverse pour le décodeur.
"""

import struct
from typing import List, Optional


class BitWriter:
    """Écrit des bits dans un flux binaire (MSB first)."""

    def __init__(self):
        self._buffer: int = 0       # bits en attente
        self._bits_in_buf: int = 0  # nombre de bits dans le buffer
        self._bytes: bytearray = bytearray()

    # ── primitives ──────────────────────────────────────────────

    def write_bits(self, value: int, n_bits: int) -> None:
        """Écrit les n_bits bits de poids faible de value (MSB first)."""
        if n_bits <= 0:
            return
        if value < 0:
            raise ValueError(f"write_bits: valeur négative {value}")
        mask = (1 << n_bits) - 1
        value &= mask
        self._buffer = (self._buffer << n_bits) | value
        self._bits_in_buf += n_bits
        self._flush_full_bytes()

    def write_bool(self, flag: bool) -> None:
        """Écrit un seul bit."""
        self.write_bits(1 if flag else 0, 1)

    def write_byte(self, value: int) -> None:
        """Écrit un octet complet."""
        self.write_bits(value & 0xFF, 8)

    def write_uint16(self, value: int) -> None:
        self.write_bits(value & 0xFFFF, 16)

    def write_uint32(self, value: int) -> None:
        self.write_bits(value & 0xFFFFFFFF, 32)

    # ── codage de longueur variable ────────────────────────────

    def write_elias_gamma(self, n: int) -> int:
        """
        Écrit n ≥ 1 en code Elias gamma. Retourne le nombre de bits écrits.
        Codage : floor(log2(n)) zéros, puis la représentation binaire de n.
        """
        if n < 1:
            raise ValueError(f"Elias gamma : n={n} < 1")
        bits_needed = n.bit_length()          # = floor(log2(n)) + 1
        self.write_bits(0, bits_needed - 1)   # zéros préfixes
        self.write_bits(n, bits_needed)        # valeur
        return 2 * bits_needed - 1

    def write_elias_delta(self, n: int) -> int:
        """Écrit n ≥ 1 en code Elias delta. Retourne le nombre de bits écrits."""
        if n < 1:
            raise ValueError(f"Elias delta : n={n} < 1")
        L = n.bit_length()          # floor(log2(n)) + 1
        len_L = L.bit_length()      # floor(log2(L)) + 1
        total = 0
        # Écrire L en Elias gamma
        self.write_bits(0, len_L - 1)
        total += len_L - 1
        self.write_bits(L, len_L)
        total += len_L
        # Écrire les (L-1) bits de poids faible de n
        if L > 1:
            self.write_bits(n, L - 1)
            total += L - 1
        return total

    # ── finalisation ───────────────────────────────────────────

    def flush(self) -> bytes:
        """Aligne sur l'octet (padding de zéros) et retourne le flux."""
        if self._bits_in_buf > 0:
            pad = 8 - self._bits_in_buf
            self._buffer <<= pad
            self._bytes.append(self._buffer & 0xFF)
            self._buffer = 0
            self._bits_in_buf = 0
        return bytes(self._bytes)

    @property
    def total_bits(self) -> int:
        return len(self._bytes) * 8 + self._bits_in_buf

    # ── interne ────────────────────────────────────────────────

    def _flush_full_bytes(self) -> None:
        while self._bits_in_buf >= 8:
            self._bits_in_buf -= 8
            byte = (self._buffer >> self._bits_in_buf) & 0xFF
            self._bytes.append(byte)
            self._buffer &= (1 << self._bits_in_buf) - 1


class BitReader:
    """Lit des bits depuis un flux binaire (MSB first)."""

    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos: int = 0
        self._bit_pos: int = 0   # position dans l'octet courant (7 = MSB)
        self._total_bits = len(data) * 8

    def read_bits(self, n_bits: int) -> int:
        """Lit n_bits bits et retourne la valeur."""
        value = 0
        for _ in range(n_bits):
            if self._byte_pos >= len(self._data):
                raise EOFError("Fin du flux binaire")
            byte = self._data[self._byte_pos]
            bit = (byte >> (7 - self._bit_pos)) & 1
            value = (value << 1) | bit
            self._bit_pos += 1
            if self._bit_pos >= 8:
                self._bit_pos = 0
                self._byte_pos += 1
        return value

    def read_bool(self) -> bool:
        return self.read_bits(1) == 1

    def read_byte(self) -> int:
        return self.read_bits(8)

    def read_uint16(self) -> int:
        return self.read_bits(16)

    def read_uint32(self) -> int:
        return self.read_bits(32)

    def read_elias_gamma(self) -> int:
        """Lit un entier codé en Elias gamma."""
        n_zeros = 0
        while self.read_bits(1) == 0:
            n_zeros += 1
        value = 1 << n_zeros
        if n_zeros > 0:
            value |= self.read_bits(n_zeros)
        return value

    def read_elias_delta(self) -> int:
        """Lit un entier codé en Elias delta."""
        len_zeros = 0
        while self.read_bits(1) == 0:
            len_zeros += 1
        L = (1 << len_zeros)
        if len_zeros > 0:
            L |= self.read_bits(len_zeros)
        if L > 1:
            value = (1 << (L - 1)) | self.read_bits(L - 1)
        else:
            value = 1
        return value

    @property
    def bits_remaining(self) -> int:
        return self._total_bits - (self._byte_pos * 8 + self._bit_pos)
