"""
agent_6_encoder.py - Agent Codeur (Dimension D)
================================================

Phase 3 - Dimension D

Fonctions:
- Header ABCDEFGH 17 bits
- encode_block() / decode_block()
- MDL 3 étapes (sélection, estimation, codage)

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from ..core.cartouche import Cartouche
from ..core.process_types import (
    ProcessType, ColorMode, Representation, CompressionMode, IntensityFamily
)
from ..core.features import BlockFeatures
from ..codecs.huffman import HuffmanCodec
from ..codecs.elias import EliasCodec
from ..codecs.arithmetic import ArithmeticCodec
from ..utils.io_utils import BitWriter, BitReader
from ..utils.math_utils import logC


@dataclass
class EncodedBlock:
    """Bloc encodé."""
    cartouche: Cartouche
    header_bits: int
    data_bits: int
    total_bits: int
    bitstream: bytes


class EncoderAgent:
    """
    Agent 6: Codeur avec Header ABCDEFGH

    Encode les blocs vidéo avec le cartouche optimal.
    """

    def __init__(self):
        self.huffman_codec = HuffmanCodec()
        self.elias_codec = EliasCodec("delta")
        self.arithmetic_codec = ArithmeticCodec()

    def encode_block(
        self,
        jump_times: np.ndarray,
        marks: np.ndarray,
        cartouche: Cartouche,
        features: BlockFeatures,
        color_dist: Optional[Dict[int, float]] = None
    ) -> EncodedBlock:
        """
        Encode un bloc vidéo complet.

        Format du bitstream:
        1. Header ABCDEFGH (17 bits)
        2. Données temporelles (selon dim. C)
        3. Données couleur (selon dim. B)

        Args:
            jump_times: Temps des sauts
            marks: Marques couleur
            cartouche: Configuration ABCDEFGH
            features: Caractéristiques
            color_dist: Distribution couleur (pour Huffman)

        Returns:
            EncodedBlock
        """
        writer = BitWriter()

        # 1. Header ABCDEFGH (17 bits)
        header = cartouche.encode()
        writer.write_bits(header, 17)
        header_bits = 17

        # 2. Métadonnées (N, r, m)
        N = len(jump_times)
        writer.write_uint16(N)
        writer.write_uint16(features.r)
        writer.write_byte(features.m)

        # 3. Données temporelles selon C
        temporal_bits_start = writer.get_bit_length()
        self._encode_temporal(jump_times, cartouche.C, features.r, writer)
        temporal_bits = writer.get_bit_length() - temporal_bits_start

        # 4. Données couleur selon B (sauf si monochromatique)
        color_bits_start = writer.get_bit_length()
        if cartouche.A != ProcessType.MONOCHROMATIC and len(marks) > 0:
            self._encode_colors(marks, cartouche.B, features.m, color_dist, writer)
        color_bits = writer.get_bit_length() - color_bits_start

        bitstream = writer.get_bytes()
        total_bits = header_bits + 40 + temporal_bits + color_bits  # 40 = metadata

        return EncodedBlock(
            cartouche=cartouche,
            header_bits=header_bits,
            data_bits=temporal_bits + color_bits,
            total_bits=total_bits,
            bitstream=bitstream
        )

    def decode_block(
        self,
        bitstream: bytes
    ) -> Tuple[Cartouche, np.ndarray, np.ndarray, BlockFeatures]:
        """
        Décode un bloc vidéo.

        Args:
            bitstream: Données encodées

        Returns:
            (cartouche, jump_times, marks, features)
        """
        reader = BitReader(bitstream)

        # 1. Header ABCDEFGH
        header = reader.read_bits(17)
        cartouche = Cartouche.decode(header)

        # 2. Métadonnées
        N = reader.read_uint16()
        r = reader.read_uint16()
        m = reader.read_byte()

        # 3. Données temporelles
        jump_times = self._decode_temporal(reader, cartouche.C, N, r)

        # 4. Données couleur
        if cartouche.A != ProcessType.MONOCHROMATIC and N > 0:
            marks = self._decode_colors(reader, cartouche.B, N, m)
        else:
            marks = np.zeros(N, dtype=int)

        features = BlockFeatures(N=N, r=r, m=m)

        return cartouche, jump_times, marks, features

    def _encode_temporal(
        self,
        jump_times: np.ndarray,
        repr_type: int,
        r: int,
        writer: BitWriter
    ):
        """Encode les données temporelles selon la représentation."""
        N = len(jump_times)

        if repr_type == Representation.TIMESTAMPS:
            # R1: Liste des temps
            bits_per_time = int(np.ceil(np.log2(max(2, r))))
            for t in jump_times:
                writer.write_bits(int(t) % r, bits_per_time)

        elif repr_type == Representation.COUNT:
            # R2: Histogramme
            counts = np.bincount(jump_times.astype(int) % r, minlength=r)
            max_count = max(counts.max(), 1)
            bits_per_count = int(np.ceil(np.log2(max_count + 1)))
            writer.write_byte(bits_per_count)
            for count in counts:
                writer.write_bits(int(count), bits_per_count)

        elif repr_type == Representation.INTERVALS:
            # R3: Intervalles
            if N > 0:
                sorted_times = np.sort(jump_times)
                intervals = np.diff(sorted_times)
                bits_per_interval = int(np.ceil(np.log2(max(2, r))))
                # Premier temps
                writer.write_bits(int(sorted_times[0]) % r, bits_per_interval)
                # Intervalles
                for delta in intervals:
                    writer.write_bits(int(delta) % r, bits_per_interval)

        elif repr_type == Representation.BOOLEAN:
            # R4a: Vecteur booléen
            for t in range(r):
                has_jump = t in jump_times.astype(int)
                writer.write_bit(1 if has_jump else 0)

        elif repr_type == Representation.COMBINATORIAL:
            # R4b: (N, Index combinatoire)
            from ..utils.math_utils import combinatorial_index
            positions = np.unique(jump_times.astype(int) % r)
            if len(positions) > 0:
                index = combinatorial_index(positions, r)
                # Encode l'index avec Elias delta
                self.elias_codec.encode(index + 1, writer)

    def _decode_temporal(
        self,
        reader: BitReader,
        repr_type: int,
        N: int,
        r: int
    ) -> np.ndarray:
        """Décode les données temporelles."""
        if N == 0:
            return np.array([])

        if repr_type == Representation.TIMESTAMPS:
            bits_per_time = int(np.ceil(np.log2(max(2, r))))
            times = np.zeros(N, dtype=int)
            for i in range(N):
                times[i] = reader.read_bits(bits_per_time)
            return times

        elif repr_type == Representation.COUNT:
            bits_per_count = reader.read_byte()
            counts = np.zeros(r, dtype=int)
            for i in range(r):
                counts[i] = reader.read_bits(bits_per_count)
            # Reconstruit les temps
            times = []
            for t, count in enumerate(counts):
                times.extend([t] * count)
            return np.array(times[:N])

        elif repr_type == Representation.INTERVALS:
            bits_per_interval = int(np.ceil(np.log2(max(2, r))))
            times = np.zeros(N, dtype=int)
            times[0] = reader.read_bits(bits_per_interval)
            for i in range(1, N):
                delta = reader.read_bits(bits_per_interval)
                times[i] = times[i-1] + delta
            return times

        elif repr_type == Representation.BOOLEAN:
            times = []
            for t in range(r):
                if reader.read_bit():
                    times.append(t)
            return np.array(times[:N])

        elif repr_type == Representation.COMBINATORIAL:
            from ..utils.math_utils import decode_combinatorial_index
            index = self.elias_codec.decode(reader) - 1
            return decode_combinatorial_index(index, N, r)

        return np.array([])

    def _encode_colors(
        self,
        marks: np.ndarray,
        color_mode: int,
        m: int,
        color_dist: Optional[Dict[int, float]],
        writer: BitWriter
    ):
        """Encode les couleurs selon le mode B."""
        log2_m = int(np.ceil(np.log2(max(2, m))))

        if color_mode == ColorMode.SEQUENTIAL:
            # Ba: Couleur initiale + transitions
            writer.write_bits(int(marks[0]), log2_m)
            for i in range(1, len(marks)):
                if marks[i] != marks[i-1]:
                    writer.write_bit(1)
                    writer.write_bits(int(marks[i]), log2_m)
                else:
                    writer.write_bit(0)

        elif color_mode == ColorMode.UNIFORM:
            # Bb: Chaque marque uniformément
            for mark in marks:
                writer.write_bits(int(mark), log2_m)

        elif color_mode == ColorMode.HUFFMAN:
            # Bc: Dictionnaire + codes
            if color_dist is None:
                counts = np.bincount(marks, minlength=m)
                total = counts.sum()
                color_dist = {c: counts[c]/total for c in range(m) if counts[c] > 0}

            self.huffman_codec.build_from_distribution(color_dist)
            self.huffman_codec.serialize_dictionary(writer)
            self.huffman_codec.encode_sequence(marks, writer)

        elif color_mode == ColorMode.ELIAS:
            # Bd: Code Elias pour chaque marque
            self.elias_codec.encode_sequence(marks, writer)

    def _decode_colors(
        self,
        reader: BitReader,
        color_mode: int,
        N: int,
        m: int
    ) -> np.ndarray:
        """Décode les couleurs selon le mode B."""
        log2_m = int(np.ceil(np.log2(max(2, m))))
        marks = np.zeros(N, dtype=int)

        if color_mode == ColorMode.SEQUENTIAL:
            current = reader.read_bits(log2_m)
            marks[0] = current
            for i in range(1, N):
                if reader.read_bit():
                    current = reader.read_bits(log2_m)
                marks[i] = current

        elif color_mode == ColorMode.UNIFORM:
            for i in range(N):
                marks[i] = reader.read_bits(log2_m)

        elif color_mode == ColorMode.HUFFMAN:
            self.huffman_codec.deserialize_dictionary(reader)
            marks = self.huffman_codec.decode_sequence(reader, N)

        elif color_mode == ColorMode.ELIAS:
            marks = self.elias_codec.decode_sequence(reader, N)

        return marks

    def mdl_3steps(
        self,
        jump_times: np.ndarray,
        marks: np.ndarray,
        features: BlockFeatures
    ) -> Tuple[float, IntensityFamily, Cartouche]:
        """
        Codage MDL en 3 étapes.

        1. Sélection de la famille F minimisant -log L + C_n(F)
        2. Estimation de α̂(t) dans F
        3. Codage arithmétique via Λ̂(t)

        Args:
            jump_times: Temps des sauts
            marks: Marques couleur
            features: Caractéristiques

        Returns:
            (longueur_totale, famille_optimale, cartouche)
        """
        N = features.N
        r = features.r

        # Étape 1: Sélection de la famille
        families = [
            IntensityFamily.HISTOGRAM,
            IntensityFamily.SPLINES,
            IntensityFamily.WAVELETS,
            IntensityFamily.TRIGONOMETRIC
        ]

        best_family = IntensityFamily.HISTOGRAM
        best_score = float('inf')

        for family in families:
            # Estimation et calcul du score MDL
            k_params = self._get_family_params(family, r)
            complexity = (k_params + 4) * np.log2(N + 1) / 2 if N > 0 else 0

            # Log-vraisemblance simplifiée (uniforme comme baseline)
            neg_log_lik = N * np.log2(r) if r > 1 else 0

            score = neg_log_lik + complexity

            if score < best_score:
                best_score = score
                best_family = family

        # Étape 2 & 3: Le coût final est le score MDL + coût couleur
        color_mode = features.suggest_color_mode()
        color_cost = self._compute_color_cost(marks, features.m, color_mode, features)

        total_length = best_score + color_cost

        # Construction du cartouche optimal
        cartouche = Cartouche(
            A=features.suggest_process_type(),
            B=color_mode,
            C=Representation.COMBINATORIAL,
            D=CompressionMode.MDL,
            E=best_family,
            F=2,  # 16 bits par défaut
            G=1,  # 8x8 pixels
            H=0   # Continu
        )

        return total_length, best_family, cartouche

    def _get_family_params(self, family: IntensityFamily, r: int) -> int:
        """Retourne le nombre de paramètres de la famille."""
        if family == IntensityFamily.HISTOGRAM:
            return min(16, r // 16)  # Nombre de bins
        elif family == IntensityFamily.SPLINES:
            return min(10, r // 25)  # Nombre de nœuds
        elif family == IntensityFamily.WAVELETS:
            return min(8, int(np.log2(r)))  # Niveaux
        elif family == IntensityFamily.TRIGONOMETRIC:
            return min(5, r // 50)  # Harmoniques
        return 4

    def _compute_color_cost(
        self,
        marks: np.ndarray,
        m: int,
        mode: ColorMode,
        features: BlockFeatures
    ) -> float:
        """Calcule le coût couleur."""
        N = len(marks)
        if N == 0 or m <= 1:
            return 0.0

        log2_m = np.log2(m)
        H = features.H_color
        N_trans = features.N_trans
        D_huf = features.get_huffman_overhead()

        if mode == ColorMode.SEQUENTIAL:
            return log2_m + N_trans * log2_m
        elif mode == ColorMode.UNIFORM:
            return N * log2_m
        elif mode == ColorMode.HUFFMAN:
            return N * H + D_huf
        elif mode == ColorMode.ELIAS:
            return sum(self.elias_codec.length(int(m) + 1) for m in marks)

        return 0.0
