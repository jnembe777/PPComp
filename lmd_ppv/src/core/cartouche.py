"""
cartouche.py - Header ABCDEFGH 17 bits
======================================

Structure du cartouche encodant la méthode de compression:
A(3b) + B(2b) + C(3b) + D(2b) + E(2b) + F(2b) + G(2b) + H(1b) = 17 bits

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

from dataclasses import dataclass, field
from typing import Optional
from .process_types import (
    ProcessType, ColorMode, Representation, CompressionMode,
    IntensityFamily, ChromaticLevel, SpatialResolution, TemporalMode
)


@dataclass
class Cartouche:
    """
    Cartouche ABCDEFGH - Header 17 bits par bloc vidéo

    Encode intégralement la méthode de compression choisie pour un bloc.
    C'est le vecteur de décision du pipeline.

    Attributes:
        A: Type de processus ponctuel (3 bits, 0-4)
        B: Mode de codage couleur (2 bits, 0-3)
        C: Représentation temporelle (3 bits, 0-4)
        D: Mode de compression (2 bits, 0-2)
        E: Famille d'intensité α̂(t) (2 bits, 0-3)
        F: Niveau chromatique (2 bits, 1-3)
        G: Résolution spatiale (2 bits, 0-3)
        H: Mode temporel (1 bit, 0-1)
    """
    A: int = field(default=ProcessType.MARKED)        # 3b: Type PP
    B: int = field(default=ColorMode.UNIFORM)         # 2b: Mode couleur
    C: int = field(default=Representation.COMBINATORIAL)  # 3b: Représentation
    D: int = field(default=CompressionMode.MDL)       # 2b: Compression
    E: int = field(default=IntensityFamily.SPLINES)   # 2b: Famille α̂
    F: int = field(default=ChromaticLevel.BITS_16)    # 2b: Chromatique
    G: int = field(default=SpatialResolution.PX_8)    # 2b: Spatiale
    H: int = field(default=TemporalMode.CONTINUOUS)   # 1b: Temporel

    def encode(self) -> int:
        """Encode le cartouche en un entier 17 bits.

        Layout: AAABBC CCDDEE FFGGH (17 bits)
        """
        return (
            ((self.A & 0x7) << 14) |  # bits 16-14
            ((self.B & 0x3) << 12) |  # bits 13-12
            ((self.C & 0x7) << 9)  |  # bits 11-9
            ((self.D & 0x3) << 7)  |  # bits 8-7
            ((self.E & 0x3) << 5)  |  # bits 6-5
            ((self.F & 0x3) << 3)  |  # bits 4-3
            ((self.G & 0x3) << 1)  |  # bits 2-1
            (self.H & 0x1)            # bit 0
        )

    @classmethod
    def decode(cls, bits: int) -> 'Cartouche':
        """Décode un entier 17 bits en Cartouche."""
        return cls(
            A=(bits >> 14) & 0x7,
            B=(bits >> 12) & 0x3,
            C=(bits >> 9) & 0x7,
            D=(bits >> 7) & 0x3,
            E=(bits >> 5) & 0x3,
            F=(bits >> 3) & 0x3,
            G=(bits >> 1) & 0x3,
            H=bits & 0x1
        )

    def to_string(self) -> str:
        """Représentation lisible du cartouche."""
        a_names = ["Aa", "Ab", "Ac", "Ad", "Ae"]
        b_names = ["Ba", "Bb", "Bc", "Bd"]
        c_names = ["R1", "R2", "R3", "R4a", "R4b"]
        d_names = ["Da", "Db", "Dc"]
        e_names = ["Ea", "Eb", "Ec", "Ed"]

        return (
            f"{a_names[self.A]} {b_names[self.B]} {c_names[self.C]} "
            f"{d_names[self.D]} {e_names[self.E]} F{self.F} G{self.G} H{self.H}"
        )

    def __repr__(self) -> str:
        return f"Cartouche({self.to_string()}, bits=0x{self.encode():05X})"

    @property
    def process_type(self) -> ProcessType:
        return ProcessType(self.A)

    @property
    def color_mode(self) -> ColorMode:
        return ColorMode(self.B)

    @property
    def representation(self) -> Representation:
        return Representation(self.C)

    @property
    def compression_mode(self) -> CompressionMode:
        return CompressionMode(self.D)

    @property
    def intensity_family(self) -> IntensityFamily:
        return IntensityFamily(self.E)

    @property
    def is_monochromatic(self) -> bool:
        """Si monochromatique, C_color = 0 quel que soit B."""
        return self.A == ProcessType.MONOCHROMATIC

    @property
    def process_type_name(self) -> str:
        """Nom du type de processus."""
        names = ["MARKED", "MONOCHROMATIC", "VECTORIAL_MARGINAL", "VECTORIAL_JOINT", "MARKOVIAN"]
        return names[self.A] if 0 <= self.A < len(names) else "UNKNOWN"

    @property
    def color_mode_name(self) -> str:
        """Nom du mode couleur."""
        names = ["SEQUENTIAL", "UNIFORM", "HUFFMAN", "ELIAS"]
        return names[self.B] if 0 <= self.B < len(names) else "UNKNOWN"

    def validate(self) -> bool:
        """Vérifie la validité des valeurs."""
        return (
            0 <= self.A <= 4 and
            0 <= self.B <= 3 and
            0 <= self.C <= 4 and
            0 <= self.D <= 2 and
            0 <= self.E <= 3 and
            1 <= self.F <= 3 and
            0 <= self.G <= 3 and
            0 <= self.H <= 1
        )


# Cartouches prédéfinis pour cas courants
CARTOUCHE_DEFAULT = Cartouche()  # Aa Bb R4b Dc Eb F2 G1 H0

CARTOUCHE_FAST = Cartouche(
    A=ProcessType.MARKED,
    B=ColorMode.UNIFORM,
    C=Representation.BOOLEAN,
    D=CompressionMode.UNIFORM,
    E=IntensityFamily.HISTOGRAM,
    F=ChromaticLevel.BITS_8,
    G=SpatialResolution.PX_16,
    H=TemporalMode.DISCRETE
)

CARTOUCHE_QUALITY = Cartouche(
    A=ProcessType.MARKED,
    B=ColorMode.HUFFMAN,
    C=Representation.COMBINATORIAL,
    D=CompressionMode.MDL,
    E=IntensityFamily.SPLINES,
    F=ChromaticLevel.BITS_24,
    G=SpatialResolution.PX_2,
    H=TemporalMode.CONTINUOUS
)
