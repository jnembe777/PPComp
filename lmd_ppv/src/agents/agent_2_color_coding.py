"""
agent_2_color_coding.py - Agent Codage Couleur (Dimension B)
=============================================================

Phase 2 - Dimension B (NOUVELLE - Correction fondamentale v6)

CORRECTION FONDAMENTALE:
Les formules L1-L4 classiques ne sont valables que pour le mode Bb (uniforme).
L_i(B) = L_temporel_i + C_color(B)

4 modes de codage:
- Ba: Séquentiel (log₂m + N_trans·log₂m)
- Bb: Uniforme (N·log₂m) ← formules L1-L4 actuelles
- Bc: Huffman (N·H(P̂) + D_huf)
- Bd: Elias universel (N·L*(m))

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..core.process_types import ColorMode
from ..core.features import BlockFeatures
from ..codecs.huffman import HuffmanCodec, huffman_threshold
from ..codecs.elias import EliasCodec
from ..utils.io_utils import BitWriter, BitReader


@dataclass
class ColorCodingResult:
    """Résultat du codage couleur."""
    mode: ColorMode
    cost: float
    all_costs: Dict[ColorMode, float]
    N_star: float  # Seuil Huffman
    gain_Bc_vs_Bb: float
    huffman_codec: Optional[HuffmanCodec] = None


class ColorCodingAgent:
    """
    Agent 2: Codage Couleur (Dimension B)

    Centralise toute la logique de la dimension B du cartouche.
    """

    def __init__(self):
        self.huffman_codec = HuffmanCodec()
        self.elias_codec = EliasCodec("delta")

    def analyze(
        self,
        marks: np.ndarray,
        m: int,
        color_dist: Optional[Dict[int, float]] = None
    ) -> ColorCodingResult:
        """
        Analyse les marques et détermine le mode B optimal.

        Args:
            marks: Séquence des marques couleur
            m: Nombre de couleurs
            color_dist: Distribution pré-calculée (optionnel)

        Returns:
            ColorCodingResult avec mode optimal et coûts
        """
        N = len(marks)

        if N == 0 or m <= 1:
            return ColorCodingResult(
                mode=ColorMode.UNIFORM,
                cost=0.0,
                all_costs={mode: 0.0 for mode in ColorMode},
                N_star=float('inf'),
                gain_Bc_vs_Bb=0.0
            )

        # Calcul de la distribution si non fournie
        if color_dist is None:
            counts = np.bincount(marks, minlength=m)
            total = counts.sum()
            color_dist = {c: counts[c] / total for c in range(m) if counts[c] > 0}

        # Calcul des métriques
        H = self._compute_entropy(color_dist)
        N_trans = self._count_transitions(marks)
        log2_m = np.log2(m)
        D_huf = m * (int(np.floor(log2_m)) + 1)

        # Calcul des coûts pour chaque mode
        costs = {
            ColorMode.SEQUENTIAL: self.color_cost_Ba(m, N_trans),
            ColorMode.UNIFORM: self.color_cost_Bb(N, m),
            ColorMode.HUFFMAN: self.color_cost_Bc(N, H, D_huf),
            ColorMode.ELIAS: self.color_cost_Bd(marks)
        }

        # Sélection du meilleur mode
        best_mode = min(costs, key=costs.get)

        # Calcul du seuil N*
        N_star = huffman_threshold(m, H, D_huf)

        # Gain Bc vs Bb
        gain = costs[ColorMode.UNIFORM] - costs[ColorMode.HUFFMAN]

        # Construction du codec Huffman si Bc est optimal
        huffman = None
        if best_mode == ColorMode.HUFFMAN:
            huffman = HuffmanCodec()
            huffman.build_from_distribution(color_dist)

        return ColorCodingResult(
            mode=best_mode,
            cost=costs[best_mode],
            all_costs=costs,
            N_star=N_star,
            gain_Bc_vs_Bb=gain,
            huffman_codec=huffman
        )

    def color_cost_Ba(self, m: int, N_trans: int) -> float:
        """
        Coût du mode Ba (Séquentiel).

        C_color(Ba) = log₂m + N_trans · log₂m

        La couleur est indiquée UNE SEULE FOIS en tête de bloc.
        Seules les transitions explicites sont encodées.

        Args:
            m: Nombre de couleurs
            N_trans: Nombre de transitions couleur

        Returns:
            Coût en bits
        """
        if m <= 1:
            return 0.0
        log2_m = np.log2(m)
        return log2_m + N_trans * log2_m

    def color_cost_Bb(self, N: int, m: int) -> float:
        """
        Coût du mode Bb (Uniforme).

        C_color(Bb) = N · log₂m

        C'est le terme des formules L1-L4 classiques.
        Chaque saut porte sa marque codée uniformément.

        Args:
            N: Nombre de sauts
            m: Nombre de couleurs

        Returns:
            Coût en bits
        """
        if m <= 1 or N == 0:
            return 0.0
        return N * np.log2(m)

    def color_cost_Bc(self, N: int, H: float, D_huf: int) -> float:
        """
        Coût du mode Bc (Huffman).

        C_color(Bc) = N · H(P̂) + D_huf

        Codes de longueur variable sur la distribution empirique.
        Avantageux si N > N* = D_huf / (log₂m - H).

        Args:
            N: Nombre de sauts
            H: Entropie de la distribution
            D_huf: Overhead du dictionnaire

        Returns:
            Coût en bits
        """
        if N == 0:
            return 0.0
        return N * H + D_huf

    def color_cost_Bd(self, marks: np.ndarray) -> float:
        """
        Coût du mode Bd (Elias universel).

        C_color(Bd) = Σ_k L_δ(m_k + 1)

        Utilisé quand m n'est pas connu au décodeur.

        Args:
            marks: Séquence des marques

        Returns:
            Coût en bits
        """
        if len(marks) == 0:
            return 0.0
        return sum(self.elias_codec.length(int(m) + 1) for m in marks)

    def _compute_entropy(self, dist: Dict[int, float]) -> float:
        """Calcule l'entropie H(P̂)."""
        H = 0.0
        for p in dist.values():
            if p > 0:
                H -= p * np.log2(p)
        return H

    def _count_transitions(self, marks: np.ndarray) -> int:
        """Compte les transitions couleur."""
        if len(marks) <= 1:
            return 0
        return int(np.sum(marks[1:] != marks[:-1]))

    def encode_colors(
        self,
        marks: np.ndarray,
        mode: ColorMode,
        writer: BitWriter,
        huffman_codec: Optional[HuffmanCodec] = None
    ):
        """
        Encode les couleurs selon le mode spécifié.

        Args:
            marks: Séquence des marques
            mode: Mode de codage
            writer: BitWriter
            huffman_codec: Codec Huffman pré-construit (pour Bc)
        """
        if len(marks) == 0:
            return

        m = int(marks.max()) + 1
        log2_m = int(np.ceil(np.log2(max(2, m))))

        if mode == ColorMode.SEQUENTIAL:
            self._encode_sequential(marks, log2_m, writer)
        elif mode == ColorMode.UNIFORM:
            self._encode_uniform(marks, log2_m, writer)
        elif mode == ColorMode.HUFFMAN:
            self._encode_huffman(marks, huffman_codec, writer)
        elif mode == ColorMode.ELIAS:
            self._encode_elias(marks, writer)

    def _encode_sequential(self, marks: np.ndarray, log2_m: int, writer: BitWriter):
        """Encode en mode séquentiel (Ba)."""
        # Couleur initiale
        writer.write_bits(int(marks[0]), log2_m)

        # Transitions seulement
        for i in range(1, len(marks)):
            if marks[i] != marks[i-1]:
                writer.write_bit(1)  # Flag transition
                writer.write_bits(int(marks[i]), log2_m)
            else:
                writer.write_bit(0)  # Pas de transition

    def _encode_uniform(self, marks: np.ndarray, log2_m: int, writer: BitWriter):
        """Encode en mode uniforme (Bb)."""
        for mark in marks:
            writer.write_bits(int(mark), log2_m)

    def _encode_huffman(
        self,
        marks: np.ndarray,
        huffman_codec: HuffmanCodec,
        writer: BitWriter
    ):
        """Encode en mode Huffman (Bc)."""
        if huffman_codec is None:
            raise ValueError("Huffman codec required for Bc mode")

        # Sérialise le dictionnaire
        huffman_codec.serialize_dictionary(writer)

        # Encode les symboles
        huffman_codec.encode_sequence(marks, writer)

    def _encode_elias(self, marks: np.ndarray, writer: BitWriter):
        """Encode en mode Elias (Bd)."""
        self.elias_codec.encode_sequence(marks, writer)

    def decode_colors(
        self,
        mode: ColorMode,
        reader: BitReader,
        n_symbols: int,
        m: int
    ) -> np.ndarray:
        """
        Décode les couleurs selon le mode spécifié.

        Args:
            mode: Mode de codage
            reader: BitReader
            n_symbols: Nombre de symboles à décoder
            m: Nombre de couleurs (pour Bb)

        Returns:
            Séquence des marques décodées
        """
        log2_m = int(np.ceil(np.log2(max(2, m))))

        if mode == ColorMode.SEQUENTIAL:
            return self._decode_sequential(reader, n_symbols, log2_m)
        elif mode == ColorMode.UNIFORM:
            return self._decode_uniform(reader, n_symbols, log2_m)
        elif mode == ColorMode.HUFFMAN:
            return self._decode_huffman(reader, n_symbols)
        elif mode == ColorMode.ELIAS:
            return self._decode_elias(reader, n_symbols)

        return np.zeros(n_symbols, dtype=int)

    def _decode_sequential(
        self,
        reader: BitReader,
        n_symbols: int,
        log2_m: int
    ) -> np.ndarray:
        """Décode le mode séquentiel (Ba)."""
        marks = np.zeros(n_symbols, dtype=int)

        # Couleur initiale
        current = reader.read_bits(log2_m)
        marks[0] = current

        # Transitions
        for i in range(1, n_symbols):
            if reader.read_bit():  # Flag transition
                current = reader.read_bits(log2_m)
            marks[i] = current

        return marks

    def _decode_uniform(
        self,
        reader: BitReader,
        n_symbols: int,
        log2_m: int
    ) -> np.ndarray:
        """Décode le mode uniforme (Bb)."""
        marks = np.zeros(n_symbols, dtype=int)
        for i in range(n_symbols):
            marks[i] = reader.read_bits(log2_m)
        return marks

    def _decode_huffman(self, reader: BitReader, n_symbols: int) -> np.ndarray:
        """Décode le mode Huffman (Bc)."""
        huffman = HuffmanCodec()
        huffman.deserialize_dictionary(reader)
        return huffman.decode_sequence(reader, n_symbols)

    def _decode_elias(self, reader: BitReader, n_symbols: int) -> np.ndarray:
        """Décode le mode Elias (Bd)."""
        return self.elias_codec.decode_sequence(reader, n_symbols)

    def should_use_huffman(
        self,
        N: int,
        m: int,
        H: float,
        D_huf: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Détermine si Huffman (Bc) est avantageux sur Uniforme (Bb).

        Bc < Bb si N > N* = D_huf / (log₂m - H)

        Args:
            N: Nombre de sauts
            m: Nombre de couleurs
            H: Entropie
            D_huf: Overhead dictionnaire (calculé si None)

        Returns:
            (utiliser_huffman, N_star)
        """
        N_star = huffman_threshold(m, H, D_huf)
        return N > N_star, N_star

    def should_use_sequential(
        self,
        N: int,
        m: int,
        N_trans: int,
        H: float
    ) -> bool:
        """
        Détermine si Séquentiel (Ba) est avantageux sur Huffman (Bc).

        Ba < Bc si N_trans < (N·H + D_huf - log₂m) / log₂m

        Args:
            N: Nombre de sauts
            m: Nombre de couleurs
            N_trans: Nombre de transitions
            H: Entropie

        Returns:
            True si Ba est meilleur que Bc
        """
        if m <= 1:
            return True

        log2_m = np.log2(m)
        D_huf = m * (int(np.floor(log2_m)) + 1)

        threshold = (N * H + D_huf - log2_m) / log2_m
        return N_trans < threshold


def compare_all_modes(
    N: int,
    m: int,
    H: float,
    N_trans: int
) -> Dict:
    """
    Compare les 4 modes de codage couleur.

    Args:
        N: Nombre de sauts
        m: Nombre de couleurs
        H: Entropie
        N_trans: Nombre de transitions

    Returns:
        Dict avec tous les coûts et recommandations
    """
    agent = ColorCodingAgent()

    log2_m = np.log2(m) if m > 1 else 0
    D_huf = m * (int(np.floor(log2_m)) + 1) if m > 1 else 0

    costs = {
        "Ba (Séquentiel)": agent.color_cost_Ba(m, N_trans),
        "Bb (Uniforme)": agent.color_cost_Bb(N, m),
        "Bc (Huffman)": agent.color_cost_Bc(N, H, D_huf),
        "Bd (Elias)": N * (log2_m + 2 * np.log2(max(1, log2_m)))
    }

    best = min(costs, key=costs.get)
    N_star = huffman_threshold(m, H, D_huf)

    return {
        "costs": costs,
        "best_mode": best,
        "best_cost": costs[best],
        "N_star": N_star,
        "gain_Bc_vs_Bb": costs["Bb (Uniforme)"] - costs["Bc (Huffman)"],
        "gain_Ba_vs_Bb": costs["Bb (Uniforme)"] - costs["Ba (Séquentiel)"],
        "parameters": {"N": N, "m": m, "H": H, "N_trans": N_trans, "D_huf": D_huf}
    }
