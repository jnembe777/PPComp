"""
arithmetic.py - Codage arithmétique pour MDL
=============================================

Codage arithmétique adaptatif basé sur l'intensité estimée α̂(t).

Utilisé dans le mode Dc (MDL statistique 3 étapes).

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..utils.io_utils import BitWriter, BitReader


# Constantes pour la précision
PRECISION = 32
WHOLE = 1 << PRECISION
HALF = WHOLE >> 1
QUARTER = WHOLE >> 2


@dataclass
class ArithmeticState:
    """État interne du codeur/décodeur arithmétique."""
    low: int = 0
    high: int = WHOLE - 1
    pending_bits: int = 0


class ArithmeticCodec:
    """
    Codec arithmétique adaptatif.

    Peut utiliser:
    - Distribution statique (pour couleurs)
    - Intensité cumulée Λ̂(t) (pour temps de sauts MDL)

    Attributes:
        precision: Nombre de bits de précision
        cumulative: Distribution cumulative
    """

    def __init__(self, precision: int = PRECISION):
        self.precision = precision
        self.whole = 1 << precision
        self.half = self.whole >> 1
        self.quarter = self.whole >> 2
        self.cumulative: Optional[np.ndarray] = None
        self.n_symbols: int = 0

    def set_distribution(self, probs: np.ndarray):
        """
        Définit la distribution des symboles.

        Args:
            probs: Vecteur de probabilités (somme = 1)
        """
        self.n_symbols = len(probs)
        # Cumulative: [0, p0, p0+p1, ..., 1]
        self.cumulative = np.zeros(self.n_symbols + 1)
        self.cumulative[1:] = np.cumsum(probs)
        # Mise à l'échelle sur [0, whole)
        self.cumulative = (self.cumulative * self.whole).astype(int)
        self.cumulative[-1] = self.whole  # Assure la borne sup

    def set_intensity_cdf(self, cdf: np.ndarray, n_bins: int):
        """
        Définit la CDF basée sur l'intensité cumulée Λ̂(t).

        Pour codage MDL des temps de sauts:
        τ_k -> Λ̂(τ_k) / Λ̂(T) ~ Uniforme[0,1]

        Args:
            cdf: CDF Λ̂(t) / Λ̂(T) pour chaque bin
            n_bins: Nombre de bins temporels
        """
        self.n_symbols = n_bins
        self.cumulative = (cdf * self.whole).astype(int)
        self.cumulative = np.clip(self.cumulative, 0, self.whole)

    def encode(self, symbols: np.ndarray, writer: BitWriter):
        """
        Encode une séquence de symboles.

        Args:
            symbols: Séquence de symboles (indices)
            writer: BitWriter
        """
        if self.cumulative is None:
            raise ValueError("Distribution not set")

        state = ArithmeticState()

        for symbol in symbols:
            self._encode_symbol(int(symbol), state, writer)

        # Finalisation
        self._finalize_encoding(state, writer)

    def _encode_symbol(self, symbol: int, state: ArithmeticState, writer: BitWriter):
        """Encode un symbole."""
        range_size = state.high - state.low + 1

        # Nouvelles bornes
        state.high = state.low + (range_size * self.cumulative[symbol + 1]) // self.whole - 1
        state.low = state.low + (range_size * self.cumulative[symbol]) // self.whole

        # Normalisation
        while True:
            if state.high < self.half:
                # Bit 0
                self._output_bit(0, state, writer)
            elif state.low >= self.half:
                # Bit 1
                self._output_bit(1, state, writer)
                state.low -= self.half
                state.high -= self.half
            elif state.low >= self.quarter and state.high < 3 * self.quarter:
                # Underflow
                state.pending_bits += 1
                state.low -= self.quarter
                state.high -= self.quarter
            else:
                break

            state.low = state.low << 1
            state.high = (state.high << 1) | 1

    def _output_bit(self, bit: int, state: ArithmeticState, writer: BitWriter):
        """Écrit un bit et les bits pending."""
        writer.write_bit(bit)
        while state.pending_bits > 0:
            writer.write_bit(1 - bit)
            state.pending_bits -= 1

    def _finalize_encoding(self, state: ArithmeticState, writer: BitWriter):
        """Finalise l'encodage."""
        state.pending_bits += 1
        if state.low < self.quarter:
            self._output_bit(0, state, writer)
        else:
            self._output_bit(1, state, writer)

    def decode(self, reader: BitReader, n_symbols: int) -> np.ndarray:
        """
        Décode une séquence de symboles.

        Args:
            reader: BitReader
            n_symbols: Nombre de symboles à décoder

        Returns:
            Séquence de symboles décodés
        """
        if self.cumulative is None:
            raise ValueError("Distribution not set")

        symbols = np.zeros(n_symbols, dtype=int)
        state = ArithmeticState()

        # Initialisation de la valeur
        value = 0
        for _ in range(self.precision):
            value = (value << 1) | reader.read_bit()

        for i in range(n_symbols):
            symbols[i] = self._decode_symbol(state, reader, value)
            # Mise à jour de value après décodage
            while True:
                if state.high < self.half:
                    pass
                elif state.low >= self.half:
                    value -= self.half
                    state.low -= self.half
                    state.high -= self.half
                elif state.low >= self.quarter and state.high < 3 * self.quarter:
                    value -= self.quarter
                    state.low -= self.quarter
                    state.high -= self.quarter
                else:
                    break

                state.low = state.low << 1
                state.high = (state.high << 1) | 1
                value = (value << 1) | (reader.read_bit() if not reader.is_eof() else 0)

        return symbols

    def _decode_symbol(self, state: ArithmeticState, reader: BitReader, value: int) -> int:
        """Décode un symbole."""
        range_size = state.high - state.low + 1
        scaled_value = ((value - state.low + 1) * self.whole - 1) // range_size

        # Recherche du symbole
        symbol = np.searchsorted(self.cumulative[1:], scaled_value, side='right')
        symbol = min(symbol, self.n_symbols - 1)

        # Mise à jour des bornes
        state.high = state.low + (range_size * self.cumulative[symbol + 1]) // self.whole - 1
        state.low = state.low + (range_size * self.cumulative[symbol]) // self.whole

        return symbol


class AdaptiveArithmeticCodec:
    """
    Codec arithmétique adaptatif avec mise à jour des fréquences.

    Les probabilités sont estimées au fur et à mesure du codage.
    """

    def __init__(self, n_symbols: int, precision: int = PRECISION):
        self.n_symbols = n_symbols
        self.precision = precision
        self.whole = 1 << precision
        self.half = self.whole >> 1
        self.quarter = self.whole >> 2

        # Fréquences initiales (uniforme + pseudo-count)
        self.frequencies = np.ones(n_symbols + 1, dtype=int)
        self.total = n_symbols

    def _get_cumulative(self) -> np.ndarray:
        """Calcule la distribution cumulative actuelle."""
        cum = np.zeros(self.n_symbols + 1, dtype=int)
        cum[1:] = np.cumsum(self.frequencies[:self.n_symbols])
        return (cum * self.whole // self.total).astype(int)

    def _update_frequencies(self, symbol: int):
        """Met à jour les fréquences après un symbole."""
        self.frequencies[symbol] += 1
        self.total += 1

        # Normalisation si overflow
        if self.total > self.half:
            self.frequencies = (self.frequencies + 1) // 2
            self.total = self.frequencies.sum()

    def encode(self, symbols: np.ndarray, writer: BitWriter):
        """Encode avec adaptation."""
        state = ArithmeticState()

        for symbol in symbols:
            cumulative = self._get_cumulative()
            # ... (similaire à ArithmeticCodec.encode)
            self._update_frequencies(int(symbol))

    def decode(self, reader: BitReader, n_symbols: int) -> np.ndarray:
        """Décode avec adaptation."""
        symbols = np.zeros(n_symbols, dtype=int)

        for i in range(n_symbols):
            cumulative = self._get_cumulative()
            # ... (similaire à ArithmeticCodec.decode)
            self._update_frequencies(int(symbols[i]))

        return symbols
