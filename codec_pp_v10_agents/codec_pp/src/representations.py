"""
Representations — Encodage et décodage des 4 représentations R1-R4.

Version 2 : chaque fonction accepte un HuffmanTable optionnel.
Si présent, les couleurs sont encodées en longueur variable.
Sinon, en largeur fixe (color_bits).
"""

from typing import List, Optional
from .bitstream import BitWriter, BitReader
from .combinatorics import (
    index_to_bool_vector, bits_for_binomial
)

# Import conditionnel pour éviter la dépendance circulaire
HuffmanTable = None  # sera résolu au runtime


def _get_htable_class():
    global HuffmanTable
    if HuffmanTable is None:
        from .huffman import HuffmanTable as HT
        HuffmanTable = HT
    return HuffmanTable


def _write_colors(writer, colors, color_bits, htable=None):
    bits = 0
    if htable is not None:
        for c in colors:
            bits += htable.encode_symbol(writer, c)
    else:
        for c in colors:
            writer.write_bits(c, color_bits)
            bits += color_bits
    return bits


def _read_colors(reader, n, color_bits, htable=None):
    if htable is not None:
        return [htable.decode_symbol(reader) for _ in range(n)]
    else:
        return [reader.read_bits(color_bits) for _ in range(n)]


# ═══════════════════════════════════════════════════════════════
#  ENCODAGE
# ═══════════════════════════════════════════════════════════════

def encode_R1(writer, sequence, color_bits, htable=None):
    return _write_colors(writer, sequence, color_bits, htable)


def encode_R2(writer, dates, colors, n_jumps, duration_bits,
              color_bits, htable=None):
    bits = 0
    bits += writer.write_elias_gamma(n_jumps)
    for t in dates:
        writer.write_bits(t, duration_bits)
        bits += duration_bits
    bits += _write_colors(writer, colors, color_bits, htable)
    return bits


def encode_R3(writer, bool_vector, colors, r, color_bits, htable=None):
    bits = 0
    for b in bool_vector:
        writer.write_bool(b == 1)
        bits += 1
    bits += _write_colors(writer, colors, color_bits, htable)
    return bits


def encode_R4(writer, n_jumps, s_index, colors, r, duration_bits,
              color_bits, htable=None):
    bits = 0
    bits += writer.write_elias_gamma(n_jumps)
    s_bits = bits_for_binomial(r, n_jumps)
    if s_bits > 0:
        writer.write_bits(s_index, s_bits)
        bits += s_bits
    bits += _write_colors(writer, colors, color_bits, htable)
    return bits


def encode_mono(writer, color, color_bits, htable=None):
    if htable is not None:
        return htable.encode_symbol(writer, color)
    else:
        writer.write_bits(color, color_bits)
        return color_bits


# ═══════════════════════════════════════════════════════════════
#  DÉCODAGE
# ═══════════════════════════════════════════════════════════════

def decode_R1(reader, r, color_bits, htable=None):
    return _read_colors(reader, r, color_bits, htable)


def decode_R2(reader, r, duration_bits, color_bits, htable=None):
    n_jumps = reader.read_elias_gamma()
    dates = [reader.read_bits(duration_bits) for _ in range(n_jumps)]
    colors = _read_colors(reader, n_jumps, color_bits, htable)

    sequence = [0] * r
    color_idx = 0
    for t in range(r):
        if color_idx < n_jumps - 1 and t >= dates[color_idx + 1]:
            color_idx += 1
        sequence[t] = colors[color_idx]
    return sequence


def decode_R3(reader, r, color_bits, htable=None):
    bvec = [reader.read_bits(1) for _ in range(r)]
    n_jumps = sum(bvec)
    colors = _read_colors(reader, n_jumps, color_bits, htable)

    sequence = [0] * r
    color_idx = 0
    current_color = 0
    for t in range(r):
        if bvec[t] == 1:
            current_color = colors[color_idx]
            color_idx += 1
        sequence[t] = current_color
    return sequence


def decode_R4(reader, r, duration_bits, color_bits, htable=None):
    n_jumps = reader.read_elias_gamma()
    s_bits = bits_for_binomial(r, n_jumps)
    s_index = reader.read_bits(s_bits) if s_bits > 0 else 0
    colors = _read_colors(reader, n_jumps, color_bits, htable)

    bvec = index_to_bool_vector(s_index, r, n_jumps)
    sequence = [0] * r
    color_idx = 0
    current_color = 0
    for t in range(r):
        if bvec[t] == 1:
            current_color = colors[color_idx]
            color_idx += 1
        sequence[t] = current_color
    return sequence


def decode_mono(reader, r, color_bits, htable=None):
    if htable is not None:
        color = htable.decode_symbol(reader)
    else:
        color = reader.read_bits(color_bits)
    return [color] * r
