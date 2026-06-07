"""
Blocks v3 — Macroblocs avec prédiction intra-bloc + classification MDL.

Pipeline encodage par plan :
  1. Choix du mode de prédiction par bloc (4 modes, 2 bits/bloc)
  2. Application de la prédiction → résidus (distribution concentrée)
  3. Construction des matrices sur les résidus
  4. Classification MDL par bloc (Mono/PP/Spatial/Markov)
  5. Encodage des résidus avec la rep optimale + Huffman

Pipeline décodage par plan :
  1. Lecture des modes de prédiction
  2. Décodage des résidus (pipeline standard)
  3. Inversion de la prédiction → valeurs originales
"""

import numpy as np
from typing import List, Dict, Optional
from math import ceil, log2
from collections import Counter

from .bitstream import BitWriter, BitReader
from .matrices import build_all_matrices, compute_representation_lengths
from .representations import (
    encode_R1, encode_R2, encode_R3, encode_R4,
    decode_R1, decode_R2, decode_R3, decode_R4,
)
from .huffman import HuffmanTable
from .format import (
    make_code_methode, parse_code_methode,
    PROC_MONO, PROC_PP, PROC_SPATIAL, PROC_MARKOV,
    REP_R1, REP_R2, REP_R3, REP_R4,
    COMP_NONE, COMP_HUFFMAN,
)
from .combinatorics import bits_for_binomial, index_to_bool_vector
from .process_types import (
    _analyze_block_pixels, classify_block,
    encode_spatial, decode_spatial,
    encode_markov_v2, decode_markov,
)
from .prediction import (
    choose_best_mode_plane,
    apply_prediction_by_blocks,
    invert_prediction_by_blocks,
    PRED_BYPASS, NUM_PRED_MODES, PRED_NAMES,
)


def _build_block_huffman(pixels, color_bits, use_huffman):
    if not use_huffman:
        return None, False
    color_freq = Counter()
    for px in pixels:
        for c in px['colors']:
            color_freq[c] += 1
    if len(color_freq) < 2:
        return None, False
    candidate = HuffmanTable.from_frequencies(dict(color_freq))
    avg = candidate.average_bits(dict(color_freq))
    total_syms = sum(color_freq.values())
    n_sym = len(color_freq)
    overhead = n_sym * (color_bits + 5) + max(2 * n_sym.bit_length() - 1, 1)
    if overhead + int(avg * total_syms) < color_bits * total_syms:
        return candidate, True
    return None, False


# ═══════════════════════════════════════════════════════════════
#  ENCODAGE
# ═══════════════════════════════════════════════════════════════

def encode_plane_blocks(
    writer: BitWriter,
    plane: np.ndarray,
    pnl: int, pnc: int, gop_r: int,
    color_bits: int, col_flag: int,
    block_size: int,
    use_huffman: bool,
    stats: dict,
    plane_name: str,
    use_prediction: bool = True,
    heuristic=None,
) -> None:
    gop_duration_bits = max(1, ceil(log2(max(gop_r, 2))))
    comp_flag = COMP_HUFFMAN if use_huffman else COMP_NONE
    nb_bi = ceil(pnl / block_size)
    nb_bj = ceil(pnc / block_size)

    # ── 1. Prédiction ──────────────────────────────────────────
    if use_prediction:
        if heuristic is not None:
            # Fast path : prédire le mode par bloc avec l'heuristique
            from .heuristic import extract_block_features
            mode_map = np.zeros((nb_bi, nb_bj), dtype=np.int32)
            for bi in range(nb_bi):
                for bj in range(nb_bj):
                    i0 = bi * block_size
                    i1 = min(i0 + block_size, pnl)
                    j0 = bj * block_size
                    j1 = min(j0 + block_size, pnc)
                    feats = extract_block_features(
                        plane, i0, i1, j0, j1, gop_r
                    )
                    _, pred_mode = heuristic.predict_method(feats)
                    mode_map[bi, bj] = pred_mode
        else:
            mode_map = choose_best_mode_plane(plane, block_size)
        residuals = apply_prediction_by_blocks(plane, mode_map, block_size)
    else:
        mode_map = np.zeros((nb_bi, nb_bj), dtype=np.int32)
        residuals = plane

    # Écrire le flag prédiction (1 bit) + modes (2 bits par bloc)
    writer.write_bool(use_prediction)
    if use_prediction:
        for bi in range(nb_bi):
            for bj in range(nb_bj):
                writer.write_bits(int(mode_map[bi, bj]), 2)

    # ── 2. Construire les matrices sur les résidus ─────────────
    # NOTE: on ne construit PAS les matrices ici — on les construit
    # par bloc APRÈS la transformation palette pour que les indices
    # soient cohérents.

    ps = stats['plane_stats'][plane_name]

    # ── 3. Encoder chaque bloc ─────────────────────────────────
    for bi in range(nb_bi):
        for bj in range(nb_bj):
            i0 = bi * block_size
            i1 = min(i0 + block_size, pnl)
            j0 = bj * block_size
            j1 = min(j0 + block_size, pnc)

            block_data = residuals[:, i0:i1, j0:j1].copy()

            # ── Palette analysis ───────────────────────────
            from .palette import (
                analyze_block_palette, palette_transform,
                estimate_dl_palette, write_palette,
            )
            pal_info = analyze_block_palette(block_data)
            m = pal_info['m']

            # Décider si la palette est avantageuse
            # Heuristique : palette si index_bits < color_bits et m ≤ 64
            use_palette = (pal_info['index_bits'] < color_bits and m <= 64)

            if use_palette:
                effective_bits = pal_info['index_bits']
                # Transformer en indices
                indexed_block = palette_transform(block_data, pal_info)
                # Injecter dans le plan résiduel temporairement
                work_plane = residuals.copy()
                work_plane[:, i0:i1, j0:j1] = indexed_block
            else:
                effective_bits = color_bits
                work_plane = residuals

            # Construire matrices sur ce bloc
            block_plane = work_plane[:, i0:i1, j0:j1]
            # On construit les matrices sur un mini-plan juste pour ce bloc
            from .matrices import build_all_matrices as _bam
            block_3d = work_plane[:, i0:i1, j0:j1].copy()
            # Créer un mini-plan pour build_all_matrices
            mini_plane = np.zeros((gop_r, i1-i0, j1-j0), dtype=np.uint8)
            mini_plane[:] = block_3d
            data_blk = _bam(mini_plane)

            pixels = _analyze_block_pixels(
                mini_plane, data_blk, gop_r, 0, i1-i0, 0, j1-j0
            )

            # Écrire le flag palette (1 bit)
            writer.write_bool(use_palette)
            if use_palette:
                write_palette(writer, pal_info, color_bits)

            if heuristic is not None:
                from .heuristic import extract_block_features
                feats = extract_block_features(
                    mini_plane, 0, i1-i0, 0, j1-j0, gop_r
                )
                proc_type, _ = heuristic.predict_method(feats)
                from .process_types import (
                    _estimate_dl_mono, _estimate_dl_pp,
                    _estimate_dl_spatial, _estimate_dl_markov_v2,
                )
                if proc_type == 0:
                    _, info = _estimate_dl_mono(pixels, effective_bits)
                    if info is None:
                        proc_type = 1
                elif proc_type == 2:
                    _, info = _estimate_dl_spatial(
                        pixels, gop_r, effective_bits, gop_duration_bits)
                    if info is None:
                        proc_type = 1
                elif proc_type == 3:
                    _, info = _estimate_dl_markov_v2(
                        pixels, gop_r, effective_bits, gop_duration_bits)
                    if info is None:
                        proc_type = 1
                if proc_type == 1:
                    _, info = _estimate_dl_pp(
                        pixels, gop_r, effective_bits, gop_duration_bits)
            else:
                proc_type, dl, info = classify_block(
                    pixels, gop_r, effective_bits, gop_duration_bits
                )

            # ── PROC_MONO ──────────────────────────────────
            if proc_type == 0:
                cm = make_code_methode(PROC_MONO, REP_R1, comp_flag, col_flag)
                writer.write_bits(cm, 7)
                writer.write_bits(info['color'], effective_bits)
                n_px = len(pixels)
                stats['mono_count'] += n_px
                ps['mono'] += n_px
                stats['total_pixels_encoded'] += n_px

            # ── PROC_PP ────────────────────────────────────
            elif proc_type == 1:
                best_rep = info['best_rep']
                rep_code = [REP_R1, REP_R2, REP_R3, REP_R4][best_rep - 1]
                cm = make_code_methode(PROC_PP, rep_code, comp_flag, col_flag)
                writer.write_bits(cm, 7)

                htable, has_huff = _build_block_huffman(
                    pixels, effective_bits, use_huffman
                )
                writer.write_bool(has_huff)
                if has_huff:
                    htable.write_table(writer, effective_bits)

                for px in pixels:
                    if best_rep == 1:
                        encode_R1(writer, px['seq'], effective_bits, htable)
                    elif best_rep == 2:
                        encode_R2(writer, px['dates'], px['colors'],
                                  px['n_jumps'], gop_duration_bits,
                                  effective_bits, htable)
                    elif best_rep == 3:
                        encode_R3(writer, px['bool_vec'], px['colors'],
                                  gop_r, effective_bits, htable)
                    elif best_rep == 4:
                        encode_R4(writer, px['n_jumps'], px['s_index'],
                                  px['colors'], gop_r, gop_duration_bits,
                                  effective_bits, htable)

                    stats['rep_counts'][best_rep] += 1
                    ps['reps'][best_rep] += 1
                    stats['total_pixels_encoded'] += 1

            # ── PROC_SPATIAL ───────────────────────────────
            elif proc_type == 2:
                cm = make_code_methode(PROC_SPATIAL, REP_R3, comp_flag, col_flag)
                writer.write_bits(cm, 7)

                htable, has_huff = _build_block_huffman(
                    pixels, effective_bits, use_huffman
                )
                writer.write_bool(has_huff)
                if has_huff:
                    htable.write_table(writer, effective_bits)

                encode_spatial(writer, pixels, info, gop_r,
                               gop_duration_bits, effective_bits, htable)

                stats['total_pixels_encoded'] += len(pixels)
                if 'proc_spatial_count' not in stats:
                    stats['proc_spatial_count'] = 0
                stats['proc_spatial_count'] += len(pixels)

            # ── PROC_MARKOV ────────────────────────────────
            elif proc_type == 3:
                cm = make_code_methode(PROC_MARKOV, REP_R1, comp_flag, col_flag)
                writer.write_bits(cm, 7)

                htable, has_huff = _build_block_huffman(
                    pixels, effective_bits, use_huffman
                )
                writer.write_bool(has_huff)
                if has_huff:
                    htable.write_table(writer, effective_bits)

                encode_markov_v2(writer, pixels, info, gop_r,
                                 effective_bits, htable)

                stats['total_pixels_encoded'] += len(pixels)
                if 'proc_markov_count' not in stats:
                    stats['proc_markov_count'] = 0
                stats['proc_markov_count'] += len(pixels)


# ═══════════════════════════════════════════════════════════════
#  DÉCODAGE
# ═══════════════════════════════════════════════════════════════

def decode_plane_blocks(
    reader: BitReader,
    buf: np.ndarray,
    pnl: int, pnc: int, gop_r: int,
    t_offset: int, r_total: int,
    color_bits: int,
    block_size: int,
    stats: dict,
) -> None:
    gop_duration_bits = max(1, ceil(log2(max(gop_r, 2))))
    nb_bi = ceil(pnl / block_size)
    nb_bj = ceil(pnc / block_size)

    # ── 1. Lire les modes de prédiction ────────────────────────
    has_prediction = reader.read_bool()
    mode_map = np.zeros((nb_bi, nb_bj), dtype=np.int32)
    if has_prediction:
        for bi in range(nb_bi):
            for bj in range(nb_bj):
                mode_map[bi, bj] = reader.read_bits(2)

    # ── 2. Décoder les résidus dans un buffer temporaire ───────
    # On décode dans un buffer temporaire pour pouvoir inverser
    # la prédiction ensuite
    res_buf = np.zeros((gop_r, pnl, pnc), dtype=np.uint8)

    for bi in range(nb_bi):
        for bj in range(nb_bj):
            i0 = bi * block_size
            i1 = min(i0 + block_size, pnl)
            j0 = bj * block_size
            j1 = min(j0 + block_size, pnc)
            n_pixels = (i1 - i0) * (j1 - j0)

            # ── Palette ────────────────────────────────
            from .palette import read_palette, inverse_palette_transform
            has_palette = reader.read_bool()
            pal_info = None
            if has_palette:
                pal_info = read_palette(reader, color_bits)
                eff_bits = pal_info['index_bits']
            else:
                eff_bits = color_bits

            cm = reader.read_bits(7)
            proc, rep, comp, col = parse_code_methode(cm)

            # ── PROC_MONO ──────────────────────────────
            if proc == PROC_MONO:
                color = reader.read_bits(eff_bits)
                res_buf[:, i0:i1, j0:j1] = color
                stats['mono_count'] += n_pixels

            # ── PROC_PP ────────────────────────────────
            elif proc == PROC_PP:
                has_huff = reader.read_bool()
                htable = None
                if has_huff:
                    htable = HuffmanTable.read_table(reader, eff_bits)

                for i in range(i0, i1):
                    for j in range(j0, j1):
                        if rep == REP_R1:
                            seq = decode_R1(reader, gop_r, eff_bits, htable)
                            stats['rep_counts'][1] += 1
                        elif rep == REP_R2:
                            seq = decode_R2(reader, gop_r, gop_duration_bits,
                                            eff_bits, htable)
                            stats['rep_counts'][2] += 1
                        elif rep == REP_R3:
                            seq = decode_R3(reader, gop_r, eff_bits, htable)
                            stats['rep_counts'][3] += 1
                        elif rep == REP_R4:
                            seq = decode_R4(reader, gop_r, gop_duration_bits,
                                            eff_bits, htable)
                            stats['rep_counts'][4] += 1
                        for t in range(gop_r):
                            res_buf[t, i, j] = seq[t]

            # ── PROC_SPATIAL ───────────────────────────
            elif proc == PROC_SPATIAL:
                has_huff = reader.read_bool()
                htable = None
                if has_huff:
                    htable = HuffmanTable.read_table(reader, eff_bits)
                sequences = decode_spatial(
                    reader, gop_r, gop_duration_bits,
                    eff_bits, n_pixels, htable,
                )
                px_idx = 0
                for i in range(i0, i1):
                    for j in range(j0, j1):
                        for t in range(gop_r):
                            res_buf[t, i, j] = sequences[px_idx][t]
                        px_idx += 1

            # ── PROC_MARKOV ────────────────────────────
            elif proc == PROC_MARKOV:
                has_huff = reader.read_bool()
                htable = None
                if has_huff:
                    htable = HuffmanTable.read_table(reader, eff_bits)
                sequences = decode_markov(
                    reader, gop_r, eff_bits, n_pixels, htable,
                )
                px_idx = 0
                for i in range(i0, i1):
                    for j in range(j0, j1):
                        for t in range(gop_r):
                            res_buf[t, i, j] = sequences[px_idx][t]
                        px_idx += 1

            # ── Inverse palette ────────────────────────
            if has_palette and pal_info is not None:
                block_decoded = res_buf[:, i0:i1, j0:j1].copy()
                res_buf[:, i0:i1, j0:j1] = inverse_palette_transform(
                    block_decoded, pal_info
                )

    # ── 3. Inverser la prédiction ──────────────────────────────
    if has_prediction:
        gop_reconstructed = invert_prediction_by_blocks(
            res_buf, mode_map, block_size
        )
    else:
        gop_reconstructed = res_buf

    # Copier dans le buffer de sortie
    for t in range(gop_r):
        t_abs = t_offset + t
        if t_abs < r_total:
            buf[t_abs, :pnl, :pnc] = gop_reconstructed[t]
