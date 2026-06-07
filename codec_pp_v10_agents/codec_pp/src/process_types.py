"""
Process Types — Classification et encodage spécialisé par type de processus.

4 types de processus ponctuels marqués pour un bloc de pixels :

  PROC_MONO (00) : Bloc constant — une seule couleur pour tout le bloc.
      Condition : tous les pixels sont constants ET partagent la même couleur.
      Encodage : [couleur]

  PROC_PP (01) : Processus ponctuel général — chaque pixel indépendant.
      Condition : fallback quand aucun autre type n'est plus compact.
      Encodage : [pixels × R_best]  (R1-R4 par pixel)

  PROC_SPATIAL (10) : Processus spatial — jump pattern partagé.
      Condition : une majorité de pixels partagent le même vecteur booléen b.
      Encodage :
        [b_dominant : r bits]           — le pattern de sauts dominant
        [n_conform : Elias gamma]       — nombre de pixels conformes
        Pour chaque pixel :
          [flag : 1 bit]  (conforme au pattern dominant ?)
          Si conforme → [couleurs aux sauts : n × Huffman]
          Si non-conforme → [R4 complet]

  PROC_MARKOV (11) : Processus markovien — transitions de couleurs.
      Condition : peu de transitions distinctes dans le bloc.
      Encodage :
        [nb_transitions : Elias gamma]
        [table transitions : nb_trans × (from:color_bits + to:color_bits)]
        Pour chaque pixel :
          [couleur_initiale : Huffman]
          [indices transitions : n_jumps × trans_bits]

La classification utilise le principe MDL : on estime la longueur de
description (DL) pour chaque type et on choisit le minimum.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from math import ceil, log2
from collections import Counter

from .bitstream import BitWriter, BitReader
from .combinatorics import (
    compute_jump_vector, bool_vector_to_index, index_to_bool_vector,
    bits_for_binomial
)
from .huffman import HuffmanTable
from .representations import (
    encode_R1, encode_R2, encode_R3, encode_R4,
    decode_R1, decode_R2, decode_R3, decode_R4,
)
from .matrices import compute_representation_lengths


# ═══════════════════════════════════════════════════════════════
#  STRUCTURES DE DONNÉES PAR PIXEL
# ═══════════════════════════════════════════════════════════════

def _analyze_block_pixels(plane, data, gop_r, i0, i1, j0, j1):
    """Collecte les données de chaque pixel dans le bloc."""
    pixels = []
    for i in range(i0, i1):
        for j in range(j0, j1):
            n_jumps = data['B'][i][j][0]
            bool_vec = data['B'][i][j][1:]
            s_index = int(data['S'][i, j])
            dates = data['MT'][i][j][1:]
            colors = data['CB'][i][j]
            seq = [int(plane[t, i, j]) for t in range(gop_r)]
            pixels.append({
                'i': i, 'j': j,
                'n_jumps': n_jumps,
                'bool_vec': bool_vec,
                'bool_tuple': tuple(bool_vec),
                's_index': s_index,
                'dates': dates,
                'colors': colors,
                'seq': seq,
            })
    return pixels


# ═══════════════════════════════════════════════════════════════
#  ESTIMATION DES LONGUEURS DE DESCRIPTION (MDL)
# ═══════════════════════════════════════════════════════════════

def _estimate_dl_mono(pixels, color_bits):
    """DL pour PROC_MONO : tous constants avec même couleur."""
    if not pixels:
        return float('inf'), None

    first_color = pixels[0]['colors'][0] if pixels[0]['colors'] else None
    for px in pixels:
        if px['n_jumps'] != 1:
            return float('inf'), None
        if px['colors'][0] != first_color:
            return float('inf'), None

    # 7 bits Code Méthode + color_bits
    return 7 + color_bits, {'color': first_color}


def _estimate_dl_pp(pixels, gop_r, color_bits, gop_duration_bits):
    """DL pour PROC_PP : R_best indépendant par pixel."""
    total = 7 + 1  # Code Méthode + Huffman flag
    # + overhead table Huffman estimé
    color_freq = Counter()
    for px in pixels:
        for c in px['colors']:
            color_freq[c] += 1

    # Estimer overhead Huffman
    huff_overhead = 0
    avg_huff_bits = color_bits  # fallback
    if len(color_freq) > 1:
        ht = HuffmanTable.from_frequencies(dict(color_freq))
        avg_huff_bits = ht.average_bits(dict(color_freq))
        n_sym = len(color_freq)
        huff_overhead = n_sym * (color_bits + 5) + max(2 * n_sym.bit_length() - 1, 1)
        fixed_cost = color_bits * sum(color_freq.values())
        huff_cost = huff_overhead + int(avg_huff_bits * sum(color_freq.values()))
        if huff_cost >= fixed_cost:
            avg_huff_bits = color_bits
            huff_overhead = 0

    total += huff_overhead

    # Meilleure rep unique pour le bloc
    total_per_rep = {1: 0, 2: 0, 3: 0, 4: 0}
    for px in pixels:
        n = px['n_jumps']
        L1, L2, L3, L4 = compute_representation_lengths(
            n, gop_r, color_bits, gop_duration_bits
        )
        # Ajuster pour Huffman sur les couleurs
        if avg_huff_bits < color_bits:
            ratio = avg_huff_bits / color_bits
            L1 = int(L1 * ratio)  # R1 = r couleurs
            # R2, R3, R4 : seules les couleurs bénéficient
            L2_struct = gop_duration_bits + n * gop_duration_bits
            L2 = L2_struct + int(n * avg_huff_bits)
            L3 = gop_r + int(n * avg_huff_bits)
            s_bits = bits_for_binomial(gop_r, n)
            L4 = gop_duration_bits + s_bits + int(n * avg_huff_bits)

        total_per_rep[1] += L1
        total_per_rep[2] += L2
        total_per_rep[3] += L3
        total_per_rep[4] += L4

    best_rep = min(total_per_rep, key=total_per_rep.get)
    total += total_per_rep[best_rep]

    return total, {'best_rep': best_rep}


def _estimate_dl_spatial(pixels, gop_r, color_bits, gop_duration_bits):
    """
    DL pour PROC_SPATIAL : pattern de sauts dominant partagé.

    On trouve le bool_vec le plus fréquent dans le bloc.
    Les pixels conformes n'ont besoin que de leurs couleurs.
    Les non-conformes sont encodés en R4 complet.
    """
    if not pixels:
        return float('inf'), None

    # Trouver le pattern dominant
    pattern_counts = Counter(px['bool_tuple'] for px in pixels)
    dominant_pattern, dominant_count = pattern_counts.most_common(1)[0]
    n_conform = dominant_count
    n_nonconform = len(pixels) - n_conform
    n_jumps_dom = sum(dominant_pattern)

    # Si < 50% conformes, pas intéressant
    if n_conform < len(pixels) * 0.4:
        return float('inf'), None

    # Coût structure
    total = 7          # Code Méthode
    total += gop_r     # b_dominant (r bits)
    total += max(2 * max(n_conform.bit_length(), 1) - 1, 1)  # Elias gamma n_conform
    total += 1         # Huffman flag

    # Estimer Huffman
    color_freq = Counter()
    for px in pixels:
        for c in px['colors']:
            color_freq[c] += 1

    avg_color_bits = color_bits
    huff_overhead = 0
    if len(color_freq) > 1:
        ht = HuffmanTable.from_frequencies(dict(color_freq))
        avg_c = ht.average_bits(dict(color_freq))
        n_sym = len(color_freq)
        ho = n_sym * (color_bits + 5) + max(2 * n_sym.bit_length() - 1, 1)
        if ho + int(avg_c * sum(color_freq.values())) < color_bits * sum(color_freq.values()):
            avg_color_bits = avg_c
            huff_overhead = ho

    total += huff_overhead

    # Pixels conformes : 1 bit flag + n_jumps_dom couleurs
    total += n_conform * (1 + int(n_jumps_dom * avg_color_bits))

    # Pixels non-conformes : 1 bit flag + R4 complet
    for px in pixels:
        if px['bool_tuple'] != dominant_pattern:
            n = px['n_jumps']
            s_bits = bits_for_binomial(gop_r, n)
            cost = 1 + gop_duration_bits + s_bits + int(n * avg_color_bits)
            total += cost

    return total, {
        'dominant_pattern': list(dominant_pattern),
        'n_jumps_dom': n_jumps_dom,
        'n_conform': n_conform,
    }


def _estimate_dl_markov(pixels, gop_r, color_bits, gop_duration_bits):
    """
    DL pour PROC_MARKOV : encodage par transitions de couleurs.

    On construit la table des transitions (from, to) observées dans le bloc.
    Chaque pixel est encodé comme : couleur initiale + séquence d'indices
    de transition.
    """
    if not pixels:
        return float('inf'), None

    # Collecter toutes les transitions
    transition_set = set()
    trans_counts = Counter()
    for px in pixels:
        seq = px['seq']
        for t in range(1, len(seq)):
            if seq[t] != seq[t-1]:
                tr = (seq[t-1], seq[t])
                transition_set.add(tr)
                trans_counts[tr] += 1

    nb_trans = len(transition_set)
    if nb_trans == 0:
        return float('inf'), None  # tout constant → MONO

    # Si trop de transitions, pas intéressant
    trans_bits = max(1, ceil(log2(max(nb_trans, 2))))

    # Coût structure
    total = 7  # Code Méthode
    total += max(2 * max(nb_trans.bit_length(), 1) - 1, 1)  # Elias gamma nb_trans
    total += nb_trans * 2 * color_bits  # table (from, to)
    total += 1  # Huffman flag

    # Huffman sur couleurs initiales
    init_freq = Counter(px['seq'][0] for px in pixels)
    avg_init_bits = color_bits
    if len(init_freq) > 1:
        ht = HuffmanTable.from_frequencies(dict(init_freq))
        avg_init_bits = ht.average_bits(dict(init_freq))

    # Coût par pixel : couleur_initiale + (n_jumps - 1) × trans_bits
    for px in pixels:
        n = px['n_jumps']
        total += int(avg_init_bits)  # couleur initiale
        total += max(n - 1, 0) * trans_bits  # indices de transition

    return total, {
        'nb_trans': nb_trans,
        'trans_bits': trans_bits,
        'transitions': sorted(transition_set),
    }


# ═══════════════════════════════════════════════════════════════
#  CLASSIFIEUR MDL
# ═══════════════════════════════════════════════════════════════

def classify_block(pixels, gop_r, color_bits, gop_duration_bits):
    """
    Classifie un bloc selon le principe MDL.

    Returns:
        (proc_type, dl, extra_info) — proc_type ∈ {0, 1, 2, 3}
    """
    dl_mono, info_mono = _estimate_dl_mono(pixels, color_bits)
    dl_pp, info_pp = _estimate_dl_pp(pixels, gop_r, color_bits, gop_duration_bits)
    dl_spatial, info_spatial = _estimate_dl_spatial(pixels, gop_r, color_bits, gop_duration_bits)
    dl_markov, info_markov = _estimate_dl_markov_v2(pixels, gop_r, color_bits, gop_duration_bits)

    candidates = [
        (0, dl_mono, info_mono),      # PROC_MONO
        (1, dl_pp, info_pp),           # PROC_PP
        (2, dl_spatial, info_spatial),  # PROC_SPATIAL
        (3, dl_markov, info_markov),    # PROC_MARKOV
    ]

    best = min(candidates, key=lambda x: x[1])
    return best[0], best[1], best[2]


# ═══════════════════════════════════════════════════════════════
#  ENCODAGE SPATIAL (PROC_SPATIAL = 10)
# ═══════════════════════════════════════════════════════════════

def encode_spatial(
    writer: BitWriter,
    pixels: list,
    info: dict,
    gop_r: int,
    gop_duration_bits: int,
    color_bits: int,
    htable: Optional[HuffmanTable],
) -> None:
    """Encode un bloc en mode spatial (pattern dominant partagé)."""
    dom_pattern = info['dominant_pattern']
    n_jumps_dom = info['n_jumps_dom']

    # Écrire le pattern dominant (r bits)
    for b in dom_pattern:
        writer.write_bool(b == 1)

    # Nombre de pixels conformes
    writer.write_elias_gamma(info['n_conform'])

    # Encoder chaque pixel
    for px in pixels:
        if list(px['bool_tuple']) == dom_pattern:
            # Conforme : flag=1, puis seulement les couleurs
            writer.write_bool(True)
            for c in px['colors']:
                if htable:
                    htable.encode_symbol(writer, c)
                else:
                    writer.write_bits(c, color_bits)
        else:
            # Non-conforme : flag=0, puis R4 complet
            writer.write_bool(False)
            n = px['n_jumps']
            writer.write_elias_gamma(n)
            s_bits = bits_for_binomial(gop_r, n)
            if s_bits > 0:
                writer.write_bits(px['s_index'], s_bits)
            for c in px['colors']:
                if htable:
                    htable.encode_symbol(writer, c)
                else:
                    writer.write_bits(c, color_bits)


def decode_spatial(
    reader: BitReader,
    gop_r: int,
    gop_duration_bits: int,
    color_bits: int,
    n_pixels: int,
    htable: Optional[HuffmanTable],
) -> List[List[int]]:
    """Décode un bloc spatial → liste de séquences."""
    # Lire le pattern dominant
    dom_pattern = [reader.read_bits(1) for _ in range(gop_r)]
    n_jumps_dom = sum(dom_pattern)
    n_conform = reader.read_elias_gamma()

    sequences = []
    for _ in range(n_pixels):
        is_conform = reader.read_bool()

        if is_conform:
            # Lire les couleurs aux positions de saut du pattern dominant
            colors = []
            for _ in range(n_jumps_dom):
                if htable:
                    colors.append(htable.decode_symbol(reader))
                else:
                    colors.append(reader.read_bits(color_bits))

            # Reconstruire la séquence
            seq = [0] * gop_r
            ci = 0
            cur = 0
            for t in range(gop_r):
                if dom_pattern[t] == 1:
                    cur = colors[ci]
                    ci += 1
                seq[t] = cur
            sequences.append(seq)
        else:
            # R4 complet
            n = reader.read_elias_gamma()
            s_bits = bits_for_binomial(gop_r, n)
            s_index = reader.read_bits(s_bits) if s_bits > 0 else 0
            colors = []
            for _ in range(n):
                if htable:
                    colors.append(htable.decode_symbol(reader))
                else:
                    colors.append(reader.read_bits(color_bits))

            bvec = index_to_bool_vector(s_index, gop_r, n)
            seq = [0] * gop_r
            ci = 0
            cur = 0
            for t in range(gop_r):
                if bvec[t] == 1:
                    cur = colors[ci]
                    ci += 1
                seq[t] = cur
            sequences.append(seq)

    return sequences


# ═══════════════════════════════════════════════════════════════
#  ENCODAGE MARKOV (PROC_MARKOV = 11)
# ═══════════════════════════════════════════════════════════════

def encode_markov(
    writer: BitWriter,
    pixels: list,
    info: dict,
    gop_r: int,
    color_bits: int,
    htable: Optional[HuffmanTable],
) -> None:
    """Encode un bloc en mode markovien (table de transitions)."""
    transitions = info['transitions']  # liste de (from, to)
    nb_trans = info['nb_trans']
    trans_bits = info['trans_bits']

    # Table de transitions
    writer.write_elias_gamma(nb_trans)
    trans_to_idx = {}
    for idx, (fr, to) in enumerate(transitions):
        writer.write_bits(fr, color_bits)
        writer.write_bits(to, color_bits)
        trans_to_idx[(fr, to)] = idx

    # Encoder chaque pixel
    for px in pixels:
        seq = px['seq']
        # Couleur initiale
        if htable:
            htable.encode_symbol(writer, seq[0])
        else:
            writer.write_bits(seq[0], color_bits)

        # Transitions
        for t in range(1, gop_r):
            if seq[t] != seq[t-1]:
                tr = (seq[t-1], seq[t])
                idx = trans_to_idx[tr]
                writer.write_bits(idx, trans_bits)


def decode_markov(
    reader: BitReader,
    gop_r: int,
    color_bits: int,
    n_pixels: int,
    htable: Optional[HuffmanTable],
) -> List[List[int]]:
    """Décode un bloc markovien → liste de séquences."""
    nb_trans = reader.read_elias_gamma()
    trans_bits = max(1, ceil(log2(max(nb_trans, 2))))

    # Lire la table
    idx_to_trans = {}
    for idx in range(nb_trans):
        fr = reader.read_bits(color_bits)
        to = reader.read_bits(color_bits)
        idx_to_trans[idx] = (fr, to)

    # Construire un lookup : pour reconstruire, on a besoin du vecteur
    # booléen des sauts (non stocké directement). On doit lire les
    # transitions séquentiellement.

    # En fait, le décodeur sait quand il y a un saut car il connaît
    # la séquence précédente. On encode TOUTES les transitions (y compris
    # "pas de changement"). Correction : on n'encode que les vrais sauts.
    # Le problème est que le décodeur ne sait pas quand un saut se produit.

    # Solution : encoder aussi le vecteur booléen.
    # Mais ça annule l'avantage. Mieux : encoder le nombre de pas entre
    # chaque saut (run-length).

    # Pour rester simple et correct, on utilise un flag 1 bit par frame :
    # 0 = pas de changement, 1 = lire un indice de transition.
    # C'est ce que fait déjà R3 en substance. La valeur ajoutée du Markov
    # est que l'indice de transition est sur trans_bits (souvent < color_bits).

    sequences = []
    for _ in range(n_pixels):
        # Couleur initiale
        if htable:
            cur = htable.decode_symbol(reader)
        else:
            cur = reader.read_bits(color_bits)

        seq = [cur]
        for t in range(1, gop_r):
            has_trans = reader.read_bits(1)
            if has_trans:
                idx = reader.read_bits(trans_bits)
                _, to = idx_to_trans[idx]
                cur = to
            seq.append(cur)

        sequences.append(seq)

    return sequences


def encode_markov_v2(
    writer: BitWriter,
    pixels: list,
    info: dict,
    gop_r: int,
    color_bits: int,
    htable: Optional[HuffmanTable],
) -> None:
    """Encode Markov v2 : flag 1 bit/frame + indice transition si saut."""
    transitions = info['transitions']
    nb_trans = info['nb_trans']
    trans_bits = info['trans_bits']

    writer.write_elias_gamma(nb_trans)
    trans_to_idx = {}
    for idx, (fr, to) in enumerate(transitions):
        writer.write_bits(fr, color_bits)
        writer.write_bits(to, color_bits)
        trans_to_idx[(fr, to)] = idx

    for px in pixels:
        seq = px['seq']
        if htable:
            htable.encode_symbol(writer, seq[0])
        else:
            writer.write_bits(seq[0], color_bits)

        for t in range(1, gop_r):
            if seq[t] != seq[t-1]:
                writer.write_bool(True)
                tr = (seq[t-1], seq[t])
                writer.write_bits(trans_to_idx[tr], trans_bits)
            else:
                writer.write_bool(False)


def _estimate_dl_markov_v2(pixels, gop_r, color_bits, gop_duration_bits):
    """DL Markov v2 : 1 bit/frame + trans_bits si saut."""
    if not pixels:
        return float('inf'), None

    transition_set = set()
    total_jumps = 0
    for px in pixels:
        seq = px['seq']
        for t in range(1, len(seq)):
            if seq[t] != seq[t-1]:
                transition_set.add((seq[t-1], seq[t]))
                total_jumps += 1

    nb_trans = len(transition_set)
    if nb_trans == 0:
        return float('inf'), None

    trans_bits = max(1, ceil(log2(max(nb_trans, 2))))

    total = 7  # Code Méthode
    total += max(2 * max(nb_trans.bit_length(), 1) - 1, 1)
    total += nb_trans * 2 * color_bits  # table
    total += 1  # Huffman flag

    # Par pixel : color_bits init + (gop_r-1) flag bits + total_jumps × trans_bits
    total += len(pixels) * color_bits  # couleurs initiales
    total += len(pixels) * (gop_r - 1)  # flags
    total += total_jumps * trans_bits   # indices

    return total, {
        'nb_trans': nb_trans,
        'trans_bits': trans_bits,
        'transitions': sorted(transition_set),
    }
