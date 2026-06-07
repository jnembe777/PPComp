"""
Decoder v4 — Décodage par macroblocs + multi-plan YCbCr.
"""

import numpy as np
import time
from typing import Dict, Any, Tuple
from math import ceil

from .bitstream import BitReader
from .format import decode_header, HEADER_SIZE
from .blocks import decode_plane_blocks
from .colorspace import merge_video_planes, SUBSAMPLE_420


class PPVDecoder:

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def decode(self, input_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:

        t_start = time.time()

        with open(input_path, 'rb') as f:
            raw = f.read()

        meta = decode_header(raw[:HEADER_SIZE])
        nl, nc = meta['nl'], meta['nc']
        r_total = meta['r']
        color_bits = meta['color_bits']
        gop_size = meta['gop_size']
        nb_gops = meta['nb_gops']
        is_color = (meta['color_space'] > 0)

        if self.verbose:
            clr = "YCbCr" if is_color else "Gris"
            print(f"  DÉCODEUR v4 — {clr}, {nl}×{nc}, "
                  f"{r_total} fr, {nb_gops} GOPs")

        body = raw[HEADER_SIZE:]
        reader = BitReader(body)

        # Buffers
        if is_color:
            nl_c, nc_c = (nl + 1) // 2, (nc + 1) // 2
            Y_buf  = np.zeros((r_total, nl, nc), dtype=np.uint8)
            Cb_buf = np.zeros((r_total, nl_c, nc_c), dtype=np.uint8)
            Cr_buf = np.zeros((r_total, nl_c, nc_c), dtype=np.uint8)
            bufs = [
                ('Y',  Y_buf,  nl,   nc),
                ('Cb', Cb_buf, nl_c, nc_c),
                ('Cr', Cr_buf, nl_c, nc_c),
            ]
        else:
            Y_buf = np.zeros((r_total, nl, nc), dtype=np.uint8)
            bufs = [('Y', Y_buf, nl, nc)]

        stats = {
            'rep_counts': {1: 0, 2: 0, 3: 0, 4: 0},
            'mono_count': 0,
        }

        for g_idx in range(nb_gops):
            gop_idx = reader.read_uint16()
            gop_r = reader.read_uint16()
            nb_planes = reader.read_bits(2)
            block_size = reader.read_bits(8)

            t_offset = g_idx * gop_size

            for p_idx in range(nb_planes):
                pname, pbuf, pnl, pnc = bufs[p_idx]

                decode_plane_blocks(
                    reader, pbuf, pnl, pnc, gop_r,
                    t_offset, r_total, color_bits,
                    block_size, stats,
                )

        # Reconstruction
        if is_color:
            M_out = merge_video_planes(Y_buf, Cb_buf, Cr_buf,
                                       nl, nc, SUBSAMPLE_420)
        else:
            M_out = Y_buf

        t_elapsed = time.time() - t_start
        if self.verbose:
            print(f"  Décodé en {t_elapsed:.3f}s")

        meta['decode_time_s'] = t_elapsed
        meta['stats'] = stats
        meta['is_color'] = is_color
        return M_out, meta
