"""
Encoder v4 — Macroblocs + multi-plan YCbCr + Huffman adaptatif.

Chaque plan est découpé en blocs de block_size × block_size.
Un seul Code Méthode (7 bits) par bloc au lieu d'un par pixel.
La table Huffman est locale au bloc.
"""

import numpy as np
import time
from typing import Optional, Dict, Any
from math import ceil

from .bitstream import BitWriter
from .format import encode_header, COL_GRAY, COL_COLOR
from .colorspace import split_video_planes, SUBSAMPLE_420, SUBSAMPLE_444
from .blocks import encode_plane_blocks
from .video_io import load_video


class PPVEncoder:

    def __init__(self, gop_size: int = 32, block_size: int = 8,
                 use_huffman: bool = True, use_prediction: bool = True,
                 verbose: bool = True, heuristic=None):
        self.gop_size = gop_size
        self.block_size = block_size
        self.use_huffman = use_huffman
        self.use_prediction = use_prediction
        self.verbose = verbose
        self.heuristic = heuristic  # DecisionHeuristic or None

    def encode(
        self,
        input_path: Optional[str] = None,
        output_path: str = "output.ppv",
        M: Optional[np.ndarray] = None,
        fps: int = 30,
        color_bits: int = 8,
        max_frames: Optional[int] = None,
        grayscale: bool = True,
        subsampling: int = SUBSAMPLE_420,
    ) -> Dict[str, Any]:

        t_start = time.time()

        # ── Chargement ─────────────────────────────────────────
        if M is not None:
            pass
        elif input_path is not None:
            M, fps = load_video(input_path, max_frames=max_frames,
                                grayscale=grayscale)
        else:
            raise ValueError("Fournir input_path ou M")

        # ── Préparer les plans ─────────────────────────────────
        is_color = (M.ndim == 4 and M.shape[3] == 3)

        if is_color:
            pi = split_video_planes(M, subsampling)
            planes = [
                ('Y',  pi['Y'],  pi['nl'],  pi['nc']),
                ('Cb', pi['Cb'], pi['nl_c'], pi['nc_c']),
                ('Cr', pi['Cr'], pi['nl_c'], pi['nc_c']),
            ]
            nl, nc, r_total = pi['nl'], pi['nc'], pi['r']
            col_flag = COL_COLOR
        else:
            r_total, nl, nc = M.shape
            planes = [('Y', M, nl, nc)]
            col_flag = COL_GRAY
            subsampling = SUBSAMPLE_444

        nb_planes = len(planes)
        nb_gops = ceil(r_total / self.gop_size)
        bs = self.block_size

        raw_bits = sum(r_total * pnl * pnc * color_bits
                       for _, _, pnl, pnc in planes)

        if self.verbose:
            mode = "HUFFMAN" if self.use_huffman else "FIXE"
            clr = "YCbCr 4:2:0" if is_color else "Gris"
            print(f"╔════════════════════════════════════════════════════╗")
            print(f"║  ENCODEUR PPV v4 — {clr}, {mode}, blocs {bs}×{bs}")
            print(f"╠════════════════════════════════════════════════════╣")
            print(f"║  {nl}×{nc}, {r_total} fr @ {fps} fps, "
                  f"{nb_planes} plans, {nb_gops} GOPs")
            if is_color:
                print(f"║  Chroma: {planes[1][2]}×{planes[1][3]}")
            print(f"╚════════════════════════════════════════════════════╝")

        # ── Stats ──────────────────────────────────────────────
        stats = {
            'nl': nl, 'nc': nc, 'r': r_total, 'fps': fps,
            'color_bits': color_bits, 'gop_size': self.gop_size,
            'block_size': bs, 'nb_gops': nb_gops,
            'nb_planes': nb_planes, 'is_color': is_color,
            'rep_counts': {1: 0, 2: 0, 3: 0, 4: 0},
            'mono_count': 0, 'total_pixels_encoded': 0,
            'bits_per_gop': [], 'raw_size_bits': raw_bits,
            'huffman_table_bits': 0,
            'plane_stats': {p[0]: {'bits': 0, 'mono': 0,
                                    'reps': {1:0,2:0,3:0,4:0}}
                            for p in planes},
        }

        # ── Encodage ──────────────────────────────────────────
        body_writer = BitWriter()

        for g_idx in range(nb_gops):
            t0 = g_idx * self.gop_size
            t1 = min(t0 + self.gop_size, r_total)
            gop_r = t1 - t0

            body_writer.write_uint16(g_idx)
            body_writer.write_uint16(gop_r)
            body_writer.write_bits(nb_planes, 2)
            body_writer.write_bits(bs, 8)  # block size

            gop_bits_start = body_writer.total_bits

            if self.verbose:
                print(f"\n  GOP {g_idx}/{nb_gops-1}  (r={gop_r})")

            for pname, pdata, pnl, pnc in planes:
                gop_plane = pdata[t0:t1]
                plane_start = body_writer.total_bits

                encode_plane_blocks(
                    body_writer, gop_plane, pnl, pnc, gop_r,
                    color_bits, col_flag, bs,
                    self.use_huffman, stats, pname,
                    self.use_prediction,
                    self.heuristic,
                )

                plane_bits = body_writer.total_bits - plane_start
                stats['plane_stats'][pname]['bits'] += plane_bits

                if self.verbose:
                    nb_blk = ceil(pnl/bs) * ceil(pnc/bs)
                    print(f"    {pname:>2s}: {plane_bits:>7,} bits  "
                          f"({nb_blk} blocs, "
                          f"{plane_bits/max(pnl*pnc,1):.1f} b/px)")

            gop_bits = body_writer.total_bits - gop_bits_start
            stats['bits_per_gop'].append(gop_bits)

        # ── Finalisation ───────────────────────────────────────
        body_data = body_writer.flush()
        total_body_bits = len(body_data) * 8

        header = encode_header(
            nl=nl, nc=nc, r=r_total,
            color_bits=color_bits, fps=fps,
            gop_size=self.gop_size, nb_gops=nb_gops,
            total_bits_body=total_body_bits,
            color_space=1 if is_color else 0,
        )

        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(body_data)

        t_elapsed = time.time() - t_start

        comp_bits = len(header) * 8 + total_body_bits
        ratio = raw_bits / comp_bits if comp_bits > 0 else 0
        savings = (1 - comp_bits / raw_bits) * 100 if raw_bits > 0 else 0

        stats.update({
            'output_path': output_path,
            'raw_size_bytes': raw_bits // 8,
            'compressed_size_bytes': len(header) + len(body_data),
            'compression_ratio': ratio,
            'savings_percent': savings,
            'encode_time_s': t_elapsed,
        })

        if self.verbose:
            print(f"\n{'═' * 58}")
            print(f"  Brut      : {raw_bits:>10,} bits  ({raw_bits//8:,} B)")
            print(f"  Compressé : {comp_bits:>10,} bits  "
                  f"({stats['compressed_size_bytes']:,} B)")
            print(f"  Ratio     : {ratio:.2f}x  ({savings:.1f}%)")
            print(f"  Temps     : {t_elapsed:.3f}s")
            tnm = sum(stats['rep_counts'].values())
            for rep in [1, 2, 3, 4]:
                cnt = stats['rep_counts'][rep]
                pct = (cnt/tnm*100) if tnm else 0
                print(f"    R{rep}: {cnt:>6}  ({pct:5.1f}%)")
            print(f"    Mono: {stats['mono_count']}")
            print(f"{'═' * 58}")

        return stats
