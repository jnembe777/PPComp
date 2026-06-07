"""
fast_encoder.py - Encodeur video optimise
==========================================

Optimisations:
- Multiprocessing pour parallelisation
- Vectorisation NumPy aggressive
- Cache LRU pour calculs repetes
- Numba JIT si disponible

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import warnings

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

from .loader import VideoLoader, VideoInfo
from .quantizer import ColorQuantizer, QuantizeMethod, Palette
from ..core.cartouche import Cartouche
from ..core.features import BlockFeatures
from ..utils.io_utils import BitWriter

# Try to import numba for JIT compilation
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range


@dataclass
class FastEncodingStats:
    """Statistiques d'encodage rapide."""
    total_frames: int = 0
    total_blocks: int = 0
    total_jumps: int = 0
    input_bytes: int = 0
    output_bits: int = 0
    compression_ratio: float = 0.0
    encoding_time_sec: float = 0.0
    fps_encoding: float = 0.0
    parallel_workers: int = 1


# === Fonctions optimisees Numba ===

@njit(cache=True)
def fast_count_changes(block: np.ndarray) -> int:
    """Compte les changements dans un bloc (optimise Numba)."""
    T, H, W = block.shape
    count = 0
    for t in range(1, T):
        for y in range(H):
            for x in range(W):
                if block[t, y, x] != block[t-1, y, x]:
                    count += 1
    return count


@njit(cache=True)
def fast_extract_jumps(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extrait les sauts rapidement."""
    T, H, W = block.shape
    max_jumps = T * H * W

    times = np.zeros(max_jumps, dtype=np.float64)
    marks = np.zeros(max_jumps, dtype=np.int32)
    count = 0

    for t in range(1, T):
        for y in range(H):
            for x in range(W):
                if block[t, y, x] != block[t-1, y, x]:
                    times[count] = float(t)
                    marks[count] = block[t, y, x]
                    count += 1

    return times[:count], marks[:count]


@njit(cache=True)
def fast_count_transitions(marks: np.ndarray) -> int:
    """Compte les transitions de couleur."""
    if len(marks) <= 1:
        return 0
    count = 0
    for i in range(1, len(marks)):
        if marks[i] != marks[i-1]:
            count += 1
    return count


def fast_entropy(counts: np.ndarray) -> float:
    """Calcule l'entropie rapidement."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def fast_logC(n: int, k: int) -> float:
    """log2(C(n,k)) optimise."""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 0.0
    if k > n - k:
        k = n - k

    # Utilise log-gamma pour precision
    from math import lgamma, log
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(2)


# === Traitement de bloc simplifie et rapide ===

def process_block_fast(block: np.ndarray, m: int) -> Tuple[int, int, int]:
    """
    Traite un bloc rapidement.

    Returns:
        (N, total_bits, N_trans)
    """
    T, H, W = block.shape
    r = T

    # Extraction rapide
    if HAS_NUMBA:
        times, marks = fast_extract_jumps(block)
    else:
        # Fallback numpy vectorise
        diff = block[1:] != block[:-1]
        positions = np.where(diff)
        times = positions[0].astype(np.float64) + 1
        marks = block[1:][diff].astype(np.int32)

    N = len(times)

    if N == 0:
        # Bloc constant - juste le cartouche
        return 0, 17, 0

    # Compte les transitions
    if HAS_NUMBA:
        N_trans = fast_count_transitions(marks)
    else:
        N_trans = np.sum(marks[1:] != marks[:-1]) if N > 1 else 0

    # Calcul rapide de l'entropie
    counts = np.bincount(marks, minlength=m)
    H_color = fast_entropy(counts)

    log2_m = np.log2(m) if m > 1 else 0

    # Calcul des couts (formules simplifiees)
    # L4 = log2(N+1) + logC(r, N) + C_color
    L_temp = np.log2(N + 1) + fast_logC(r, min(N, r))

    # Meilleur cout couleur
    C_Ba = log2_m + N_trans * log2_m if N_trans > 0 else log2_m
    C_Bb = N * log2_m
    C_Bc = N * H_color + m * (int(log2_m) + 1)  # Avec overhead Huffman

    C_color = min(C_Ba, C_Bb, C_Bc)

    # Monochromatique ?
    unique_colors = len(np.unique(marks))
    if unique_colors == 1:
        C_color = 0  # Pas de cout couleur

    total_bits = int(17 + L_temp + C_color)  # 17 bits cartouche

    return N, total_bits, N_trans


def process_block_batch(args: Tuple) -> List[Tuple[int, int, int]]:
    """Traite un batch de blocs (pour multiprocessing)."""
    blocks, m = args
    results = []
    for block in blocks:
        result = process_block_fast(block, m)
        results.append(result)
    return results


# === Quantificateur rapide ===

@njit(cache=True, parallel=True)
def _build_lut_numba(palette: np.ndarray, lut_size: int = 32) -> np.ndarray:
    """Construit la LUT avec Numba (parallele)."""
    lut = np.zeros((lut_size, lut_size, lut_size), dtype=np.uint8)
    n_colors = len(palette)

    for r in prange(lut_size):
        for g in range(lut_size):
            for b in range(lut_size):
                color_r = r * 8
                color_g = g * 8
                color_b = b * 8

                min_dist = 1e9
                best_idx = 0

                for k in range(n_colors):
                    dr = float(palette[k, 0]) - color_r
                    dg = float(palette[k, 1]) - color_g
                    db = float(palette[k, 2]) - color_b
                    dist = dr * dr + dg * dg + db * db

                    if dist < min_dist:
                        min_dist = dist
                        best_idx = k

                lut[r, g, b] = best_idx

    return lut


@njit(cache=True, parallel=True)
def _quantize_batch_numba(frames: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Quantifie un batch avec Numba (parallele)."""
    T, H, W, _ = frames.shape
    result = np.zeros((T, H, W), dtype=np.uint8)

    for t in prange(T):
        for y in range(H):
            for x in range(W):
                r = frames[t, y, x, 2] >> 3
                g = frames[t, y, x, 1] >> 3
                b = frames[t, y, x, 0] >> 3
                result[t, y, x] = lut[r, g, b]

    return result


class FastQuantizer:
    """Quantificateur optimise avec Numba."""

    def __init__(self, n_colors: int = 64):
        self.n_colors = n_colors
        self.palette = None
        self._lut = None

    def fit(self, frames: np.ndarray):
        """Calcule la palette avec k-means rapide."""
        n_samples = min(5000, frames.size // 3)
        flat = frames.reshape(-1, 3)
        indices = np.random.choice(len(flat), n_samples, replace=False)
        samples = flat[indices].astype(np.float32)

        # K-means simplifie (5 iterations suffisent)
        centers = samples[np.random.choice(n_samples, self.n_colors, replace=False)]

        for _ in range(5):
            diff = samples[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = (diff ** 2).sum(axis=2)
            labels = dist.argmin(axis=1)

            for k in range(self.n_colors):
                mask = labels == k
                if mask.sum() > 0:
                    centers[k] = samples[mask].mean(axis=0)

        self.palette = centers.astype(np.uint8)
        self._build_lut()

    def _build_lut(self):
        """Construit la LUT (utilise Numba si disponible)."""
        if HAS_NUMBA:
            self._lut = _build_lut_numba(self.palette, 32)
        else:
            # Fallback vectorise
            lut_size = 32
            r = np.arange(lut_size)[:, None, None] * 8
            g = np.arange(lut_size)[None, :, None] * 8
            b = np.arange(lut_size)[None, None, :] * 8

            colors = np.stack([r + np.zeros((lut_size, lut_size, lut_size)),
                              g + np.zeros((lut_size, lut_size, lut_size)),
                              b + np.zeros((lut_size, lut_size, lut_size))], axis=-1)

            colors_flat = colors.reshape(-1, 3)
            diff = colors_flat[:, np.newaxis, :] - self.palette[np.newaxis, :, :]
            dist = (diff ** 2).sum(axis=2)
            indices = dist.argmin(axis=1)

            self._lut = indices.reshape(lut_size, lut_size, lut_size).astype(np.uint8)

    def quantize_batch(self, frames: np.ndarray) -> np.ndarray:
        """Quantifie un batch de frames."""
        if HAS_NUMBA:
            return _quantize_batch_numba(frames, self._lut)
        else:
            # Vectorise sans Numba
            r = frames[:, :, :, 2] >> 3
            g = frames[:, :, :, 1] >> 3
            b = frames[:, :, :, 0] >> 3
            return self._lut[r, g, b]


# === Encodeur rapide ===

class FastVideoEncoder:
    """
    Encodeur video haute performance.

    Utilise multiprocessing et optimisations NumPy/Numba.
    """

    def __init__(
        self,
        block_size: int = 16,
        block_frames: int = 32,
        n_colors: int = 64,
        n_workers: Optional[int] = None
    ):
        self.block_size = block_size
        self.block_frames = block_frames
        self.n_colors = n_colors
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)

        self.quantizer = FastQuantizer(n_colors=n_colors)

    def encode(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> FastEncodingStats:
        """
        Encode une video rapidement.

        Args:
            video_path: Chemin de la video
            max_frames: Limite de frames
            progress_callback: Callback(progress, message)

        Returns:
            Statistiques d'encodage
        """
        start_time = time.time()

        # Charge la video
        loader = VideoLoader(video_path)
        info = loader.info

        total_frames = min(info.frame_count, max_frames) if max_frames else info.frame_count

        if progress_callback:
            progress_callback(0.0, f"Chargement {info.width}x{info.height}...")

        print(f"[FAST] Video: {info.width}x{info.height}, {total_frames} frames")
        print(f"[FAST] Workers: {self.n_workers}, Numba: {HAS_NUMBA}")

        # Phase 1: Fit quantizer on sample
        t_load = time.time()
        if progress_callback:
            progress_callback(0.05, "Echantillonnage...")

        loader.seek(0)
        sample_frames = loader.read_frames(min(20, total_frames))
        self.quantizer.fit(sample_frames)
        del sample_frames

        print(f"[FAST] Init quantizer: {time.time() - t_load:.1f}s")

        # Phase 2: Process in chunks to avoid memory issues
        H, W = info.height, info.width
        chunk_size = 256  # Frames per chunk
        n_blocks_spatial = ((H + self.block_size - 1) // self.block_size) * \
                          ((W + self.block_size - 1) // self.block_size)

        blocks = []
        t_quant = time.time()

        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)

            if progress_callback:
                progress_callback(0.05 + 0.20 * chunk_start / total_frames,
                                f"Chunk {chunk_start}-{chunk_end}...")

            # Read chunk
            loader.seek(chunk_start)
            chunk_frames = loader.read_frames(chunk_end - chunk_start)

            if len(chunk_frames) == 0:
                break

            # Quantize chunk
            indexed_chunk = self.quantizer.quantize_batch(chunk_frames)
            del chunk_frames

            # Extract blocks from chunk
            T_chunk = len(indexed_chunk)

            for t_start in range(0, T_chunk, self.block_frames):
                t_end = min(t_start + self.block_frames, T_chunk)
                temporal = indexed_chunk[t_start:t_end]

                for y in range(0, H, self.block_size):
                    for x in range(0, W, self.block_size):
                        y_end = min(y + self.block_size, H)
                        x_end = min(x + self.block_size, W)

                        block = temporal[:, y:y_end, x:x_end].copy()
                        blocks.append(block)

            del indexed_chunk

        loader.close()

        n_blocks = len(blocks)
        print(f"[FAST] Quantification: {time.time() - t_quant:.1f}s")
        print(f"[FAST] Blocs: {n_blocks}")

        # Phase 3: Traitement parallele
        t_process = time.time()
        if progress_callback:
            progress_callback(0.30, f"Encodage {n_blocks} blocs...")

        total_N = 0
        total_bits = 0
        total_trans = 0

        if self.n_workers > 1 and n_blocks > 100:
            # Multiprocessing par batches
            batch_size = max(10, n_blocks // (self.n_workers * 4))
            batches = []

            for i in range(0, n_blocks, batch_size):
                batch = blocks[i:i + batch_size]
                batches.append((batch, self.n_colors))

            # ThreadPoolExecutor est plus rapide pour ce cas (pas de pickle overhead)
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(process_block_batch, b) for b in batches]

                completed = 0
                for future in as_completed(futures):
                    results = future.result()
                    for N, bits, trans in results:
                        total_N += N
                        total_bits += bits
                        total_trans += trans

                    completed += 1
                    if progress_callback:
                        progress_callback(0.30 + 0.65 * completed / len(batches),
                                         f"Batch {completed}/{len(batches)}")
        else:
            # Traitement sequentiel
            for i, block in enumerate(blocks):
                N, bits, trans = process_block_fast(block, self.n_colors)
                total_N += N
                total_bits += bits
                total_trans += trans

                if progress_callback and i % 100 == 0:
                    progress_callback(0.30 + 0.65 * i / n_blocks,
                                     f"Bloc {i}/{n_blocks}")

        print(f"[FAST] Traitement: {time.time() - t_process:.1f}s")

        # Stats finales
        elapsed = time.time() - start_time
        input_bytes = total_frames * H * W * 3
        output_bytes = total_bits // 8

        stats = FastEncodingStats(
            total_frames=total_frames,
            total_blocks=n_blocks,
            total_jumps=total_N,
            input_bytes=input_bytes,
            output_bits=total_bits,
            compression_ratio=input_bytes / output_bytes if output_bytes > 0 else 0,
            encoding_time_sec=elapsed,
            fps_encoding=total_frames / elapsed if elapsed > 0 else 0,
            parallel_workers=self.n_workers
        )

        if progress_callback:
            progress_callback(1.0, "Termine!")

        return stats

    def print_report(self, stats: FastEncodingStats):
        """Affiche le rapport."""
        print("\n" + "=" * 60)
        print("RAPPORT ENCODAGE RAPIDE LMD-PPV")
        print("=" * 60)

        print(f"\nFrames:       {stats.total_frames}")
        print(f"Blocs:        {stats.total_blocks}")
        print(f"Sauts:        {stats.total_jumps}")

        print(f"\nEntree:       {stats.input_bytes / 1024 / 1024:.2f} MB")
        print(f"Sortie:       {stats.output_bits / 8 / 1024:.2f} KB")
        print(f"Compression:  {stats.compression_ratio:.1f}x")

        print(f"\nTemps:        {stats.encoding_time_sec:.1f}s")
        print(f"Vitesse:      {stats.fps_encoding:.1f} fps")
        print(f"Workers:      {stats.parallel_workers}")
        print(f"Numba:        {'Oui' if HAS_NUMBA else 'Non'}")

        print("=" * 60)


def fast_encode(
    video_path: str,
    max_frames: Optional[int] = None,
    n_colors: int = 64,
    block_size: int = 16
) -> FastEncodingStats:
    """
    Fonction utilitaire pour encodage rapide.
    """
    encoder = FastVideoEncoder(
        block_size=block_size,
        n_colors=n_colors
    )

    stats = encoder.encode(video_path, max_frames)
    encoder.print_report(stats)

    return stats
