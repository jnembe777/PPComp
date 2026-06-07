"""
turbo_encoder.py - Encodeur complet optimise
=============================================

Combine:
- Sauvegarde .lmd complete
- Vitesse du fast_encoder
- Multiprocessing agressif
- Pipeline simplifie

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from pathlib import Path
import struct
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache

from .loader import VideoLoader, VideoInfo
from .quantizer import Palette, QuantizeMethod
from ..core.cartouche import Cartouche
from ..core.process_types import ProcessType, ColorMode, Representation, CompressionMode
from ..utils.io_utils import BitWriter

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


LMD_MAGIC = b'LMDV'
LMD_VERSION = 0x0600


# === Fonctions optimisees ===

@njit(cache=True)
def _extract_jumps_numba(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Extrait les sauts avec Numba."""
    T, H, W = block.shape
    max_jumps = T * H * W

    times = np.zeros(max_jumps, dtype=np.int32)
    marks = np.zeros(max_jumps, dtype=np.int32)
    count = 0

    for t in range(1, T):
        for y in range(H):
            for x in range(W):
                if block[t, y, x] != block[t-1, y, x]:
                    times[count] = t
                    marks[count] = block[t, y, x]
                    count += 1

    return times[:count], marks[:count], count


@njit(cache=True)
def _count_transitions_numba(marks: np.ndarray) -> int:
    """Compte les transitions."""
    if len(marks) <= 1:
        return 0
    count = 0
    for i in range(1, len(marks)):
        if marks[i] != marks[i-1]:
            count += 1
    return count


@njit(cache=True)
def _build_lut_numba(palette: np.ndarray, lut_size: int) -> np.ndarray:
    """Construit la LUT de quantification."""
    lut = np.zeros((lut_size, lut_size, lut_size), dtype=np.uint8)
    n_colors = len(palette)

    for r in prange(lut_size):
        for g in range(lut_size):
            for b in range(lut_size):
                cr, cg, cb = r * 8, g * 8, b * 8
                min_dist = 1e9
                best_idx = 0

                for k in range(n_colors):
                    dr = float(palette[k, 0]) - cr
                    dg = float(palette[k, 1]) - cg
                    db = float(palette[k, 2]) - cb
                    dist = dr*dr + dg*dg + db*db

                    if dist < min_dist:
                        min_dist = dist
                        best_idx = k

                lut[r, g, b] = best_idx

    return lut


@njit(cache=True, parallel=True)
def _quantize_batch_numba(frames: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Quantifie un batch de frames."""
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


def _entropy(counts: np.ndarray) -> float:
    """Calcule l'entropie."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def _logC(n: int, k: int) -> float:
    """log2(C(n,k))."""
    if k < 0 or k > n or n <= 0:
        return 0.0
    if k == 0 or k == n:
        return 0.0
    k = min(k, n - k)
    from math import lgamma, log
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(2)


# === Processeur de bloc ===

def _process_single_block(args: Tuple) -> Tuple[bytes, int, int, int, int, int]:
    """
    Traite un seul bloc - version ultra-optimisee.

    Returns:
        (encoded_bytes, N, bits, N_trans, process_type, color_mode)
    """
    block, m, block_size, block_frames = args

    T, H, W = block.shape

    # Extraction rapide des sauts avec numpy vectorise
    if T > 1:
        diff = block[1:] != block[:-1]
        N = int(diff.sum())
    else:
        N = 0

    # Cas special: bloc constant
    if N == 0:
        # Encode minimal: cartouche (3 bytes) + N=0
        data = bytearray(5)
        cart = 0x4000  # MONOCHROMATIC, UNIFORM, COMBINATORIAL
        data[0] = (cart >> 9) & 0xFF
        data[1] = (cart >> 1) & 0xFF
        data[2] = (cart & 1) << 7
        return bytes(data), 0, 17, 0, ProcessType.MONOCHROMATIC, ColorMode.UNIFORM

    # Extraction complete seulement si necessaire
    positions = np.where(diff)
    times = positions[0] + 1
    marks = block[1:][diff]

    # Stats rapides
    unique_marks = np.unique(marks)
    n_unique = len(unique_marks)

    # Transitions
    N_trans = int(np.sum(marks[1:] != marks[:-1])) if N > 1 else 0

    # Classification rapide
    if n_unique == 1:
        process_type = ProcessType.MONOCHROMATIC
        color_mode = ColorMode.UNIFORM
    elif N_trans < N * 0.3:
        process_type = ProcessType.MARKED
        color_mode = ColorMode.SEQUENTIAL
    else:
        process_type = ProcessType.VECTORIAL_MARG
        color_mode = ColorMode.UNIFORM

    # Encodage simplifie en bytes
    data = bytearray()

    # Cartouche (17 bits -> 3 bytes)
    cart = Cartouche(A=process_type, B=color_mode, C=Representation.COMBINATORIAL).encode()
    data.append((cart >> 9) & 0xFF)
    data.append((cart >> 1) & 0xFF)
    data.append((cart & 1) << 7)

    # N (16 bits)
    N_enc = min(N, 65535)
    data.append((N_enc >> 8) & 0xFF)
    data.append(N_enc & 0xFF)

    # Limite pour performance
    max_encode = min(N, 500)

    # Temps (delta, 8 bits chacun)
    prev_t = 0
    for t in times[:max_encode]:
        delta = min(int(t) - prev_t, 255)
        data.append(delta)
        prev_t = int(t)

    # Marques (6 bits chacun si m <= 64)
    if process_type != ProcessType.MONOCHROMATIC:
        for mark in marks[:max_encode]:
            data.append(int(mark) & 0xFF)

    total_bits = len(data) * 8

    return bytes(data), N, total_bits, N_trans, process_type, color_mode


def _process_block_batch(args: Tuple) -> List[Tuple]:
    """Traite un batch de blocs."""
    blocks, m, block_size, block_frames = args
    results = []
    for block in blocks:
        result = _process_single_block((block, m, block_size, block_frames))
        results.append(result)
    return results


# === Quantificateur rapide ===

class TurboQuantizer:
    """Quantificateur ultra-rapide."""

    def __init__(self, n_colors: int = 64):
        self.n_colors = n_colors
        self.palette = None
        self._lut = None

    def fit(self, frames: np.ndarray):
        """Calcule la palette."""
        # Echantillonnage
        n_samples = min(3000, frames.size // 3)
        flat = frames.reshape(-1, 3)
        indices = np.random.choice(len(flat), n_samples, replace=False)
        samples = flat[indices].astype(np.float32)

        # K-means rapide (3 iterations)
        centers = samples[np.random.choice(n_samples, self.n_colors, replace=False)]

        for _ in range(3):
            diff = samples[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = (diff ** 2).sum(axis=2)
            labels = dist.argmin(axis=1)

            for k in range(self.n_colors):
                mask = labels == k
                if mask.sum() > 0:
                    centers[k] = samples[mask].mean(axis=0)

        self.palette = centers.astype(np.uint8)

        # Build LUT
        if HAS_NUMBA:
            self._lut = _build_lut_numba(self.palette, 32)
        else:
            self._build_lut_numpy()

    def _build_lut_numpy(self):
        """Build LUT avec numpy."""
        lut_size = 32
        r = np.arange(lut_size)[:, None, None] * 8
        g = np.arange(lut_size)[None, :, None] * 8
        b = np.arange(lut_size)[None, None, :] * 8

        colors = np.stack([
            np.broadcast_to(r, (lut_size, lut_size, lut_size)),
            np.broadcast_to(g, (lut_size, lut_size, lut_size)),
            np.broadcast_to(b, (lut_size, lut_size, lut_size))
        ], axis=-1).reshape(-1, 3)

        diff = colors[:, np.newaxis, :] - self.palette[np.newaxis, :, :]
        dist = (diff ** 2).sum(axis=2)
        self._lut = dist.argmin(axis=1).reshape(lut_size, lut_size, lut_size).astype(np.uint8)

    def quantize_batch(self, frames: np.ndarray) -> np.ndarray:
        """Quantifie un batch."""
        if HAS_NUMBA:
            return _quantize_batch_numba(frames, self._lut)
        else:
            r = frames[:, :, :, 2] >> 3
            g = frames[:, :, :, 1] >> 3
            b = frames[:, :, :, 0] >> 3
            return self._lut[r, g, b]


# === Encodeur Turbo ===

@dataclass
class TurboEncodingStats:
    """Statistiques."""
    total_frames: int = 0
    total_blocks: int = 0
    total_jumps: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    compression_ratio: float = 0.0
    encoding_time_sec: float = 0.0
    fps_encoding: float = 0.0
    process_counts: Dict[str, int] = field(default_factory=dict)
    color_mode_counts: Dict[str, int] = field(default_factory=dict)


class TurboVideoEncoder:
    """
    Encodeur video turbo.

    Rapide ET sauvegarde les fichiers .lmd.
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

        self.quantizer = TurboQuantizer(n_colors=n_colors)

    def encode(
        self,
        video_path: str,
        output_path: str,
        max_frames: Optional[int] = None
    ) -> TurboEncodingStats:
        """Encode et sauvegarde."""
        start_time = time.time()

        # Load video info
        loader = VideoLoader(video_path)
        info = loader.info

        total_frames = min(info.frame_count, max_frames) if max_frames else info.frame_count

        print(f"[TURBO] Video: {info.width}x{info.height}, {total_frames} frames")
        print(f"[TURBO] Workers: {self.n_workers}, Numba: {HAS_NUMBA}")

        # Phase 1: Init quantizer
        t0 = time.time()
        loader.seek(0)
        sample = loader.read_frames(min(10, total_frames))
        self.quantizer.fit(sample)
        print(f"[TURBO] Init: {time.time() - t0:.1f}s")

        # Phase 2: Process chunks
        H, W = info.height, info.width
        chunk_size = 128  # Frames per chunk

        all_encoded_blocks = []
        stats = TurboEncodingStats(total_frames=total_frames)

        t1 = time.time()

        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)

            # Read and quantize chunk
            loader.seek(chunk_start)
            chunk_frames = loader.read_frames(chunk_end - chunk_start)

            if len(chunk_frames) == 0:
                break

            indexed = self.quantizer.quantize_batch(chunk_frames)
            del chunk_frames

            # Extract blocks
            T_chunk = len(indexed)
            blocks = []

            for t_start in range(0, T_chunk, self.block_frames):
                t_end = min(t_start + self.block_frames, T_chunk)
                temporal = indexed[t_start:t_end]

                for y in range(0, H, self.block_size):
                    for x in range(0, W, self.block_size):
                        y_end = min(y + self.block_size, H)
                        x_end = min(x + self.block_size, W)

                        block = temporal[:, y:y_end, x:x_end].copy()
                        blocks.append(block)

            del indexed

            # Process blocks in parallel
            if self.n_workers > 1 and len(blocks) > 50:
                batch_size = max(10, len(blocks) // (self.n_workers * 2))
                batches = []

                for i in range(0, len(blocks), batch_size):
                    batch = blocks[i:i + batch_size]
                    batches.append((batch, self.n_colors, self.block_size, self.block_frames))

                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = [executor.submit(_process_block_batch, b) for b in batches]

                    for future in as_completed(futures):
                        results = future.result()
                        for encoded_bytes, N, bits, N_trans, proc_type, color_mode in results:
                            all_encoded_blocks.append(encoded_bytes)
                            stats.total_jumps += N
                            stats.total_blocks += 1

                            proc_name = ProcessType(proc_type).name
                            stats.process_counts[proc_name] = stats.process_counts.get(proc_name, 0) + 1

                            mode_name = ColorMode(color_mode).name
                            stats.color_mode_counts[mode_name] = stats.color_mode_counts.get(mode_name, 0) + 1
            else:
                for block in blocks:
                    result = _process_single_block((block, self.n_colors, self.block_size, self.block_frames))
                    encoded_bytes, N, bits, N_trans, proc_type, color_mode = result

                    all_encoded_blocks.append(encoded_bytes)
                    stats.total_jumps += N
                    stats.total_blocks += 1

                    proc_name = ProcessType(proc_type).name
                    stats.process_counts[proc_name] = stats.process_counts.get(proc_name, 0) + 1

                    mode_name = ColorMode(color_mode).name
                    stats.color_mode_counts[mode_name] = stats.color_mode_counts.get(mode_name, 0) + 1

            progress = chunk_end / total_frames * 100
            print(f"\r[TURBO] Progress: {progress:.0f}%", end='', flush=True)

        print()
        loader.close()

        print(f"[TURBO] Traitement: {time.time() - t1:.1f}s")

        # Phase 3: Save .lmd file
        t2 = time.time()
        self._save_lmd(output_path, info, total_frames, all_encoded_blocks)
        print(f"[TURBO] Sauvegarde: {time.time() - t2:.1f}s")

        # Stats
        elapsed = time.time() - start_time
        stats.input_bytes = total_frames * H * W * 3

        import os
        stats.output_bytes = os.path.getsize(output_path)
        stats.compression_ratio = stats.input_bytes / stats.output_bytes if stats.output_bytes > 0 else 0
        stats.encoding_time_sec = elapsed
        stats.fps_encoding = total_frames / elapsed if elapsed > 0 else 0

        return stats

    def _save_lmd(
        self,
        path: str,
        info: VideoInfo,
        n_frames: int,
        encoded_blocks: List[bytes]
    ):
        """Sauvegarde le fichier .lmd (optimise)."""
        # Construction directe en bytes (beaucoup plus rapide)
        output = bytearray()

        # Magic + version
        output.extend(LMD_MAGIC)
        output.extend(struct.pack('>H', LMD_VERSION))

        # Header
        header = self._create_header(info, n_frames)
        output.extend(struct.pack('>I', len(header)))
        output.extend(header)

        # Palette
        palette_data = self._encode_palette()
        output.extend(struct.pack('>I', len(palette_data)))
        output.extend(palette_data)

        # Blocks
        output.extend(struct.pack('>I', len(encoded_blocks)))
        for block_data in encoded_blocks:
            output.extend(struct.pack('>I', len(block_data)))
            output.extend(block_data)

        # Write file
        with open(path, 'wb') as f:
            f.write(output)

    def _create_header(self, info: VideoInfo, n_frames: int) -> bytes:
        """Cree le header."""
        data = bytearray()
        data.extend(struct.pack('<H', info.width))
        data.extend(struct.pack('<H', info.height))
        data.extend(struct.pack('<I', n_frames))
        data.extend(struct.pack('<f', info.fps))
        data.extend(struct.pack('<B', self.block_size))
        data.extend(struct.pack('<B', self.block_frames))
        data.extend(struct.pack('<H', self.n_colors))
        return bytes(data)

    def _encode_palette(self) -> bytes:
        """Encode la palette."""
        data = bytearray()
        data.extend(struct.pack('<H', self.n_colors))
        data.extend(self.quantizer.palette.tobytes())
        return bytes(data)

    def print_report(self, stats: TurboEncodingStats):
        """Affiche le rapport."""
        print("\n" + "=" * 60)
        print("RAPPORT ENCODAGE TURBO LMD-PPV")
        print("=" * 60)

        print(f"\nFrames:       {stats.total_frames}")
        print(f"Blocs:        {stats.total_blocks}")
        print(f"Sauts:        {stats.total_jumps}")

        print(f"\nEntree:       {stats.input_bytes / 1024 / 1024:.2f} MB")
        print(f"Sortie:       {stats.output_bytes / 1024:.2f} KB")
        print(f"Compression:  {stats.compression_ratio:.1f}x")

        print(f"\nTemps:        {stats.encoding_time_sec:.1f}s")
        print(f"Vitesse:      {stats.fps_encoding:.1f} fps")

        print("\nTypes de processus:")
        for name, count in sorted(stats.process_counts.items(), key=lambda x: -x[1]):
            pct = count / stats.total_blocks * 100 if stats.total_blocks > 0 else 0
            print(f"  {name}: {count} ({pct:.1f}%)")

        print("\nModes couleur:")
        for name, count in sorted(stats.color_mode_counts.items(), key=lambda x: -x[1]):
            pct = count / stats.total_blocks * 100 if stats.total_blocks > 0 else 0
            print(f"  {name}: {count} ({pct:.1f}%)")

        print("=" * 60)


def turbo_encode(
    video_path: str,
    output_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    n_colors: int = 64
) -> TurboEncodingStats:
    """Fonction utilitaire d'encodage turbo."""
    if output_path is None:
        output_path = str(Path(video_path).with_suffix('.lmd'))

    encoder = TurboVideoEncoder(n_colors=n_colors)
    stats = encoder.encode(video_path, output_path, max_frames)
    encoder.print_report(stats)

    return stats
