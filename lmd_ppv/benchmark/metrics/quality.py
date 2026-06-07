"""
quality.py - Metriques de qualite video
========================================

Metriques implementees:
- PSNR: Peak Signal-to-Noise Ratio (dB)
- SSIM: Structural Similarity Index (0-1)

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
import subprocess
import json
import tempfile


@dataclass
class QualityMetrics:
    """Metriques de qualite d'une video."""
    psnr: float  # dB, plus c'est haut, mieux c'est
    ssim: float  # 0-1, plus c'est haut, mieux c'est
    psnr_min: float = 0.0
    psnr_max: float = 0.0
    ssim_min: float = 0.0
    ssim_max: float = 0.0

    # Metriques par frame (optionnel)
    psnr_per_frame: Optional[List[float]] = None
    ssim_per_frame: Optional[List[float]] = None

    @property
    def psnr_variance(self) -> float:
        if self.psnr_per_frame:
            return np.var(self.psnr_per_frame)
        return 0.0

    @property
    def ssim_variance(self) -> float:
        if self.ssim_per_frame:
            return np.var(self.ssim_per_frame)
        return 0.0

    def to_dict(self) -> Dict:
        return {
            'psnr': self.psnr,
            'psnr_min': self.psnr_min,
            'psnr_max': self.psnr_max,
            'psnr_variance': self.psnr_variance,
            'ssim': self.ssim,
            'ssim_min': self.ssim_min,
            'ssim_max': self.ssim_max,
            'ssim_variance': self.ssim_variance,
        }


def compute_psnr(
    original: np.ndarray,
    compressed: np.ndarray,
    max_value: float = 255.0
) -> float:
    """
    Calcule le PSNR entre deux images/frames.

    Args:
        original: Image originale (H, W) ou (H, W, C)
        compressed: Image compressee
        max_value: Valeur maximale des pixels (255 pour 8 bits)

    Returns:
        PSNR en dB (inf si identiques)
    """
    if original.shape != compressed.shape:
        raise ValueError("Les images doivent avoir la meme taille")

    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)

    if mse == 0:
        return float('inf')

    psnr = 10 * np.log10((max_value ** 2) / mse)
    return psnr


def compute_ssim(
    original: np.ndarray,
    compressed: np.ndarray,
    win_size: int = 7,
    data_range: float = 255.0
) -> float:
    """
    Calcule le SSIM entre deux images/frames.

    Implementation simplifiee du Structural Similarity Index.

    Args:
        original: Image originale
        compressed: Image compressee
        win_size: Taille de la fenetre (doit etre impair)
        data_range: Plage dynamique des donnees

    Returns:
        SSIM entre 0 et 1
    """
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(
            original, compressed,
            win_size=win_size,
            data_range=data_range,
            channel_axis=-1 if len(original.shape) == 3 else None
        )
    except ImportError:
        # Fallback: implementation simplifiee
        return _compute_ssim_simple(original, compressed, data_range)


def _compute_ssim_simple(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: float = 255.0
) -> float:
    """Implementation simplifiee du SSIM."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim


def compute_video_quality_ffmpeg(
    original_path: Path,
    compressed_path: Path,
    metrics: List[str] = ['psnr', 'ssim']
) -> QualityMetrics:
    """
    Calcule les metriques de qualite via FFmpeg.

    Utilise les filtres FFmpeg pour calculer PSNR et SSIM
    de maniere efficace.

    Args:
        original_path: Chemin de la video originale
        compressed_path: Chemin de la video compressee
        metrics: Liste des metriques a calculer

    Returns:
        QualityMetrics
    """
    results = {'psnr': 0.0, 'ssim': 0.0}

    # Construire le filtre FFmpeg
    filters = []
    if 'psnr' in metrics:
        filters.append('psnr')
    if 'ssim' in metrics:
        filters.append('ssim')

    if not filters:
        return QualityMetrics(psnr=0.0, ssim=0.0)

    filter_str = ','.join(filters)

    # Executer FFmpeg
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name

        cmd = [
            'ffmpeg', '-y',
            '-i', str(compressed_path),
            '-i', str(original_path),
            '-lavfi', f'[0:v][1:v]{filter_str}',
            '-f', 'null', '-'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        # Parser la sortie
        stderr = result.stderr

        # Parser PSNR
        if 'psnr' in metrics:
            psnr_values = []
            for line in stderr.split('\n'):
                if 'PSNR' in line and 'average' in line.lower():
                    try:
                        # Format: "PSNR y:XX.XX u:XX.XX v:XX.XX average:XX.XX"
                        avg_match = line.split('average:')[-1].strip().split()[0]
                        results['psnr'] = float(avg_match)
                    except (ValueError, IndexError):
                        pass
                elif '[Parsed_psnr' in line:
                    try:
                        # Parser les valeurs par frame
                        parts = line.split()
                        for part in parts:
                            if 'psnr_avg:' in part:
                                val = float(part.split(':')[1])
                                psnr_values.append(val)
                    except (ValueError, IndexError):
                        pass

        # Parser SSIM
        if 'ssim' in metrics:
            ssim_values = []
            for line in stderr.split('\n'):
                if 'SSIM' in line and 'All:' in line:
                    try:
                        # Format: "SSIM Y:X.XXXX U:X.XXXX V:X.XXXX All:X.XXXX"
                        all_match = line.split('All:')[-1].strip().split()[0]
                        results['ssim'] = float(all_match)
                    except (ValueError, IndexError):
                        pass
                elif '[Parsed_ssim' in line:
                    try:
                        parts = line.split()
                        for part in parts:
                            if 'All:' in part:
                                val = float(part.split(':')[1])
                                ssim_values.append(val)
                    except (ValueError, IndexError):
                        pass

    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass

    return QualityMetrics(
        psnr=results.get('psnr', 0.0),
        ssim=results.get('ssim', 0.0)
    )


def compute_video_quality_numpy(
    original_path: Path,
    compressed_path: Path,
    max_frames: Optional[int] = None
) -> QualityMetrics:
    """
    Calcule les metriques de qualite frame par frame avec NumPy.

    Plus lent que FFmpeg mais ne necessite pas FFmpeg installe.

    Args:
        original_path: Video originale
        compressed_path: Video compressee
        max_frames: Limite de frames a analyser

    Returns:
        QualityMetrics
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV requis pour compute_video_quality_numpy")

    cap_orig = cv2.VideoCapture(str(original_path))
    cap_comp = cv2.VideoCapture(str(compressed_path))

    psnr_values = []
    ssim_values = []
    frame_count = 0

    while True:
        ret1, frame1 = cap_orig.read()
        ret2, frame2 = cap_comp.read()

        if not ret1 or not ret2:
            break

        if max_frames and frame_count >= max_frames:
            break

        # Convertir en niveaux de gris si couleur
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2

        # Redimensionner si necessaire
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        psnr = compute_psnr(gray1, gray2)
        ssim = compute_ssim(gray1, gray2)

        if not np.isinf(psnr):
            psnr_values.append(psnr)
        ssim_values.append(ssim)

        frame_count += 1

    cap_orig.release()
    cap_comp.release()

    if not psnr_values:
        return QualityMetrics(psnr=0.0, ssim=0.0)

    return QualityMetrics(
        psnr=np.mean(psnr_values),
        ssim=np.mean(ssim_values),
        psnr_min=np.min(psnr_values),
        psnr_max=np.max(psnr_values),
        ssim_min=np.min(ssim_values),
        ssim_max=np.max(ssim_values),
        psnr_per_frame=psnr_values,
        ssim_per_frame=ssim_values
    )


def compute_quality(
    original_path: Path,
    compressed_path: Path,
    method: str = 'ffmpeg',
    **kwargs
) -> QualityMetrics:
    """
    Interface principale pour calculer la qualite.

    Args:
        original_path: Video originale
        compressed_path: Video compressee
        method: 'ffmpeg' ou 'numpy'
        **kwargs: Arguments supplementaires

    Returns:
        QualityMetrics
    """
    if method == 'ffmpeg':
        return compute_video_quality_ffmpeg(original_path, compressed_path)
    elif method == 'numpy':
        return compute_video_quality_numpy(original_path, compressed_path, **kwargs)
    else:
        raise ValueError(f"Methode inconnue: {method}")
