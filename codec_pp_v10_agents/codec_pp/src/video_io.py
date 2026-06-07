"""
Video I/O — Chargement de vidéo vers la matrice M et reconstruction.

Supporte :
  - Fichiers AVI/MP4 via OpenCV (si disponible)
  - Fichiers numpy .npy (pour tests)
  - Génération de vidéos synthétiques (pour tests)
"""

import numpy as np
from typing import Optional, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_video(
    path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    max_frames: Optional[int] = None,
    grayscale: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Charge une vidéo et retourne M[r, nl, nc] + fps.

    Args:
        path: chemin vers le fichier vidéo ou .npy
        width, height: redimensionnement optionnel
        max_frames: nombre max de frames à charger
        grayscale: convertir en niveaux de gris

    Returns:
        (M, fps) — M est de shape (r, nl, nc), dtype uint8 ou uint16
    """
    if path.endswith('.npy'):
        M = np.load(path)
        return M, 30  # fps par défaut

    if not HAS_CV2:
        raise ImportError(
            "OpenCV non disponible. Installez : pip install opencv-python"
        )

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Limites
    n_frames = min(total_frames, max_frames) if max_frames else total_frames
    out_h = height or orig_h
    out_w = width or orig_w

    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (out_h != orig_h) or (out_w != orig_w):
            frame = cv2.resize(frame, (out_w, out_h))
        frames.append(frame)

    cap.release()

    M = np.array(frames, dtype=np.uint8)
    if grayscale and len(M.shape) == 3:
        # shape = (r, nl, nc)
        pass
    elif not grayscale and len(M.shape) == 4:
        # Pour l'instant, on ne gère que le niveau de gris
        # Prendre la luminance Y = 0.299R + 0.587G + 0.114B
        M = np.round(
            0.299 * M[:, :, :, 2] +
            0.587 * M[:, :, :, 1] +
            0.114 * M[:, :, :, 0]
        ).astype(np.uint8)

    return M, fps


def save_video(
    M: np.ndarray,
    path: str,
    fps: int = 30
) -> None:
    """
    Reconstruit une vidéo AVI depuis la matrice M[r, nl, nc].
    """
    if not HAS_CV2:
        raise ImportError("OpenCV requis pour sauvegarder la vidéo")

    r, nl, nc = M.shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(path, fourcc, fps, (nc, nl), isColor=False)

    for t in range(r):
        frame = M[t].astype(np.uint8)
        out.write(frame)

    out.release()


def generate_synthetic_video(
    nl: int = 8,
    nc: int = 8,
    r: int = 16,
    n_colors: int = 4,
    change_prob: float = 0.15,
    seed: int = 42
) -> np.ndarray:
    """
    Génère une vidéo synthétique pour les tests.

    Chaque pixel a une probabilité change_prob de changer de couleur
    à chaque frame. Les couleurs sont tirées dans [0, n_colors-1].

    Returns:
        M de shape (r, nl, nc), dtype uint8
    """
    rng = np.random.RandomState(seed)
    M = np.zeros((r, nl, nc), dtype=np.uint8)

    # Frame initiale
    M[0] = rng.randint(0, n_colors, (nl, nc), dtype=np.uint8)

    # Frames suivantes
    for t in range(1, r):
        change_mask = rng.random((nl, nc)) < change_prob
        new_colors = rng.randint(0, n_colors, (nl, nc), dtype=np.uint8)
        M[t] = np.where(change_mask, new_colors, M[t - 1])

    return M
