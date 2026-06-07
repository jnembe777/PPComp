#!/usr/bin/env python3
"""
test_video.py - Test de l'integration video
============================================

Cree une video synthetique et teste l'encodage/decodage.
"""

import numpy as np
import sys
from pathlib import Path

# Ajoute le repertoire au path
sys.path.insert(0, str(Path(__file__).parent))


def create_test_video(path: str, n_frames: int = 60, width: int = 128, height: int = 128):
    """Cree une video de test avec des formes animees."""
    try:
        import cv2
    except ImportError:
        print("[ERREUR] OpenCV requis: pip install opencv-python")
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height))

    for t in range(n_frames):
        # Fond qui change de couleur
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Couleur de fond cyclique
        bg_color = [
            int(50 + 50 * np.sin(t * 0.1)),
            int(50 + 50 * np.sin(t * 0.15)),
            int(100 + 50 * np.sin(t * 0.2))
        ]
        frame[:, :] = bg_color

        # Rectangle qui bouge
        x = int(20 + 60 * np.sin(t * 0.1))
        y = int(20 + 40 * np.sin(t * 0.15))
        cv2.rectangle(frame, (x, y), (x + 30, y + 30), (255, 100, 50), -1)

        # Cercle qui bouge
        cx = int(width // 2 + 40 * np.cos(t * 0.12))
        cy = int(height // 2 + 30 * np.sin(t * 0.12))
        cv2.circle(frame, (cx, cy), 15, (50, 255, 100), -1)

        # Ligne
        cv2.line(frame, (0, t % height), (width, (t + 50) % height), (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"[OK] Video de test creee: {path}")
    return True


def test_full_pipeline():
    """Test complet du pipeline video."""
    print("=" * 60)
    print("TEST INTEGRATION VIDEO LMD-PPV")
    print("=" * 60)

    # Cree la video de test
    test_path = "test_sample.mp4"
    print("\n1. Creation de la video de test...")

    if not create_test_video(test_path, n_frames=60, width=128, height=128):
        return False

    # Encode
    print("\n2. Encodage LMD-PPV...")
    from src.video.encoder import VideoEncoder
    from src.video.quantizer import QuantizeMethod

    encoder = VideoEncoder(
        block_size=16,
        block_frames=32,
        n_colors=64,
        quantize_method=QuantizeMethod.KMEANS
    )

    output_path = "test_sample.lmd"
    encoded = encoder.encode(test_path, output_path)

    print(f"\n   Ratio de compression: {encoded.stats.compression_ratio:.1f}x")
    print(f"   Taille: {encoded.stats.output_bytes / 1024:.1f} KB")

    # Info
    print("\n3. Verification du fichier...")
    from src.video.decoder import VideoDecoder

    decoder = VideoDecoder()
    header = decoder.load(output_path)

    print(f"   Resolution: {header.width}x{header.height}")
    print(f"   Frames: {header.n_frames}")
    print(f"   Blocs: {len(decoder.blocks)}")

    # Decode une frame
    print("\n4. Decodage de la premiere frame...")
    frame = decoder.decode_frame(0)
    print(f"   Shape: {frame.shape}")
    print(f"   Range: [{frame.min()}, {frame.max()}]")

    # Nettoyage (optionnel)
    # Path(test_path).unlink()
    # Path(output_path).unlink()

    print("\n" + "=" * 60)
    print("[OK] Test d'integration video reussi!")
    print("=" * 60)

    return True


def test_existing_video(video_path: str):
    """Test avec une video existante."""
    print(f"Test avec: {video_path}")

    from src.video.loader import VideoLoader
    from src.video.quantizer import ColorQuantizer, QuantizeMethod

    # Charge la video
    loader = VideoLoader(video_path)
    print(f"Info: {loader.info}")

    # Lit quelques frames
    frames = loader.read_frames(10)
    print(f"Frames lues: {frames.shape}")

    # Quantifie
    quantizer = ColorQuantizer(n_colors=64, method=QuantizeMethod.KMEANS)
    palette = quantizer.fit(frames)
    print(f"Palette: {palette.n_colors} couleurs")

    indexed = quantizer.quantize(frames[0])
    print(f"Frame indexee: {indexed.shape}, max={indexed.max()}")

    loader.close()
    print("[OK] Test video existante reussi")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # Test avec video fournie
        test_existing_video(sys.argv[1])
    else:
        # Test complet avec video synthetique
        test_full_pipeline()
