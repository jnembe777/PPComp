"""
pipeline.py - Pipeline ABCDEFGH Complet
========================================

Pipeline de compression vidéo LMD-PPV intégrant les 9 agents.

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .core.cartouche import Cartouche
from .core.process_types import ProcessType, ColorMode, Representation, CompressionMode
from .core.features import BlockFeatures

from .agents.agent_0_extraction import ExtractionAgent, ExtractionResult
from .agents.agent_1_classification import ClassificationAgent, ClassificationResult
from .agents.agent_2_color_coding import ColorCodingAgent, ColorCodingResult
from .agents.agent_3_structures import StructuresAgent, ProcessData
from .agents.agent_4_precoding import PrecodingAgent, PrecodingResult
from .agents.agent_5_metrics import MetricsAgent, MetricsResult
from .agents.agent_6_encoder import EncoderAgent, EncodedBlock
from .agents.agent_7_hierarchy import HierarchyAgent, SpatialProcessTree
from .agents.agent_8_tests import TestsAgent


@dataclass
class PipelineResult:
    """Résultat complet du pipeline."""
    # Cartouche optimal
    cartouche: Cartouche

    # Résultats des agents
    extraction: ExtractionResult
    classification: ClassificationResult
    color_coding: ColorCodingResult
    precoding: PrecodingResult
    metrics: MetricsResult
    encoded: EncodedBlock

    # Statistiques
    compression_ratio: float
    total_bits: int
    processing_time_ms: float


class LMDPipeline:
    """
    Pipeline de compression LMD-PPV.

    Orchestre les 9 agents pour compresser un bloc vidéo:
    1. Extraction bitwise
    2. Classification du type
    3. Codage couleur (dim B)
    4. Structures de données
    5. Précodage R1-R4b (dim C)
    6. Métriques LMD
    7. Encodeur (dim D)
    8. Hiérarchie + Zoom
    9. Tests (validation)
    """

    def __init__(self, block_size: int = 16):
        """
        Initialise le pipeline.

        Args:
            block_size: Taille des blocs (16x16 par défaut)
        """
        self.block_size = block_size

        # Agents
        self.agent_0_extraction = ExtractionAgent(block_size, block_size)
        self.agent_1_classification = ClassificationAgent()
        self.agent_2_color_coding = ColorCodingAgent()
        self.agent_3_structures = StructuresAgent()
        self.agent_4_precoding = PrecodingAgent()
        self.agent_5_metrics = MetricsAgent()
        self.agent_6_encoder = EncoderAgent()
        self.agent_7_hierarchy = HierarchyAgent()
        self.agent_8_tests = TestsAgent()

    def process_block(self, video_block: np.ndarray) -> PipelineResult:
        """
        Traite un bloc vidéo complet.

        Args:
            video_block: Bloc (T, H, W) avec indices de couleur

        Returns:
            PipelineResult avec tous les résultats
        """
        import time
        start_time = time.time()

        # Phase 1: Extraction + Classification
        extraction = self.agent_0_extraction.extract(video_block)
        features = extraction.features

        classification = self.agent_1_classification.classify(features)

        # Phase 2: Codage couleur + Structures + Précodage
        marks = np.array([j[3] for j in extraction.jump_positions])
        color_coding = self.agent_2_color_coding.analyze(
            marks=marks,
            m=features.m,
            color_dist=features.color_dist
        )

        jump_times = np.array([j[0] for j in extraction.jump_positions], dtype=float)
        precoding = self.agent_4_precoding.analyze(
            jump_times=jump_times,
            N=features.N,
            r=features.r,
            m=features.m,
            features=features
        )

        # Phase 3: Métriques + Encodage
        metrics = self.agent_5_metrics.compute_all_metrics(features)

        # Construction du cartouche optimal
        cartouche = self._build_optimal_cartouche(
            classification, color_coding, precoding, features
        )

        # Encodage
        encoded = self.agent_6_encoder.encode_block(
            jump_times=jump_times,
            marks=marks,
            cartouche=cartouche,
            features=features,
            color_dist=features.color_dist
        )

        # Statistiques
        raw_bits = video_block.size * 8  # Estimation
        compression_ratio = raw_bits / encoded.total_bits if encoded.total_bits > 0 else 0

        processing_time = (time.time() - start_time) * 1000

        return PipelineResult(
            cartouche=cartouche,
            extraction=extraction,
            classification=classification,
            color_coding=color_coding,
            precoding=precoding,
            metrics=metrics,
            encoded=encoded,
            compression_ratio=compression_ratio,
            total_bits=encoded.total_bits,
            processing_time_ms=processing_time
        )

    def _build_optimal_cartouche(
        self,
        classification: ClassificationResult,
        color_coding: ColorCodingResult,
        precoding: PrecodingResult,
        features: BlockFeatures
    ) -> Cartouche:
        """Construit le cartouche optimal."""
        return Cartouche(
            A=classification.process_type,
            B=color_coding.mode if not classification.is_mono_better else ColorMode.UNIFORM,
            C=precoding.best_repr,
            D=CompressionMode.MDL,
            E=1,  # Splines par défaut
            F=2,  # 16 bits
            G=1,  # 8x8 px
            H=0   # Continu
        )

    def process_video(
        self,
        video: np.ndarray,
        use_quadtree: bool = False
    ) -> List[PipelineResult]:
        """
        Traite une vidéo complète par blocs.

        Args:
            video: Vidéo (T, H, W)
            use_quadtree: Utiliser la subdivision quadtree

        Returns:
            Liste de résultats par bloc
        """
        T, H, W = video.shape
        results = []

        for y in range(0, H, self.block_size):
            for x in range(0, W, self.block_size):
                # Extrait le bloc
                block = video[
                    :,
                    y:min(y + self.block_size, H),
                    x:min(x + self.block_size, W)
                ]

                # Traite
                result = self.process_block(block)
                results.append(result)

        return results

    def run_tests(self) -> str:
        """Exécute la suite de tests."""
        self.agent_8_tests.run_all_tests()
        return self.agent_8_tests.generate_report()


def run_demo():
    """Démonstration du pipeline."""
    from .agents.agent_0_extraction import create_test_video_block

    print("=" * 60)
    print("DÉMONSTRATION - Pipeline LMD-PPV v6.0")
    print("=" * 60)

    # Crée un bloc de test
    print("\n1. Création d'un bloc vidéo de test...")
    video_block = create_test_video_block(T=64, H=16, W=16, m=8, jump_rate=0.08)
    print(f"   Dimensions: {video_block.shape}")

    # Pipeline
    print("\n2. Exécution du pipeline...")
    pipeline = LMDPipeline(block_size=16)
    result = pipeline.process_block(video_block)

    # Résultats
    print("\n3. Résultats:")
    print(f"   Cartouche: {result.cartouche.to_string()}")
    print(f"   Type processus: {ProcessType(result.cartouche.A).name}")
    print(f"   Mode couleur: {ColorMode(result.cartouche.B).name}")
    print(f"   Représentation: {Representation(result.cartouche.C).name}")
    print(f"\n   Features:")
    print(f"     N (sauts): {result.extraction.features.N}")
    print(f"     m (couleurs): {result.extraction.features.m}")
    print(f"     H_color: {result.extraction.features.H_color:.3f} bits")
    print(f"     N_trans: {result.extraction.features.N_trans}")
    print(f"\n   Codage:")
    print(f"     Header: {result.encoded.header_bits} bits")
    print(f"     Data: {result.encoded.data_bits} bits")
    print(f"     Total: {result.encoded.total_bits} bits")
    print(f"     Ratio: {result.compression_ratio:.2f}x")
    print(f"\n   Temps: {result.processing_time_ms:.1f} ms")

    # Tests
    print("\n4. Exécution des tests...")
    report = pipeline.run_tests()
    print(report)

    return result


if __name__ == "__main__":
    run_demo()
