"""
test_pipeline.py - Tests pytest pour le pipeline LMD-PPV
=========================================================
"""

import pytest
import numpy as np

from src.core.cartouche import Cartouche
from src.core.process_types import ColorMode, ProcessType, Representation
from src.core.features import BlockFeatures
from src.codecs.huffman import HuffmanCodec
from src.utils.math_utils import logC, entropy
from src.agents.agent_0_extraction import create_test_video_block


class TestCartouche:
    """Tests pour le cartouche ABCDEFGH."""

    def test_encode_decode_identity(self):
        """Test encode/decode = identité."""
        cartouche = Cartouche(A=2, B=1, C=4, D=2, E=1, F=2, G=1, H=0)
        encoded = cartouche.encode()
        decoded = Cartouche.decode(encoded)

        assert decoded.A == cartouche.A
        assert decoded.B == cartouche.B
        assert decoded.C == cartouche.C
        assert decoded.D == cartouche.D
        assert decoded.E == cartouche.E
        assert decoded.F == cartouche.F
        assert decoded.G == cartouche.G
        assert decoded.H == cartouche.H

    def test_encode_17_bits(self):
        """Vérifie que l'encodage tient sur 17 bits."""
        cartouche = Cartouche(A=4, B=3, C=4, D=2, E=3, F=3, G=3, H=1)
        encoded = cartouche.encode()
        assert encoded < (1 << 17)

    def test_validate(self):
        """Test de validation des valeurs."""
        valid = Cartouche(A=0, B=0, C=0, D=0, E=0, F=1, G=0, H=0)
        assert valid.validate()

        invalid = Cartouche(A=5, B=0, C=0, D=0, E=0, F=1, G=0, H=0)  # A > 4
        assert not invalid.validate()


class TestHuffman:
    """Tests pour le codec Huffman."""

    def test_prefix_free(self):
        """Vérifie que le code est sans préfixe."""
        dist = {0: 0.5, 1: 0.25, 2: 0.15, 3: 0.10}
        codec = HuffmanCodec()
        codec.build_from_distribution(dist)
        assert codec.verify_prefix_free()

    def test_average_length_bound(self):
        """Vérifie L_avg ≤ H + 1."""
        dist = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
        codec = HuffmanCodec()
        codec.build_from_distribution(dist)

        H = entropy(dist)
        L_avg = codec.get_average_length(dist)

        assert L_avg <= H + 1.01  # Tolérance

    def test_roundtrip(self):
        """Test encode/decode."""
        dist = {0: 0.5, 1: 0.3, 2: 0.2}
        codec = HuffmanCodec()
        codec.build_from_distribution(dist)

        symbols = np.array([0, 1, 0, 2, 1, 0, 0, 1])

        from src.utils.io_utils import BitWriter, BitReader

        writer = BitWriter()
        codec.encode_sequence(symbols, writer)
        data = writer.get_bytes()

        reader = BitReader(data)
        decoded = codec.decode_sequence(reader, len(symbols))

        np.testing.assert_array_equal(symbols, decoded)


class TestColorCosts:
    """Tests pour les coûts couleur."""

    def test_huffman_threshold(self):
        """Vérifie le seuil N*."""
        from src.codecs.huffman import huffman_threshold

        m = 16
        H = 2.8
        D_huf = m * 5  # Approximation

        N_star = huffman_threshold(m, H, D_huf)

        # Vérification: au-delà de N*, Bc < Bb
        N_above = int(N_star) + 10
        cost_Bb = N_above * np.log2(m)
        cost_Bc = N_above * H + D_huf

        assert cost_Bc < cost_Bb

    def test_mono_no_color_cost(self):
        """Test que monochromatique a C_color = 0."""
        from src.core.point_process import MonochromaticProcess

        process = MonochromaticProcess(
            processes={0: [1.0, 2.0, 3.0]},
            r=256
        )

        for mode in ColorMode:
            assert process.color_cost(mode) == 0.0


class TestMathUtils:
    """Tests pour les utilitaires mathématiques."""

    def test_logC_symmetry(self):
        """Test symétrie C(n,k) = C(n,n-k)."""
        n, k = 100, 30
        assert abs(logC(n, k) - logC(n, n-k)) < 0.001

    def test_logC_bounds(self):
        """Test bornes de logC."""
        assert logC(10, 0) == 0.0
        assert logC(10, 10) == 0.0
        assert logC(10, 5) > 0

    def test_entropy_uniform(self):
        """Entropie d'une distribution uniforme = log(n)."""
        n = 8
        uniform = {i: 1/n for i in range(n)}
        H = entropy(uniform)
        assert abs(H - np.log2(n)) < 0.001


class TestPipeline:
    """Tests d'intégration du pipeline."""

    def test_basic_pipeline(self):
        """Test pipeline de base."""
        from src.pipeline import LMDPipeline

        video_block = create_test_video_block(T=32, H=8, W=8, m=4, jump_rate=0.05)
        pipeline = LMDPipeline(block_size=8)

        result = pipeline.process_block(video_block)

        assert result.cartouche is not None
        assert result.encoded.total_bits > 0
        assert result.compression_ratio >= 0

    def test_empty_block(self):
        """Test bloc sans changements."""
        from src.pipeline import LMDPipeline

        # Bloc constant
        video_block = np.zeros((32, 8, 8), dtype=int)
        pipeline = LMDPipeline(block_size=8)

        result = pipeline.process_block(video_block)

        assert result.extraction.features.N == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
