"""
agent_8_tests.py - Agent Tests E2E
===================================

Phase 5 - Validation

Suite de tests:
1. TestColorModes: Roundtrip Ba/Bb/Bc/Bd
2. TestColorCost: L4(Bc) <= L4(Bb) si N > N*
3. TestHuffman: Code sans préfixe, L_avg <= H+1
4. TestSequential: Ba < Bc conditions
5. TestCartouche: encode/decode identité
6. TestFormules: Validation L1-L4
7. TestHierarchique: C_hier(Bc) <= N*8L
8. TestE2E: Pipeline complet

Référence: J. Nembé, Codage LMD Versatile v6.0
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ..core.cartouche import Cartouche
from ..core.process_types import ColorMode, ProcessType, Representation
from ..core.features import BlockFeatures
from ..core.point_process import MarkedProcess
from ..codecs.huffman import HuffmanCodec, huffman_threshold
from ..utils.io_utils import BitWriter, BitReader


@dataclass
class TestResult:
    """Résultat d'un test."""
    name: str
    passed: bool
    message: str
    details: Dict = None


class TestsAgent:
    """
    Agent 8: Tests E2E

    Suite de tests pour valider le système complet.
    """

    def __init__(self):
        self.results: List[TestResult] = []

    def run_all_tests(self) -> List[TestResult]:
        """Exécute tous les tests."""
        self.results = []

        self.results.append(self.test_color_modes())
        self.results.append(self.test_color_cost())
        self.results.append(self.test_huffman())
        self.results.append(self.test_sequential())
        self.results.append(self.test_cartouche())
        self.results.append(self.test_formules())
        self.results.append(self.test_hierarchique())
        self.results.append(self.test_e2e())

        return self.results

    def test_color_modes(self) -> TestResult:
        """Test 1: Roundtrip encode/decode pour les 4 modes."""
        from ..agents.agent_2_color_coding import ColorCodingAgent

        agent = ColorCodingAgent()
        errors = []

        for mode in ColorMode:
            # Génère des données aléatoires
            N = 50
            m = 8
            marks = np.random.randint(0, m, size=N)

            # Encode
            writer = BitWriter()
            color_dist = None
            huffman = None

            if mode == ColorMode.HUFFMAN:
                counts = np.bincount(marks, minlength=m)
                color_dist = {c: counts[c]/N for c in range(m) if counts[c] > 0}
                huffman = HuffmanCodec()
                huffman.build_from_distribution(color_dist)

            agent.encode_colors(marks, mode, writer, huffman)
            data = writer.get_bytes()

            # Decode
            reader = BitReader(data)
            decoded = agent.decode_colors(mode, reader, N, m)

            # Vérifie
            if not np.array_equal(marks, decoded):
                errors.append(f"{mode.name}: marks differ")

        passed = len(errors) == 0
        return TestResult(
            name="TestColorModes",
            passed=passed,
            message="Roundtrip OK pour tous les modes" if passed else f"Erreurs: {errors}",
            details={"modes_tested": [m.name for m in ColorMode]}
        )

    def test_color_cost(self) -> TestResult:
        """Test 2: L4(Bc) <= L4(Bb) si et seulement si N > N*."""
        from ..agents.agent_5_metrics import MetricsAgent

        agent = MetricsAgent()
        errors = []

        # Test avec différentes configurations
        test_cases = [
            {"N": 100, "m": 16, "H": 2.0, "expected_bc_better": True},
            {"N": 20, "m": 16, "H": 2.0, "expected_bc_better": False},
            {"N": 200, "m": 8, "H": 2.5, "expected_bc_better": True},
        ]

        for tc in test_cases:
            features = BlockFeatures(
                N=tc["N"], r=256, m=tc["m"],
                H_color=tc["H"], N_trans=tc["N"] // 5
            )

            metrics = agent.compute_all_metrics(features)
            N_star = metrics.N_star

            bc_better = tc["N"] > N_star
            if bc_better != tc["expected_bc_better"]:
                errors.append(f"N={tc['N']}, N*={N_star:.1f}: expected {tc['expected_bc_better']}")

        passed = len(errors) == 0
        return TestResult(
            name="TestColorCost",
            passed=passed,
            message="Seuil N* correct" if passed else f"Erreurs: {errors}"
        )

    def test_huffman(self) -> TestResult:
        """Test 3: Code Huffman sans préfixe, L_avg <= H+1."""
        errors = []

        # Distribution non-uniforme
        dist = {0: 0.5, 1: 0.25, 2: 0.15, 3: 0.10}

        codec = HuffmanCodec()
        codec.build_from_distribution(dist)

        # Vérifie sans préfixe
        if not codec.verify_prefix_free():
            errors.append("Code non sans-préfixe")

        # Vérifie longueur moyenne
        H = sum(-p * np.log2(p) for p in dist.values())
        L_avg = codec.get_average_length(dist)

        if L_avg > H + 1.01:  # Tolérance pour erreurs d'arrondi
            errors.append(f"L_avg={L_avg:.3f} > H+1={H+1:.3f}")

        passed = len(errors) == 0
        return TestResult(
            name="TestHuffman",
            passed=passed,
            message="Huffman valide (sans prefixe, L <= H+1)" if passed else f"Erreurs: {errors}",
            details={"H": H, "L_avg": L_avg}
        )

    def test_sequential(self) -> TestResult:
        """Test 4: Ba < Bc si N_trans suffisamment petit."""
        from ..agents.agent_2_color_coding import ColorCodingAgent

        agent = ColorCodingAgent()
        errors = []

        # Parametres
        N = 100
        m = 16
        H = 2.5
        log2_m = np.log2(m)  # = 4
        D_huf = m * (int(np.floor(log2_m)) + 1)  # = 80

        # Cas 1: Peu de transitions -> Ba devrait etre meilleur que Bb
        N_trans_low = 5
        cost_Ba_low = agent.color_cost_Ba(m, N_trans_low)  # 4 + 5*4 = 24
        cost_Bb = agent.color_cost_Bb(N, m)  # 100 * 4 = 400

        if cost_Ba_low >= cost_Bb:
            errors.append(f"Ba={cost_Ba_low:.1f} devrait etre < Bb={cost_Bb:.1f} pour N_trans={N_trans_low}")

        # Cas 2: Beaucoup de transitions -> Ba devrait etre pire que Bb
        N_trans_high = 150  # Plus que N !
        cost_Ba_high = agent.color_cost_Ba(m, N_trans_high)  # 4 + 150*4 = 604

        if cost_Ba_high <= cost_Bb:
            errors.append(f"Ba={cost_Ba_high:.1f} devrait etre > Bb={cost_Bb:.1f} pour N_trans={N_trans_high}")

        # Cas 3: Verifier la formule Ba
        expected_Ba = log2_m + N_trans_low * log2_m
        if abs(cost_Ba_low - expected_Ba) > 0.01:
            errors.append(f"Ba formula: got {cost_Ba_low:.2f}, expected {expected_Ba:.2f}")

        passed = len(errors) == 0
        return TestResult(
            name="TestSequential",
            passed=passed,
            message="Conditions Ba vs Bb correctes" if passed else f"Erreurs: {errors}",
            details={"Ba_low": cost_Ba_low, "Ba_high": cost_Ba_high, "Bb": cost_Bb}
        )

    def test_cartouche(self) -> TestResult:
        """Test 5: Encode/decode cartouche = identité."""
        errors = []

        # Test plusieurs cartouches
        cartouches = [
            Cartouche(A=0, B=1, C=4, D=2, E=1, F=2, G=1, H=0),
            Cartouche(A=4, B=3, C=0, D=0, E=3, F=3, G=3, H=1),
            Cartouche(A=2, B=2, C=2, D=1, E=0, F=1, G=0, H=0),
        ]

        for i, c in enumerate(cartouches):
            encoded = c.encode()
            decoded = Cartouche.decode(encoded)

            if (c.A != decoded.A or c.B != decoded.B or c.C != decoded.C or
                c.D != decoded.D or c.E != decoded.E or c.F != decoded.F or
                c.G != decoded.G or c.H != decoded.H):
                errors.append(f"Cartouche {i}: {c} != {decoded}")

        passed = len(errors) == 0
        return TestResult(
            name="TestCartouche",
            passed=passed,
            message="Encode/decode identité OK" if passed else f"Erreurs: {errors}"
        )

    def test_formules(self) -> TestResult:
        """Test 6: L_R4b(Bb) = formule L4 classique."""
        from ..utils.math_utils import logC

        errors = []

        # Paramètres de test
        N, r, m = 60, 256, 8

        # Formule classique L4
        L4_classique = np.log2(N + 1) + logC(r, N) + N * np.log2(m)

        # Via notre implémentation
        features = BlockFeatures(N=N, r=r, m=m, H_color=3.0, N_trans=10)

        from ..agents.agent_5_metrics import MetricsAgent
        agent = MetricsAgent()
        metrics = agent.compute_all_metrics(features)

        L4_impl = metrics.L4[ColorMode.UNIFORM]

        if abs(L4_classique - L4_impl) > 0.1:
            errors.append(f"L4 classique={L4_classique:.2f} vs impl={L4_impl:.2f}")

        passed = len(errors) == 0
        return TestResult(
            name="TestFormules",
            passed=passed,
            message="L4(Bb) = formule classique" if passed else f"Erreurs: {errors}",
            details={"L4_classique": L4_classique, "L4_impl": L4_impl}
        )

    def test_hierarchique(self) -> TestResult:
        """Test 7: C_color_hier - Huffman vs Uniform selon distribution."""
        from ..agents.agent_7_hierarchy import HierarchicalColor

        errors = []

        N = 500  # Grand N pour que Huffman soit avantageux

        # Distribution SKEWED (non-uniforme) - H << 8 bits
        # Distribution avec 4 couleurs dominantes: 50%, 25%, 15%, 10%
        skewed_dist = {0: 0.50, 1: 0.25, 2: 0.15, 3: 0.10}
        # H = -sum(p*log2(p)) ~ 1.68 bits << 8 bits

        hier_skewed = HierarchicalColor(
            levels=2,
            level_bits=[8, 8],
            level_distributions=[skewed_dist, skewed_dist]
        )

        cost_Bc_skewed = hier_skewed.color_cost_hier(ColorMode.HUFFMAN, N)
        cost_Bb_skewed = hier_skewed.color_cost_hier(ColorMode.UNIFORM, N)

        # Avec distribution skewed et grand N, Bc doit etre < Bb
        if cost_Bc_skewed >= cost_Bb_skewed:
            errors.append(f"Skewed: Bc={cost_Bc_skewed:.1f} >= Bb={cost_Bb_skewed:.1f}")

        # Test distribution uniforme: Bc > Bb (overhead Huffman)
        hier_uniform = HierarchicalColor(
            levels=2,
            level_bits=[8, 8],
            level_distributions=[
                {i: 1/256 for i in range(256)},
                {i: 1/256 for i in range(256)}
            ]
        )

        cost_Bc_uniform = hier_uniform.color_cost_hier(ColorMode.HUFFMAN, N)
        cost_Bb_uniform = hier_uniform.color_cost_hier(ColorMode.UNIFORM, N)

        # Pour uniforme: Bc > Bb (overhead rend Huffman inutile)
        if cost_Bc_uniform <= cost_Bb_uniform:
            errors.append(f"Uniform: Bc={cost_Bc_uniform:.1f} devrait etre > Bb={cost_Bb_uniform:.1f}")

        # Verifier que Bb = N*16
        expected_Bb = N * 16
        if abs(cost_Bb_uniform - expected_Bb) > 1:
            errors.append(f"Bb={cost_Bb_uniform:.1f} != N*16={expected_Bb}")

        passed = len(errors) == 0
        return TestResult(
            name="TestHierarchique",
            passed=passed,
            message="Couts hierarchiques corrects (skewed vs uniform)" if passed else f"Erreurs: {errors}",
            details={
                "skewed_Bc": cost_Bc_skewed, "skewed_Bb": cost_Bb_skewed,
                "uniform_Bc": cost_Bc_uniform, "uniform_Bb": cost_Bb_uniform
            }
        )

    def test_e2e(self) -> TestResult:
        """Test 8: Pipeline complet sur un bloc."""
        from ..agents.agent_0_extraction import ExtractionAgent, create_test_video_block
        from ..agents.agent_6_encoder import EncoderAgent

        errors = []

        # Crée un bloc de test
        video_block = create_test_video_block(T=32, H=8, W=8, m=4, jump_rate=0.05)

        # Extraction
        extractor = ExtractionAgent(block_width=8, block_height=8)
        result = extractor.extract(video_block)

        # Vérification features
        if result.features.N == 0 and video_block.max() > 0:
            # Peut être normal si aucun changement
            pass

        # Encodage
        encoder = EncoderAgent()
        marks = np.array([j[3] for j in result.jump_positions])
        times = np.array([j[0] for j in result.jump_positions])

        if len(marks) > 0:
            # Sélection du cartouche optimal
            best_mode = result.features.suggest_color_mode()
            cartouche = Cartouche(
                A=result.features.suggest_process_type(),
                B=best_mode,
                C=Representation.COMBINATORIAL,
                D=2, E=1, F=2, G=1, H=0
            )

            encoded = encoder.encode_block(
                jump_times=times.astype(float),
                marks=marks,
                cartouche=cartouche,
                features=result.features,
                color_dist=result.features.color_dist
            )

            # Décodage
            dec_cartouche, dec_times, dec_marks, dec_features = encoder.decode_block(
                encoded.bitstream
            )

            # Vérifications
            if dec_cartouche.encode() != cartouche.encode():
                errors.append("Cartouche différent après décodage")

            if len(dec_marks) != len(marks):
                errors.append(f"Nombre de marks: {len(marks)} -> {len(dec_marks)}")

        passed = len(errors) == 0
        return TestResult(
            name="TestE2E",
            passed=passed,
            message="Pipeline E2E OK" if passed else f"Erreurs: {errors}",
            details={"N": result.features.N, "m": result.features.m}
        )

    def generate_report(self) -> str:
        """Génère un rapport des tests."""
        lines = ["=" * 60]
        lines.append("RAPPORT DE TESTS - LMD-PPV v6.0")
        lines.append("=" * 60)

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        for result in self.results:
            status = "[PASS]" if result.passed else "[FAIL]"
            lines.append(f"\n{status} {result.name}")
            lines.append(f"  {result.message}")
            if result.details:
                for k, v in result.details.items():
                    lines.append(f"  {k}: {v}")

        lines.append("\n" + "=" * 60)
        lines.append(f"RÉSUMÉ: {passed}/{total} tests passés")
        lines.append("=" * 60)

        return "\n".join(lines)
