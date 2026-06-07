"""
statistics.py - Tests statistiques
===================================

Tests de significativite pour les comparaisons de codecs.

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class TestResult:
    """Resultat d'un test statistique."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # p < alpha
    alpha: float = 0.05
    interpretation: str = ""

    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'significant': self.significant,
            'alpha': self.alpha,
            'interpretation': self.interpretation,
        }


class StatisticalAnalysis:
    """
    Analyse statistique des resultats de benchmark.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialise l'analyseur.

        Args:
            alpha: Seuil de significativite
        """
        self.alpha = alpha

    def paired_t_test(
        self,
        values1: List[float],
        values2: List[float],
        codec1: str = "Codec1",
        codec2: str = "Codec2"
    ) -> TestResult:
        """
        Test t apparie pour comparer deux codecs.

        Args:
            values1: Valeurs du codec 1
            values2: Valeurs du codec 2
            codec1: Nom du codec 1
            codec2: Nom du codec 2

        Returns:
            TestResult
        """
        if not SCIPY_AVAILABLE:
            return TestResult(
                test_name="Paired t-test",
                statistic=0,
                p_value=1,
                significant=False,
                interpretation="scipy not available"
            )

        if len(values1) != len(values2):
            return TestResult(
                test_name="Paired t-test",
                statistic=0,
                p_value=1,
                significant=False,
                interpretation="Sample sizes must match"
            )

        stat, p_value = stats.ttest_rel(values1, values2)
        significant = p_value < self.alpha

        mean_diff = np.mean(np.array(values1) - np.array(values2))

        if significant:
            if mean_diff > 0:
                interp = f"{codec1} significantly better than {codec2}"
            else:
                interp = f"{codec2} significantly better than {codec1}"
        else:
            interp = f"No significant difference between {codec1} and {codec2}"

        return TestResult(
            test_name="Paired t-test",
            statistic=stat,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            interpretation=interp
        )

    def wilcoxon_test(
        self,
        values1: List[float],
        values2: List[float],
        codec1: str = "Codec1",
        codec2: str = "Codec2"
    ) -> TestResult:
        """
        Test de Wilcoxon (non-parametrique).

        Args:
            values1: Valeurs du codec 1
            values2: Valeurs du codec 2
            codec1: Nom du codec 1
            codec2: Nom du codec 2

        Returns:
            TestResult
        """
        if not SCIPY_AVAILABLE:
            return TestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0,
                p_value=1,
                significant=False,
                interpretation="scipy not available"
            )

        try:
            stat, p_value = stats.wilcoxon(values1, values2)
        except ValueError as e:
            return TestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0,
                p_value=1,
                significant=False,
                interpretation=str(e)
            )

        significant = p_value < self.alpha
        median_diff = np.median(np.array(values1) - np.array(values2))

        if significant:
            if median_diff > 0:
                interp = f"{codec1} significantly better than {codec2}"
            else:
                interp = f"{codec2} significantly better than {codec1}"
        else:
            interp = f"No significant difference between {codec1} and {codec2}"

        return TestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=stat,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            interpretation=interp
        )

    def anova_test(
        self,
        groups: Dict[str, List[float]]
    ) -> TestResult:
        """
        ANOVA a un facteur pour comparer plusieurs codecs.

        Args:
            groups: Dict codec -> values

        Returns:
            TestResult
        """
        if not SCIPY_AVAILABLE:
            return TestResult(
                test_name="One-way ANOVA",
                statistic=0,
                p_value=1,
                significant=False,
                interpretation="scipy not available"
            )

        group_values = list(groups.values())

        if len(group_values) < 2:
            return TestResult(
                test_name="One-way ANOVA",
                statistic=0,
                p_value=1,
                significant=False,
                interpretation="Need at least 2 groups"
            )

        stat, p_value = stats.f_oneway(*group_values)
        significant = p_value < self.alpha

        if significant:
            interp = "Significant differences exist between codecs"
        else:
            interp = "No significant differences between codecs"

        return TestResult(
            test_name="One-way ANOVA",
            statistic=stat,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            interpretation=interp
        )

    def kruskal_wallis_test(
        self,
        groups: Dict[str, List[float]]
    ) -> TestResult:
        """
        Test de Kruskal-Wallis (ANOVA non-parametrique).

        Args:
            groups: Dict codec -> values

        Returns:
            TestResult
        """
        if not SCIPY_AVAILABLE:
            return TestResult(
                test_name="Kruskal-Wallis H-test",
                statistic=0,
                p_value=1,
                significant=False,
                interpretation="scipy not available"
            )

        group_values = list(groups.values())

        if len(group_values) < 2:
            return TestResult(
                test_name="Kruskal-Wallis H-test",
                statistic=0,
                p_value=1,
                significant=False,
                interpretation="Need at least 2 groups"
            )

        stat, p_value = stats.kruskal(*group_values)
        significant = p_value < self.alpha

        if significant:
            interp = "Significant differences exist between codecs"
        else:
            interp = "No significant differences between codecs"

        return TestResult(
            test_name="Kruskal-Wallis H-test",
            statistic=stat,
            p_value=p_value,
            significant=significant,
            alpha=self.alpha,
            interpretation=interp
        )

    def effect_size_cohens_d(
        self,
        values1: List[float],
        values2: List[float]
    ) -> float:
        """
        Calcule l'effet de taille (Cohen's d).

        Args:
            values1: Valeurs du groupe 1
            values2: Valeurs du groupe 2

        Returns:
            Cohen's d
        """
        n1, n2 = len(values1), len(values2)
        mean1, mean2 = np.mean(values1), np.mean(values2)
        var1, var2 = np.var(values1, ddof=1), np.var(values2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    def confidence_interval(
        self,
        values: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calcule l'intervalle de confiance.

        Args:
            values: Valeurs
            confidence: Niveau de confiance

        Returns:
            (lower, upper)
        """
        n = len(values)
        mean = np.mean(values)
        se = np.std(values, ddof=1) / np.sqrt(n)

        if SCIPY_AVAILABLE:
            h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        else:
            # Approximation z
            z = 1.96 if confidence == 0.95 else 2.576
            h = se * z

        return (mean - h, mean + h)

    def compare_all_codecs(
        self,
        data: Dict[str, List[float]],
        metric: str = "PSNR"
    ) -> Dict:
        """
        Compare tous les codecs entre eux.

        Args:
            data: Dict codec -> values
            metric: Nom de la metrique

        Returns:
            Resultats de tous les tests
        """
        results = {
            'metric': metric,
            'global_test': None,
            'pairwise_tests': [],
            'effect_sizes': {},
            'confidence_intervals': {},
        }

        # Test global (ANOVA ou Kruskal-Wallis)
        results['global_test'] = self.kruskal_wallis_test(data)

        # Tests apparies
        codecs = list(data.keys())
        for i, codec1 in enumerate(codecs):
            for codec2 in codecs[i+1:]:
                test = self.wilcoxon_test(
                    data[codec1], data[codec2],
                    codec1, codec2
                )
                results['pairwise_tests'].append({
                    'codec1': codec1,
                    'codec2': codec2,
                    'result': test.to_dict()
                })

                # Effect size
                d = self.effect_size_cohens_d(data[codec1], data[codec2])
                results['effect_sizes'][f"{codec1}_vs_{codec2}"] = d

        # Intervalles de confiance
        for codec, values in data.items():
            ci = self.confidence_interval(values)
            results['confidence_intervals'][codec] = {
                'lower': ci[0],
                'upper': ci[1],
                'mean': np.mean(values)
            }

        return results

    def generate_report(
        self,
        data: Dict[str, List[float]],
        metric: str = "PSNR"
    ) -> str:
        """
        Genere un rapport statistique.

        Args:
            data: Dict codec -> values
            metric: Nom de la metrique

        Returns:
            Rapport formate
        """
        results = self.compare_all_codecs(data, metric)

        lines = [
            "=" * 60,
            f"STATISTICAL ANALYSIS - {metric}",
            "=" * 60,
            "",
            "GLOBAL TEST",
            "-" * 40,
        ]

        if results['global_test']:
            gt = results['global_test']
            lines.append(f"Test: {gt.test_name}")
            lines.append(f"Statistic: {gt.statistic:.4f}")
            lines.append(f"P-value: {gt.p_value:.4f}")
            lines.append(f"Significant (alpha={gt.alpha}): {gt.significant}")
            lines.append(f"Interpretation: {gt.interpretation}")

        lines.extend(["", "PAIRWISE COMPARISONS", "-" * 40])

        for pair in results['pairwise_tests']:
            r = pair['result']
            lines.append(f"\n{pair['codec1']} vs {pair['codec2']}:")
            lines.append(f"  P-value: {r['p_value']:.4f}")
            lines.append(f"  Significant: {r['significant']}")
            lines.append(f"  {r['interpretation']}")

            es_key = f"{pair['codec1']}_vs_{pair['codec2']}"
            if es_key in results['effect_sizes']:
                d = results['effect_sizes'][es_key]
                effect_interp = "negligible" if abs(d) < 0.2 else \
                               "small" if abs(d) < 0.5 else \
                               "medium" if abs(d) < 0.8 else "large"
                lines.append(f"  Effect size (Cohen's d): {d:.3f} ({effect_interp})")

        lines.extend(["", "CONFIDENCE INTERVALS (95%)", "-" * 40])

        for codec, ci in results['confidence_intervals'].items():
            lines.append(f"{codec}: {ci['mean']:.2f} [{ci['lower']:.2f}, {ci['upper']:.2f}]")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
