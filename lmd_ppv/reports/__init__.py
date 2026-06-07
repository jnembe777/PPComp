"""
reports - Generation de rapports
=================================

- generator: Generateur principal
- charts: Graphiques matplotlib
- tables: Tableaux formates
- statistics: Tests statistiques

Reference: J. Nembe, Codage LMD Versatile v6.0
"""

from .generator import ReportGenerator, ReportConfig
from .charts import ChartGenerator
from .tables import TableGenerator
from .statistics import StatisticalAnalysis

__all__ = [
    'ReportGenerator', 'ReportConfig',
    'ChartGenerator', 'TableGenerator', 'StatisticalAnalysis'
]
