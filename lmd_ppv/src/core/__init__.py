"""
Core module - Structures de données et algorithmes fondamentaux
"""

from .cartouche import Cartouche
from .process_types import ProcessType, ColorMode, Representation, CompressionMode
from .point_process import MarkedProcess, MonochromaticProcess, VectorialProcess, MarkovianProcess
from .features import BlockFeatures
from .lookup_tables import (
    predict_dimension_C,
    select_representation_algo1,
    get_gamma_bounds,
    get_delta_threshold,
    get_lambda_bound,
    RepresentationSelection,
)
