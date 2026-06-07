"""
Video module - Integration avec fichiers video reels
"""

from .loader import VideoLoader
from .quantizer import ColorQuantizer
from .encoder import VideoEncoder
from .decoder import VideoDecoder
from .fast_encoder import FastVideoEncoder, fast_encode

__all__ = [
    'VideoLoader', 'ColorQuantizer',
    'VideoEncoder', 'VideoDecoder',
    'FastVideoEncoder', 'fast_encode'
]
